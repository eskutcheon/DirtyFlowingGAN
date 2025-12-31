import math
from typing import Optional, Callable, Union, Tuple, Any
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
from transformers.modeling_outputs import SemanticSegmenterOutput



class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        #self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty(1, requires_grad=True))  # Lipschitz constant
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        # He initialization for better convergence
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization
        # might need to add an unsqueeze(0) to self.c.data:
        # self.c.data = W_abs_row_sum.max().unsqueeze(0)

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        # Handle multi-dimensional inputs (batch size, seq_len, features)
        # if input.ndim > 2:
        #     input = input.flatten(0, -2)  # Flatten spatial and batch dims into one
        # compute Lipschitz constant lipc
        lipc = self.get_lipschitz_constant()
        scale = lipc / torch.abs(self.weight).sum(1) # normalize with l1 norm of weights
        scale = torch.clamp(scale, max=1.0) # prevent scaling weights by a factor larger than 1
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


# TODO: refactor to be an actual MLP with multiple layers - replacing the intitialization in the constructor of SegformerLipschitzDecoderHead
class LipschitzSegformerMLP(torch.nn.Module):
    """ Linear Embedding with Lipschitz-regularized linear layer. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = LipschitzLinear(input_dim, output_dim)

    def get_lipschitz_loss(self):
        # Return the Lipschitz constant of the projection layer
        return self.proj.get_lipschitz_constant()

    def forward(self, hidden_states):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, N, C)
        hidden_states = self.proj(hidden_states)  # Apply the Lipschitz linear layer
        return hidden_states


class SegformerLipschitzDecoderHead(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        #^ UPDATE: replaced the SegformerMLP instances with LipschitzSegformerMLP (exclusively for the decoder layers, not the other feedforward layers)
        mlps = []
        for i in range(config.num_encoder_blocks): # default num_encoder_blocks is 4
            mlp = LipschitzSegformerMLP(
                input_dim=config.hidden_sizes[i],
                output_dim=config.decoder_hidden_size
            )
            mlps.append(mlp)
        # essentially the only change to the original SegformerDecodeHead is overwriting self.linear_c below
        self.linear_c = torch.nn.ModuleList(mlps)

    def get_lipschitz_loss(self):
        lipschitz_constants = torch.tensor([mlp.get_lipschitz_loss() for mlp in self.linear_c])
        # return numerically stable geometric mean of the Lipschitz constants
        loss = torch.exp(torch.mean(torch.log(lipschitz_constants), dim=0))
        #print("lipschitz loss (geometric mean of constants) before scaling: ", loss)
        return loss



class PartitionedSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        #^ added the following to overwrite the original decode_head with the Lipschitz version
        self.decode_head = SegformerLipschitzDecoderHead(config)

    # still wondering if I might as well use the original segformer since I can query outputs.last_hidden_state as needed in MADAug
        # the task model isn't called anywhere else in the MADAug code since I replaced the old usage with returning the mixed images then calling forward()
    def encode(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        """if torch.any(torch.isnan(encoder_hidden_states[-1])):
            raise ValueError("NaNs detected in encoder hidden states in segformer_custom.py")"""
        #print("CHECKING: is last_hidden_state the same as encoder_hidden_states[-1] : ", torch.allclose(outputs.last_hidden_state, encoder_hidden_states[-1]))
        """if output_attentions is not None:
            print("output_attentions shape: ", output_attentions)"""
        return encoder_hidden_states, outputs

    def partial_decoding(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        """ partial decoding mostly copied from the original implementation Decoder head of Segformer up until MLP decoding - used to retrieve late-stage feature representations """
        batch_size = encoder_hidden_states[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.decode_head.linear_c):
            if self.decode_head.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                #print("encoder_hidden_state shape in `if` block before reshape: ", encoder_hidden_state.shape)
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )
                #print("reshaped encoder_hidden_state shape in `if` block of SegformerDecodeHead: ", encoder_hidden_state.shape)
            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            #print("height and width in next step: ", (height, width))
            encoder_hidden_state = mlp(encoder_hidden_state)
            #print("mlp(encoder_hidden_state) shape: ", tuple(encoder_hidden_state.shape))
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            #print("encoder_hidden_state.permute(0, 2, 1) shape: ", tuple(encoder_hidden_state.shape))
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            #print("encoder_hidden_state after reshape: ", tuple(encoder_hidden_state.shape))
            # upsampling
            encoder_hidden_state = torch.nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            #print("encoder_hidden_state shape after interpolation: ", tuple(encoder_hidden_state.shape))
            all_hidden_states += (encoder_hidden_state,)
        hidden_states = torch.cat(all_hidden_states[::-1], dim=1)
        #print("torch.cat(all_hidden_states[::-1], dim=1) shape: ", tuple(hidden_states.shape))
        hidden_states = self.decode_head.linear_fuse(hidden_states)
        #print("hidden_states.shape (after linear fusion of concatenated states): ", tuple(hidden_states.shape))
        hidden_states = self.decode_head.batch_norm(hidden_states)
        #print("hidden_states.shape (after batch_norm): ", tuple(hidden_states.shape))
        hidden_states = self.decode_head.activation(hidden_states)
        #hidden_states = self.decode_head.dropout(hidden_states)
        return hidden_states


    def get_fused_hidden_states(self,
                                pixel_values: torch.FloatTensor,
                                output_attentions: Optional[bool] = None,
                                output_hidden_states: Optional[bool] = None,
                                return_dict: Optional[bool] = None):
        return self.partial_decoding(self.encode(pixel_values, output_attentions, output_hidden_states, return_dict)[0])

    def decode(self, encoder_hidden_states):
        logits = self.decode_head(encoder_hidden_states)
        return logits

    def compute_loss(self, logits, labels):
        # upsample logits to the images' original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = None
        if self.config.num_labels > 1:
            loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            loss = loss_fct(upsampled_logits, labels.long())
        elif self.config.num_labels == 1:
            valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
            loss_fct = BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
            loss = (loss * valid_mask).mean()
        return loss


    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Any:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
                Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
            Returns: SemanticSegmenterOutput
            Examples:
            ```python
            >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            >>> from PIL import Image
            >>> import requests
            >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> inputs = image_processor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
            >>> list(logits.shape)
            [1, 150, 128, 128]
            ```
        """
        # ? NOTE: self.config.use_return_dict default is True from transformers.PretrainedConfig
            # https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/configuration#transformers.PretrainedConfig
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 1. Encoding
        encoder_hidden_states, outputs = self.encode(
            pixel_values, output_attentions, output_hidden_states, return_dict
        )
        # for idx, state in enumerate(encoder_hidden_states):
        #     print(f"hidden state for encoder block {idx}: {state.shape}")  # Debugging line
        # 2. Decoding
        logits = self.decode(encoder_hidden_states)
        # 3. Loss calculation
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        # 4. Return output
        if not return_dict:
            output = (logits,) + outputs[1:] if output_hidden_states else (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        # from transformers.modeling_outputs, this is literally just a simple dataclass with no class methods
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )