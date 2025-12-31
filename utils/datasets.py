from typing import Union, List, Tuple, Dict, Callable, Any, Sequence
import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as TT
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# project files


def get_randomize_pipe():
    return TT.RandomHorizontalFlip(p = 0.2)


# TODO: update for the new dataset structure
class ImageDataset(Dataset):
    def __init__(self, root: str, mode: str = 'train', transform: List[callable] = None):
        """ Initialize the dataset with the root directory and mode (train/test).
            Args:
                root (str): The root directory of the dataset.
                mode (str): The mode in which the dataset is used ('train' or 'test').
                transform (List[callable]): Optional list of transformations to apply.
        """
        self.transform = TT.Compose(transform) if transform else None #TT.Compose([TT.ToTensor()])
        self.files_A = sorted(glob.glob(os.path.join(root, f'{mode}/clean/*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root, f'{mode}/dirty/*.*')))
        # TODO: this is a dumb implementation (from ChatGPT) - incorporate the dataloaders later and wrap the shorter one in itertools.cycle
        if len(self.files_A) > len(self.files_B):
            self.files_B = self.files_B * (len(self.files_A) // len(self.files_B) + 1)
        elif len(self.files_B) > len(self.files_A):
            self.files_A = self.files_A * (len(self.files_B) // len(self.files_A) + 1)

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """ Load and return a pair of images (A and B) at the given index.
            Args:
                index (int): The index of the images to load.
            Returns:
                dict: A dictionary containing 'A' and 'B' images.
        """
        image_A = read_image(self.files_A[index], ImageReadMode.RGB)
        image_B = read_image(self.files_B[index], ImageReadMode.RGB)
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
        return {'A': image_A, 'B': image_B}




def get_dataloader(
        paths: Dict[str, List[os.PathLike]],         # dictionary (w/ keys 'A' and 'B') of paths specifying the data to be fed into the DataLoader
        train_mode: bool,                            # bool indicating running in 'train' or 'test' modes - affects transformations and shuffling
        device: torch.device,                        # device to initialize the dataloader's Generator object with - FIXME: unused for now
        batch_size: int,                             # the size of minibatches (number of images) returned by the dataloader's iterator
        out_size: Union[int, Tuple[int]],            # the shape to resize the spatial dimensions of all images and masks (assuming square images only)
        metadata_dict: Dict[str, Any]|None = None,   # a large dictionary of metadata loaded from a JSON file which contains information about each image
        sample_prop: float = None,                   # a sample proportion <= 1, which when not None, is used to sample a smaller portion of the dataset with a SubsetRandomSampler
        using_hf = False                             # whether to use a HuggingFace dataset or custom dataset
    ) -> DataLoader:
    """ specifies the shuffling, sampling, and transformations applied to data loaded into a pytorch DataLoader, created and returned to the calling location
        Returns:
            loader : pytorch DataLoader (https://datagy.io/pytorch-dataloader/) using global batch_size, shuffling flag, and optional generator
    """
    # ? NOTE: generator parameter generates random indices and a base_seed for multiprocessing; may be useless without giving an explicit num_workers arg, w/ loader running in the main process
    shuffle = train_mode
    sampler = None
    out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
    dataset_args = [paths, out_size, train_mode, metadata_dict]
    dataset = DatasetBase(*dataset_args) if using_hf else DatasetBase(*dataset_args)
    # sampler only works with debug mode for now
    if sample_prop is not None:
        shuffle = False
        loader_size = len(dataset)
        sampler = SubsetRandomSampler(torch.randperm(loader_size)[:int(loader_size*sample_prop)])
    ''' note on num_workers: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
            "...you may be able to move data to gpu in your collate_fn function. Assuming that this function happens in parallel as well, it could speed things up.
            The potential problem being that you now have >= n_workers batches on the gpu so memory could be restricted"
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=True, num_workers=4, pin_memory_device=device if device=="cuda" else "")




# FIXME: it's possible I may need to use IterableDataset as the subclass to work correctly with custom loaders
class DatasetBase(Dataset):
    def __init__(self,
                 path_dict: Dict[str, List[str]],
                 out_size: Union[int, Tuple[int]],
                 train_mode: bool = False,
                 #augment_manager: AugmentationManager = None,
                 metadata_dict: Dict[str, Any] = None):
        super().__init__()
        self.OUTPUT_SIZE: Tuple[int,int] = out_size if isinstance(out_size, tuple) else (out_size, out_size)
        #self.augment_manager: AugmentationManager = augment_manager
        self.train_mode = train_mode
        self.preprocessing: Sequence[Callable] = TT.Compose([
            TT.Resize(self.OUTPUT_SIZE),
            TT.ToDtype(torch.float32),
        ])
        # ? NOTE: Should add ToPureTensor at the end of any post-processing pipes to change from TVTensor to pure tensor
        self.img_paths: List[os.PathLike] = path_dict["img"]
        self.mask_paths: List[os.PathLike] = path_dict["mask"]
        self.num_images: int = len(self.img_paths)
        if metadata_dict is not None:
            # strip the occlusion scores only
            self.metadata: Dict[str, Any] = metadata_dict
        # provide new view of data to extend dataset - for now, this is just TT.RandomHorizontalFlip(p = 0.2) but I wanted to keep it a little more abstracted
        self.reflector: Sequence[Callable] = get_randomize_pipe() # not needed but it cuts down redundant imports like torchvision.transforms.v2

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Union[str, Union['tv_tensors.Image', 'tv_tensors.Mask']]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        sample = {
            'img': self.preprocessing['img'](read_image(img_path, ImageReadMode.RGB)),
            'mask': read_image(mask_path, ImageReadMode.UNCHANGED),
        }
        if self.train_mode:
            sample: dict = self.reflector(sample)
        rgb_mask_path = os.path.join(os.path.dirname(mask_path), "..", "rgbLabels", os.path.basename(mask_path))
        sample.update({"img_path": img_path, "mask_path": rgb_mask_path, "augmentation": "none"})
        return sample

    def __len__(self) -> int:
        return self.num_images
