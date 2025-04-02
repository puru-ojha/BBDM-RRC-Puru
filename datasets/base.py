from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


class ImagePathDataset(Dataset):
    def __init__(
        self,
        image_paths,
        target_mask_paths=None,
        image_size=(256, 256),
        flip=False,
        to_normal=False,
    ):
        self.image_size = image_size
        self.image_paths = image_paths
        self.target_mask_paths = target_mask_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal  # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        img_path = self.image_paths[index]
        mask_path = self.target_mask_paths[index] if self.target_mask_paths else None

        image = None
        mask = None
        try:
            image = Image.open(img_path)
            if mask_path is not None:
                mask = Image.open(mask_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        if mask and (not mask.mode == "L"):
            mask = mask.convert("L")

        image = transform(image)
        if mask is not None:
            mask = transform(mask)

        if self.to_normal:
            image = (image - 0.5) * 2.0
            image.clamp_(-1.0, 1.0)

            if mask is not None:
                mask = (mask - 0.5) * 2.0
                mask.clamp_(-1.0, 1.0)

        image_name = Path(img_path).stem
        mask_name = Path(mask_path).stem if mask_path else None

        # NOTE: mask and name are None in case of valid and test sets
        return image, image_name, mask, mask_name
