from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Protocol, Tuple, Union

from PIL.Image import Image
import timm
import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import torchvision.transforms.functional as TVF
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.utils import TensorType


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...


# === Custom Torchvision Image Transforms ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


# === Abstract Base Class for arbitrary Vision Backbones ===
class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size

        # Instance attributes for a Vision Backbone
        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]:
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def num_patches(self) -> int:
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        ...


# === Image Processing ===
def letterbox_pad_transform(image: Image, padding_fill_value: Tuple[int, int, int]) -> Image:
    """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; this transform will be
        created by TIMM, and edited to follow our custom `image_resize_strategy` logic.
        @param use_fused_vision_backbone: Boolean indicating single or fused (dual) vision backbone
        @param image_resize_strategy: Prismatic image resize strategy in < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: `data_cfg`] Input image size as tuple (channels, width, height)
        @param interpolation: [TIMM :: `data_cfg`] Interpolation as string (default: "bicubic")
        @param mean: [TIMM :: `data_cfg`] Normalization mean as float tuple (or two-tuple if `fused_backbone`)
        @param std: [TIMM :: `data_cfg`] Normalization std as float tuple (or two-tuple if `fused_backbone`)
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # Handle `None` default values
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        # TIMM `data_cfg` Parameters
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # Grab torchvision transforms via TIMM =>> need to parse for specific "functional" transform values!
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
                is_training=False,  # No image augmentations when loading the transform!
            )

            # [Validation] Ensure appropriate transform structure, expected sizes
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.
            #   => Instead, we're going to parse the transform and call "torchvision.transforms.functional" (`tvf`)
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        # Dispatch **kwargs to super()
        super().__init__(**kwargs)

    def apply_transform(self, img: Image) -> torch.Tensor:
        """Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image, List[Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image instances for simplicity.
        @param images: A (batch of) PIL.Image instance(s) to preprocess.
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # Return BatchFeature =>> note that for compatibility, constructor expects Dict[str, np.ndarray], so we convert
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image, List[Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)
