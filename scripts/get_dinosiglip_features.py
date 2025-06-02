from PIL import Image
import torch

from lumos.utils.dinosiglip_vit import DinoSigLIPViTBackbone
from lumos.utils.prismatic import PrismaticImageProcessor


def get_dinosiglip_features(pil_image: Image) -> torch.Tensor:
    dinosig = DinoSigLIPViTBackbone(
        vision_backbone_id="dinosiglip-vit-so-224px", image_resize_strategy="resize-naive", default_image_size=224
    )
    image_processor = PrismaticImageProcessor(
        use_fused_vision_backbone=True,
        image_resize_strategy="resize-naive",
        input_sizes=[[3, 224, 224], [3, 224, 224]],
        interpolations=["bicubic", "bicubic"],
        means=[[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]],
        stds=[[0.229, 0.224, 0.225], [0.229, 0.224, 0.225]],
    )
    pixel_values = image_processor(pil_image, return_tensors="pt")["pixel_values"]
    img_dino, img_siglip = torch.split(pixel_values, [3, 3], dim=1)
    pixel_values_dict = {"dino": img_dino, "siglip": img_siglip}
    feat = dinosig(pixel_values_dict)
    return feat


if __name__ == "__main__":
    input_image = Image.open("/path/to/image/file")  # Replace with your image path
    pil_image_list = [input_image]
    feat = get_dinosiglip_features(pil_image_list)
    print(feat.shape)
