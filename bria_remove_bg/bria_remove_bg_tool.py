import pathlib

import numpy as np
import torch
from invokeai.app.services.config.config_default import get_config
from PIL import Image

from .bria_rmbg import BriaRMBG
from .utils import postprocess_image, preprocess_image

config = get_config()


def resize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


class BriaRMBGTool:
    def __init__(self, model: BriaRMBG, device: torch.device) -> None:
        self.network = model
        self.device = device

    @staticmethod
    def load_model(model_path: pathlib.Path, device: torch.device) -> BriaRMBG:
        network = BriaRMBG()
        network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        network.eval()
        return network

    def remove_background(self, image: Image.Image) -> Image.Image:
        model_size = [1024, 1024]

        image = image.convert("RGB") if image.mode != "RGB" else image  # Ensure image is RGB
        image_width, image_height = image.size

        # Resize image to match model sizing requirements
        image = image.resize(tuple(model_size), Image.BILINEAR)

        # Process the image
        np_image = np.array(image)

        tensor_image = preprocess_image(np_image, model_size)

        self.network.to(self.device.type)
        tensor_image = tensor_image.to(self.device)

        bg_removed_tensor = self.network(tensor_image)
        tensor_image = postprocess_image(bg_removed_tensor[0][0], (image_height, image_width))

        final_result_image = Image.fromarray(tensor_image)

        # Create new transparent image to composite the result
        bg_removed_image = Image.new("RGBA", final_result_image.size, (0, 0, 0, 0))
        image = image.resize(
            (image_width, image_height), Image.BILINEAR
        )  # Resize original image back coz we resized it earlier
        bg_removed_image.paste(image, mask=final_result_image)

        return bg_removed_image
