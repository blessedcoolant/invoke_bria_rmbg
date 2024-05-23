import pathlib
from typing import Any, Literal, Optional

import numpy as np
import torch
from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.download_with_progress import download_with_progress_bar
from invokeai.backend.util.devices import choose_torch_device
from PIL import Image

from .bria_rmbg import BriaRMBG
from .utils import postprocess_image, preprocess_image

config = get_config()

BRIA_RMBG_MODELS = {
    "1.4": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth?download=true",
        "local": "any/bria-rmbg/model.pth",
    },
    "Open RMBG": {
        "url": "https://huggingface.co/schirrmacher/ormbg/resolve/main/models/ormbg.pth?download=true",
        "local": "any/bria-rmbg/ormbg.pth",
    },
}


def resize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


RMBG_MODEL_TYPES = Optional[Literal["1.4", "Open RMBG"]]


class BriaRMBGTool:
    def __init__(self) -> None:
        self.network: Optional[BriaRMBG] = None
        self.device = choose_torch_device()
        self.current_model: RMBG_MODEL_TYPES = None

    def load_model(self, model_type: RMBG_MODEL_TYPES = "1.4"):
        self.current_model = model_type
        BRIA_RMBG_MODEL_PATH = pathlib.Path(config.models_path / BRIA_RMBG_MODELS[model_type]["local"])
        if not BRIA_RMBG_MODEL_PATH.exists():
            download_with_progress_bar(
                "BRIA RMBG Checkpoint", BRIA_RMBG_MODELS[model_type]["url"], BRIA_RMBG_MODEL_PATH
            )

        self.network = BriaRMBG()
        self.network.load_state_dict(torch.load(BRIA_RMBG_MODEL_PATH, map_location=self.device))
        self.network.eval()

    def remove_background(self, image: Image.Image) -> Any:
        model_size = [1024, 1024]

        image = image.convert("RGB") if image.mode != "RGB" else image  # Ensure image is RGB
        image_width, image_height = image.size

        # Resize image to match model sizing requirements
        image = image.resize(tuple(model_size), Image.BILINEAR)

        # Process the image
        np_image = np.array(image)

        tensor_image = preprocess_image(np_image, model_size)
        if self.device == "cuda":
            tensor_image = tensor_image.cuda()
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
