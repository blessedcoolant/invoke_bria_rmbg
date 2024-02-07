import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.util.devices import choose_torch_device
from invokeai.backend.util.util import download_with_progress_bar
from PIL import Image
from torchvision.transforms.functional import normalize

from .bria_rmbg import BriaRMBG

config = InvokeAIAppConfig.get_config()

BRIA_RMBG_MODELS = {
    "1.4": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth?download=true",
        "local": "any/bria-rmbg/model.pth",
    },
}


def resize_image(image):
    image = image.convert("RGB")
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


class BriaRMBGTool:
    def __init__(self) -> None:
        self.network = None
        self.device = choose_torch_device()

    def load_model(self):
        BRIA_RMBG_MODEL_PATH = pathlib.Path(config.models_path / BRIA_RMBG_MODELS["1.4"]["local"])
        if not BRIA_RMBG_MODEL_PATH.exists():
            download_with_progress_bar(BRIA_RMBG_MODELS["1.4"]["url"], BRIA_RMBG_MODEL_PATH)

        self.network = BriaRMBG()
        self.network.load_state_dict(torch.load(BRIA_RMBG_MODEL_PATH, map_location=self.device))
        self.network.eval()

    def remove_background(self, image: Image.Image) -> Any:
        image = image.convert("RGB") if image.mode != "RGB" else image  # Ensure image is RGB
        image_width, image_height = image.size

        # Resize image to match model sizing requirements
        image = image.resize((1024, 1024), Image.BILINEAR)

        # Process the image
        np_image = np.array(image)
        tensor_image = torch.tensor(np_image, dtype=torch.float32).permute(2, 0, 1)
        tensor_image = torch.unsqueeze(tensor_image, 0)
        tensor_image = torch.divide(tensor_image, 255.0)
        tensor_image = normalize(tensor_image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        if self.device == "cuda":
            tensor_image = tensor_image.cuda()

        bg_removed_tensor = self.network(tensor_image)
        final_result = torch.squeeze(
            F.interpolate(bg_removed_tensor[0][0], size=(image_height, image_width), mode="bilinear"), 0
        )
        ma = torch.max(final_result)
        mi = torch.min(final_result)
        final_result = (final_result - mi) / (ma - mi)
        final_result_array = (final_result * 255).cpu().data.numpy().astype(np.uint8)
        final_result_image = Image.fromarray(np.squeeze(final_result_array))

        # Create new transparent image to composite the result
        bg_removed_image = Image.new("RGBA", final_result_image.size, (0, 0, 0, 0))
        image = image.resize(
            (image_width, image_height), Image.BILINEAR
        )  # Resize original image back coz we resized it earlier
        bg_removed_image.paste(image, mask=final_result_image)

        return bg_removed_image
