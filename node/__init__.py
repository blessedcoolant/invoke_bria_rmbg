import pathlib
from typing import Literal, Optional

from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.backend.util.devices import TorchDevice
from invokeai.invocation_api import BaseInvocation, InputField, InvocationContext, WithMetadata, invocation

from ..bria_remove_bg.bria_remove_bg_tool import BriaRMBGTool

BRIA_RMBG_MODELS = {
    "1.4": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth",
    "Open RMBG": "https://huggingface.co/schirrmacher/ormbg/resolve/main/models/ormbg.pth",
}

RMBG_MODEL_TYPES = Optional[Literal["1.4", "Open RMBG"]]


@invocation(
    "bria_bg_remove",
    title="BRIA AI Background Removal",
    tags=["image", "background", "removal", "bria"],
    category="image",
    version="1.0.2",
)
class BriaRemoveBackgroundInvocation(BaseInvocation, WithMetadata):
    """Uses the new Bria 1.4 model to remove backgrounds from images."""

    image: ImageField = InputField(description="The image to crop")
    model: RMBG_MODEL_TYPES = InputField(default="1.4", description="The model to use for background removal")

    def load_model(self, context):
        def loader(model_path: pathlib.Path):
            return BriaRMBGTool.load_model(model_path, TorchDevice.choose_torch_device())

        with context.models.load_remote_model(source=BRIA_RMBG_MODELS[self.model], loader=loader) as model:
            bria_remover = BriaRMBGTool(model, TorchDevice.choose_torch_device())
            return bria_remover

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        bria_remover = self.load_model(context)
        bg_removed_image = bria_remover.remove_background(image)

        image_dto = context.images.save(image=bg_removed_image)

        return ImageOutput.build(image_dto)
