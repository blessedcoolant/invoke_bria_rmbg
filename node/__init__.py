from typing import Literal

from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.invocation_api import BaseInvocation, InputField, InvocationContext, WithMetadata, invocation

from ..bria_remove_bg.bria_remove_bg_tool import RMBG_MODEL_TYPES, BriaRMBGTool

bria_remover = None


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

    def invoke(self, context: InvocationContext) -> ImageOutput:
        global bria_remover
        image = context.images.get_pil(self.image.image_name)

        if not isinstance(bria_remover, BriaRMBGTool):
            bria_remover = BriaRMBGTool()

        if bria_remover.network is None or bria_remover.current_model != self.model:
            bria_remover.load_model(self.model)

        bg_removed_image = bria_remover.remove_background(image)

        image_dto = context.images.save(image=bg_removed_image)

        return ImageOutput.build(image_dto)
