from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

from ..bria_remove_bg.bria_remove_bg_tool import BriaRMBGTool

bria_remover = None


@invocation(
    "bria_bg_remove",
    title="BRIA AI Background Removal",
    tags=["image", "background", "removal", "bria"],
    category="image",
    version="1.0.0",
)
class BriaRemoveBackgroundInvocation(BaseInvocation, WithMetadata):
    """Uses the new Bria 1.4 model to remove backgrounds from images."""

    image: ImageField = InputField(description="The image to crop")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        global bria_remover
        image = context.services.images.get_pil_image(self.image.image_name)

        if not bria_remover:
            bria_remover = BriaRMBGTool()
            bria_remover.load_model()

        bg_removed_image = bria_remover.remove_background(image)

        image_dto = context.services.images.create(
            image=bg_removed_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
