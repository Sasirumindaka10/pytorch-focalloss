"""Set up torch_focalloss package"""

# make the classes available directly from the top level
from .losses import BinaryFocalLoss, MultiClassFocalLoss  # type: ignore
