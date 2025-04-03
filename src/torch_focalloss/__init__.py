"""Set up torch_focalloss package"""

# make the classes available directly from the top level
from torch_focalloss.losses import BinaryFocalLoss, MultiClassFocalLoss

__all__ = ["BinaryFocalLoss", "MultiClassFocalLoss"]
