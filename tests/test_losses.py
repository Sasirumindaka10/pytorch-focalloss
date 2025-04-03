"""Tests of focal loss function implementations in torch_focalloss"""

# pylint: disable=no-name-in-module
from torch import (
    Tensor,
    float32,
    equal,
    full,
    isclose,
    isnan,
    rand,
    randint,
    randn,
    tensor,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torch_focalloss import BinaryFocalLoss, MultiClassFocalLoss


class TestBinaryFocalLoss:
    """Tests for BinaryFocalLoss"""

    def test_initialization(self) -> None:
        """Test initialization with various parameters"""
        # Default initialization
        bfl = BinaryFocalLoss()
        assert bfl.gamma == 2.0
        assert bfl.alpha is None
        assert bfl.reduction == "mean"
        assert bfl.weight is None

        # Custom parameters
        gamma = 3.0
        alpha = 0.75
        bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha, reduction="sum")
        assert bfl.gamma == gamma
        assert isinstance(bfl.alpha, Tensor)
        assert bfl.alpha.item() == alpha
        assert bfl.reduction == "sum"

        # Using pos_weight instead of alpha
        pos_weight = tensor([1.5])
        bfl = BinaryFocalLoss(pos_weight=pos_weight)
        assert bfl.alpha is not None
        assert equal(bfl.alpha, pos_weight)

        # Using alpha tensor
        alpha_tensor = tensor([0.5, 1.0, 1.5])
        bfl = BinaryFocalLoss(alpha=alpha_tensor)
        assert equal(bfl.alpha, alpha_tensor)  # type: ignore

    def test_equals_bce_when_gamma_zero(self) -> None:
        """Test equivalence to BCE when gamma=0"""
        # Binary classification
        batch_size = 10
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)

        # No weighting
        bce = BCEWithLogitsLoss()
        bfl = BinaryFocalLoss(gamma=0)

        bce_loss = bce(preds, target)
        focal_loss = bfl(preds, target)

        assert isclose(bce_loss, focal_loss)

        # With alpha weighting
        alpha = 1.5
        bce = BCEWithLogitsLoss(pos_weight=tensor([alpha]))
        bfl = BinaryFocalLoss(gamma=0, alpha=alpha)

        bce_loss = bce(preds, target)
        focal_loss = bfl(preds, target)

        assert isclose(bce_loss, focal_loss)

    def test_multi_label_classification(self) -> None:
        """Test multi-label classification"""
        batch_size = 10
        num_classes = 3

        preds = randn(batch_size, num_classes)
        target = randint(2, size=(batch_size, num_classes), dtype=float32)

        # Test with different alphas for each label
        alpha = tensor([0.5, 1.0, 1.5])
        bce = BCEWithLogitsLoss(pos_weight=alpha)
        bfl = BinaryFocalLoss(gamma=0, alpha=alpha)

        bce_loss = bce(preds, target)
        focal_loss = bfl(preds, target)

        assert isclose(bce_loss, focal_loss)

        # Test with focal effect (gamma > 0)
        gamma = 2.0
        bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha)
        focal_loss = bfl(preds, target)

        # Loss should be lower with gamma > 0
        assert focal_loss < bce_loss

    def test_reduction_options(self) -> None:
        """Test different reduction options"""
        batch_size = 5
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)

        # Test 'none' reduction
        bfl = BinaryFocalLoss(reduction="none")
        loss = bfl(preds, target)
        assert loss.shape == (batch_size,)

        # Test 'sum' reduction
        bfl = BinaryFocalLoss(reduction="sum")
        loss = bfl(preds, target)
        assert loss.shape == ()  # scalar

        # Test 'mean' reduction
        bfl = BinaryFocalLoss(reduction="mean")
        loss = bfl(preds, target)
        assert loss.shape == ()  # scalar

    def test_weight_parameter(self) -> None:
        """Test the weight parameter works correctly"""
        batch_size = 5
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)
        weight = rand(batch_size)

        bfl = BinaryFocalLoss(gamma=0, weight=weight)
        bce = BCEWithLogitsLoss(weight=weight)

        bfl_loss = bfl(preds, target)
        bce_loss = bce(preds, target)

        assert isclose(bfl_loss, bce_loss)

    def test_gamma_effect(self) -> None:
        """Test that increasing gamma decreases the loss value"""
        batch_size = 10
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)

        # Create a sequence of losses with increasing gamma
        gammas = [0, 1, 2, 3]
        losses: list[float] = []

        for gamma in gammas:
            bfl = BinaryFocalLoss(gamma=gamma)
            losses.append(bfl(preds, target).item())

        # Check that loss decreases as gamma increases
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1]

    def test_all_correct_predictions(self) -> None:
        """Test behavior when all predictions are correct"""
        # batch_size = 5

        # Create perfect predictions
        # (large positive logits for positive examples,
        # large negative logits for negative examples)
        target = tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        preds = tensor([-10.0, 10.0, -10.0, 10.0, -10.0])

        bfl = BinaryFocalLoss()
        loss = bfl(preds, target)

        # Loss should be very small but positive
        assert loss > 0
        assert loss < 0.01

    def test_all_incorrect_predictions(self) -> None:
        """Test behavior when all predictions are incorrect"""
        # batch_size = 5

        # Create completely wrong predictions
        target = tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        preds = tensor([10.0, -10.0, 10.0, -10.0, 10.0])

        bfl = BinaryFocalLoss()
        loss = bfl(preds, target)

        # Loss should be large
        assert loss > 5


class TestMultiClassFocalLoss:
    """Tests for MultiClassFocalLoss"""

    def test_initialization(self) -> None:
        """Test initialization with various parameters"""
        # Default initialization
        mcfl = MultiClassFocalLoss()
        assert mcfl.gamma == 2.0
        assert mcfl.alpha is None
        assert mcfl.reduction == "mean"
        assert mcfl.ignore_index == -100
        assert mcfl.label_smoothing == 0.0

        # Custom parameters
        gamma = 3.0
        num_classes = 4
        alpha = rand(num_classes)
        mcfl = MultiClassFocalLoss(
            gamma=gamma,
            alpha=alpha,
            reduction="sum",
            ignore_index=-1,
            label_smoothing=0.1,
        )
        assert mcfl.gamma == gamma
        assert equal(mcfl.alpha, alpha)  # type: ignore
        assert mcfl.reduction == "sum"
        assert mcfl.ignore_index == -1
        assert mcfl.label_smoothing == 0.1

        # Using weight instead of alpha
        weight = rand(num_classes)
        mcfl = MultiClassFocalLoss(weight=weight)
        assert equal(mcfl.alpha, weight)  # type: ignore

    def test_equals_ce_when_gamma_zero(self) -> None:
        """Test equivalence to CrossEntropy when gamma=0"""
        batch_size = 10
        num_classes = 4

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # No weighting
        ce = CrossEntropyLoss()
        mcfl = MultiClassFocalLoss(gamma=0)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

        # With class weighting
        alpha = rand(num_classes)
        ce = CrossEntropyLoss(weight=alpha)
        mcfl = MultiClassFocalLoss(gamma=0, alpha=alpha)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

    def test_reduction_options(self) -> None:
        """Test different reduction options"""
        batch_size = 5
        num_classes = 3

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # Test 'none' reduction
        mcfl = MultiClassFocalLoss(reduction="none")
        loss = mcfl(preds, target)
        assert loss.shape == (batch_size,)

        # Test 'sum' reduction
        mcfl = MultiClassFocalLoss(reduction="sum")
        loss = mcfl(preds, target)
        assert loss.shape == ()  # scalar

        # Test 'mean' reduction
        mcfl = MultiClassFocalLoss(reduction="mean")
        loss = mcfl(preds, target)
        assert loss.shape == ()  # scalar

    def test_gamma_effect(self) -> None:
        """Test that increasing gamma decreases the loss value"""
        batch_size = 10
        num_classes = 4

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # Create a sequence of losses with increasing gamma
        gammas = [0, 1, 2, 3]
        losses: list[float] = []

        for gamma in gammas:
            mcfl = MultiClassFocalLoss(gamma=gamma)
            losses.append(mcfl(preds, target).item())

        # Check that loss decreases as gamma increases
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1]

    def test_ignore_index(self) -> None:
        """Test ignore_index parameter works correctly"""
        batch_size = 5
        num_classes = 3
        ignore_idx = 2

        preds = randn(batch_size, num_classes)
        # Create targets with some elements equal to ignore_idx
        target = tensor([0, 1, ignore_idx, 0, 1])

        # Standard CE loss with ignore_index
        ce = CrossEntropyLoss(ignore_index=ignore_idx)
        # Focal loss with same ignore_index
        mcfl = MultiClassFocalLoss(gamma=0, ignore_index=ignore_idx)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

        # Create a target with all elements equal to ignore_idx
        all_ignored_target = full((batch_size,), ignore_idx)

        ce_loss = ce(preds, all_ignored_target)
        focal_loss = mcfl(preds, all_ignored_target)

        # Both losses should be nan when all targets are ignored
        assert isnan(ce_loss)
        assert focal_loss == 0

    def test_label_smoothing(self) -> None:
        """Test label smoothing parameter works correctly"""
        batch_size = 5
        num_classes = 3
        smoothing = 0.1

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # Standard CE with label smoothing
        ce = CrossEntropyLoss(label_smoothing=smoothing)
        # Focal loss with same label smoothing
        mcfl = MultiClassFocalLoss(gamma=0, label_smoothing=smoothing)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

    def test_all_correct_predictions(self) -> None:
        """Test behavior when all predictions are correct"""
        batch_size = 5
        num_classes = 3

        # Create one-hot encoded target
        target = tensor([0, 1, 2, 0, 1])

        # Create perfect predictions (very large values at target positions)
        preds = full((batch_size, num_classes), -100.0)
        for i, t in enumerate(target):
            preds[i, t] = 100.0

        mcfl = MultiClassFocalLoss()
        loss = mcfl(preds, target)

        # Loss should be very small, possibly 0 due to numerical precision
        assert loss >= 0
        assert loss < 0.01

    def test_all_incorrect_predictions(self) -> None:
        """Test behavior when all predictions are incorrect"""
        batch_size = 5
        num_classes = 3

        # Create targets
        target = tensor([0, 1, 2, 0, 1])

        # Create completely wrong predictions
        # (very negative values at target positions)
        preds = full((batch_size, num_classes), 100.0)
        for i, t in enumerate(target):
            preds[i, t] = -100.0

        mcfl = MultiClassFocalLoss()
        loss = mcfl(preds, target)

        # Loss should be large
        assert loss > 5

    def test_weighted_loss_mean_reduction(self) -> None:
        """Test that mean reduction respects class weights"""
        batch_size = 6
        num_classes = 3

        # Create a balanced dataset with 2 examples of each class
        target = tensor([0, 0, 1, 1, 2, 2])
        preds = randn(batch_size, num_classes)

        # Create weights that heavily favor class 0
        alpha = tensor([10.0, 1.0, 1.0])

        # Create loss with these weights
        mcfl = MultiClassFocalLoss(gamma=0, alpha=alpha)
        weighted_loss = mcfl(preds, target)

        # Create a separate loss for each class
        class0_loss = CrossEntropyLoss()(
            preds[target == 0], target[target == 0]
        )
        class1_loss = CrossEntropyLoss()(
            preds[target == 1], target[target == 1]
        )
        class2_loss = CrossEntropyLoss()(
            preds[target == 2], target[target == 2]
        )

        # weighted mean should be closer to class0_loss due to higher weight
        manual_weighted_mean = (
            10.0 * class0_loss + 1.0 * class1_loss + 1.0 * class2_loss
        ) / 12.0

        assert isclose(weighted_loss, manual_weighted_mean)
