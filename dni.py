import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

from contextlib import contextmanager
from functools import partial


class UnidirectionalInterface(torch.nn.Module):
    """Basic `Interface` for unidirectional communication.

    Can be used to manually pass `messages` with methods `send` and `receive`.

    Args:
        synthesizer: `Synthesizer` to use to generate `messages`.
    """

    def __init__(self, synthesizer):
        super().__init__()

        self.synthesizer = synthesizer

    def receive(self, trigger):
        """Synthesizes a `message` based on `trigger`.

        Detaches `message` so no gradient will go through it during the
        backward pass.

        Args:
            trigger: `trigger` to use to synthesize a `message`.

        Returns:
            The synthesized `message`.
        """
        return self.synthesizer(
            trigger, _Manager.get_current_context()
        ).detach()

    def send(self, message, trigger):
        """Updates the estimate of synthetic `message` based on `trigger`.

        Synthesizes a `message` based on `trigger`, computes the MSE between it
        and the input `message` and backpropagates it to compute its gradient
        w.r.t. `Synthesizer` parameters. Does not backpropagate through
        `trigger`.

        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger`.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        """
        synthetic_message = self.synthesizer(
            trigger.detach(), _Manager.get_current_context()
        )
        loss = F.mse_loss(synthetic_message, message.detach())
        _Manager.backward(loss)


class ForwardInterface(UnidirectionalInterface):
    """`Interface` for synthesizing activations in the forward pass.

    Can be used to achieve a forward unlock. It does not make too much sense to
    use it on its own, as it breaks backpropagation (no gradients pass through
    `ForwardInterface`). To achieve both forward and update unlock, use
    `BidirectionalInterface`.

    Args:
        synthesizer: `Synthesizer` to use to generate `messages`.
    """

    def forward(self, message, trigger):
        """Synthetic forward pass, no backward pass.

        Convenience method combining `send` and `receive`. Updates the
        `message` estimate based on `trigger` and returns a synthetic
        `message`.

        Works only in `training` mode, otherwise just returns the input
        `message`.

        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger`.
            trigger: `trigger` that the `message` should be synthesized based
                on.

        Returns:
            The synthesized `message`.
        """
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message


class BackwardInterface(UnidirectionalInterface):
    """`Interface` for synthesizing gradients in the backward pass.

    Can be used to achieve an update unlock.

    Args:
        synthesizer: `Synthesizer` to use to generate gradients.
    """

    def forward(self, trigger):
        """Normal forward pass, synthetic backward pass.

        Convenience method combining `backward` and `make_trigger`. Can be
        used when we want to backpropagate synthetic gradients from and
        intercept real gradients at the same `Variable`, for example for
        update decoupling feed-forward networks.

        Backpropagates synthetic gradient from `trigger` and returns a copy of
        `trigger` with a synthetic gradient update operation attached.

        Works only in `training` mode, otherwise just returns the input
        `trigger`.

        Args:
            trigger: `trigger` to backpropagate synthetic gradient from and
                intercept real gradient at.

        Returns:
            A copy of `trigger` with a synthetic gradient update operation
            attached.
        """
        if self.training:
            self.backward(trigger)
            return self.make_trigger(trigger.detach())
        else:
            return trigger

    def backward(self, trigger, factor=1):
        """Backpropagates synthetic gradient from `trigger`.

        Computes synthetic gradient based on `trigger`, scales it by `factor`
        and backpropagates it from `trigger`.

        Works only in `training` mode, otherwise is a no-op.

        Args:
            trigger: `trigger` to compute synthetic gradient based on and to
                backpropagate it from.
            factor (optional): Factor by which to scale the synthetic gradient.
                Defaults to 1.
        """
        if self.training:
            synthetic_gradient = self.receive(trigger)
            _Manager.backward(trigger, synthetic_gradient.data * factor)

    def make_trigger(self, trigger):
        """Attaches a synthetic gradient update operation to `trigger`.

        Returns a `Variable` with the same `data` as `trigger`, that during
        the backward pass will intercept gradient passing through it and use
        this gradient to update the `Synthesizer`'s estimate.

        Works only in `training` mode, otherwise just returns the input
        `trigger`.

        Returns:
            A copy of `trigger` with a synthetic gradient update operation
            attached.
        """
        if self.training:
            return _SyntheticGradientUpdater.apply(
                trigger,
                self.synthesizer(trigger, _Manager.get_current_context())
            )
        else:
            return trigger


class _SyntheticGradientUpdater(torch.autograd.Function):

    @staticmethod
    def forward(ctx, trigger, synthetic_gradient):
        (_, needs_synthetic_gradient_grad) = ctx.needs_input_grad
        if not needs_synthetic_gradient_grad:
            raise ValueError(
                'synthetic_gradient should need gradient but it does not'
            )

        ctx.save_for_backward(synthetic_gradient)
        # clone trigger to force creating a new Variable with
        # requires_grad=True
        return trigger.clone()

    @staticmethod
    def backward(ctx, true_gradient):
        (synthetic_gradient,) = ctx.saved_variables
        # compute MSE gradient manually to avoid dependency on PyTorch
        # internals
        (batch_size, *_) = synthetic_gradient.size()
        grad_synthetic_gradient = (
            2 / batch_size * (synthetic_gradient - true_gradient)
        )
        return (true_gradient, grad_synthetic_gradient)


class BidirectionalInterface(torch.nn.Module):
    """`Interface` for synthesizing both activations and gradients w.r.t. them.

    Can be used to achieve a full unlock.

    Args:
        forward_synthesizer: `Synthesizer` to use to generate `messages`.
        backward_synthesizer: `Synthesizer` to use to generate gradients w.r.t.
            `messages`.
    """

    def __init__(self, forward_synthesizer, backward_synthesizer):
        super().__init__()

        self.forward_interface = ForwardInterface(forward_synthesizer)
        self.backward_interface = BackwardInterface(backward_synthesizer)

    def forward(self, message, trigger):
        """Synthetic forward pass, synthetic backward pass.

        Convenience method combining `send` and `receive`. Can be used when we
        want to `send` and immediately `receive` using the same `trigger`. For
        more complex scenarios, `send` and `receive` need to be used
        separately.

        Updates the `message` estimate based on `trigger`, backpropagates
        synthetic gradient from `message` and returns a synthetic `message`
        with a synthetic gradient update operation attached.

        Works only in `training` mode, otherwise just returns the input
        `message`.
        """
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message

    def receive(self, trigger):
        """Combination of `ForwardInterface.receive` and
        `BackwardInterface.make_trigger`.

        Generates a synthetic `message` based on `trigger` and attaches to it
        a synthetic gradient update operation.

        Args:
            trigger: `trigger` to use to synthesize a `message`.

        Returns:
            The synthesized `message` with a synthetic gradient update
            operation attached.
        """
        message = self.forward_interface.receive(trigger)
        return self.backward_interface.make_trigger(message)

    def send(self, message, trigger):
        """Combination of `ForwardInterface.send` and
        `BackwardInterface.backward`.

        Updates the estimate of synthetic `message` based on `trigger` and
        backpropagates synthetic gradient from `message`.

        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger` and that synthetic gradient should be backpropagated
                from.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        """
        self.forward_interface.send(message, trigger)
        self.backward_interface.backward(message)


class BasicSynthesizer(torch.nn.Module):
    """Basic `Synthesizer` based on an MLP with ReLU activation.

    Args:
        output_dim: Dimensionality of the synthesized `messages`.
        n_hidden (optional): Number of hidden layers. Defaults to 0.
        hidden_dim (optional): Dimensionality of the hidden layers. Defaults to
            `output_dim`.
        trigger_dim (optional): Dimensionality of the trigger. Defaults to
            `output_dim`.
        context_dim (optional): Dimensionality of the context. If `None`, do
            not use context. Defaults to `None`.
    """

    def __init__(self, output_dim, n_hidden=0, hidden_dim=None,
                 trigger_dim=None, context_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = output_dim
        if trigger_dim is None:
            trigger_dim = output_dim

        top_layer_dim = output_dim if n_hidden == 0 else hidden_dim

        self.input_trigger = torch.nn.Linear(
            in_features=trigger_dim, out_features=top_layer_dim
        )

        if context_dim is not None:
            self.input_context = torch.nn.Linear(
                in_features=context_dim, out_features=top_layer_dim
            )
        else:
            self.input_context = None

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=(
                    hidden_dim if layer_index < n_hidden - 1 else output_dim
                )
            )
            for layer_index in range(n_hidden)
        ])

        # zero-initialize the last layer, as in the paper
        if n_hidden > 0:
            init.constant(self.layers[-1].weight, 0)
        else:
            init.constant(self.input_trigger.weight, 0)
            if context_dim is not None:
                init.constant(self.input_context.weight, 0)

    def forward(self, trigger, context):
        """Synthesizes a `message` based on `trigger` and `context`.

        Args:
            trigger: `trigger` to synthesize the `message` based on. Size:
                (`batch_size`, `trigger_dim`).
            context: `context` to condition the synthesizer. Ignored if
                `context_dim` has not been specified in the constructor. Size:
                (`batch_size`, `context_dim`).

        Returns:
            The synthesized `message`.
        """
        last = self.input_trigger(trigger)

        if self.input_context is not None:
            last += self.input_context(context)

        for layer in self.layers:
            last = layer(F.relu(last))

        return last


@contextmanager
def defer_backward():
    """Defers backpropagation until the end of scope.

    Accumulates all gradients passed to `dni.backward` inside the scope and
    backpropagates them all in a single `torch.autograd.backward` call.

    Use it and `dni.backward` whenever you want to backpropagate multiple times
    through the same nodes in the computation graph, for example when mixing
    real and synthetic gradients. Otherwise, PyTorch will complain about
    backpropagating more than once through the same graph.

    Scopes of this context manager cannot be nested.
    """
    if _Manager.defer_backward:
        raise RuntimeError('cannot nest defer_backward')
    _Manager.defer_backward = True

    try:
        yield

        if _Manager.deferred_gradients:
            (variables, gradients) = zip(*_Manager.deferred_gradients)
            torch.autograd.backward(variables, gradients)
    finally:
        _Manager.reset_defer_backward()


@contextmanager
def synthesizer_context(context):
    """Conditions `Synthesizer` calls within the scope on the given `context`.

    All `Synthesizer.forward` calls within the scope will receive `context`
    as an argument.

    Scopes of this context manager can be nested.
    """
    _Manager.context_stack.append(context)
    yield
    _Manager.context_stack.pop()


class _Manager:

    defer_backward = False
    deferred_gradients = []
    context_stack = []

    @classmethod
    def reset_defer_backward(cls):
        cls.defer_backward = False
        cls.deferred_gradients = []

    @classmethod
    def backward(cls, variable, gradient=None):
        if gradient is None:
            gradient = _ones_like(variable.data)

        if cls.defer_backward:
            cls.deferred_gradients.append((variable, gradient))
        else:
            variable.backward(gradient)

    @classmethod
    def get_current_context(cls):
        if cls.context_stack:
            return cls.context_stack[-1]
        else:
            return None


"""A simplified variant of `torch.autograd.backward` influenced by
`defer_backward`.

Inside of `defer_backward` scope, accumulates passed gradient to backpropagate
it at the end of scope. Outside of `defer_backward`, backpropagates the
gradient immediately.

Use it and `defer_backward` whenever you want to backpropagate multiple times
through the same nodes in the computation graph.

Args:
    variable: `Variable` to backpropagate the gradient from.
    gradient (optional): Gradient to backpropagate from `variable`. Defaults
        to a `Tensor` of the same size as `variable`, filled with 1.
"""
backward = _Manager.backward


def _ones_like(tensor):
    return tensor.new().resize_(tensor.size()).fill_(1)
