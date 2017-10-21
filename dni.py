import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

from contextlib import contextmanager
from functools import partial


class UnidirectionalInterface(torch.nn.Module):

    def __init__(self, synthesizer):
        super().__init__()

        self.synthesizer = synthesizer

    def receive(self, trigger):
        return self.synthesizer(
            trigger, _Manager.get_current_context()
        ).detach()

    def send(self, message, trigger):
        synthetic_message = self.synthesizer(
            trigger.detach(), _Manager.get_current_context()
        )
        loss = F.mse_loss(synthetic_message, message.detach())
        _Manager.backward(loss)


class ForwardInterface(UnidirectionalInterface):

    def forward(self, message, trigger):
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message


class BackwardInterface(UnidirectionalInterface):

    def forward(self, trigger):
        if self.training:
            self.backward(trigger)
            return self.make_trigger(trigger.detach())
        else:
            return trigger

    def backward(self, trigger, factor=1):
        if self.training:
            synthetic_gradient = self.receive(trigger)
            _Manager.backward(trigger, synthetic_gradient.data * factor)

    def make_trigger(self, trigger):
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

    def __init__(self, forward_synthesizer, backward_synthesizer):
        super().__init__()

        self.forward_interface = ForwardInterface(forward_synthesizer)
        self.backward_interface = BackwardInterface(backward_synthesizer)

    def forward(self, message, trigger):
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message

    def receive(self, trigger):
        message = self.forward_interface.receive(trigger)
        return self.backward_interface.make_trigger(message)

    def send(self, message, trigger):
        self.forward_interface.send(message, trigger)
        self.backward_interface.backward(message)


class BasicSynthesizer(torch.nn.Module):

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
        last = self.input_trigger(trigger)

        if self.input_context is not None:
            last += self.input_context(context)

        for layer in self.layers:
            last = layer(F.relu(last))

        return last


@contextmanager
def defer_backward():
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


backward = _Manager.backward


def _ones_like(tensor):
    return tensor.new().resize_(tensor.size()).fill_(1)
