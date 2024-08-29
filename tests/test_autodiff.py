from typing import Tuple

import pytest

import minitorch
from minitorch import Context, ScalarFunction, ScalarHistory
from minitorch.autodiff import topological_sort

# from minitorch.scalar_functions import Log, Exp
# from minitorch.scalar import Scalar
import math

# ## Task 1.3 - Tests for the autodifferentiation machinery.

# Simple sanity check and debugging tests.


class Function1(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        "$f(x, y) = x + y + 10$"
        return x + y + 10

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        "Derivatives are $f'_x(x, y) = 1$ and $f'_y(x, y) = 1$"
        return d_output, d_output


class Function2(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        "$f(x, y) = x \times y + x$"
        ctx.save_for_backward(x, y)
        return x * y + x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        "Derivatives are $f'_x(x, y) = y + 1$ and $f'_y(x, y) = x$"
        x, y = ctx.saved_values
        return d_output * (y + 1), d_output * x


class Function3(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        "$f(x, y) = log(xy) + exp(xy)$"
        return math.log(x * y) + math.exp(x * y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        "Derivatives are $f'_x(x, y) = \frac{1}{x} + exp(xy) \times y$ and $f'_y(x, y) = \frac{1}{y} + exp(xy) \times x$"
        x, y = ctx.saved_values
        return d_output * (1 / x + math.exp(x * y) * y), d_output * (
            1 / y + math.exp(x * y) * x
        )


# Checks for the chain rule function.


@pytest.mark.task1_3
def test_chain_rule1() -> None:
    "Check that constants are ignored."
    constant = minitorch.Scalar(0.0, None)

    y = Function1.apply(constant, constant)

    back = y.chain_rule(d_output=5)
    assert len(list(back)) == 0


@pytest.mark.task1_3
def test_chain_rule2() -> None:
    "Check that constants are ignored and variables get derivatives."
    var = minitorch.Scalar(0.0)
    constant = minitorch.Scalar(0.0, None)

    y = Function1.apply(var, constant)

    back = y.chain_rule(d_output=5)
    back = list(back)
    assert len(back) == 1
    variable, deriv = back[0]
    assert deriv == 5


@pytest.mark.task1_3
def test_chain_rule3() -> None:
    "Check that constants are ignored and variables get derivatives."
    var = minitorch.Scalar(5)
    constant = minitorch.Scalar(10, None)

    y = Function2.apply(constant, var)

    back = y.chain_rule(d_output=5)
    back = list(back)
    assert len(back) == 1
    variable, deriv = back[0]
    # assert variable.name == var.name
    assert deriv == 5 * 10


@pytest.mark.task1_3
def test_chain_rule4() -> None:
    var1 = minitorch.Scalar(5)
    var2 = minitorch.Scalar(10)

    y = Function2.apply(var1, var2)

    back = y.chain_rule(d_output=5)
    back = list(back)
    assert len(back) == 2
    variable, deriv = back[0]
    # assert variable.name == var1.name
    assert deriv == 5 * (10 + 1)
    variable, deriv = back[1]
    # assert variable.name == var2.name
    assert deriv == 5 * 5


# ## Task 1.4 - Run some simple backprop tests

# Main tests are in test_scalar.py


@pytest.mark.task1_4
def test_backprop1() -> None:
    # Example 1: F1(0, v)
    var = minitorch.Scalar(0)
    var2 = Function1.apply(0, var)
    var2.backward(d_output=5)
    assert var.derivative == 5


@pytest.mark.task1_4
def test_backprop2() -> None:
    # Example 2: F1(0, 0)
    var = minitorch.Scalar(0)
    var2 = Function1.apply(0, var)
    var3 = Function1.apply(0, var2)
    var3.backward(d_output=5)
    assert var.derivative == 5


@pytest.mark.task1_4
def test_backprop3() -> None:
    # Example 3: F1(F1(0, v1), F1(0, v1))
    var1 = minitorch.Scalar(0)
    var2 = Function1.apply(0, var1)
    var3 = Function1.apply(0, var1)
    var4 = Function1.apply(var2, var3)
    var4.backward(d_output=5)
    assert var1.derivative == 10


@pytest.mark.task1_4
def test_backprop4() -> None:
    # Example 4: F1(F1(0, v1), F1(0, v1))
    var0 = minitorch.Scalar(0)
    var1 = Function1.apply(0, var0)
    var2 = Function1.apply(0, var1)
    var3 = Function1.apply(0, var1)
    var4 = Function1.apply(var2, var3)
    var4.backward(d_output=5)
    assert var0.derivative == 10


@pytest.mark.task1_4
def test_backprop5() -> None:
    # Example 5: F2(F2(0, v1), F2(0, v1))
    var1 = minitorch.Scalar(2)
    var2 = Function2.apply(1, var1)
    var3 = Function2.apply(1, var1)
    var4 = Function2.apply(var2, var3)
    var4.backward(d_output=1)
    assert var1.derivative == 7
