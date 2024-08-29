from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.
    $f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2 \epsilon}a$

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # raise NotImplementedError("Need to implement for Task 1.1")
    vals_1, vals_0 = list(vals), list(vals)
    vals_1[arg] += epsilon
    vals_0[arg] -= epsilon
    return (f(*vals_1) - f(*vals_0)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # raise NotImplementedError("Need to implement for Task 1.4")

    #### Pseudocode for topo sort
    ## visit(last)
    ## function visit(node n):
    ##     if n has a mark then return
    ##     for each node m with an edge from n to m, do:
    ##         visit(m)
    ##     mark n with a permanent mark
    ##     add n to list

    result: List[Variable] = []
    visited = set()

    def visit(var: Variable) -> None:
        id = var.unique_id
        if id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        visited.add(id)
        result.insert(0, var)

    visit(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # raise NotImplementedError("Need to implement for Task 1.4")

    # get sorted computational graph, where the first element is the output (right of the computational graph)
    queue = topological_sort(variable)
    derivatives = {}  # key: var_id; value: $\diffp{variable}{var_id}$
    derivatives[variable.unique_id] = deriv

    for var in queue:
        # for each variable `var`, find its derivative `deriv`
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            # if `var` is a leaf, update its `derivative` attribute
            var.accumulate_derivative(deriv)
        else:
            # if `var` is created by a function,
            # calculate derivatives for all inputs using chain rule.
            # `deriv` is the partial derivative of output, w.r.t. `var`
            for v, d in var.chain_rule(deriv):
                # if input is a constant, ignore
                if v.is_constant():
                    continue
                # if input is a variable, accumulate its derivative
                derivatives[v.unique_id] = derivatives.get(v.unique_id, 0.0) + d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
