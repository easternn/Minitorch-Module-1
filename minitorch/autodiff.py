from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from toposort import toposort_flatten

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus_epsilon = []
    vals_minus_epsilon = []
    i = 0
    for item in vals:
        if i == arg:
            vals_plus_epsilon.append(item + epsilon / 2)
            vals_minus_epsilon.append(item - epsilon / 2)
        else:
            vals_plus_epsilon.append(item)
            vals_minus_epsilon.append(item)
        i += 1
    delta = f(*vals_plus_epsilon) - f(*vals_minus_epsilon)
    approximation = delta / epsilon
    return approximation


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
    unique_id_to_variable = {}
    graph = []
    used = set()
    queue = [variable]

    while len(queue) != 0:
        var = queue.pop()
        if var.is_constant():
            continue
        if var.unique_id in used:
            continue
        used.add(var.unique_id)
        unique_id_to_variable[var.unique_id] = var
        if var.is_leaf():
            continue
        parents = var.parents
        for parent in parents:
            if parent.is_constant():
                continue
            graph.append((parent.unique_id, var.unique_id))
            queue.append(parent)

    graph_topsort_form = {}
    for edge in graph:
        if edge[1] not in graph_topsort_form:
            graph_topsort_form[edge[1]] = []
        graph_topsort_form[edge[1]].append(edge[0])
    topsort_reverse = toposort_flatten(graph_topsort_form)
    topsort_reverse.reverse()
    variable_processing_order = []
    for unique_id in topsort_reverse:
        variable_processing_order.append(unique_id_to_variable[unique_id])
    return variable_processing_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    processing_order = topological_sort(variable)
    unique_id_to_d_output = {variable.unique_id: deriv}
    for var in processing_order:
        d_output = unique_id_to_d_output[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d_output)
            continue
        chain_rule_info = var.chain_rule(d_output)
        for (var1, d_output1) in chain_rule_info:
            if var1.unique_id not in unique_id_to_d_output:
                unique_id_to_d_output[var1.unique_id] = 0
            unique_id_to_d_output[var1.unique_id] += d_output1


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
