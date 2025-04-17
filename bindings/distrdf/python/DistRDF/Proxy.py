#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import logging
import textwrap
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import singledispatch
from typing import Any, List, Optional, Union

import ROOT

from DistRDF import Operation
from DistRDF.Node import Node

logger = logging.getLogger(__name__)


@contextmanager
def _managed_tcontext():
    """
    Factory function, decorated with `contextlib.contextmanager` to make it
    work in a `with` context manager. It creates a `ROOT.TDirectory.TContext`
    that will store the current `ROOT.gDirectory` variable. At the end of the
    context, the C++ destructor of the `TContext` object will be explicitly
    called, thanks to the `__destruct__` dunder method implemented in PyROOT.
    This will restore the `gDirectory` variable to its initial value, allowing
    changing it in the context manager without permanent effects.
    """
    try:
        ctxt = ROOT.TDirectory.TContext()
        yield None
    finally:
        ctxt.__destruct__()


def execute_graph(node: Node) -> None:
    """
    Executes the distributed RDataFrame computation graph the input node
    belongs to. If the node already has a value, this is a no-op.
    """
    if node.value is None:  # If event-loop not triggered
        # Creating a ROOT.TDirectory.TContext in a context manager so that
        # ROOT.gDirectory won't be changed by the event loop execution.
        with _managed_tcontext():
            # All the information needed to reconstruct the computation graph on
            # the workers is contained in the head node
            node.get_head().execute_graph()

def _update_internal_df_with_transformation(node:Node, operation: Operation) -> None:
    """Propagate transform operations to the headnode internal RDataFrame"""
    # The parent node is None only if the node is the head node
    parent_node = node.parent if node.parent is not None else node
    # Retrieve correct C++ transformation to call
    rdf_operation = getattr(parent_node.rdf_node, operation.name)
    # Call and inject the result in the Python node
    node.rdf_node = rdf_operation(*operation.args, **operation.kwargs)

def _create_new_node(parent: Node, operation: Operation.Operation) -> Node:
    """Creates a new node and inserts it in the computation graph"""

    headnode = parent.get_head()
    headnode.node_counter += 1

    newnode = Node(parent.get_head, headnode.node_counter, operation, parent)

    parent.nchildren += 1

    headnode.graph_nodes.appendleft(newnode)

    return newnode


class Proxy(ABC):
    """
    Abstract class for proxies objects. These objects help to keep track of
    nodes' variable assignment. That is, when a node is no longer assigned
    to a variable by the user, the role of the proxy is to show that. This is
    done via changing the value of the :obj:`has_user_references` of the
    proxied node from :obj:`True` to :obj:`False`.
    """

    def __init__(self, node: Node):
        """
        Creates a new `Proxy` object for a given node.

        Args:
            proxied_node: The node that the current Proxy should wrap.
        """
        self.proxied_node = node

    @abstractmethod
    def __getattr__(self, attr):
        """
        Proxies have to declare the way they intercept calls to attributes
        and methods of the proxied node.
        """
        pass

    def __del__(self):
        """
        This function is called right before the current Proxy gets deleted by
        Python. Its purpose is to show that the wrapped node has no more
        user references, which is one of the conditions for the node to be
        pruned from the computational graph.
        """
        self.proxied_node.has_user_references = False


class ResultMapProxy(Proxy):
    """
    Instances of ResultMapProxy act as futures of the result produced
    by a call to DistRDF.VariationsFor. The aim is to mimic the functionality of
    ROOT::RDF::Experimental::RResultMap to provide the same API usage.
    """

    def __init__(self, node: Node):
        super().__init__(node)
        self._keys: Optional[List[str]] = None

    def __getattr__(self, attr):
        """
        The __getattr__ of the Proxy base class is an abstract method. This
        class has no attributes to present to the user.
        """
        raise AttributeError(f"'ResultMapProxy' object has no attribute '{attr}'")

    def __getitem__(self, key: str):
        """
        Equivalent of 'operator[]' of the RResultMap. Triggers the computation
        graph, then returns the varied value linked to the 'key' name.
        """
        execute_graph(self.proxied_node)
        try:
            return self.proxied_node.value.GetVariation(key)
        except ROOT.std.runtime_error as e:
            raise KeyError(f"'{key}' is not a valid variation name in this branch of the graph. "
                           f"Available variations are {self.GetKeys()}") from e

    def GetKeys(self) -> List[str]:
        """
        Equivalent of 'GetKeys' of the RResultMap. Unlike its C++ counterpart,
        at the moment we cannot retrieve the list of variation names for a
        certain action without triggering the distributed computation graph. For
        this reason, the function raises an error if the keys are accessed
        before computations have been triggered. In the future the behaviour
        should be aligned with the C++ counterpart.
        """
        if self.proxied_node.value is None:
            # TODO:
            # The event loop has not been triggered yet. Currently we can't retrieve
            # the list of variation names without starting the distributed computations
            raise RuntimeError(textwrap.dedent(
                """
                A list of names of systematic variations was requested, but the corresponding map of variations is not
                present. The variation names cannot be retrieved unless the computation graph has properly run and
                finished. Something may have gone wrong in the distributed execution, or no variation values were
                explicitly requested. In the future, it will be possible to get the variation names without triggering.
                """))
        else:
            if self._keys is None:
                self._keys = [str(key) for key in self.proxied_node.value.GetKeys()]
            return self._keys


class ResultPtrProxy(Proxy):
    """
    Instances of ResultPtrProxy act as futures of the result produced
    by some action node. The aim is to mimic the functionality of
    ROOT::RDF::RResultPtr to provide the same API usage.
    """

    def __getattr__(self, attr):
        """
        Intercepts calls on the result of
        the action node.

        Returns:
            function: A method to handle an operation call to the
            current action node.
        """
        self._cur_attr = attr  # Stores the name of operation call
        return self._call_action_result

    def GetValue(self):
        """
        Returns the result value of the current action node if it was executed
        before, else triggers the execution of the distributed graph before
        returning the value.
        """
        execute_graph(self.proxied_node)
        return self.proxied_node.value

    def _call_action_result(self, *args, **kwargs):
        """
        Handles an operation call to the current action node and returns
        result of the current action node.
        """
        return getattr(self.GetValue(), self._cur_attr)(*args, **kwargs)

    def create_variations(self) -> ResultMapProxy:
        """
        Creates a node responsible to signal the creation of variations in the
        distributed computation graph, returning a specialized proxy to that
        node. This function is usually called from DistRDF.VariationsFor.
        """
        return ResultMapProxy(_create_new_node(self.proxied_node, Operation.create_op("VariationsFor")))


class NodeProxy(Proxy):
    """
    A proxy object to an non-action node. It implements acces to attributes
    and methods of the proxied node. It is also in charge of the creation of
    a new operation node in the graph. The aim is to mimic the functionality of
    ROOT::RDF::RNode to provide the same API usage.
    """

    def __getattr__(self, attr):
        """
        Intercepts calls to attributes and methods of the proxied node and
        returns the appropriate object(s).

        Args:
            attr (str): The name of the attribute or method of the proxied
                node the user wants to access.
        """

        # if attr is a supported operation, start
        # operation and node creation
        if attr in Operation.SUPPORTED_OPERATIONS:
            self.proxied_node._new_op_name = attr  # Stores new operation name
            return self._create_new_op
        else:
            try:
                return getattr(self.proxied_node, attr)
            except AttributeError:
                if self.proxied_node.operation:
                    msg = "'{0}' object has no attribute '{1}'".format(
                        str(self.proxied_node.operation.name),
                        attr
                    )
                else:
                    msg = "'RDataFrame' object has no attribute '{}'".format(
                        attr
                    )
                raise AttributeError(msg)

    def GetColumnNames(self):
        """Forward call to the internal RDataFrame object"""
        return self.proxied_node.rdf_node.GetColumnNames()
    
    def GetColumnType(self, column):
        """Forward call to the internal RDataFrame object"""
        return self.proxied_node.rdf_node.GetColumnType(column)

    def _create_new_op(self, *args, **kwargs):
        """
        Handles an operation call to the current node and returns the new node
        built using the operation call.
        """
        op = Operation.create_op(self.proxied_node._new_op_name, *args, **kwargs)
        newnode = _create_new_node(self.proxied_node, op)
        return get_proxy_for(op, newnode)


@singledispatch
def get_proxy_for(operation: Operation.Transformation, node: Node) -> NodeProxy:
    """"Returns appropriate proxy for the input node"""
    _update_internal_df_with_transformation(node, operation)
    return NodeProxy(node)


@get_proxy_for.register
def _(operation: Operation.Action, node: Node) -> ResultPtrProxy:
    return ResultPtrProxy(node)


@get_proxy_for.register
def _(operation: Operation.InstantAction, node: Node) -> Any:
    execute_graph(node)
    return node.value


@get_proxy_for.register
def _(operation: Operation.Snapshot, node: Node) -> Union[ResultPtrProxy, Any]:
    if len(operation.args) == 4:
        # An RSnapshotOptions instance was passed as fourth argument
        if operation.args[3].fLazy:
            return get_proxy_for.dispatch(Operation.Action)(operation, node)

    return get_proxy_for.dispatch(Operation.InstantAction)(operation, node)


@get_proxy_for.register
def _(operation: Operation.AsNumpy, node: Node) -> Union[ResultPtrProxy, Any]:
    if operation.kwargs.get("lazy", False):
        return get_proxy_for.dispatch(Operation.Action)(operation, node)

    return get_proxy_for.dispatch(Operation.InstantAction)(operation, node)
