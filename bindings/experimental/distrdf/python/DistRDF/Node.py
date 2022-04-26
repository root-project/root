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
from typing import Callable, Optional, TYPE_CHECKING

# Type hints only
if TYPE_CHECKING:
    # Avoid circular imports
    from DistRDF.HeadNode import HeadNode
    from DistRDF.Operation import Operation

logger = logging.getLogger(__name__)


class Node(object):
    """
    A Class that represents a node in RDataFrame operations graph. A Node
    houses an operation and has references to children nodes.

    Attributes:
        get_head: A function returning the head node in the graph.

        node_id: The id of this node, given sequentially in the order of
            creation with respect to the head node of the graph.

        operation: The operation that this node represents. `None` if the node
            is the head node.

        parent: A reference to the parent node of this node. `None` if the node
            is the head node.

        nchildren: A counter of how many children this node has.

        _new_op_name (str): The name of the new incoming operation of the next
            child, which is the last child node among the current node's
            children.

        value: The computed value after executing the operation in the current
            node for a particular DistRDF graph. This is permanently :obj:`None`
            for transformation nodes and the action nodes get a
            :obj:`ROOT.RResultPtr` after event-loop execution.

        has_user_references (bool): A flag to check whether the node has
            direct user references, that is if it is assigned to a variable.
            Default value is :obj:`True`, turns to :obj:`False` if the proxy
            that wraps the node gets garbage collected by Python.

        rdf_node: A reference to the result of calling a function of the
            RDataFrame API with the current operation. This is practically a
            node of the true computation graph, which is being executed in some
            distributed task. It is a transient attribute. On the client, it
            is always None. The value is computed and stored only during a task
            on a worker.
    """

    def __init__(self, get_head: Callable[[], "HeadNode"], node_id: int = 0,
                 operation: "Operation" = None, parent: Node = None):
        self.get_head = get_head
        self.node_id = node_id
        self.operation = operation
        self.parent = parent
        self.nchildren: int = 0
        self._new_op_name: str = ""
        self.value = None
        self.has_user_references: bool = True
        self.rdf_node = None

        # This is the internal attribute for the 'parent_id' property. It is
        # serialized from the local information then deserialized and set in a
        # distributed task.
        self._parent_id: Optional[int] = parent.node_id if parent is not None else None

    @property
    def parent_id(self) -> Optional[int]:
        """Retrieves the id of the parent node."""
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: Optional[int]):
        """
        Sets the id of the parent node. Only present to enable setting this
        attribute when deserializing the node in a distributed task.
        """
        self._parent_id = value

    def __getstate__(self):
        """
        Serialize the minimum amount of information needed in a distributed task
        to execute the operation corresponding to this node of the graph.
        """
        return {"operation": self.operation, "parent_id": self.parent_id}

    def __setstate__(self, state):
        self.operation = state["operation"]
        self.parent_id = state["parent_id"]

    def is_prunable(self) -> bool:
        """
        Checks whether the current node can be pruned from the computational
        graph.

        Returns:
            bool: True if the node has no children and no user references or
            its value has already been computed, False otherwise.
        """
        logger.debug(f"Checking node {self.node_id} for pruning")

        if self.nchildren == 0 and (not self.has_user_references or self.value is not None):

            # ***** Condition 1 *****
            # If the node does not have children, it might be prunable.

            # ***** Condition 2 *****
            # If the node is wrapped by a proxy which is not directly
            # assigned to a variable, then it will be flagged for pruning

            # ***** Condition 3 *****
            # If the current node's value was already computed, it should
            # get pruned. Only action nodes may possess a value attribute
            # which is not None

            logger.debug(f"node {self.node_id} can be pruned")

            # Decrement children count of the parent node of this node, so that
            # when the pruning will consider the parent, this child will not be
            # counted.
            self.parent.nchildren -= 1

            return True

        logger.debug(f"node {self.node_id} should not be pruned")

        return False


class VariationsNode(Node):
    """
    Helper tag for a node of the computation graph that is responsible for
    querying systematic variations from another action.
    """
    pass
