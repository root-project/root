import ROOT
from PyRDF.backend.Backend import Backend
import logging

logger = logging.getLogger(__name__)


class Local(Backend):
    """
    Backend that relies on the C++ implementation of RDataFrame to locally
    execute the current graph.

    Attributes:
        config (dict): The config object for the Local backend.
    """

    def __init__(self, config={}):
        """
        Creates a new instance of the
        Local implementation of `Backend`.

        Args:
            config (dict, optional): The config object for the required
                backend. The default value is an empty Python dictionary:
                :obj:`{}`.
        """
        super(Local, self).__init__(config)
        operations_not_supported = [
            'Take',
            'Foreach',
            'Reduce',
            'Aggregate',
        ]
        if ROOT.ROOT.IsImplicitMTEnabled():
            # Add `Range` operation only if multi-threading
            # is disabled.
            operations_not_supported.append('Range')

        self.supported_operations = [op for op in self.supported_operations
                                     if op not in operations_not_supported]
        logger.debug("Creating `Local` backend. Available operations:\n{}"
                     .format(self.supported_operations))
        self.pyroot_rdf = None

    def execute(self, generator):
        """
        Executes locally the current RDataFrame graph.

        Args:
            generator (PyRDF.CallableGenerator): An instance of
                :obj:`CallableGenerator` that is responsible for generating
                the callable function.

        """
        mapper = generator.get_callable()  # Get the callable

        # if the RDataFrame has not been created yet or if a new one
        # is created by the user in the same session
        if (not self.pyroot_rdf) or \
           (self.pyroot_rdf is not generator.head_node):
            self.pyroot_rdf = ROOT.ROOT.RDataFrame(*generator.head_node.args)

        values = mapper(self.pyroot_rdf)  # Execute the mapper function

        # Get the action nodes in the same order as values
        nodes = generator.get_action_nodes()

        # Retrieve action names and output to debug logger
        action_names = [node.operation.name for node in nodes]
        logger.debug("Action operations in the loop:\n{}".format(action_names))

        # values[0].GetValue()  # Trigger event-loop

        for i in range(len(values)):
            if nodes[i].operation.name in ["AsNumpy", "Snapshot"]:
                nodes[i].value = values[i]
            else:
                # Set the obtained values and
                # 'RResultPtr's of action nodes
                nodes[i].value = values[i].GetValue()
                # We store the 'RResultPtr's because,
                # those should be in scope while doing
                # a 'GetValue' call on them
                nodes[i].ResultPtr = values[i]
