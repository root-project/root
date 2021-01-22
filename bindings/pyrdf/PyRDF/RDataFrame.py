from __future__ import print_function
from PyRDF.Node import Node
import ROOT
from PyRDF.Proxy import TransformationProxy
import logging

logger = logging.getLogger(__name__)


class RDataFrame(object):
    """
    User interface to the object containing the Python equivalent of ROOT
    C++'s RDataFrame class. The purpose of this class is to kickstart the
    head node of the computational graph, together with a proxy wrapping it.
    """
    def __new__(cls, *args):
        """
        Creates the head node of the graph with the arguments provided by the
        user, then returns a proxy to that node.

        Args:
            *args (list): A list of arguments that were provided by the user
                to construct the RDataFrame object.
        """
        head_node = HeadNode(*args)
        proxy_head = TransformationProxy(head_node)

        # Logger debug statements
        logger.debug("Created RDataFrame head node and proxy")
        return proxy_head


class HeadNode(Node):
    """
    The Python equivalent of ROOT C++'s
    RDataFrame class.

    Attributes:
        args (list): A list of arguments that were provided to construct
            the RDataFrame object.


    PyRDF's RDataFrame constructor accepts the same arguments as the ROOT's
    RDataFrame constructor (see
    `RDataFrame <https://root.cern/doc/master/classROOT_1_1RDataFrame.html>`_)

    In addition, PyRDF allows you to use Python lists in place of C++ vectors
    as arguments of the constructor, example::

        PyRDF.RDataFrame("myTree", ["file1.root", "file2.root"])

    Raises:
        RDataFrameException: An exception raised when input arguments to
            the RDataFrame constructor are incorrect.

    """
    def __init__(self, *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (list): Variable length argument list to construct the
                RDataFrame object.
        """
        super(HeadNode, self).__init__(None, None, *args)

        args = list(args)  # Make args mutable
        num_params = len(args)

        for i in range(num_params):
            # Convert Python list to ROOT CPP vector
            if isinstance(args[i], list):
                args[i] = self._get_vector_from_list(args[i])

        try:
            ROOT.ROOT.RDataFrame(*args)  # Check if the args are correct
        except TypeError as e:
            msg = "Error creating the RDataFrame !"
            rdf_exception = RDataFrameException(e, msg)
            rdf_exception.__cause__ = None
            # The above line is to supress the traceback of error 'e'
            raise rdf_exception

        self.args = args

    def get_branches(self):
        """Gets list of default branches if passed by the user."""
        # ROOT Constructor:
        # RDataFrame(TTree& tree, defaultBranches = {})
        if len(self.args) == 2 and isinstance(self.args[0], ROOT.TTree):
            return self.args[1]
        # ROOT Constructors:
        # RDataFrame(treeName, filenameglob, defaultBranches = {})
        # RDataFrame(treename, filenames, defaultBranches = {})
        # RDataFrame(treeName, dirPtr, defaultBranches = {})
        if len(self.args) == 3:
            return self.args[2]

        return None

    def _get_vector_from_list(self, arg):
        """Converts a python list of strings to a vector."""
        reqd_vec = ROOT.std.vector('string')()

        for elem in arg:
            reqd_vec.push_back(elem)

        return reqd_vec

    def get_num_entries(self):
        """
        Gets the number of entries in the given dataset.

        Returns:
            int: This is the computed number of entries in the input dataset.

        """
        first_arg = self.args[0]
        if isinstance(first_arg, int):
            # If there's only one argument
            # which is an integer, return it.
            return first_arg
        elif isinstance(first_arg, ROOT.TTree):
            # If the argument is a TTree or TChain,
            # get the number of entries from it.
            return first_arg.GetEntries()

        second_arg = self.args[1]

        # Construct a ROOT.TChain object
        chain = ROOT.TChain(first_arg)

        if isinstance(second_arg, str):
            # If the second argument is a string
            chain.Add(second_arg)
        else:
            # If the second argument is a list or vector
            for fname in second_arg:
                chain.Add(str(fname))

        return chain.GetEntries()

    def get_treename(self):
        """
        Get name of the TTree.

        Returns:
            (str, None): Name of the TTree, or :obj:`None` if there is no tree.

        """
        first_arg = self.args[0]
        if isinstance(first_arg, ROOT.TChain):
            # Get name from a given TChain
            return first_arg.GetName()
        elif isinstance(first_arg, ROOT.TTree):
            # Get name directly from the TTree
            return first_arg.GetUserInfo().At(0).GetName()
        elif isinstance(first_arg, str):
            # First argument was the name of the tree
            return first_arg
        # RDataFrame may have been created without any TTree or TChain
        return None

    def get_tree(self):
        """
        Get ROOT.TTree instance used as an argument to PyRDF.RDataFrame()

        Returns:
            (ROOT.TTree, None): instance of the tree used to instantiate the
            RDataFrame, or `None` if another object was used. ROOT.Tchain
            inherits from ROOT.TTree so that can be the return value as well.
        """
        first_arg = self.args[0]
        if isinstance(first_arg, ROOT.TTree):
            return first_arg

        return None

    def get_inputfiles(self):
        """
        Get list of input files.

        This list can be extracted from a given TChain or from the list of
        arguments.

        Returns:
            (str, list, None): Name of a single file, list of files (both may
            contain globbing characters), or None if there are no input files.

        """
        first_arg = self.args[0]
        if isinstance(first_arg, ROOT.TChain):
            # Extract file names from a given TChain
            chain = first_arg
            return [chainElem.GetTitle()
                    for chainElem in chain.GetListOfFiles()]
        if len(self.args) > 1:
            second_arg = self.args[1]
            if (isinstance(second_arg, str) or
               isinstance(second_arg, ROOT.std.vector('string'))):
                # Get file(s) from second argument
                # (may contain globbing characters)
                return second_arg
        # RDataFrame may have been created with no input files
        return None


class RDataFrameException(Exception):
    """
    A special type of Exception that shows up for incorrect arguments to
    RDataFrame.
    """
    def __init__(self, exception, msg):
        """
        Creates a new `RDataFrameException`.

        Args:
            exception: An exception of type :obj:`Exception` or any child
                class of :obj:`Exception`.

            msg (str): Message to be printed while raising exception.
        """
        super(RDataFrameException, self).__init__(exception)
        print(msg)
