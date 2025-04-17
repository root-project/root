#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import concurrent.futures
import logging
import textwrap
import types
import warnings
from typing import TYPE_CHECKING, Iterable

from DistRDF.Backends import build_backends_submodules
from DistRDF.LiveVisualize import LiveVisualize

if TYPE_CHECKING:
    from DistRDF.Proxy import ResultMapProxy, ResultPtrProxy

logger = logging.getLogger(__name__)


def initialize(fun, *args, **kwargs):
    """
    Set a function that will be executed as a first step on every backend before
    any other operation. This method also executes the function on the current
    user environment so changes are visible on the running session.

    This allows users to inject and execute custom code on the worker
    environment without being part of the RDataFrame computational graph.

    Args:
        fun (function): Function to be executed.

        *args (list): Variable length argument list used to execute the
            function.

        **kwargs (dict): Keyword arguments used to execute the function.
    """
    from DistRDF.Backends import Base

    Base.BaseBackend.register_initialization(fun, *args, **kwargs)


def DistributeCppCode(code_to_declare: str) -> None:
    """
    Declare the C++ code that has to be processed on each worker.
    Args:
        codeToDeclare (str): cpp code to be declared on the workers

    """
    from DistRDF.Backends import Base

    Base.BaseBackend.register_declaration(code_to_declare)


def DistributeHeaders(paths_to_headers: Iterable[str]):
    """
    This function allows users to directly load C++ custom headers
    onto the workers. The headers are declared locally first.

    Args:
        paths_to_headers (list): list of paths to headers to be distributed to each worker

    """
    from DistRDF.Backends import Base

    Base.BaseBackend.register_headers(paths_to_headers)


def DistributeFiles(paths_to_files: Iterable[str]):
    """
    This function allows users to directly load arbitrary files
    onto the workers.

    Args:
        paths_to_files (list): list of paths to files to be distributed

    """
    from DistRDF.Backends import Base

    Base.BaseBackend.register_files(paths_to_files)


def DistributeSharedLibs(paths_to_shared_libraries: Iterable[str]) -> None:
    """
    This function allows users to directly load pre-compiled shared libraries
    onto the workers. The shared libraries are loaded locally first.

    Args:
        paths_to_shared_libraries (list): list of paths to shared libraries to be distributed

    """
    from DistRDF.Backends import Base

    Base.BaseBackend.register_shared_lib(paths_to_shared_libraries)


def RunGraphs(proxies: Iterable) -> int:
    """
    Trigger the execution of multiple RDataFrame computation graphs on a certain
    distributed backend. If the backend doesn't support multiple job
    submissions concurrently, the distributed computation graphs will be
    executed sequentially.

    Args:
        proxies(list): List of action proxies that should be triggered. Only
            actions belonging to different RDataFrame graphs will be
            triggered to avoid useless calls.

    Return:
        (int): The number of unique computation graphs executed by this call.


    Example:

        @code{.py}
        import ROOT

        # Create 3 different dataframes and book an histogram on each one
        histoproxies = [
            ROOT.RDataFrame(100, executor=SupportedExecutor(...))
                .Define("x", "rdfentry_")
                .Histo1D(("name", "title", 10, 0, 100), "x")
            for _ in range(4)
        ]

        # Execute the 3 computation graphs
        n_graphs_run = ROOT.RDF.RunGraphs(histoproxies)
        # Retrieve all the histograms in one go
        histos = [histoproxy.GetValue() for histoproxy in histoproxies]
        @endcode


    """
    # Import here to avoid circular dependencies in main module
    from DistRDF.Proxy import execute_graph

    if not proxies:
        logger.warning("RunGraphs: Got an empty list of handles, now quitting.")
        return 0

    # Get proxies belonging to distinct computation graphs
    uniqueproxies = list({proxy.proxied_node.get_head(): proxy for proxy in proxies}.values())

    # Submit all computation graphs concurrently from multiple Python threads.
    # The submission is not computationally intensive
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(uniqueproxies)) as executor:
        futures = [executor.submit(execute_graph, proxy.proxied_node) for proxy in uniqueproxies]
        concurrent.futures.wait(futures)

    return len(uniqueproxies)


def VariationsFor(actionproxy: ResultPtrProxy) -> ResultMapProxy:
    """
    Equivalent of ROOT.RDF.Experimental.VariationsFor in distributed mode.
    """
    # similar to resPtr.fActionPtr->MakeVariedAction()
    return actionproxy.create_variations()


def FromSpec(jsonfile: str, *args, **kwargs) -> RDataFrame:
    """
    Equivalent of ROOT.RDF.Experimental.FromSpec in distributed mode.
    """
    import ROOT

    spec = ROOT.Internal.RDF.RetrieveSpecFromJson(jsonfile)

    executor = kwargs.get("executor", None)
    if executor is None:
        raise ValueError(
            "Missing keyword argument 'executor'. Please provide a connection object "
            "to one of the schedulers supported by distributed RDataFrame."
        )
    # Try to dispatch to the correct distributed scheduler implementation
    try:
        from distributed import Client

        from DistRDF.Backends.Dask import RDataFrame

        if isinstance(executor, Client):
            return RDataFrame(spec, *args, **kwargs)
    except ImportError:
        pass

    try:
        from pyspark import SparkContext

        from DistRDF.Backends.Spark import RDataFrame

        if isinstance(executor, SparkContext):
            return RDataFrame(spec, *args, **kwargs)
    except ImportError:
        pass

    raise TypeError(
        f"The client object of type '{type(executor)}' is not a supported connection type for distributed RDataFrame."
    )


class _DeprecatedModule(types.ModuleType):
    """A simple module type to raise a warning before usage."""

    def __getattribute__(self, name):
        msg_warng = textwrap.dedent(
            """
            In ROOT 6.36, the ROOT.RDF.Experimental.Distributed module has become just ROOT.RDF.Distributed. ROOT 6.38
            will remove the 'Experimental' keyword completely, so it is suggested to move to the stable API in user 
            code. You can now change lines such as:
            ```
            connection = ... # your distributed Dask client or SparkContext
            RDataFrame = ROOT.RDF.Experimental.Distributed.[Backend].RDataFrame
            df = RDataFrame(..., [daskclient,sparkcontext] = connection)
            ```
            to simply:
            ```
            connection = ... # your distributed Dask client or SparkContext
            df = ROOT.RDataFrame(..., executor = connection)
            ```
            """
        )
        warnings.warn(msg_warng, FutureWarning)
        return super().__getattribute__(name)


def create_distributed_module(parentmodule, experimental: bool = False):
    """
    Helper function to create the ROOT.RDF.Distributed module.

    Users will see this module as the entry point of functions to create and
    run an RDataFrame computation distributedly.
    """
    distributed = types.ModuleType("ROOT.RDF.Distributed")

    # PEP302 attributes
    distributed.__file__ = "<module ROOT.RDF>"
    # distributed.__name__ is the constructor argument
    distributed.__path__ = []  # this makes it a package
    # distributed.__loader__ is not defined
    distributed.__package__ = parentmodule

    distributed = build_backends_submodules(distributed)

    # Inject top-level functions
    distributed.initialize = initialize
    distributed.RunGraphs = RunGraphs
    distributed.VariationsFor = VariationsFor
    distributed.LiveVisualize = LiveVisualize
    distributed.DistributeHeaders = DistributeHeaders
    distributed.DistributeFiles = DistributeFiles
    distributed.DistributeSharedLibs = DistributeSharedLibs
    distributed.DistributeCppCode = DistributeCppCode
    distributed.FromSpec = FromSpec

    if experimental:
        distributed.__class__ = _DeprecatedModule

    return distributed


def RDataFrame(*args, **kwargs):
    executor = kwargs.get("executor", None)
    if executor is None:
        raise ValueError(
            "Missing keyword argument 'executor'. Please provide a connection object "
            "to one of the schedulers supported by distributed RDataFrame."
        )

    # Try to dispatch to the correct distributed scheduler implementation
    try:
        from distributed import Client

        from DistRDF.Backends.Dask import RDataFrame

        if isinstance(executor, Client):
            return RDataFrame(*args, **kwargs)
    except ImportError:
        pass

    try:
        from pyspark import SparkContext

        from DistRDF.Backends.Spark import RDataFrame

        if isinstance(executor, SparkContext):
            return RDataFrame(*args, **kwargs)
    except ImportError:
        pass

    raise TypeError(
        f"The client object of type '{type(executor)}' is not a supported connection type for distributed RDataFrame."
    )
