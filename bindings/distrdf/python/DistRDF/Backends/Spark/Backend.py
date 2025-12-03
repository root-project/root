# @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import ntpath  # Filename from path (should be platform-independent)
import warnings
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING, Union, Tuple

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF.Backends import Base
from DistRDF.Backends import Utils

try:
    import pyspark
except ImportError:
    raise ImportError(("cannot import module 'pyspark'. Refer to the Apache Spark documentation "
                       "for installation instructions."))


class SparkBackend(Base.BaseBackend):
    """
    Backend that executes the computational graph using using `Spark` framework
    for distributed execution.

    """

    def __init__(self, sparkcontext=None):
        """
        Creates an instance of the Spark backend class.

        Args:
            config (dict, optional): The config options for Spark backend.
                The default value is an empty Python dictionary :obj:`{}`.
                :obj:`config` should be a dictionary of Spark configuration
                options and their values with :obj:'npartitions' as the only
                allowed extra parameter.

        Example::

            config = {
                'npartitions':20,
                'spark.master':'myMasterURL',
                'spark.executor.instances':10,
                'spark.app.name':'mySparkAppName'
            }

        Note:
            If a SparkContext is already set in the current environment, the
            Spark configuration parameters from :obj:'config' will be ignored
            and the already existing SparkContext would be used.

        """
        super(SparkBackend, self).__init__()

        if sparkcontext is not None:
            self.sc = sparkcontext
        else:
            self.sc = pyspark.SparkContext.getOrCreate()

    def optimize_npartitions(self) -> int:
        """
        The SparkContext.defaultParallelism property roughly translates to the
        available amount of logical cores on the cluster. Some examples:
        - spark.master = local[n]: returns n.
        - spark.executor.instances = m and spark.executor.cores = n: returns `n*m`.
        By default, the minimum number this returns is 2 if the context
        doesn't know any better. For example, if dynamic allocation is enabled.
        """
        return self.sc.defaultParallelism

    def ProcessAndMerge(self,
                        ranges: List[Any],
                        mapper: Callable[[Ranges.DataRange,
                                        Callable[[Union[Ranges.EmptySourceRange, Ranges.TreeRangePerc]],
                                                    Base.TaskObjects],
                                        Callable[[ROOT.RDF.RNode, int], List],
                                        Callable],
                                        Base.TaskResult],
                        reducer: Callable[[Base.TaskResult, Base.TaskResult], Base.TaskResult],
                        ) -> Base.TaskResult:
        """
        Performs map-reduce using Spark framework.

        Args:
            mapper (function): A function that runs the computational graph
                and returns a list of values.

            reducer (function): A function that merges two lists that were
                returned by the mapper.

        Returns:
            list: A list representing the values of action nodes returned
            after computation (Map-Reduce).
        """

        # These need to be passed as variables and not as class attributes
        # otherwise the `spark_mapper` function would be referencing
        # this instance of the Spark backend along with the referenced
        # SparkContext. This would cause the errors described in SPARK-5063.
        headers = self.headers
        shared_libraries = self.shared_libraries
        pcms = self.pcms
        files = self.files
        
        def spark_mapper(current_range):
            """
            Gets the paths to the file(s) in the current executor, then
            declares the headers found.

            Args:
                current_range (tuple): A pair that contains the starting and
                    ending values of the current range.

            Returns:
                function: The map function to be executed on each executor,
                complete with all headers needed for the analysis.
            """
            # Get and declare headers on each worker
            headers_on_executor = [
                pyspark.SparkFiles.get(ntpath.basename(filepath))
                for filepath in headers
            ]
            Utils.distribute_headers(headers_on_executor)

            # Get and declare shared libraries on each worker
            shared_libs_on_ex = [
                pyspark.SparkFiles.get(ntpath.basename(filepath))
                for filepath in shared_libraries
            ]
            Utils.distribute_shared_libraries(shared_libs_on_ex)

            return mapper(current_range)
        
        self.distribute_unique_paths(headers)
        self.distribute_unique_paths(shared_libraries)
        self.distribute_unique_paths(pcms)
        self.distribute_unique_paths(files)

        # Build parallel collection
        parallel_collection = self.sc.parallelize(ranges, len(ranges))

        # Map-Reduce using Spark
        return parallel_collection.map(spark_mapper).treeReduce(reducer)

    def ProcessAndMergeLive(self, ranges, mapper, reducer, drawables_info_dict):
        """
        Informs the user that the live visualization feature is not supported for the Spark backend 
        and refers to ProcessAndMerge to proceed with the standard map-reduce workflow.
        """
        warnings.warn("The live visualization feature is not supported for the Spark backend. Skipping LiveVisualize.")
        return self.ProcessAndMerge(ranges, mapper, reducer)

    def distribute_unique_paths(self, paths):
        """
        Spark supports sending files to the executors via the
        `SparkContext.addFile` method. This method receives in input the path
        to the file (relative to the path of the current python session). The
        file is initially added to the Spark driver and then sent to the
        workers when they are initialized.

        Args:
            paths (set): A set of paths to files that should be sent to the
                distributed workers.
        """
        for filepath in paths:
            self.sc.addFile(filepath)

    def make_dataframe(self, *args, **kwargs):
        """
        Creates an instance of distributed RDataFrame that can send computations
        to a Spark cluster.
        """
        # Set the number of partitions for this dataframe, one of the following:
        # 1. User-supplied `npartitions` optional argument
        npartitions = kwargs.pop("npartitions", None)
        headnode = HeadNode.get_headnode(self, npartitions, *args)
        return DataFrame.RDataFrame(headnode)
