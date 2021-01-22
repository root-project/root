from __future__ import print_function
from PyRDF.backend.Dist import Dist
from PyRDF.backend.Utils import Utils
from pyspark import SparkConf, SparkContext
from pyspark import SparkFiles
import ntpath  # Filename from path (should be platform-independent)


class Spark(Dist):
    """
    Backend that executes the computational graph using using `Spark` framework
    for distributed execution.

    """

    MIN_NPARTITIONS = 2

    def __init__(self, config={}):
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
        super(Spark, self).__init__(config)

        sparkConf = SparkConf().setAll(config.items())
        self.sparkContext = SparkContext.getOrCreate(sparkConf)

        # Set the value of 'npartitions' if it doesn't exist
        self.npartitions = self._get_partitions()

    def _get_partitions(self):
        npart = (self.npartitions or
                 self.sparkContext.getConf().get('spark.executor.instances') or
                 Spark.MIN_NPARTITIONS)
        # getConf().get('spark.executor.instances') could return a string
        return int(npart)

    def ProcessAndMerge(self, mapper, reducer):
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
        from PyRDF import includes_headers
        from PyRDF import includes_shared_libraries

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
                SparkFiles.get(ntpath.basename(filepath))
                for filepath in includes_headers
            ]
            Utils.declare_headers(headers_on_executor)

            # Get and declare shared libraries on each worker
            shared_libs_on_ex = [
                SparkFiles.get(ntpath.basename(filepath))
                for filepath in includes_shared_libraries
            ]
            Utils.declare_shared_libraries(shared_libs_on_ex)

            return mapper(current_range)

        ranges = self.build_ranges()  # Get range pairs

        # Build parallel collection
        sc = self.sparkContext
        parallel_collection = sc.parallelize(ranges, self.npartitions)

        # Map-Reduce using Spark
        return parallel_collection.map(spark_mapper).treeReduce(reducer)

    def distribute_files(self, includes_list):
        """
        Spark supports sending files to the executors via the
        `SparkContext.addFile` method. This method receives in input the path
        to the file (relative to the path of the current python session). The
        file is initially added to the Spark driver and then sent to the
        workers when they are initialized.

        Args:
            includes_list (list): A list consisting of all necessary C++
                files as strings, created one of the `include` functions of
                the PyRDF API.
        """
        for filepath in includes_list:
            self.sparkContext.addFile(filepath)
