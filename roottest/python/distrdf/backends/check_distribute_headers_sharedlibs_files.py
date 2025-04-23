import ROOT
import subprocess
from pathlib import Path
from distributed import get_worker
import os
import pytest
import math


class TestInterfaceHeadersLibrariesFiles:
    """
    Check that the interface functions(distribute headers, shared libraries and files) work properly.
    """

    def _create_shared_libs(self):
        subprocess.run(
            [
                "g++",
                "-fPIC",
                "../test_shared_libs/mysource6.cpp",
                "-shared",
                "-o",
                "../test_shared_libs/mylib6.so",
            ]
        )
        subprocess.run(
            [
                "g++",
                "-fPIC",
                "../test_shared_libs/mysource7.cpp",
                "-shared",
                "-o",
                "../test_shared_libs/mylib7.so",
            ]
        )

    def _remove_shared_libs(self):
        os.remove("../test_shared_libs/mylib6.so")
        os.remove("../test_shared_libs/mylib7.so")

    def _check_rdf_histos_5(self, rdf):
        # This filters out all numbers less than 5
        rdf_filtered = rdf.Filter("check_number_less_than_5(rdfentry_)")
        histo = rdf_filtered.Histo1D(("name", "title", 10, 0, 10), "rdfentry_")
        # The expected results after filtering
        # The actual set of numbers required after filtering
        required_numbers = range(5)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(
            sum((x - required_mean) ** 2 for x in required_numbers) / required_size
        )
        # Compare the sizes of equivalent set of numbers
        assert histo.GetEntries() == required_size
        # Compare the means of equivalent set of numbers
        assert histo.GetMean() == required_mean
        # Compare the standard deviations of equivalent set of numbers
        assert histo.GetStdDev() == required_stdDev

    def _check_rdf_histos_6(self, rdf):
        rdf_filtered = rdf.Filter("check_number_less_than_6(rdfentry_)")
        histo = rdf_filtered.Histo1D(("name", "title", 10, 0, 10), "rdfentry_")

        required_numbers = range(6)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(
            sum((x - required_mean) ** 2 for x in required_numbers) / required_size
        )
        assert histo.GetEntries() == required_size
        assert histo.GetMean() == required_mean
        assert histo.GetStdDev() == required_stdDev

    def _check_rdf_histos_7(self, rdf):
        rdf_filtered = rdf.Filter("check_number_less_than_7(rdfentry_)")
        histo = rdf_filtered.Histo1D(("name", "title", 15, 0, 15), "rdfentry_")
        required_numbers = range(7)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(
            sum((x - required_mean) ** 2 for x in required_numbers) / required_size
        )
        assert histo.GetEntries() == required_size
        assert histo.GetMean() == required_mean
        assert histo.GetStdDev() == required_stdDev

    def _distribute_header_check_filter_and_histo(self, connection):
        """
        Check that the filter operation is able to use C++ functions that
        were included using header files.
        """
        rdf = ROOT.RDataFrame(10, executor=connection)

        ROOT.RDF.Distributed.DistributeHeaders(
            "../test_headers/header1.hxx"
        )

        self._check_rdf_histos_5(rdf)

    def _extend_ROOT_include_path(self, connection):
        """
        Check that the include path of ROOT is extended with the directories
        specified in `DistRDF.include_headers()` so references between headers
        are correctly solved.
        """
        header_folder = "../test_headers/headers_folder"

        # Create an RDataFrame with 100 integers from 0 to 99
        rdf = ROOT.RDataFrame(100, executor=connection)

        ROOT.RDF.Distributed.DistributeHeaders(header_folder)
        # Get list of include paths seen by ROOT
        ROOT_include_path = ROOT.gInterpreter.GetIncludePath().split(" ")

        # Create new include folder token
        new_folder_include = '-I"{}"'.format(header_folder)

        # Check that new folder is in ROOT include paths
        assert new_folder_include in ROOT_include_path

        # Filter numbers less than 10 and create an histogram
        rdf_less_than_10 = rdf.Filter("check_number_less_than_10(rdfentry_)")
        histo1 = rdf_less_than_10.Histo1D(
            ("name", "title", 10, 0, 100), "rdfentry_")

        # Check that histogram has 10 entries and mean 4.5
        assert histo1.GetEntries() == 10
        assert histo1.GetMean() == pytest.approx(4.5)

    def _distribute_shared_lib_check_filter_and_histo(self, connection):
        """
        Check that the filter operation is able to use C++ functions that
        were included using a single shared library.
        """
        rdf = ROOT.RDataFrame(15, executor=connection)

        ROOT.RDF.Distributed.DistributeHeaders(
            "../test_shared_libs/myheader7.h"
        )
        ROOT.RDF.Distributed.DistributeSharedLibs(
            "../test_shared_libs/mylib7.so"
        )
        self._check_rdf_histos_7(rdf)

    def _distribute_shared_lib_folder_check_filter_and_histo(self, connection):
        """
        Check that the filter operation is able to use C++ functions that
        were included using a single shared library which is in a folder of multiple libraries and only the folder is distributed.
        """
        rdf = ROOT.RDataFrame(15, executor=connection)

        ROOT.RDF.Distributed.DistributeHeaders(
            "../test_shared_libs/myheader6.h"
        )
        ROOT.RDF.Distributed.DistributeSharedLibs(
            "../test_shared_libs/")
        self._check_rdf_histos_6(rdf)

    def _distribute_multiple_shared_lib_check_filter_and_histo(
        self, connection
    ):
        """
        Check that the filter operation is able to use C++ functions that
        were included using multiple shared libraries.
        """
        rdf = ROOT.RDataFrame(15, executor=connection)

        ROOT.RDF.Distributed.DistributeHeaders(
            ["../test_shared_libs/myheader7.h", "../test_shared_libs/myheader6.h"]
        )
        ROOT.RDF.Distributed.DistributeSharedLibs(
            ["../test_shared_libs/mylib7.so", "../test_shared_libs/mylib6.so"]
        )
        self._check_rdf_histos_6(rdf)
        self._check_rdf_histos_7(rdf)

    def _distribute_multiple_shared_lib_folder_check_filter_and_histo(
        self, connection
    ):
        """
        Check that the filter operation is able to use C++ functions that
        were included using multiple shared libraries while only distributing the folder with shared libraries.
        """
        rdf = ROOT.RDataFrame(15, executor=connection)

        ROOT.RDF.Distributed.DistributeHeaders(
            ["../test_shared_libs/myheader7.h", "../test_shared_libs/myheader6.h"]
        )
        ROOT.RDF.Distributed.DistributeSharedLibs(
            "../test_shared_libs/")
        self._check_rdf_histos_6(rdf)
        self._check_rdf_histos_7(rdf)

    def _distribute_single_file(self, connection, backend):
        # For spark we are using spark "addFile" function directly
        # hence this is not tested here directly.

        rdf = ROOT.RDataFrame(10, executor=connection)

        ROOT.RDF.Distributed.DistributeFiles(
            "../test_files/file.txt")

        if backend == "dask":

            def Foo():
                try:
                    localdir = get_worker().local_directory

                    if os.path.exists(Path(localdir) / "file.txt"):
                        os.environ["ENV"] = localdir
                except ValueError:
                    pass
                ROOT.gInterpreter.Declare(
                    """
                #ifndef CODE_ENV
                #define CODE_ENV
                bool isEnv(){
                    return  gSystem->Getenv("ENV");
                }
                #endif
                """
                )

            ROOT.RDF.Distributed.initialize(Foo)
            df_flag = rdf.Define("flags", "isEnv()")
            countFlags = df_flag.Sum("flags").GetValue()
            assert countFlags == 10.0

    def _distribute_multiple_files(self, connection, backend):
        # For spark we are using spark "addFile" function directly
        # hence this is not tested here directly.

        rdf = ROOT.RDataFrame(10, executor=connection)

        ROOT.RDF.Distributed.DistributeFiles(
            ["../test_files/file.txt", "../test_files/file_1.txt"]
        )

        if backend == "dask":

            def Foo():
                try:
                    localdir = get_worker().local_directory
                    if os.path.exists(Path(localdir) / "file.txt") and os.path.exists(
                        Path(localdir) / "file_1.txt"
                    ):
                        os.environ["ENV"] = localdir
                except ValueError:
                    pass
                ROOT.gInterpreter.Declare(
                    """
                #ifndef CODE_ENV
                #define CODE_ENV
                bool isEnv(){
                    return  gSystem->Getenv("ENV");
                }
                #endif
                """
                )

            ROOT.RDF.Distributed.initialize(Foo)
            df_flag = rdf.Define("flags", "isEnv()")
            countFlags = df_flag.Sum("flags").GetValue()
            assert countFlags == 10.0

    @pytest.fixture(autouse=True)
    def setup_and_clean(self):
        self._create_shared_libs()
        yield
        from DistRDF.Backends.Base import BaseBackend

        BaseBackend.headers = set()
        BaseBackend.shared_libraries = set()
        BaseBackend.pcms = set()
        BaseBackend.files = set()
        self._remove_shared_libs()

    def test_check_single_headers(self, payload):
        connection, _ = payload
        self._distribute_header_check_filter_and_histo(connection)

    def test_distribute_multiple_headers(self, payload):
        connection, _ = payload
        self._extend_ROOT_include_path(connection)

    def test_distribute_single_library(self, payload):
        connection, _ = payload
        self._distribute_shared_lib_check_filter_and_histo(connection)

    def test_distribute_single_library_from_folder(self, payload):
        connection, _ = payload
        self._distribute_shared_lib_folder_check_filter_and_histo(connection)

    def test_distribute_multiple_libraries(self, payload):
        connection, _ = payload
        self._distribute_multiple_shared_lib_check_filter_and_histo(connection)

    def test_distribute_multiple_libraries_from_folder(self, payload):
        connection, _ = payload
        self._distribute_multiple_shared_lib_folder_check_filter_and_histo(
            connection
        )

    def test_distribute_single_file(self, payload):
        connection, backend = payload
        self._distribute_single_file(connection, backend)

    def test_distribute_multiple_files(self, payload):
        connection, backend = payload
        self._distribute_multiple_files(connection, backend)


if __name__ == "__main__":
    pytest.main(args=[__file__])
