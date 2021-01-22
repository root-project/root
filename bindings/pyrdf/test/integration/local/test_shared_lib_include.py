import unittest
import PyRDF
import ROOT
import subprocess
import os


class SharedLibrariesIncludeLocalTest(unittest.TestCase):
    """
    Check that the required shared libraries files are properly included.
    """

    def tearDown(self):
        """remove included libraries after analysis"""
        PyRDF.includes_headers.clear()
        PyRDF.includes_shared_libraries.clear()

    def test_includes_shared_lib_with_filter_op(self):
        """
        Check that the filter operation is able to use C++ functions that
        were include using header files.
        """
        # Paths to the cpp file that has to be compiled into a shared library
        # and the path with the output name of the library.
        # Both are relative to the current directory of this file
        cpp_path = "test_shared_libraries/a.cpp"

        library_path = "test_shared_libraries/liba.so"

        library_code = (
            "`root-config --cxx` "
            "`root-config --cflags --libs` "
            "-fPIC -shared {cpp}"
            " -o {lib}"
        ).format(cpp=cpp_path, lib=library_path)
        # This creates the shared library
        subprocess.call(library_code, shell=True)

        # Path to the shared library relative to the main PyRDF directory.
        so_path = "test_shared_libraries/liba.so"

        PyRDF.include_shared_libraries(so_path)

        # The user can include directly the header related to the library
        # or choose to declare functions or objects later
        header_path = "test_shared_libraries/a.h"
        PyRDF.include_headers(header_path)

        # Creates an RDataFrame with 5 integers [0...4]
        rdf = PyRDF.RDataFrame(5)

        # This filters out all numbers less than 3
        filter1 = rdf.Filter("tdfentry_ < 3")

        # This defines a new variable x with all entries squared
        # then filters out all x values less than 3
        filter2 = rdf.Define("x", "f(tdfentry_)").Filter("x < 3")

        count1 = filter1.Count().GetValue()
        count2 = filter2.Count().GetValue()

        # The final answer should be the number of integers
        # less than 5, which is 3, and the number of squared integers less
        # than 5, which is 2.
        self.assertEqual(count1, 3)
        self.assertEqual(count2, 2)

        # Remove unnecessary files at the end
        os.remove(so_path)

    def test_includes_shared_lib_with_pcm(self):
        """
        Check that a pcm file is correctly included with its shared library.
        """
        # Get path of current working directory
        root_dir = os.getcwd()
        # Change working directory to create shared library
        wd_path = "test_shared_libraries/"
        os.chdir(wd_path)

        h_path = "myclass.h"
        cpp_path = "myclass.cpp"
        library_path = "libmyclass.so"
        pcm_path = "myclass_rdict.pcm"

        # Create the pcm file from the header
        pcm_code = "genreflex {h} -o {cpp}".format(h=h_path, cpp=cpp_path)
        subprocess.call(pcm_code, shell=True)

        # Create the shared library in the same folder as the pcm
        library_code = (
            "`root-config --cxx` "
            "`root-config --cflags --libs` "
            "-fPIC -shared {cpp} "
            "-o {lib}"
        ).format(cpp=cpp_path, lib=library_path)
        subprocess.call(library_code, shell=True)

        # Change back to initial working directory
        os.chdir(root_dir)

        # Disable ROOT autoparsing, otherwise ROOT will be able to infer the
        # data members of the class through AST, even without reading the
        # pcm file.
        apState = ROOT.gInterpreter.SetClassAutoparsing(0)

        # Include the shared library. ROOT will know the class declared
        # inside, but not its data members (those are in the pcm file and
        # in the header).
        PyRDF.include_shared_libraries(wd_path + library_path)

        # Ask ROOT to return the representation of the class in ROOT's
        # typesystem.
        c = ROOT.TClass.GetClass("myclass")

        # Get the name of the data member from the class through ROOT
        # if succesful, it means that ROOT read it from the pcm file.
        self.assertEqual(c.GetRealData("three").GetName(), "three")

        # Re-enable ROOT AutoParsing and then parse the contents of the
        # included shared library, so that free functions will be reachable
        # by ROOT.
        ROOT.gInterpreter.SetClassAutoparsing(apState)
        ROOT.gInterpreter.AutoParse("myclass")

        # Creates an RDataFrame with 5 integers [0...4]
        rdf = PyRDF.RDataFrame(5)
        # Defines a new column with the integers doubled
        # Then filters integers less than 3
        filtered_rdf = rdf.Define("x", "n_times_two(tdfentry_)").Filter("x < 3")
        # Count integers less than 3
        count = filtered_rdf.Count().GetValue()
        # Answer should be 2
        self.assertEqual(count, 2)

        # Remove unnecessary files at the end
        os.remove(wd_path + library_path)
        os.remove(wd_path + cpp_path)
        os.remove(wd_path + pcm_path)
