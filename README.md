# roottest

Tests in the roottest repository can be added and executed using CMake and
CTest. These tests are executed as part of ROOT's Continuous Integration setup.

## Run tests locally


### Configuration and Build

There are two ways to generate and execute the tests:

#### Option 1: roottest as part of a ROOT build

Before building, enable the 'testing' and 'roottest' option:

       cmake -Dtesting=ON -Droottest=ON $PATH_TO_ROOT_SOURCES
       cmake --build . -j8

#### Option 2: roottest as a stand-alone project

Set the ROOT environment to an existing build / installation:

       . ${ROOTSYS}/thisroot.[c]sh

Create a build directory, change into it and execute

       cmake $PATH_TO_ROOTTEST
       cmake --build . -j8

### Running the test suite

Tests can then be executed using the ctest command in the build directory:

    ctest -N                (list all tests)
    ctest -j4               (run all tests in parallel)
    ctest -R regex          (run all tests matching regex)
    ctest -E regex          (run all tests not matching regex)
    ctest -V                (verbose output)
    ctest -L regex          (run all tests that contain the label regex)
    ctest --print-labels    (list all existing labels)

You can combine most of the ctest options, e.g. ctest -V -j4 -R root-meta is a
valid call.


## Adding tests

Tests can be defined in a CMakeLists.txt file via custom CMake functions and
macros. Either put the new test into an existing directory and edit its
CMakeLists.txt or create a fresh directory and make it known to the testing
system. roottest detects test directories by calling the function
ROOTTEST_ADD_TESTDIRS(). Each subdirectory of the current directory is
automatically added, if it contains a CMakeLists.txt file.

In order to add a test, the function ROOTTEST_ADD_TEST has been introduced.
A basic description follows, for details take a look at the implementation
in roottest/cmake/modules/RootCTestMacros.cmake).

Synopsis (shortened):

    function ROOTTEST_ADD_TEST(testname
                               MACRO|EXEC|COMMAND macro_or_command
                               [MACROARG arg1 arg2 ...]
                               [OPTS opt1 opt2 ...]
                               [OUTREF stdout_reference_file]
                               [ERRREF stderr reference_file]
                               [WORKING_DIR dir]
                               [COPY_TO_BUILDDIR fil1 file2 ...]
                               [PRECMD command args ...]
                               [POSTCMD command args ...]
                               [OUTCNVCMD script_or_program]
                               [FAILREGEX regexp]
                               [PASSREGEX regexp]
                               [DEPENDS dependency1 dependency2 ...]
                               [WILLFAIL])

Description:

    ROOTTEST_ADD_TEST creates a new test case testname that executes either
    a provided ROOT macro or calls an executable macro_or_command when called
    by ctest.

Options:

    MACRO               ROOT macro file. If set, the resulting test will then
                        execute root.exe -q -l -b macro_file. It only accepts
                        macros that have a .C/.C+/.cxx/.cxx+/.py file extension.
                        If macro_file does not follow this requirement it is
                        simply redirected to root.exe (i.e. no special treatment
                        like resolving the full path of the macro is done).

    EXEC                Execute program in $PATH or current directory.
    
    COMMAND             Execute a command (program or script) with its arguments

    MACROARG            Arguments that shall be passed to the specified
                        ROOT macro (root.exe macro(MACROARGS)).

    OPTS                Options that will be appended to ROOT or an executable.

    OUTREF              File that references the stdout output of the test. If
                        the systems architecture influences the test output, a
                        *.ref32 / *.ref64 file can be created. They will be
                        preferred to a *.ref file.  
                        
    ERRREF              File that references the stderr output of the test.
                        For system architecture specific output, see option
                        OUTREF.

    WORKING_DIR         Set the tests working directory. As a default, the
                        current CMAKE build directory is used.

    COPY_TO_BUILDDIR    Copy files into the current build directory, so tests
                        find them without the need for full paths.
    
    PRECMD              Command to be executed before the macro or the executable.
                        Only the return code is checked. If error code is returned
                        the overall test fails.
                        
    POSTCMD             Command to be executed after the macro or the executable.
                        Only the return code is checked.
                        
    OUTCNVCMD           Possibility to process the output before is given to the
                        diff utility to check it against the reference file.
                        
    PASSREGEX           Property to verify that the output of the test contains
                        certain strings (regular expression) to pass
                        
    FAILREGEX           Property to verify that the output of the test contains
                        certain strings (regular expression) to fail

    DEPENDS             Specify tests that must execute before the new test
                        is run.

    WILLFAIL            Flag that marks the test as expected to fail.

Examples:

1. Simply add a ROOT macro test.

        ROOTTEST_ADD_TEST(assertSparseToTHn
                          MACRO assertSparseToTHn.C)


2. Add a compiled ROOT macro test (note the +) and compare its stdout and
       stderr output to the given reference files. Associate the test to the
       labels roottest, regression and cling.

        ROOTTEST_ADD_TEST(compiled
                          MACRO runvbase.C+
                          OUTREF vbase-c.out.ref
                          ERRREF vbase-c.err.ref
                          LABELS roottest regression cling)

Some tests may require ROOT dictionaries or rootmaps. They can be generated
using the ROOTTEST_GENERATE_DICTIONARY() and
ROOTTEST_GENERATE_REFLEX_DICTIONARY() macros, e.g.

    1. ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes          # dictionary name
                                           classes.h        # header files
                                           SELECTION classes_selection.xml)

    2. ROOTTEST_GENERATE_DICTIONARY(scopeDict scopeProblem.C
                                    LINKDEF linkdef.h)

Invoking these macros also sets the variables GENERATE_REFLEX_TEST and
GENERATE_DICTIONARY_TEST to the internal CTest test name. They can be
used to manage dependencies on dictionaries:

    1. ROOTTEST_GENERATE_DICTIONARY(TTestClass_h LINKDEF TTestClass.h)

    2. ROOTTEST_ADD_TEST(runTTestClass
                         MACRO runTTestClass.C
                         OUTREF TTestClass.ref
                         DEPENDS ${GENERATE_DICTIONARY_TEST}
                         LABELS roottest regression cling)


## Advanced / developers' features

### Adding definitions and ClingWorkarounds

Definitions to root macro tests can be added by calling add_definitions(-DDEF).
ClingWorkarounds are supposed to be specified in
roottest/cmake/modules/RoottestCTest.cmake. Always add a pair consisting of the
workaround definition and a introduce a variable that can be checked to see,
if a workaround is active.


### Set test owner

The owner of a test can be set by calling ROOTTEST_SET_TESTOWNER("Test Owner").


### Ignoring tests

An existing test may be marked to be ignored. This is done by adding its name
to the CTEST_CUSTOM_TESTS_IGNORE variable in roottest/CTestCustom.cmake.
