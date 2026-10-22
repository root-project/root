if(NOT test_distrdf_dask AND NOT test_distrdf_pyspark)
    return()
endif()

if(test_distrdf_dask)
    list(APPEND DISTRDF_BACKENDS_IN_USE "dask")
    list(APPEND DISTRDF_RESOURCE_LOCKS "dask_resource_lock")
endif()

if(test_distrdf_pyspark)

    # Define environment variables needed in all pyspark tests
    set(PYSPARK_ENV_VARS PYSPARK_PYTHON=${Python3_EXECUTABLE})

    if(CMAKE_SYSTEM_NAME STREQUAL Darwin AND CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL 10.13)
        # MacOS has changed rules about forking processes after 10.13
        # Running pyspark tests with XCode Python3 throws crashes with errors like:
        # `objc[17271]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.`
        # This issue should have been fixed after Python 3.8 (see https://bugs.python.org/issue33725)
        # Indeed, any other Python 3.8+ executable does not show this crash. It is
        # specifically the XCode Python executable that triggers this.
        # For now, there seems no other way than this workaround,
        # which effectively sets the behaviour of `fork` back to MacOS 10.12
        list(APPEND PYSPARK_ENV_VARS OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES)
    endif()

    list(APPEND DISTRDF_ENVIRONMENT_VARS ${PYSPARK_ENV_VARS})
    list(APPEND DISTRDF_BACKENDS_IN_USE "spark")
    list(APPEND DISTRDF_RESOURCE_LOCKS "spark_resource_lock")
endif()


# Signal which backends are in use to the test environment
# Change semicolon to comma here to avoid breaking ENVIRONMENT argument later on
string(REPLACE ";" "," DISTRDF_BACKENDS_IN_USE "${DISTRDF_BACKENDS_IN_USE}")
list(APPEND DISTRDF_ENVIRONMENT_VARS DISTRDF_BACKENDS_IN_USE=${DISTRDF_BACKENDS_IN_USE})

# Disable the XRootD client's fork handlers in the test processes. Some tests
# read files from root://eospublic.cern.ch, which leaves an idle XrdCl
# connection in the driver process. The driver later forks (the PySpark JVM is
# launched through Popen with a preexec_fn, which runs the pthread_atfork
# handlers), and with XRootD 6.0.x and 6.1.0 the fork handling confuses the
# client's poller: when the idle connection is then closed (TTL expiry or a
# server-side disconnect) the poller thread deadlocks and the process never
# exits, so the test times out even though all tests passed. This is XRootD
# issue https://github.com/xrootd/xrootd/issues/2879, fixed in XRootD 6.1.1.
# The handlers are only needed by children that keep using inherited XrdCl
# connections, which none of the processes spawned here do.
# TODO: Remove once no CI platform ships an affected XRootD, most likely when
# Ubuntu 26.10 is end-of-life.
list(APPEND DISTRDF_ENVIRONMENT_VARS XRD_RUNFORKHANDLER=0)

# We use the PROCESSORS property to tell cmake how many cores we will be
# using in the following test. In this folder, mock cluster connections will be
# created, each using 2 cores with multiprocessing, we give it more room by
# setting the property to 4. The test also locks a resource for the creation of
# the clusters, depending on how many backends are active. The resource lock
# is shared with the "common" folder and the tutorials of the main repository.
ROOT_ADD_PYUNITTEST(test_all test_all.py GENERIC ENVIRONMENT "${DISTRDF_ENVIRONMENT_VARS}")

# This test has to take multiple resource locks. This means that they should
# be passed as a cmake list (semi-colon separated strings) after the
# RESOURCE_LOCK argument to `set_tests_properties`. The issue lies in
# forwarding this argument to ROOTTEST_ADD_TEST and then ROOT_ADD_TEST.
# Argument forwarding and list parsing in cmake is utterly broken.
# The safest and most reliable thing to do is to call the final cmake
# function directly here, so we can be sure that the PROPERTIES argument
# will be properly parsed.
set_tests_properties(
    pyunittests-roottest-python-distrdf-backends-all
    PROPERTIES RESOURCE_LOCK "${DISTRDF_RESOURCE_LOCKS}" PROCESSORS 4 TIMEOUT 1200)

ROOT_ADD_PYUNITTEST(test_standalone_fromspec test_standalone_fromspec.py GENERIC ENVIRONMENT "${DISTRDF_ENVIRONMENT_VARS}")
set_tests_properties(
    pyunittests-roottest-python-distrdf-backends-standalone-fromspec
    PROPERTIES RESOURCE_LOCK "${DISTRDF_RESOURCE_LOCKS}" PROCESSORS 4)
