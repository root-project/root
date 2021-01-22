"""!
@package PyRDF
__init__ file of the PyRDF module
"""
from PyRDF.RDataFrame import RDataFrame  # noqa
from PyRDF.RDataFrame import RDataFrameException  # noqa
from PyRDF.CallableGenerator import CallableGenerator  # noqa
from PyRDF.backend.Local import Local
from PyRDF.backend.Backend import Backend
from PyRDF.backend.Utils import Utils
import os
import logging
import sys

current_backend = Local()

includes_headers = set()  # All headers included in the analysis
includes_shared_libraries = set()  # All shared libraries included
includes_files = set()  # All other generic files included

logger = logging.getLogger(__name__)


def create_logger(level="WARNING", log_path="./PyRDF.log"):
    """!
    \brief PyRDF basic logger
    \param level (str): Logging level as defined in the Python logging module.
    \param log_path (str): Path for the log file.
    """
    logger = logging.getLogger(__name__)

    level = getattr(logging, level)

    logger.setLevel(level)

    format_string = ("%(levelname)s: %(name)s[%(asctime)s]: %(message)s")
    formatter = logging.Formatter(format_string)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def use(backend_name, conf={}):
    """!
    \brief Allows the user to choose the execution backend.

    \param backend_name (str): This is the name of the chosen backend.
    \param conf (str, optional): This should be a dictionary with
            necessary configuration parameters. Its default value is an empty
            dictionary {}.
    """
    future_backends = [
        "dask"
    ]

    global current_backend

    if backend_name in future_backends:
        msg = "This backend environment will be considered in the future !"
        raise NotImplementedError(msg)
    elif backend_name == "local":
        current_backend = Local(conf)
    elif backend_name == "spark":
        from PyRDF.backend.Spark import Spark
        current_backend = Spark(conf)
    else:
        msg = "Incorrect backend environment \"{}\"".format(backend_name)
        raise Exception(msg)


def _get_paths_set_from_string(path_string):
    """!
    \brief Retrieves paths to files (directory or single file) from a string.

    \param path_string (str): The string to the path of the file or directory
            to be recursively searched for files.

    \returns set: The set with all paths returned from the directory, or a set
            with only the path of the string.
    """
    logger.debug("Retrieving paths from {}".format(path_string))

    if os.path.isdir(path_string):
        # Create a set with all the headers in the directory
        paths_set = {
            os.path.join(rootpath, filename)
            for rootpath, dirs, filenames
            in os.walk(path_string)
            for filename
            in filenames
        }
        logger.debug("\nInitial path: {} \nPaths retrieved: {}".format(
            path_string,
            paths_set
        ))
        return paths_set
    elif os.path.isfile(path_string):
        # Convert to set if this is a string
        logger.debug("File path retrieved: {}".format(path_string))
        return {path_string}


def _check_pcm_in_library_path(shared_library_path):
    """!
    \brief Retrieves paths to shared libraries and pcm file(s) in a directory.

    \param shared_library_path (str): The string to the path of the file or
            directory to be recursively searched for files.

    \returns list, list: Two lists, the first with all paths to pcm files, the
            second with all paths to shared libraries.
    """
    all_paths = _get_paths_set_from_string(
        shared_library_path
    )

    pcm_paths = {
        filepath
        for filepath in all_paths
        if filepath.endswith(".pcm")
    }

    shared_library_formats = (".so", ".dll", ".dylib")
    libraries_path = {
        filepath
        for filepath in all_paths
        if filepath.endswith(shared_library_formats)
    }

    return pcm_paths, libraries_path


def include_headers(headers_paths):
    """!
    \brief Includes the C++ headers to be declared before execution. Each
    header is also declared on the current running session.

    \param headers_paths (str, iter): A string or an iterable (such as a
            list, set...) containing the paths to all necessary C++ headers as
            strings. This function accepts both paths to the headers
            themselves and paths to directories containing the headers.
    """
    global current_backend, includes_headers
    headers_to_include = set()

    if isinstance(headers_paths, str):
        headers_to_include.update(_get_paths_set_from_string(headers_paths))
    else:
        for path_string in headers_paths:
            headers_to_include.update(_get_paths_set_from_string(path_string))

    # If not on the local backend, distribute files to executors
    if not isinstance(current_backend, Local):
        current_backend.distribute_files(headers_to_include)

    # Declare the headers in ROOT
    Utils.declare_headers(headers_to_include)

    # Finally, add everything to the includes set
    includes_headers.update(headers_to_include)


def include_shared_libraries(shared_libraries_paths):
    """!
    \brief Includes the C++ shared libraries to be declared before execution.
    Each library is also declared on the current running session. If any pcm
    file is present in the same folder as the shared libraries, the function
    will try to retrieve them (and distribute them if working on a distributed
    backend).

    \param shared_libraries_paths (str, iter): A string or an iterable (such as
            a list, set...) containing the paths to all necessary C++ shared
            libraries as strings. This function accepts both paths to the
            libraries themselves and paths to directories containing the
            libraries.
    """
    global current_backend, includes_shared_libraries
    libraries_to_include = set()
    pcm_to_include = set()

    if isinstance(shared_libraries_paths, str):
        pcm_to_include, libraries_to_include = _check_pcm_in_library_path(
            shared_libraries_paths
        )
    else:
        for path_string in shared_libraries_paths:
            pcm, libraries = _check_pcm_in_library_path(
                path_string
            )
            libraries_to_include.update(libraries)
            pcm_to_include.update(pcm)

    # If not on the local backend, distribute files to executors
    if not isinstance(current_backend, Local):
        current_backend.distribute_files(libraries_to_include)
        current_backend.distribute_files(pcm_to_include)

    # Declare the shared libraries in ROOT
    Utils.declare_shared_libraries(libraries_to_include)

    # Finally, add everything to the includes set
    includes_shared_libraries.update(includes_shared_libraries)


def send_generic_files(files_paths):
    """!
    \brief Sends to the workers the generic files needed by the user.

    \param files_paths (str, iter): Paths to the files to be sent to the
            distributed workers.
    """
    global current_backend, includes_files
    files_to_include = set()

    if isinstance(files_paths, str):
        files_to_include.update(_get_paths_set_from_string(files_paths))
    else:
        for path_string in files_paths:
            files_to_include.update(_get_paths_set_from_string(path_string))

    # If not on the local backend, distribute files to executors
    if not isinstance(current_backend, Local):
        current_backend.distribute_files(files_to_include)


def initialize(fun, *args, **kwargs):
    """!
    \brief Set a function that will be executed as a first step on every backend
    before any other operation. This method also executes the function on the
    current user environment so changes are visible on the running session.

    This allows users to inject and execute custom code on the worker
    environment without being part of the RDataFrame computational graph.

    \param fun (function): Function to be executed.
    """
    Backend.register_initialization(fun, *args, **kwargs)
