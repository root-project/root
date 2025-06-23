#.rst:
# FindLibClang
# ------------
#
# Find LibClang
#
# Find LibClang headers and library
#
# ::
#
#   LibClang_FOUND             - True if libclang is found.
#   LibClang_LIBRARY           - Clang library to link against.
#   LibClang_VERSION           - Version number as a string (e.g. "9.0").
#   LibClang_PYTHON_EXECUTABLE - Compatible python version.

#
# Find libclang.so/.dll to be used by the clang Python bindings
#
if (NOT LibClang_LIBRARY)
    set(LibClang_LIBRARY $ENV{LibClang_LIBRARY})
    if (NOT LibClang_LIBRARY)
        find_library(LibClang_LIBRARY libclang${CMAKE_SHARED_LIBRARY_SUFFIX})
        if (NOT LibClang_LIBRARY)
            find_program(LibClang_LLVM_CONFIG "llvm-config")
            if (NOT LibClang_LLVM_CONFIG)
                set(llvm_config_versioned llvm-config)
                foreach(version RANGE 13 6)
                    list(APPEND llvm_config_versioned "llvm-config-${version}")
                endforeach ()
                find_program(LibClang_LLVM_CONFIG NAMES ${llvm_config_versioned})
            endif ()

            set(LibClang_PREFIX "")
            if (LibClang_LLVM_CONFIG)
                execute_process(COMMAND ${LibClang_LLVM_CONFIG} --prefix OUTPUT_VARIABLE LibClang_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            endif()

            set(LibClang_SUFFIX_versioned llvm)
            foreach(version RANGE 13 6)
                list(APPEND LibClang_SUFFIX_versioned "llvm-${version}/lib" "llvm/${version}/lib")
            endforeach ()

            find_library(LibClang_LIBRARY libclang${CMAKE_SHARED_LIBRARY_SUFFIX}
                HINTS ${LibClang_PREFIX} $ENV{CONDA_PREFIX}
                PATH_SUFFIXES lib lib64 x86_64-linux-gnu ${LibClang_SUFFIX_versioned}
            )
       endif()
    endif()
endif()

function(_find_libclang_python python_executable)
    #
    # Prefer python3 explicitly or implicitly over python2.
    #
    foreach(exe IN ITEMS python3 python python2)
        execute_process(
            COMMAND ${exe} -c "from clang.cindex import Config; Config.set_library_file(\"${LibClang_LIBRARY}\"); Config().lib"
            ERROR_VARIABLE _stderr
            RESULT_VARIABLE _rc
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if("${_rc}" STREQUAL "0")
            set(pyexe ${exe})
            break()
        endif()
    endforeach()
    set(${python_executable} "${pyexe}" PARENT_SCOPE)
endfunction(_find_libclang_python)

_find_libclang_python(LibClang_PYTHON_EXECUTABLE)
if(LibClang_LIBRARY)
    set(LibClang_LIBRARY ${LibClang_LIBRARY})
    string(REGEX REPLACE ".*clang-\([0-9]+.[0-9]+\).*" "\\1" LibClang_VERSION_TMP "${LibClang_LIBRARY}")
    set(LibClang_VERSION ${LibClang_VERSION_TMP} CACHE STRING "LibClang version" FORCE)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibClang REQUIRED_VARS  LibClang_LIBRARY LibClang_PYTHON_EXECUTABLE
                                           VERSION_VAR    LibClang_VERSION)

find_program(CLANG_EXE clang++)
execute_process(COMMAND ${CLANG_EXE} --version OUTPUT_VARIABLE clang_full_version_string OUTPUT_STRIP_TRAILING_WHITESPACE)
string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING "${clang_full_version_string}")

set(CLANG_VERSION_STRING ${CLANG_VERSION_STRING} PARENT_SCOPE)

mark_as_advanced(LibClang_VERSION)
unset(_filename)
unset(_find_libclang_filename)
