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
#   LibClang_VERSION           - Version number as a string (e.g. "3.9").
#   LibClang_PYTHON_EXECUTABLE - Compatible python version.

#
# Python support for clang might not be available for Python3. We need to
# find what we have.
#
function(_find_libclang_filename python_executable filename)
    #
    # Prefer python3 explicitly or implicitly over python2.
    #
    foreach(exe IN ITEMS python3 python python2)
        execute_process(
            COMMAND ${exe} -c "from clang.cindex import Config; Config().lib; print(Config().get_filename())"
            OUTPUT_VARIABLE lib
            ERROR_VARIABLE _stderr
            RESULT_VARIABLE _rc
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if("${_rc}" STREQUAL "0")
            set(pyexe ${exe})
            break()
        endif()
    endforeach()
    set(${python_executable} "${pyexe}" PARENT_SCOPE)
    set(${filename} "${lib}" PARENT_SCOPE)
endfunction(_find_libclang_filename)


_find_libclang_filename(LibClang_PYTHON_EXECUTABLE _filename)
find_library(LibClang_LIBRARY ${_filename})

if(LibClang_LIBRARY)
    set(LibClang_LIBRARY ${LibClang_LIBRARY})
    string(REGEX REPLACE ".*clang-\([0-9]+.[0-9]+\).*" "\\1" LibClang_VERSION_TMP "${LibClang_LIBRARY}")
    set(LibClang_VERSION ${LibClang_VERSION_TMP} CACHE STRING "LibClang version" FORCE)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibClang REQUIRED_VARS  LibClang_LIBRARY LibClang_PYTHON_EXECUTABLE
                                           VERSION_VAR    LibClang_VERSION)

mark_as_advanced(LibClang_VERSION)
unset(_filename)
unset(_find_libclang_filename)
