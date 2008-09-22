IF (UNIX)

FILE(WRITE "${_filename}"
"#!/bin/sh
# created by cmake, don't edit, changes will be lost

${_library_path_variable}=${_ld_library_path}\${${_library_path_variable}:+:\$${_library_path_variable}} \"${_executable}\" \"$@\"
")

# make it executable
# since this is only executed on UNIX, it is safe to call chmod
EXEC_PROGRAM(chmod ARGS ug+x \"${_filename}\" OUTPUT_VARIABLE _dummy)

ELSE (UNIX)

FILE(TO_NATIVE_PATH "${_ld_library_path}" win_path)
FILE(TO_NATIVE_PATH "${_executable}" _executable)

FILE(WRITE "${_filename}"
"
set PATH=${win_path};$ENV{PATH}
cd \"${_executable}\\..\"
\"${_executable}\" %*
")

ENDIF (UNIX)

