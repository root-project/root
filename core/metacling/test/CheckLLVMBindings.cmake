# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Script mode: cmake -DOUT=<directory> [-DEXPECT=<regex>] -P CheckLLVMBindings.cmake -- <command> [args...]
#
# Runs CMD with glibc's LD_DEBUG=bindings, which logs every symbol binding the
# dynamic linker performs, and fails if a symbol involving LLVM or Clang types
# (apart from the cling API) is bound across libCling's boundary: from libCling
# to another object, or from another object to libCling.

set(CMD)
set(in_command FALSE)
math(EXPR last "${CMAKE_ARGC} - 1")
foreach(i RANGE ${last})
  if(in_command)
    list(APPEND CMD "${CMAKE_ARGV${i}}")
  elseif(CMAKE_ARGV${i} STREQUAL "--")
    set(in_command TRUE)
  endif()
endforeach()

file(REMOVE_RECURSE "${OUT}")
file(MAKE_DIRECTORY "${OUT}")
execute_process(COMMAND ${CMAKE_COMMAND} -E env LD_DEBUG=bindings LD_DEBUG_OUTPUT=${OUT}/bindings ${CMD}
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
message("${output}")
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Command failed with ${result}: ${CMD}")
endif()
if(EXPECT AND NOT output MATCHES "${EXPECT}")
  message(FATAL_ERROR "Output does not match '${EXPECT}'")
endif()

file(GLOB logs "${OUT}/bindings.*")
set(n_bindings 0)
set(crossing)
foreach(log IN LISTS logs)
  file(STRINGS "${log}" lines REGEX "binding file ")
  list(LENGTH lines n)
  math(EXPR n_bindings "${n_bindings} + ${n}")
  foreach(line IN LISTS lines)
    if(NOT line MATCHES "symbol `_Z[^']*(4llvm|5clang)" OR line MATCHES "symbol `_Z[A-Z]*N?K?5cling")
      continue()
    endif()
    if(line MATCHES "binding file ([^ ]+) [^ ]+ to ([^ ]+) ")
      string(FIND "${CMAKE_MATCH_1}" "libCling" from_cling)
      string(FIND "${CMAKE_MATCH_2}" "libCling" to_cling)
      if((from_cling EQUAL -1) AND NOT (to_cling EQUAL -1) OR NOT (from_cling EQUAL -1) AND (to_cling EQUAL -1))
        list(APPEND crossing "${line}")
      endif()
    endif()
  endforeach()
endforeach()

# Without any bindings logged, LD_DEBUG did not work and the check would be meaningless.
if(n_bindings EQUAL 0)
  message(FATAL_ERROR "No symbol bindings were logged; LD_DEBUG=bindings is not supported here")
endif()
if(crossing)
  list(LENGTH crossing n_crossing)
  list(SUBLIST crossing 0 5 examples)
  string(REPLACE ";" "\n" examples "${examples}")
  message(FATAL_ERROR "${n_crossing} LLVM/Clang symbol bindings cross libCling's boundary, e.g.:\n${examples}")
endif()
message("${n_bindings} symbol bindings logged, none involving LLVM or Clang crosses libCling's boundary")
