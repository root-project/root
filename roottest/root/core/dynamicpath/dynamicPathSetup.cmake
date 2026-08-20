# CMake re-implementation of the former `test_dynpath_setup.sh` shell script.
#
# It verifies the way `TSystem::GetDynamicPath()` is assembled out of
#   * the `ROOT_LIBRARY_PATH` environment variable,
#   * the platform specific loader search path
#     (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH`),
#   * the `<Unix|WinNT>.*.Root.DynamicPath` resource of the local `.rootrc`,
#   * and the ROOT library (resp. binary) directory.
#
# Being a plain CMake script (executed via `cmake -P`) it runs on all the
# platforms supported by ROOT, including Windows.
#
# Required arguments (passed with -D):
#   ROOT_EXE : full path to the `root` executable
#   WORKDIR  : scratch directory used to run the checks

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

foreach(var ROOT_EXE WORKDIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "${var} must be defined (cmake -D${var}=...)")
  endif()
endforeach()

#---Platform dependent settings-------------------------------------------------
# `sep`    : separator used in path lists
# `rc_key` : name of the .rootrc resource holding the dynamic path
# `ld_var` : environment variable used by the system loader
if(WIN32)
  set(sep ";")
  set(rc_key "WinNT.*.Root.DynamicPath")
  set(ld_var "PATH")
elseif(APPLE)
  set(sep ":")
  set(rc_key "Unix.*.Root.DynamicPath")
  set(ld_var "DYLD_LIBRARY_PATH")
else()
  set(sep ":")
  set(rc_key "Unix.*.Root.DynamicPath")
  set(ld_var "LD_LIBRARY_PATH")
endif()

#---Helpers---------------------------------------------------------------------

# Run `root` on the given C++ expression and return the last line of its output.
# The statement is wrapped into a block so that `root` exits with 0 rather than
# with the (truncated) value of the last expression.
function(root_eval expr outvar)
  set(stmt "{ std::cout << ${expr} << std::endl; }")
  execute_process(COMMAND "${ROOT_EXE}" -b -q -l -e "${stmt}"
                  WORKING_DIRECTORY "${WORKDIR}"
                  OUTPUT_VARIABLE out
                  ERROR_VARIABLE err
                  RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "'${ROOT_EXE} -b -q -l -e ${stmt}' failed with ${rc}\n${out}\n${err}")
  endif()
  string(REPLACE "\r" "" out "${out}")
  string(STRIP "${out}" out)
  # Keep only the last line (the value printed by `expr`), like `tail -1`.
  string(REGEX MATCH "[^\n]*$" out "${out}")
  set(${outvar} "${out}" PARENT_SCOPE)
endfunction()

# Write the local .rootrc with the requested dynamic path; an empty value
# results in the resource being commented out.
function(set_rootrc value)
  if("${value}" STREQUAL "")
    file(WRITE "${WORKDIR}/.rootrc" "# ${rc_key}:\n")
  else()
    file(WRITE "${WORKDIR}/.rootrc" "${rc_key}: ${value}\n")
  endif()
endfunction()

# Query the dynamic path as seen by ROOT.
macro(get_dynpath)
  root_eval("gSystem->GetDynamicPath()" cur_dynpath)
endmacro()

# The dynamic path must start with `expected`.
function(check_begin expected)
  string(FIND "${cur_dynpath}${sep}" "${expected}${sep}" pos)
  if(NOT pos EQUAL 0)
    message(FATAL_ERROR "dynamic path: ${cur_dynpath}\n"
                        "dynamic path should start with: ${expected}")
  endif()
endfunction()

# The dynamic path must contain `expected` as a whole sequence of entries.
function(check_mid expected)
  string(FIND "${sep}${cur_dynpath}${sep}" "${sep}${expected}${sep}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "dynamic path: ${cur_dynpath}\n"
                        "dynamic path should contain: ${expected}")
  endif()
endfunction()

function(check_begin_and_mid begin mid)
  check_begin("${begin}")
  check_mid("${mid}")
endfunction()

# Assemble the "environment part" of the dynamic path the same way
# TUnixSystem/TWinNTSystem do, i.e. ROOT_LIBRARY_PATH, followed by the loader
# search path, followed by the `.rootrc` entries.
function(root_env_part rdynpath outvar)
  if(WIN32)
    set(ld "$ENV{PATH}")
  elseif(APPLE)
    # On macOS ROOT concatenates all three of these.
    set(ld "$ENV{DYLD_LIBRARY_PATH}${sep}$ENV{LD_LIBRARY_PATH}${sep}$ENV{DYLD_FALLBACK_LIBRARY_PATH}")
  else()
    set(ld "$ENV{LD_LIBRARY_PATH}")
  endif()
  set(${outvar} "$ENV{ROOT_LIBRARY_PATH}${sep}${ld}${sep}${rdynpath}" PARENT_SCOPE)
endfunction()

# ROOT guarantees that the library directory is part of the dynamic path. It is
# appended right after the `.rootrc` entries, but *only* if it does not already
# occur in the environment derived part -- see the
# `if (!dynpath_envpart.Contains(TROOT::GetLibDir()))` guard in TUnixSystem.cxx
# (a plain substring test, hence the plain string(FIND) below).
#
# Both situations occur in practice: when the test is run through ctest the
# driver puts $ROOTSYS/lib into (DY)LD_LIBRARY_PATH, so the library directory is
# already present and nothing gets appended.
function(check_rootrc_then_libdir rdynpath)
  root_env_part("${rdynpath}" envpart)
  string(FIND "${envpart}" "${libdir}" pos)
  if(pos EQUAL -1)
    # Not seen yet: it must be appended right behind the .rootrc entries.
    check_mid("${rdynpath}${sep}${libdir}")
  else()
    # Already provided by the environment: both must still be present.
    check_mid("${rdynpath}")
    check_mid("${libdir}")
  endif()
endfunction()

#---Set up the scratch area-----------------------------------------------------
file(REMOVE_RECURSE "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}/rootlibpath")
file(MAKE_DIRECTORY "${WORKDIR}/rootrcpath")
file(MAKE_DIRECTORY "${WORKDIR}/ldpath")

set(rootlibpath "${WORKDIR}/rootlibpath")
set(rootrcpath  "${WORKDIR}/rootrcpath")
set(ldpath      "${WORKDIR}/ldpath")

# Start from a well defined state.
unset(ENV{ROOT_LIBRARY_PATH})
set_rootrc("")

root_eval("TROOT::GetLibDir()" libdir)
if(WIN32)
  # On Windows the built-in default is `.;<bindir>` while on Unix it is
  # `.:<libdir>` (see TWinNTSystem.cxx / TUnixSystem.cxx).
  root_eval("TROOT::GetBinDir()" defaultdir)
else()
  set(defaultdir "${libdir}")
endif()

message(STATUS "ROOT library directory: ${libdir}")
message(STATUS "default dynamic path  : .${sep}${defaultdir}")

#---Checks without an explicit loader search path-------------------------------

set(ENV{ROOT_LIBRARY_PATH} "${rootlibpath}")
set_rootrc("${rootrcpath}")
get_dynpath()
check_begin("${rootlibpath}")
check_rootrc_then_libdir("${rootrcpath}")

set(ENV{ROOT_LIBRARY_PATH} "${rootlibpath}")
set_rootrc("${libdir}${sep}${rootrcpath}")
get_dynpath()
check_begin_and_mid("${rootlibpath}" "${libdir}${sep}${rootrcpath}")

set(ENV{ROOT_LIBRARY_PATH} "${rootlibpath}")
set_rootrc("")
get_dynpath()
check_begin_and_mid("${rootlibpath}" ".${sep}${defaultdir}")

unset(ENV{ROOT_LIBRARY_PATH})
set_rootrc("")
get_dynpath()
check_mid(".${sep}${defaultdir}")

#---Checks with the loader search path (LD_LIBRARY_PATH & Co.)------------------

if(DEFINED ENV{${ld_var}} AND NOT "$ENV{${ld_var}}" STREQUAL "")
  set(ENV{${ld_var}} "${ldpath}${sep}$ENV{${ld_var}}")
else()
  set(ENV{${ld_var}} "${ldpath}")
endif()

set(ENV{ROOT_LIBRARY_PATH} "${rootlibpath}")
set_rootrc("${rootrcpath}")
get_dynpath()
check_begin("${rootlibpath}${sep}${ldpath}")
check_rootrc_then_libdir("${rootrcpath}")

set(ENV{ROOT_LIBRARY_PATH} "${rootlibpath}")
set_rootrc("${libdir}${sep}${rootrcpath}")
get_dynpath()
check_begin_and_mid("${rootlibpath}${sep}${ldpath}" "${libdir}${sep}${rootrcpath}")

set(ENV{ROOT_LIBRARY_PATH} "${rootlibpath}")
set_rootrc("")
get_dynpath()
check_begin_and_mid("${rootlibpath}${sep}${ldpath}" ".${sep}${defaultdir}")

unset(ENV{ROOT_LIBRARY_PATH})
set_rootrc("")
get_dynpath()
check_begin_and_mid("${ldpath}" ".${sep}${defaultdir}")

message(STATUS "dynamic path setup: all checks passed")
