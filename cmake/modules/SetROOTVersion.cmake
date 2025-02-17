# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Sets the following variables:
# - ROOT_MAJOR_VERSION, ROOT_MINOR_VERSION, ROOT_PATCH_VERSION, e.g. "6", "30", "00", respectively.
# - ROOT_VERSION: "6.30.00"
# - ROOT_FULL_VERSION: 6.29.02-pre1
# - GIT_DESCRIBE_ALWAYS: output of `git describe --always` if source directory is git repo
# - GIT_DESCRIBE_ALL: output of `git describe --all` if source directory is git repo


cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Git)

function(SET_VERSION_FROM_FILE)
  # See https://stackoverflow.com/questions/47066115/cmake-get-version-from-multiline-text-file
  file(READ "${CMAKE_SOURCE_DIR}/core/foundation/inc/ROOT/RVersion.hxx" versionstr)
  string(REGEX MATCH "#define ROOT_VERSION_MAJOR ([0-9]*)" _ ${versionstr})
  set(ROOT_MAJOR_VERSION ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define ROOT_VERSION_MINOR ([0-9]*)" _ ${versionstr})
  if (CMAKE_MATCH_1 LESS 10)
    set(ROOT_MINOR_VERSION "0${CMAKE_MATCH_1}")
  else()
    set(ROOT_MINOR_VERSION ${CMAKE_MATCH_1})
  endif()
  string(REGEX MATCH "#define ROOT_VERSION_PATCH ([0-9]*)" _ ${versionstr})
  if (CMAKE_MATCH_1 LESS 10)
    set(ROOT_PATCH_VERSION "0${CMAKE_MATCH_1}")
  else()
    set(ROOT_PATCH_VERSION ${CMAKE_MATCH_1})
  endif()

  set(ROOT_MAJOR_VERSION "${ROOT_MAJOR_VERSION}" PARENT_SCOPE)
  set(ROOT_MINOR_VERSION "${ROOT_MINOR_VERSION}" PARENT_SCOPE)
  set(ROOT_PATCH_VERSION "${ROOT_PATCH_VERSION}" PARENT_SCOPE)
endfunction()

function(SET_ROOT_VERSION)
  if(Git_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
    execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_SOURCE_DIR}/.git describe --all
                    OUTPUT_VARIABLE GIT_DESCRIBE_ALL
                    RESULT_VARIABLE GIT_DESCRIBE_ERRCODE
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(GIT_DESCRIBE_ERRCODE "NoGit")
  endif()

  SET_VERSION_FROM_FILE()

  set(ROOT_VERSION "${ROOT_MAJOR_VERSION}.${ROOT_MINOR_VERSION}.${ROOT_PATCH_VERSION}")
  set(ROOT_FULL_VERSION "${ROOT_VERSION}")

  math(EXPR ROOT_PATCH_VERSION_ODD ${ROOT_PATCH_VERSION}%2)
  # For release versions (even patch version number) we use the number from
  # core/foundation/inc/ROOT/RVersion.hxx, not that of git: it's more stable / reliable.
  if(${ROOT_PATCH_VERSION_ODD} EQUAL 1)
    if(NOT GIT_DESCRIBE_ERRCODE)
      execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_SOURCE_DIR}/.git describe --always
                      OUTPUT_VARIABLE GIT_DESCRIBE_ALWAYS
                      ERROR_QUIET
                      OUTPUT_STRIP_TRAILING_WHITESPACE)

      if("${GIT_DESCRIBE_ALL}" MATCHES "^tags/v[0-9]+-[0-9]+-[0-9]+.*")
        # GIT_DESCRIBE_ALWAYS: v6-16-00-rc1
        # GIT_DESCRIBE_ALL: tags/v6-16-00-rc1
        # tag might end on "-rc1" or similar; parse version number in front.
        string(REGEX REPLACE "^tags/v([0-9]+)-.*" "\\1" ROOT_MAJOR_VERSION ${GIT_DESCRIBE_ALL})
        string(REGEX REPLACE "^tags/v[0-9]+-([0-9]+).*" "\\1" ROOT_MINOR_VERSION ${GIT_DESCRIBE_ALL})
        string(REGEX REPLACE "^tags/v[0-9]+-[0-9]+-([0-9]+).*" "\\1" ROOT_PATCH_VERSION ${GIT_DESCRIBE_ALL})
        string(REGEX REPLACE "^v([0-9]+)-([0-9]+)-(.*)" "\\1.\\2.\\3" ROOT_FULL_VERSION ${GIT_DESCRIBE_ALWAYS})
      elseif("${GIT_DESCRIBE_ALL}" MATCHES "/v[0-9]+-[0-9]+.*-patches$")
        # GIT_DESCRIBE_ALWAYS: v6-16-00-rc1-47-g9ba56ef4a3
        # GIT_DESCRIBE_ALL: heads/v6-16-00-patches
        string(REGEX REPLACE "^.*/v([0-9]+)-.*" "\\1" ROOT_MAJOR_VERSION ${GIT_DESCRIBE_ALL})
        string(REGEX REPLACE "^.*/v[0-9]+-([0-9]+).*" "\\1" ROOT_MINOR_VERSION ${GIT_DESCRIBE_ALL})
        set(ROOT_PATCH_VERSION "99") # aka head of ...-patches
      else()
        # GIT_DESCRIBE_ALWAYS: v6-13-04-2163-g7e8d27ea66
        # GIT_DESCRIBE_ALL: heads/master or remotes/origin/master

        # Use what was set above in SET_VERSION_FROM_FILE().
      endif()
    endif()
  else()
    if (${GIT_DESCRIBE_ALL} MATCHES "^tags/")
      string(REGEX REPLACE "^tags/" "" GIT_DESCRIBE_ALWAYS ${GIT_DESCRIBE_ALL})
    else()
      set(ROOT_GIT_VERSION "${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-${ROOT_PATCH_VERSION}")
      set(GIT_DESCRIBE_ALL "tags/${ROOT_GIT_VERSION}")
      set(GIT_DESCRIBE_ALWAYS ${ROOT_GIT_VERSION})
    endif()
  endif()

  set(ROOT_MAJOR_VERSION "${ROOT_MAJOR_VERSION}" PARENT_SCOPE)
  set(ROOT_MINOR_VERSION "${ROOT_MINOR_VERSION}" PARENT_SCOPE)
  set(ROOT_PATCH_VERSION "${ROOT_PATCH_VERSION}" PARENT_SCOPE)
  set(ROOT_VERSION "${ROOT_VERSION}" PARENT_SCOPE)
  set(ROOT_FULL_VERSION "${ROOT_FULL_VERSION}" PARENT_SCOPE)
  set(GIT_DESCRIBE_ALWAYS "${GIT_DESCRIBE_ALWAYS}" PARENT_SCOPE)
  set(GIT_DESCRIBE_ALL "${GIT_DESCRIBE_ALL}" PARENT_SCOPE)
endfunction()


SET_ROOT_VERSION()
