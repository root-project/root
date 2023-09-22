# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Writes etc/gitinfo.txt based on the source directory's git commit - if available -
# or the version info.

# INPUT variables: none
# OUTPUT variables: none

# In script mode, CMake doesn't get CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR right.
if(SRCDIR)
  set(CMAKE_SOURCE_DIR ${SRCDIR})
endif()
if(BINDIR)
  set(CMAKE_BINARY_DIR ${BINDIR})
endif()

include(${CMAKE_SOURCE_DIR}/cmake/modules/SetROOTVersion.cmake)

function(UPDATE_GIT_VERSION)
  string(TIMESTAMP PSEUDO_GIT_TIMESTAMP "%b %d %Y, %H:%M:%S" UTC)
  if(GIT_DESCRIBE_ALL)
    file(WRITE ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
      "${GIT_DESCRIBE_ALL}\n${GIT_DESCRIBE_ALWAYS}\n${PSEUDO_GIT_TIMESTAMP}\n")
  else()
    math(EXPR ROOT_PATCH_VERSION_ODD ${ROOT_PATCH_VERSION}%2)
    if(${ROOT_PATCH_VERSION} EQUAL 0)
      # A release.
      math(EXPR ROOT_MINOR_VERSION_ODD ${ROOT_MINOR_VERSION}%2)
      if(${ROOT_MINOR_VERSION_ODD} EQUAL 1)
        # Dev release.
        file(WRITE ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
          "heads/master\ntags/v${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-${ROOT_PATCH_VERSION}\n${PSEUDO_GIT_TIMESTAMP}\n")
      else()
        # Production release / patch release.
        file(WRITE ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
          "heads/v${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-patches\ntags/v${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-${ROOT_PATCH_VERSION}\n${PSEUDO_GIT_TIMESTAMP}\n")
      endif()
    else()
      file(WRITE ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
        "heads/master\ntags/v${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-${ROOT_PATCH_VERSION}\n${PSEUDO_GIT_TIMESTAMP}\n")
      message(WARNING "Cannot determine git revision info: source is not a release and not a git repo. Noting v${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-${ROOT_PATCH_VERSION} as commit.")
    endif()
  endif()
endfunction()

UPDATE_GIT_VERSION()
