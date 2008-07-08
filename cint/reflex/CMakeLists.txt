PROJECT(Reflex)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0 FATAL_ERROR)

#####################   modules   ####################

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
INCLUDE (MacroLibrary)

#####################   version   ####################

SET(REFLEX_VERSION_MAJOR 5)
SET(REFLEX_VERSION_MINOR 08)
SET(REFLEX_VERSION_RELEASE 00)
SET(REFLEX_VERSION_BUILD 2)

SET(REFLEX_VERSION "${REFLEX_VERSION_MAJOR}.${REFLEX_VERSION_MINOR}")
SET(REFLEX_VERSION_FULL "${REFLEX_VERSION}.${REFLEX_VERSION_RELEASE}.${REFLEX_VERSION_BUILD}")

#####################   checks    ####################

# disallow in-source builds
MACRO_ENSURE_OUT_OF_SOURCE_BUILD("${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there.")

#####################   options   ####################

OPTION(BUILD_SHARED_LIBS "inhibits building of the static version of the Reflex lib" OFF)
OPTION(BUILD_TESTS "builds and runs the unit test suite" OFF)

#####################   config    ####################

INCLUDE (SelectConfig)

IF (REFLEX_CXX_FLAGS)
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${REFLEX_CXX_FLAGS}")
ENDIF (REFLEX_CXX_FLAGS)

FOREACH (m SHARED MODULE EXE)
   SET(CMAKE_${m}_LINKER_FLAGS "${CMAKE_${m}_LINKER_FLAGS} ${REFLEX_LINKER_FLAGS} ${REFLEX_${m}_LINKER_FLAGS}")
ENDFOREACH (m SHARED MODULE EXE)

# set include dir and library name
SET(REFLEX_INCLUDE_DIR ${Reflex_SOURCE_DIR}/inc)
SET(REFLEX_LIBRARY Reflex)

# set the includes for all sub directories
INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})

# add definitions required for shared libraries
IF (NOT BUILD_SHARED_LIBS)
   ADD_DEFINITIONS(-DREFLEX_DLL_VETO)
ENDIF (NOT BUILD_SHARED_LIBS)

#####################   install   ####################

IF (WIN32)

   SET(BIN_INSTALL_DIR      "bin"                             ) # The install dir for executables (default ${EXEC_INSTALL_PREFIX}/bin)
   SET(LIB_INSTALL_DIR      "lib"                             ) # The subdirectory relative to the install prefix where libraries will be installed (default is ${EXEC_INSTALL_PREFIX}/lib${LIB_SUFFIX})
   SET(INCLUDE_INSTALL_DIR  "include"                         ) # The subdirectory to the header prefix

ELSE (WIN32)

   SET(EXEC_INSTALL_PREFIX  "${CMAKE_INSTALL_PREFIX}"         ) # Base directory for executables and libraries

   SET(BIN_INSTALL_DIR      "${EXEC_INSTALL_PREFIX}/bin"      ) # The install dir for executables (default ${EXEC_INSTALL_PREFIX}/bin)
   SET(LIB_INSTALL_DIR      "${EXEC_INSTALL_PREFIX}/lib"      ) # The subdirectory relative to the install prefix where libraries will be installed (default is ${EXEC_INSTALL_PREFIX}/lib${LIB_SUFFIX})
   SET(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/include" ) # The subdirectory to the header prefix

ENDIF (WIN32)

# The INSTALL_TARGETS_DEFAULT_ARGS variable should be used when libraries are installed.
# The arguments are also ok for regular executables, i.e. executables which don't go
# into sbin/ or libexec/, but for installing executables the basic syntax 
# INSTALL(TARGETS genreflex DESTINATION "${BIN_INSTALL_DIR}")
# is enough, so using this variable there doesn't help a lot.
# The variable must not be used for installing plugins.
# Usage is like this:
#    install(TARGETS Reflex ${INSTALL_TARGETS_DEFAULT_ARGS} )
#
# This will install libraries correctly under UNIX, OSX and Windows (i.e. dll's go
# into bin/.
SET(INSTALL_TARGETS_DEFAULT_ARGS  RUNTIME DESTINATION "${BIN_INSTALL_DIR}"
                                  LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
                                  ARCHIVE DESTINATION "${LIB_INSTALL_DIR}")

#####################   sub dirs  ####################

ADD_SUBDIRECTORY(doc)
ADD_SUBDIRECTORY(inc)
ADD_SUBDIRECTORY(python)
ADD_SUBDIRECTORY(src)

# enable tests only if specified
IF (BUILD_TESTS)
   ENABLE_TESTING()
   ADD_SUBDIRECTORY(test)
ENDIF (BUILD_TESTS)
