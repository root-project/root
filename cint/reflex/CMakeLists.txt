############################################################################
# CMakeLists.txt file for building ROOT cint/reflex package
############################################################################
if(CMAKE_PROJECT_NAME STREQUAL ROOT)
  ROOT_USE_PACKAGE(core/base)
  ROOT_USE_PACKAGE(core/clib)
  ROOT_USE_PACKAGE(core/meta)
  ROOT_USE_PACKAGE(core/cont)  
  ROOT_USE_PACKAGE(cint/cint)
  
  file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc 
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/Reflex/*.h 
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/Reflex/Builder/*.h )
  list(REMOVE_ITEM headers Reflex/Builder/ReflexBuilder.h Reflex/Reflex.h 
                           Reflex/SharedLibrary.h Reflex/DictionaryGenerator.h)
  #---Rootcint dictionary ---------------------------------------------------
  ROOT_GENERATE_DICTIONARY(G__Reflex ${headers} LINKDEF LinkDef.h OPTIONS -p)
  ROOT_GENERATE_ROOTMAP(Reflex LINKDEF LinkDef.h DEPENDENCIES )
  ROOT_MODULE_LIBRARY(ReflexDict G__Reflex.cxx LIBRARIES Reflex Cint Core)
  #---Reflex library---------------------------------------------------------
  ROOT_LINKER_LIBRARY(Reflex *.cxx LIBRARIES ${CMAKE_DL_LIBS})  
  #---genreflex files--------------------------------------------------------
  install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/lib/python/genreflex/gccxmlpath.py \"gccxmlpath = '${GCCXML_EXECUTABLE}'\n\")")
  install(DIRECTORY python/genreflex DESTINATION lib/python FILES_MATCHING PATTERN "*.py" PATTERN ".svn" EXCLUDE)
  install(CODE "execute_process(COMMAND python -m compileall  ${CMAKE_INSTALL_PREFIX}/lib/python/genreflex)")
  #---genmap executable------------------------------------------------------
  ROOT_EXECUTABLE(genmap genmap/genmap.cxx LIBRARIES Reflex)
  #---install headers--------------------------------------------------------
  ROOT_INSTALL_HEADERS()
    
else()
  PROJECT(Reflex)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0 FATAL_ERROR)

  #####################   modules   ####################

  SET(REFLEX_MODULE_DIR "${CMAKE_SOURCE_DIR}/cmake/modules")
  SET(REFLEX_TEMPLATE_DIR "${CMAKE_SOURCE_DIR}/cmake/templates")
  SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${REFLEX_MODULE_DIR}")

  INCLUDE (MacroLibrary)
  INCLUDE (ReflexMacros)

  #####################   version   ####################

  SET(REFLEX_VERSION_MAJOR 5)
  SET(REFLEX_VERSION_MINOR 08)
  SET(REFLEX_VERSION_RELEASE 00)
  SET(REFLEX_VERSION_BUILD 2)

  SET(REFLEX_VERSION "${REFLEX_VERSION_MAJOR}.${REFLEX_VERSION_MINOR}")
  SET(REFLEX_VERSION_FULL "${REFLEX_VERSION}.${REFLEX_VERSION_RELEASE}.${REFLEX_VERSION_BUILD}")

  #####################   options   ####################

  OPTION(BUILD_SHARED_LIBS "inhibits building of the static version of the Reflex lib" ON)
  OPTION(FORBID_IN_SOURCE_BUILDS "forbids in-source builds of Reflex" OFF)
  OPTION(REFLEX_ENABLE_TESTS "enables all testing-related activities" OFF)
  OPTION(REFLEX_BUILD_TESTS "builds the unit test suite" ON)

  #####################   checks    ####################

  IF (FORBID_IN_SOURCE_BUILDS)
    MACRO_ENSURE_OUT_OF_SOURCE_BUILD("${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there.")
  ENDIF (FORBID_IN_SOURCE_BUILDS)

  #####################   config    ####################

  INCLUDE (SelectConfig)

  IF (REFLEX_CXX_FLAGS)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${REFLEX_CXX_FLAGS}")
  ENDIF (REFLEX_CXX_FLAGS)

  FOREACH (m SHARED MODULE EXE)
    SET(CMAKE_${m}_LINKER_FLAGS "${CMAKE_${m}_LINKER_FLAGS} ${REFLEX_LINKER_FLAGS} ${REFLEX_${m}_LINKER_FLAGS}")
  ENDFOREACH (m SHARED MODULE EXE)

  # add definitions required for shared libraries
  IF (NOT BUILD_SHARED_LIBS)
    ADD_DEFINITIONS(-DREFLEX_DLL_VETO)
  ENDIF (NOT BUILD_SHARED_LIBS)

  SET(DL_LIBRARY ${CMAKE_DL_LIBS})

  #####################   install   ####################

  IF (WIN32)

    SET(BIN_INSTALL_DIR      "bin"     ) # The install dir for executables (default ${EXEC_INSTALL_PREFIX}/bin)
    SET(LIB_INSTALL_DIR      "lib"     ) # The subdirectory relative to the install prefix where libraries will be installed (default is ${EXEC_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    SET(INCLUDE_INSTALL_DIR  "include" ) # The subdirectory to the header prefix

  ELSE (WIN32)

    SET(BIN_INSTALL_DIR      "bin"     ) # The install dir for executables (default ${EXEC_INSTALL_PREFIX}/bin)
    SET(LIB_INSTALL_DIR      "lib"     ) # The subdirectory relative to the install prefix where libraries will be installed (default is ${EXEC_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    SET(INCLUDE_INSTALL_DIR  "include" ) # The subdirectory to the header prefix

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

    #####################   package   ####################

    SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ Reflection Package")
    SET(CPACK_PACKAGE_VENDOR "European Organization for Nuclear Research (CERN)")
    SET(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/doc/README)
    SET(CPACK_PACKAGE_CONTACT axel.naumann@cern.ch)

    SET(CPACK_PACKAGE_VERSION_MAJOR ${REFLEX_VERSION_MAJOR})
    SET(CPACK_PACKAGE_VERSION_MINOR ${REFLEX_VERSION_MINOR})
    SET(CPACK_PACKAGE_VERSION_PATCH ${REFLEX_VERSION_RELEASE})

    SET(CPACK_STRIP_FILES TRUE)
    SET(CPACK_SOURCE_IGNORE_FILES "/CVS/" "/\\\\.svn/" "\\\\.swp$" "\\\\.#" "/#" ".*~" "cscope.*")
    SET(CPACK_SOURCE_STRIP_FILES TRUE)

    # debian
    SET(DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
    SET(DEBIAN_PACKAGE_DEPENDS "gccxml (>= 0.7), python (>= 2.5)")
    SET(DEBIAN_PACKAGE_BUILDS_DEPENDS "gccxml (>= 0.7.0), python (>= 2.5), libcppunit (>= 1.12.0), libcppunit-dev (>= 1.12.0)")

    INCLUDE (CPack)

    #####################   sub dirs  ####################

    ADD_SUBDIRECTORY(doc)
    ADD_SUBDIRECTORY(inc)
    ADD_SUBDIRECTORY(python)
    ADD_SUBDIRECTORY(src)

    # enable tests only if specified
      IF (REFLEX_ENABLE_TESTS)
        ENABLE_TESTING()
        ADD_SUBDIRECTORY(test)
      ENDIF (REFLEX_ENABLE_TESTS)
endif()