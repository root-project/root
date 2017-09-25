#---Check if cmake has the required version-----------------------------------------------------

cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)

set(policy_new CMP0068)
foreach(policy ${policy_new})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

include(cmake/modules/CaptureCommandLine.cmake)

#---Set name of the project to "ROOT". Has to be done after check of cmake version--------------
project(ROOT)
set(IntegratedBuild ON)

#---Set the locale to default C to prevent issued due to localization of commands---------------
# This is necessary as we for example call `clang -v` and parse its output. But on a localized
# program, the output parsing is much more error prone as certrain strings we're looking for
# could be missing or be in a different order. To prevent those errors, let's just force all
# output to use the default C locale which is more or less identical on all systems.
set(ENV{LANG} C)

#---Set pathes where to put the libraries, executables and headers------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(ROOTSYS ${CMAKE_BINARY_DIR})
set(HEADER_OUTPUT_PATH ${CMAKE_BINARY_DIR}/include)
set(ROOT_INCLUDE_DIR ${HEADER_OUTPUT_PATH})

#---Set the library version in the main CMakeLists.txt------------------------------------------
file(READ ${CMAKE_SOURCE_DIR}/build/version_number versionstr)
string(STRIP ${versionstr} versionstr)
string(REGEX REPLACE "([0-9]+)[.][0-9]+[/][0-9]+" "\\1" ROOT_MAJOR_VERSION ${versionstr})
string(REGEX REPLACE "[0-9]+[.]([0-9]+)[/][0-9]+" "\\1" ROOT_MINOR_VERSION ${versionstr})
string(REGEX REPLACE "[0-9]+[.][0-9]+[/]([0-9]+)" "\\1" ROOT_PATCH_VERSION ${versionstr})
set(ROOT_VERSION "${ROOT_MAJOR_VERSION}.${ROOT_MINOR_VERSION}.${ROOT_PATCH_VERSION}")

#---Where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/ is checked-------------
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#---Enable Folders in IDE like Visual Studio----------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---Load some basic macros which are needed later for the confiuration and build----------------
include(RootBuildOptions)
include(RootNewMacros)
include(CheckCompiler)
include(MacroEnsureVersion)

#---Enable CCache ------------------------------------------------------------------------------
if(ccache)
   set(CMAKE_IGNORE_PATH_TMP ${CMAKE_IGNORE_PATH})
   unset(CMAKE_IGNORE_PATH)

   find_program(ccache_cmd NAMES ccache ccache-swig)
   mark_as_advanced(ccache_cmd ${ccache_cmd})

   set(CMAKE_IGNORE_PATH ${CMAKE_IGNORE_PATH_TMP})

   if(ccache_cmd)
      message(STATUS "Using ccache for building")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
   else()
      message(STATUS "Could NOT find ccache")
   endif()
endif()

#---Enable test coverage -----------------------------------------------------------------------
if(coverage)
  set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
  set(GCC_COVERAGE_LINK_FLAGS    "-fprofile-arcs")
  set(CMAKE_CXX_FLAGS            "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHAREDLINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

#--- Enable build timing -----------------------------------------------------------------------
if (build_timing)
  # FIXME: This currently will override the use of ccache if -Dbuild_timing=On -Dccache=On is passed.
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
  #set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time") 
endif()

#---Enable CTest package -----------------------------------------------------------------------
#include(CTest)
if(testing)
  enable_testing()
endif()

#---Check if the user wants to build the project in the source directory------------------------
ROOT_CHECK_OUT_OF_SOURCE_BUILD()

#---Here we look for installed software and switch on and of the different build options--------
include(SearchInstalledSoftware)
ROOT_SHOW_OPTIONS()

#---Here we add tcmalloc to the linker flags if needed------------------------------------------
if (TCMALLOC_FOUND)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ltcmalloc -L ${TCMALLOC_LIBRARY_PATH}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ltcmalloc -L ${TCMALLOC_LIBRARY_PATH}")
endif()

#---Here we add jemalloc to the linker flags if needed------------------------------------------
if (JEMALLOC_FOUND)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ljemalloc -L ${JEMALLOC_LIBRARY_PATH}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ljemalloc -L ${JEMALLOC_LIBRARY_PATH}")
endif()

#---Populate the configure arguments returned by 'root-config --config'-------------------------
get_cmake_property(variables CACHE_VARIABLES)
foreach(var ${variables})
  if((var MATCHES "_(LIBRARIES|LIBRARY|INCLUDE)") AND
     (NOT ${${var}} STREQUAL "") AND
     (NOT ${var} MATCHES "NOTFOUND"))
    if (var MATCHES "^QT_")
      # filter out the very long list of Qt libraries and include dirs
      if (var MATCHES "(QT_LIBRARY_DIR|QT_QTCORE_INCLUDE_DIR)")
        set(ROOT_CONFIGARGS "${ROOT_CONFIGARGS}${var}=${${var}} ")
      endif()
    else()
      if ((NOT var MATCHES "_(DOCS|TESTS|INSTALL)") AND (NOT var MATCHES "^_"))
        set(ROOT_CONFIGARGS "${ROOT_CONFIGARGS}${var}=${${var}} ")
      endif()
    endif()
  endif()
endforeach()

#---Move (copy) directories to binary tree------------------------------------------------------
set(stamp_file ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/move_artifacts.stamp)
add_custom_command(OUTPUT ${stamp_file}
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/etc ${CMAKE_BINARY_DIR}/etc
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/test ${CMAKE_BINARY_DIR}/test
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/icons ${CMAKE_BINARY_DIR}/icons
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/fonts ${CMAKE_BINARY_DIR}/fonts
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/macros ${CMAKE_BINARY_DIR}/macros
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/LICENSE ${CMAKE_BINARY_DIR}/LICENSE
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/README ${CMAKE_BINARY_DIR}/README
                   COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
                   COMMENT "Copying directories such as etc, icons, fonts, etc. to build area")

#---Copy the the tutorials to the build area, with dependency---------------------------------
file(GLOB_RECURSE tutorial_files RELATIVE ${CMAKE_SOURCE_DIR} tutorials/*)
set(tutorial_files_builddir)
foreach(tutorial_file ${tutorial_files})
  # Filter out hsimple.root; someone might have created it in the src dir, and the hsimple.root
  # target below will interfere.
  if (NOT (tutorial_file STREQUAL "tutorials/hsimple.root"))
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${tutorial_file}
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${tutorial_file} ${CMAKE_BINARY_DIR}/${tutorial_file}
      COMMENT "Copying ${CMAKE_SOURCE_DIR}/${tutorial_file}"
      DEPENDS ${CMAKE_SOURCE_DIR}/${tutorial_file})
    list(APPEND tutorial_files_builddir ${CMAKE_BINARY_DIR}/${tutorial_file})
  endif()
endforeach()
add_custom_target(move_artifacts DEPENDS ${stamp_file} ${tutorial_files_builddir})

add_subdirectory (interpreter)
# Add the compilation flags from LLVM to the flags we use to compile cling-based tools.
set(CLING_CXXFLAGS "${CLING_CXXFLAGS} ${ROOT_LLVM_FLAGS}")

#---CXX MODULES-----------------------------------------------------------------------------------
if(cxxmodules)
  # Copy-pasted from HandleLLVMOptions.cmake, please keep up to date.
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fmodules -fcxx-modules")
  # Check that we can build code with modules enabled, and that repeatedly
  # including <cassert> still manages to respect NDEBUG properly.
  CHECK_CXX_SOURCE_COMPILES("#undef NDEBUG
                             #include <cassert>
                             #define NDEBUG
                             #include <cassert>
                             int main() { assert(this code is not compiled); }"
                             CXX_SUPPORTS_MODULES)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  if(NOT CXX_SUPPORTS_MODULES)
    message(FATAL_ERROR "cxxmodules is not supported by this compiler")
  endif()

  set(ROOT_CXXMODULES_COMMONFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -fmodules -fmodules-cache-path=${CMAKE_BINARY_DIR}/include/pcms/ -fno-autolink -fdiagnostics-show-note-include-stack -Wno-module-import-in-extern-c")

  # FIXME: We should remove this once libc++ supports -fmodules-local-submodule-visibility.
  if (APPLE)
    # FIXME: TGLIncludes and alike depend on glew.h doing special preprocessor
    # trickery to override the contents of system's OpenGL.
    # On OSX #include TGLIncludes.h will trigger the creation of the system
    # OpenGL.pcm. Once it is built, glew cannot use preprocessor trickery to 'fix'
    # the translation units which it needs to 'rewrite'. The translation units
    # which need glew support are in graf3d. However, depending on the modulemap
    # organization we could request it implicitly (eg. one big module for ROOT).
    # In these cases we need to 'prepend' this include path to the compiler in order
    # for glew.h to it its trick.
    set(ROOT_CXXMODULES_COMMONFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -isystem ${CMAKE_SOURCE_DIR}/graf3d/glew/isystem")
  endif()
endif(cxxmodules)

add_custom_target(copymodulemap DEPENDS "${CMAKE_BINARY_DIR}/include/module.modulemap")
add_custom_command(
                  OUTPUT "${CMAKE_BINARY_DIR}/include/module.modulemap"
                  DEPENDS build/unix/module.modulemap "${CMAKE_BINARY_DIR}/include/module.modulemap.extra"
                  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/build/unix/module.modulemap" "${CMAKE_BINARY_DIR}/include/module.modulemap"
                  COMMAND cat "${CMAKE_BINARY_DIR}/include/module.modulemap.extra" >> "${CMAKE_BINARY_DIR}/include/module.modulemap"
)
install(FILES "${CMAKE_BINARY_DIR}/include/module.modulemap" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)

add_dependencies(move_artifacts copymodulemap)

# Provide our own modulemap for implementations other than libcxx.
if (NOT libcxx)
  # Write a empty overlay file to the output directory that CMake can run its compiler tests.
  # We will create the actual overlay later in the configuration.
  file(WRITE ${CMAKE_BINARY_DIR}/include/modulemap.overlay.yaml "{'version' : 0, 'roots' : []}")
  set(__vfs_overlay "-ivfsoverlay ${CMAKE_BINARY_DIR}/include/modulemap.overlay.yaml")

  if (cxxmodules)
    set(ROOT_CXXMODULES_COMMONFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} ${__vfs_overlay}")
  endif()

  # Only use the first path from the HEADERS_LOCATION (which is separated by colons).
  get_property(__libcpp_full_paths GLOBAL PROPERTY ROOT_CLING_CXX_HEADERS_LOCATION)
  string(REGEX MATCHALL "[^:]+" __libcpp_full_paths_list "${__libcpp_full_paths}")
  if (__libcpp_full_paths_list)
    list(GET __libcpp_full_paths_list 0 __libcpp_full_path)

    configure_file(${CMAKE_SOURCE_DIR}/build/unix/modulemap.overlay.yaml.in ${CMAKE_BINARY_DIR}/include/modulemap.overlay.yaml @ONLY)

    configure_file(${CMAKE_SOURCE_DIR}/build/unix/stl.modulemap ${CMAKE_BINARY_DIR}/include/stl.modulemap)
    configure_file(${CMAKE_SOURCE_DIR}/build/unix/libc.modulemap ${CMAKE_BINARY_DIR}/include/libc.modulemap)
  else()
    message(SEND_ERROR "Couldn't find c++ library paths, no modulemap overlay will be installed! (__libcpp_full_paths = '${__libcpp_full_paths}')")
  endif()
endif()

if (cxxmodules)
  # These vars are useful when we want to compile things without cxxmodules.
  set(ROOT_CXXMODULES_CXXFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -fcxx-modules" CACHE STRING "Useful to filter out the modules-related cxxflags.")
  if (NOT APPLE)
    set(ROOT_CXXMODULES_CXXFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -Xclang -fmodules-local-submodule-visibility" CACHE STRING "Useful to filter out the modules-related cxxflags." FORCE)
  endif()
  set(ROOT_CXXMODULES_CFLAGS "${ROOT_CXXMODULES_COMMONFLAGS}" CACHE STRING "Useful to filter out the modules-related cflags.")


  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ROOT_CXXMODULES_CFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXXMODULES_CXXFLAGS}")
endif(cxxmodules)

#---Recurse into the given subdirectories.  This does not actually cause another cmake executable
#  to run. The same process will walk through the project's entire directory structure.
add_subdirectory (core)
add_subdirectory (build)
add_subdirectory (math)
add_subdirectory (hist)
add_subdirectory (tree)
add_subdirectory (io)
add_subdirectory (net)
add_subdirectory (graf2d)
add_subdirectory (graf3d)
add_subdirectory (gui)
add_subdirectory (proof)
add_subdirectory (html)
add_subdirectory (montecarlo)
add_subdirectory (geom)
add_subdirectory (rootx)
add_subdirectory (misc)
add_subdirectory (main)
add_subdirectory (bindings)
add_subdirectory (sql)
if(tmva)
  add_subdirectory(tmva)
endif()
if(roofit)
  add_subdirectory(roofit)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(tutorials)

#---CXX MODULES-----------------------------------------------------------------------------------
# Take all the modulemap contents we collected from the packages and append them to our modulemap.
# We have to delay this because the ROOT_CXXMODULES_EXTRA_MODULEMAP_CONTENT is filled in the
# add_subdirectory calls above.
get_property(__modulemap_extra_content GLOBAL PROPERTY ROOT_CXXMODULES_EXTRA_MODULEMAP_CONTENT)
string(REPLACE ";" "" __modulemap_extra_content "${__modulemap_extra_content}")
file(WRITE "${CMAKE_BINARY_DIR}/include/module.modulemap.extra" "${__modulemap_extra_content}")


get_property(__allHeaders GLOBAL PROPERTY ROOT_HEADER_TARGETS)
add_custom_target(move_headers ALL DEPENDS ${__allHeaders})

#---Global PCH-----------------------------------------------------------------------------------
get_property(__allTargets GLOBAL PROPERTY ROOT_DICTIONARY_TARGETS)
get_property(__allFiles GLOBAL PROPERTY ROOT_DICTIONARY_FILES)
get_property(__clingetcpch GLOBAL PROPERTY CLINGETCPCH)


set (CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS}")
if(cxxmodules)
  # rootcling uses our internal version of clang. Passing the modules flags here
  # would allow rootcling to find module files built by the external compiler
  # (eg. $CXX or $CC). This, in turn, would cause problems if we are using
  # different clang version (even different commit revision) as the modules files
  # are not guaranteed to be compatible among clang revisions.
  string(REPLACE "${ROOT_CXXMODULES_CXXFLAGS}" "" CMAKE_CXX_FLAGS_SEPARATE ${CMAKE_CXX_FLAGS_SEPARATE})
endif(cxxmodules)
string(REGEX REPLACE "[ ]-" ";-" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE} ${CORE_OS_DICT_CXX_FLAGS}")

add_custom_command(OUTPUT etc/dictpch/allLinkDefs.h
                          etc/dictpch/allHeaders.h
                          etc/dictpch/allCppflags.txt
                   COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/unix/makepchinput.py ${CMAKE_SOURCE_DIR} . ${__clingetcpch} -- ${CMAKE_CXX_FLAGS_SEPARATE}
                   DEPENDS ${CMAKE_SOURCE_DIR}/build/unix/makepchinput.py ${__allFiles})

get_property(incdirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
foreach(d ${incdirs})
  if(NOT "${d}" MATCHES "AFTER|BEFORE|INTERFACE|PRIVATE|PUBLIC|SYSTEM")
    set(__allIncludes ${__allIncludes} -I${d})
  endif()
endforeach()

add_custom_command(OUTPUT etc/allDict.cxx.pch
                   COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/etc/dictpch/makepch.py etc/allDict.cxx.pch ${__allIncludes} -I${CMAKE_BINARY_DIR}/include
                   DEPENDS ${CMAKE_SOURCE_DIR}/etc/dictpch/makepch.py
                           etc/dictpch/allLinkDefs.h
                           etc/dictpch/allHeaders.h
                           etc/dictpch/allCppflags.txt
                           rootcling)
add_custom_target(onepcm ALL DEPENDS etc/allDict.cxx.pch)
set_source_files_properties(${__allFiles} PROPERTIES GENERATED TRUE)
add_dependencies(onepcm ${__allTargets})
install(FILES ${CMAKE_BINARY_DIR}/etc/allDict.cxx.pch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})
install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/dictpch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})

#---hsimple.root---------(use the executable for clearer dependencies and proper return code)---
add_custom_target(hsimple ALL DEPENDS tutorials/hsimple.root)
add_dependencies(hsimple onepcm)
if(WIN32)
  add_custom_command(OUTPUT tutorials/hsimple.root
                     COMMAND set PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY} &&
                             set ROOTIGNOREPREFIX=1 &&
                             $<TARGET_FILE:root.exe> -l -q -b -n -x hsimple.C -e return
                     WORKING_DIRECTORY tutorials
                     DEPENDS $<TARGET_FILE:root.exe> Cling Hist Tree Gpad Graf HistPainter move_artifacts)
else()
  add_custom_command(OUTPUT tutorials/hsimple.root
                     COMMAND ${ld_library_path}=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{${ld_library_path}}
                             ROOTIGNOREPREFIX=1
                             $<TARGET_FILE:root.exe> -l -q -b -n -x hsimple.C -e return
                     WORKING_DIRECTORY tutorials
                     DEPENDS $<TARGET_FILE:root.exe> Cling Hist Tree Gpad Graf HistPainter move_artifacts)
endif()
install(FILES ${CMAKE_BINARY_DIR}/tutorials/hsimple.root DESTINATION ${CMAKE_INSTALL_TUTDIR} COMPONENT tests)

#---version--------------------------------------------------------------------------------------
add_custom_target(version COMMAND ${CMAKE_SOURCE_DIR}/build/unix/makeversion.sh ${CMAKE_BINARY_DIR}
                          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
#add_dependencies(version root.exe)

#---distribution commands------------------------------------------------------------------------
add_custom_target(distsrc COMMAND ${CMAKE_SOURCE_DIR}/build/unix/makedistsrc.sh ${CMAKE_SOURCE_DIR}
                  DEPENDS ${CMAKE_BINARY_DIR}/include/RGitCommit.h)
add_custom_target(dist COMMAND cpack --config CPackConfig.cmake)

#---Configure and install various files neded later and for clients -----------------------------
include(RootConfiguration)

#---Installation of project-wise artifacts-------------------------------------------------------
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
  install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
  if(gnuinstall)
    install(DIRECTORY README/ DESTINATION ${CMAKE_INSTALL_DOCDIR})
  else()
    install(DIRECTORY README DESTINATION ${CMAKE_INSTALL_DOCDIR})
  endif()
  install(DIRECTORY etc/ DESTINATION ${CMAKE_INSTALL_SYSCONFDIR} USE_SOURCE_PERMISSIONS
                         REGEX system.rootrc EXCLUDE
                         REGEX system.rootauthrc EXCLUDE
                         REGEX system.rootdaemonrc EXCLUDE
                         REGEX rootd.rc.d EXCLUDE
                         REGEX proofd.rc.d EXCLUDE
                         REGEX rootd.xinetd EXCLUDE
                         REGEX proofd.xinetd EXCLUDE
                         REGEX root.mimes EXCLUDE
                         REGEX cmake EXCLUDE )
  install(DIRECTORY fonts/  DESTINATION ${CMAKE_INSTALL_FONTDIR})
  install(DIRECTORY icons/  DESTINATION ${CMAKE_INSTALL_ICONDIR})
  install(DIRECTORY macros/ DESTINATION ${CMAKE_INSTALL_MACRODIR})
  install(DIRECTORY man/    DESTINATION ${CMAKE_INSTALL_MANDIR})
  install(DIRECTORY test/      DESTINATION ${CMAKE_INSTALL_TESTDIR} COMPONENT tests)
  install(DIRECTORY tutorials/ DESTINATION ${CMAKE_INSTALL_TUTDIR} COMPONENT tests)
  install(DIRECTORY cmake/modules DESTINATION ${CMAKE_INSTALL_CMAKEDIR} PATTERN "Find*.cmake" EXCLUDE)
  install(FILES build/misc/root.m4 DESTINATION ${CMAKE_INSTALL_ACLOCALDIR})
endif()

#---Configure Testing using CTest----------------------------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/CTestCustom.cmake ${CMAKE_BINARY_DIR} COPYONLY)
if(testing)
  include(RootCTest)
  if(roottest)
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    #---Is the roottest source directory around?
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      set(roottestdir ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../roottest)
      get_filename_component(roottestdir ${CMAKE_CURRENT_SOURCE_DIR}/../roottest ABSOLUTE)
    endif()
    if(roottestdir)
      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/roottest)
      add_subdirectory(${roottestdir} roottest)
    else()
      message("-- Could not find roottest directory! Cloning from the repository...")
      execute_process(COMMAND ${GIT_EXECUTABLE} clone -b ${GIT_BRANCH} http://root.cern.ch/git/roottest.git
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      add_subdirectory(roottest)
    endif()
  endif()
endif()

#---Packaging-------------------------------------------------------------------------------------
include(RootCPack)
