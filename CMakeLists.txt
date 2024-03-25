# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

if(WIN32)
  # Set CMP0091 (MSVC runtime library flags are selected by an abstraction) to OLD
  # to keep the old way of selecting the runtime library with the -MD/-MDd compiler flag
  cmake_policy(SET CMP0091 OLD)
  set(CMAKE_SKIP_TEST_ALL_DEPENDENCY TRUE)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT ALLOW_IN_SOURCE)
  message(FATAL_ERROR
    " ROOT must be built out-of-source.\n"
    " Please see README/INSTALL for more information.")
endif()

set(policy_new CMP0072 CMP0076 CMP0077 CMP0079)
foreach(policy ${policy_new})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

set(policy_old CMP0116 CMP0135 CMP0144)
foreach(policy ${policy_old})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} OLD)
  endif()
endforeach()

include(cmake/modules/CaptureCommandLine.cmake)

project(ROOT)

#---Set the locale to default C to prevent issued due to localization of commands---------------
# This is necessary as we for example call `clang -v` and parse its output. But on a localized
# program, the output parsing is much more error prone as certrain strings we're looking for
# could be missing or be in a different order. To prevent those errors, let's just force all
# output to use the default C locale which is more or less identical on all systems.
set(ENV{LANG} C)

#---Set paths where to put the libraries, executables and headers------------------------------
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # prevent mkdir races
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set permissions for installed folders and subfolders that come from the source tree in case
# the permissions in the source tree are wrong since the install command will preserve them
set(DIR_PERMISSIONS DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Before setting ROOTSYS, make sure that the environment isn't polluted by a different
# ROOT build. This is significant e,g. for roottest, which will otherwise have libraries
# of a different ROOT build available / visible / reachable.
if(NOT $ENV{ROOTSYS} STREQUAL "")
  string(REPLACE "$ENV{ROOTSYS}/bin" "" ENV_PATH "$ENV{PATH}")
  string(REPLACE "$ENV{ROOTSYS}/lib" "" ENV_LD_LIBRARY_PATH "$ENV{LD_LIBRARY_PATH}")
  string(REPLACE "$ENV{ROOTSYS}/lib" "" ENV_PYTHONPATH "$ENV{PYTHONPATH}")
  string(REPLACE "$ENV{ROOTSYS}" "" ENV_CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
  set(ENV{PATH} "${ENV_PATH}")
  set(ENV{LD_LIBRARY_PATH} "${ENV_LD_LIBRARY_PATH}")
  set(ENV{PYTHONPATH} "${ENV_PYTHONPATH}")
  set(ENV{CMAKE_PREFIX_PATH} "${ENV_CMAKE_PREFIX_PATH}")
  set(ENV{ROOTSYS} ${CMAKE_BINARY_DIR})
endif()

set(ROOTSYS ${CMAKE_BINARY_DIR})
set(HEADER_OUTPUT_PATH ${CMAKE_BINARY_DIR}/include)

#---Set the ROOT version--------------------------------------------------------------------
include(cmake/modules/SetROOTVersion.cmake)

message(STATUS "Building ROOT version ${ROOT_FULL_VERSION}")

#---Where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/ is checked-------------
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#---Enable Folders in IDE like Visual Studio----------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---Load some basic macros which are needed later for the confiuration and build----------------
include(CheckCompiler)
include(RootBuildOptions)
include(RootMacros)
include(CheckAssembler)
include(CheckIntrinsics)

# relatedrepo_GetClosestMatch(REPO_NAME <repo> ORIGIN_PREFIX <originp> UPSTREAM_PREFIX <upstreamp>
#                             FETCHURL_VARIABLE <output_url> FETCHREF_VARIABLE <output_ref>)
# Return the clone URL and head/tag of the closest match for `repo` (e.g. roottest), based on the
# current head name.
function(relatedrepo_GetClosestMatch)
  cmake_parse_arguments(_ "" "REPO_NAME;ORIGIN_PREFIX;UPSTREAM_PREFIX;FETCHURL_VARIABLE;FETCHREF_VARIABLE" "" ${ARGN})

  set(${__FETCHURL_VARIABLE} ${__UPSTREAM_PREFIX}/${__REPO_NAME} PARENT_SCOPE)

  if(NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    set(${__FETCHREF_VARIABLE} v${ROOT_MAJOR_VERSION}-${ROOT_MINOR_VERSION}-${ROOT_PATCH_VERSION} PARENT_SCOPE)
    return()
  endif()

  execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git
                  rev-parse --abbrev-ref HEAD
                  OUTPUT_VARIABLE current_head OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${__FETCHREF_VARIABLE} ${current_head} PARENT_SCOPE)

  # `current_head` is a well-known branch, e.g. master, or v6-28-00-patches.  Use the matching branch
  # upstream as the fork repository may be out-of-sync
  string(REGEX MATCH "^(master|latest-stable|v[0-9]+-[0-9]+-[0-9]+(-patches)?)$" known_head ${current_head})
  if(NOT "${known_head}" STREQUAL "")
    if("${current_head}" STREQUAL "latest-stable")
      # Resolve the 'latest-stable' branch to the latest merged head/tag
      execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git
                      for-each-ref --points-at=latest-stable^2 --format=%\(refname:short\)
                      OUTPUT_VARIABLE current_head OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(${__FETCHREF_VARIABLE} ${current_head} PARENT_SCOPE)
    endif()
    return()
  endif()

  # Otherwise, try to use a branch that matches `current_head` in the fork repository
  execute_process(COMMAND ${GIT_EXECUTABLE} ls-remote --heads --tags
                  ${__ORIGIN_PREFIX}/${__REPO_NAME} ${current_head} OUTPUT_VARIABLE matching_refs ERROR_QUIET)
  if(NOT "${matching_refs}" STREQUAL "")
    set(${__FETCHURL_VARIABLE} ${__ORIGIN_PREFIX}/${__REPO_NAME} PARENT_SCOPE)
    return()
  endif()

  # Finally, try upstream using the closest head/tag below the parent commit of the current head
  execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git
                  describe --all --abbrev=0 HEAD^
                  OUTPUT_VARIABLE closest_ref ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE "^(heads|tags)/" "" candidate_head "${closest_ref}")
  execute_process(COMMAND ${GIT_EXECUTABLE} ls-remote --heads --tags
                  ${__UPSTREAM_PREFIX}/${__REPO_NAME} "${candidate_head}" OUTPUT_VARIABLE matching_refs)
  if(NOT "${matching_refs}" STREQUAL "")
    set(${__FETCHREF_VARIABLE} ${candidate_head} PARENT_SCOPE)
    return()
  endif()
  set(${__FETCHREF_VARIABLE} "" PARENT_SCOPE)
endfunction()

# relaterepo_Checkout(REPO_NAME <repo> FETCHURL <url> FETCHREF <ref>
#                     [FORCE] [REPO_DIR_VARIABLE <repo_dir>])
# Clone/fetch and checkout the specified `ref` from the given repository at `url`.  If a local clone of the
# repository is found, refuse to update it unless the `FORCE` option is in place.
function(relatedrepo_Checkout)
  cmake_parse_arguments(_ "FORCE" "REPO_NAME;FETCHURL;FETCHREF;REPO_DIR_VARIABLE" "" ${ARGN})

  # Check whether the repository exists in the source directory or its parent
  get_filename_component(source_dir ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
  if(IS_DIRECTORY ${source_dir}/${__REPO_NAME})
    set(repo_dir ${source_dir}/${__REPO_NAME})
  elseif(IS_DIRECTORY ${source_dir}/../${__REPO_NAME})
    set(repo_dir ${source_dir}/../${__REPO_NAME})
  endif()

  if(NOT DEFINED repo_dir)
    message(STATUS "Cloning ${__REPO_NAME} from '${__FETCHURL}' (${__FETCHREF})")
    if(NOT "${__FETCHREF}" STREQUAL "")
      string(PREPEND __FETCHREF "-b")
    endif()
    execute_process(COMMAND ${GIT_EXECUTABLE} clone ${__FETCHREF} ${__FETCHURL}
                    WORKING_DIRECTORY ${source_dir})
    set(repo_dir ${source_dir}/${__REPO_NAME})
  else()
    message(STATUS "Found ${__REPO_NAME}: ${repo_dir}")
    if(${__FORCE})
      message(STATUS "Updating ${__REPO_NAME} from '${__FETCHURL}' (${__FETCHREF})")
      execute_process(COMMAND ${GIT_EXECUTABLE} fetch -fu ${__FETCHURL} ${__FETCHREF}:${__FETCHREF}
                      WORKING_DIRECTORY ${repo_dir})
      execute_process(COMMAND ${GIT_EXECUTABLE} checkout -f ${__FETCHREF}
                      WORKING_DIRECTORY ${repo_dir})
    endif()
  endif()
  if(DEFINED __REPO_DIR_VARIABLE)
    set(${__REPO_DIR_VARIABLE} ${repo_dir} PARENT_SCOPE)
  endif()
endfunction()

#---Enable asserts------------------------------------------------------------------------------
if(asserts)
  string(REGEX REPLACE "-[UD]NDEBUG(=.*)?" "" "CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}" "${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}")
  string(REGEX REPLACE "-[UD]NDEBUG(=.*)?" "" "CMAKE_C_FLAGS_${_BUILD_TYPE_UPPER}" "${CMAKE_C_FLAGS_${_BUILD_TYPE_UPPER}}")
endif()

#---Enable CCache ------------------------------------------------------------------------------
if(ccache)
  find_program(CCACHE_COMMAND NAMES ccache ccache-swig)
  mark_as_advanced(CCACHE_COMMAND ${CCACHE_COMMAND})

  if(EXISTS ${CCACHE_COMMAND})
    message(STATUS "Found ccache: ${CCACHE_COMMAND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_COMMAND})
    execute_process(COMMAND ${CCACHE_COMMAND} "-V" OUTPUT_VARIABLE CCACHE_VERSION)
    string(REGEX REPLACE "ccache version ([0-9\\.]+).*" "\\1" CCACHE_VERSION ${CCACHE_VERSION})
  else()
    message(STATUS "Could NOT find ccache")
    set(ccache OFF CACHE BOOL "Use ccache (disabled since ccache was not found)" FORCE)
  endif()
endif()

#---Enable distcc ------------------------------------------------------------------------------
if(distcc)
  find_program(DISTCC_COMMAND NAMES distcc)
  mark_as_advanced(DISTCC_COMMAND ${DISTCC_COMMAND})
  if(EXISTS ${DISTCC_COMMAND})
    message(STATUS "Found distcc: ${DISTCC_COMMAND}")
    if (ccache)
        # If ccache is enabled, use distcc as CCACHE_PREFIX
        message(STATUS "Because ccache is enabled, CCACHE_PREFIX is set to ${DISTCC_COMMAND}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "CCACHE_PREFIX=${DISTCC_COMMAND} ${CCACHE_COMMAND}")
    else()
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${DISTCC_COMMAND})
    endif()
    execute_process(COMMAND ${DISTCC_COMMAND} "--version" OUTPUT_VARIABLE DISTCC_VERSION)
    string(REGEX REPLACE "distcc ([0-9\\.]+).*" "\\1" DISTCC_VERSION ${DISTCC_VERSION})
  else()
    message(STATUS "Could NOT find distcc")
    set(distcc OFF CACHE BOOL "Use distcc (disabled since distcc was not found)" FORCE)
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

#--- Set up address sanitizer builds ----------------------------------------------------------
if(asan)
  if(NOT CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
    message(WARNING "Address sanitizer builds only tested with gcc and Clang")
  endif()

  if(NOT MSVC)
    set(ASAN_EXTRA_LD_PRELOAD "${CMAKE_BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ROOTSanitizerConfig${CMAKE_SHARED_LIBRARY_SUFFIX}:${ASAN_RUNTIME_LIBRARY}")
  endif()

  foreach(item IN LISTS ASAN_EXTRA_CXX_FLAGS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${item}>)
  endforeach()
  #add_link_options() not available in our CMake version:
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_EXTRA_SHARED_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_EXTRA_EXE_LINKER_FLAGS}")
endif()

#---Enable CTest package -----------------------------------------------------------------------
#include(CTest)
if(testing)
  enable_testing()
endif()

#---Here we look for installed software and switch on and of the different build options--------
include(SearchInstalledSoftware)

#---Here we add tcmalloc to the linker flags if needed------------------------------------------
if (TCMALLOC_FOUND)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ltcmalloc -L${TCMALLOC_LIBRARY_PATH}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ltcmalloc -L${TCMALLOC_LIBRARY_PATH}")
endif()

#---Here we add jemalloc to the linker flags if needed------------------------------------------
if (JEMALLOC_FOUND)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ljemalloc -L${JEMALLOC_LIBRARY_PATH}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ljemalloc -L${JEMALLOC_LIBRARY_PATH}")
endif()

#---Populate the configure arguments returned by 'root-config --config'-------------------------
get_cmake_property(variables CACHE_VARIABLES)
foreach(var ${variables})
  if((var MATCHES "_(LIBRARIES|LIBRARY|INCLUDE|VERSION)") AND
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
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/LICENSE ${CMAKE_BINARY_DIR}/LICENSE
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/README ${CMAKE_BINARY_DIR}/README
                   COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
                   COMMENT "Copying directories such as etc, icons, fonts, js, ui5, etc. to build area")

if(http)
set(jsroot_files js/*)
endif()
if(webgui)
set(openui5_files ui5/*)
endif()

#---Copy files to the build area, with dependency---------------------------------
file(GLOB_RECURSE artifact_files RELATIVE ${CMAKE_SOURCE_DIR} tutorials/* etc/* test/* icons/* fonts/* macros/* ${jsroot_files} ${openui5_files})
set(artifact_files_builddir)
foreach(artifact_file ${artifact_files})
  # Filter out hsimple.root; someone might have created it in the src dir, and the hsimple.root
  # target below will interfere.
  if (NOT (artifact_file STREQUAL "tutorials/hsimple.root"))
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${artifact_file}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/${artifact_file} ${CMAKE_BINARY_DIR}/${artifact_file}
      COMMENT "Copying ${CMAKE_SOURCE_DIR}/${artifact_file}"
      DEPENDS ${CMAKE_SOURCE_DIR}/${artifact_file})
    list(APPEND artifact_files_builddir ${CMAKE_BINARY_DIR}/${artifact_file})
  endif()
endforeach()
add_custom_target(move_artifacts DEPENDS ${stamp_file} ${artifact_files_builddir})

add_subdirectory (interpreter)

# Update etc/gitinfo.txt for every build.
add_custom_target(gitinfotxt
  ALL
  COMMAND ${CMAKE_COMMAND} -DSRCDIR=${CMAKE_SOURCE_DIR} -DBINDIR=${CMAKE_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/modules/UpdateGitInfo.cmake
  COMMENT "Updating etc/gitinfo.txt."
  BYPRODUCTS ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
)
install(FILES ${CMAKE_BINARY_DIR}/etc/gitinfo.txt DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})

#---Recurse into the given subdirectories.  This does not actually cause another cmake executable
#  to run. The same process will walk through the project's entire directory structure.
add_subdirectory (core)
add_subdirectory (math)
add_subdirectory (hist)
add_subdirectory (tree)
add_subdirectory (io)
add_subdirectory (net)
add_subdirectory (graf2d)
add_subdirectory (graf3d)
add_subdirectory (gui)
if(html)
  add_subdirectory (html)
endif()
add_subdirectory (montecarlo)
if(geom)
  add_subdirectory (geom)
endif()
if(NOT WIN32)
  add_subdirectory (rootx)
endif()
add_subdirectory (misc)
add_subdirectory (main)
add_subdirectory (bindings)
add_subdirectory (sql)
if(proof)
  add_subdirectory(proof)
endif()
if(tmva)
  add_subdirectory(tmva)
endif()
if(roofit)
  add_subdirectory(roofit)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(tutorials)

get_property(__allHeaders GLOBAL PROPERTY ROOT_HEADER_TARGETS)
get_property(__allBuiltins GLOBAL PROPERTY ROOT_BUILTIN_TARGETS)
add_custom_target(move_headers ALL DEPENDS ${__allHeaders} ${__allBuiltins} gitinfotxt)

#---CXX MODULES-----------------------------------------------------------------------------------
if(MSVC)
  set(_os_cat "type")
else()
  set(_os_cat "cat")
endif()
file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/include/module.modulemap.extra" _from_native)
file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/include/ROOT.modulemap" _to_native)

add_custom_target(copymodulemap DEPENDS "${CMAKE_BINARY_DIR}/include/ROOT.modulemap")
add_custom_command(
		  OUTPUT "${CMAKE_BINARY_DIR}/include/ROOT.modulemap"
                  DEPENDS cmake/unix/module.modulemap "${CMAKE_BINARY_DIR}/include/module.modulemap.extra"
		  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/cmake/unix/module.modulemap" "${CMAKE_BINARY_DIR}/include/ROOT.modulemap"
                  COMMAND ${_os_cat} "${_from_native}" >> "${_to_native}"
)
install(FILES "${CMAKE_BINARY_DIR}/include/ROOT.modulemap" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)

add_dependencies(move_headers copymodulemap)

# Take all the modulemap contents we collected from the packages and append them to our modulemap.
# We have to delay this because the ROOT_CXXMODULES_EXTRA_MODULEMAP_CONTENT is filled in the
# add_subdirectory calls above.
get_property(__modulemap_extra_content GLOBAL PROPERTY ROOT_CXXMODULES_EXTRA_MODULEMAP_CONTENT)
string(REPLACE ";" "" __modulemap_extra_content "${__modulemap_extra_content}")
# Write module.modulemap.extra to a temporary file first, to not touch module.modulemap.extra
# if it's unchanged.
file(WRITE "${CMAKE_BINARY_DIR}/include/module.modulemap.extra.tmp" "${__modulemap_extra_content}")
configure_file("${CMAKE_BINARY_DIR}/include/module.modulemap.extra.tmp"
    "${CMAKE_BINARY_DIR}/include/module.modulemap.extra"
    COPYONLY)

# From now on we handled all exposed module and want to make all new modulemaps private to ROOT.
set(ROOT_CXXMODULES_WRITE_TO_CURRENT_DIR ON)

set (CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS}")

string(REGEX REPLACE "[ ]-" ";-" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE}")
if(MSVC)
  string(REPLACE "-Zc:__cplusplus" "" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE}")
  string(REPLACE "-nologo" "" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE}")
  string(REPLACE "-EHsc-" "" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE}")
  string(REPLACE "-GR" "" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE}")
  string(REPLACE "-MDd" "" CMAKE_CXX_FLAGS_SEPARATE "${CMAKE_CXX_FLAGS_SEPARATE}")
endif()

if(runtime_cxxmodules)
  # Dummy target that does nothing, we don't need a PCH for modules.
  # Onepcm target has all dependencies needed for allDict.cxx.pch, which allow
  # to test hsimple.C after all C++ modules are updated.
  add_custom_target(onepcm)
  foreach(target_dependency ${ROOT_LIBRARY_TARGETS})
    add_dependencies(onepcm ${target_dependency})
  endforeach()
  unset(ROOT_LIBRARY_TARGETS CACHE)
else()
  get_property(incdirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)

  if(WIN32)
    list(APPEND incdirs
      ${CMAKE_SOURCE_DIR}/graf2d/win32gdk/gdk/src
      ${CMAKE_SOURCE_DIR}/graf2d/win32gdk/gdk/src/gdk
      ${CMAKE_SOURCE_DIR}/graf2d/win32gdk/gdk/src/glib
    )
  endif()

  foreach(d ${incdirs})
    if(NOT "${d}" MATCHES "AFTER|BEFORE|INTERFACE|PRIVATE|PUBLIC|SYSTEM")
      set(__allIncludes ${__allIncludes} -I${d})
    endif()
  endforeach()

  get_property(__cling_pch GLOBAL PROPERTY CLINGETCPCH)
  get_property(__pch_dependencies GLOBAL PROPERTY ROOT_PCH_DEPENDENCIES)
  get_property(__pch_dictionaries GLOBAL PROPERTY ROOT_PCH_DICTIONARIES)

  add_custom_command(OUTPUT etc/allDict.cxx.pch
    BYPRODUCTS
      etc/dictpch/allCppflags.txt
      etc/dictpch/allHeaders.h
      etc/dictpch/allLinkDefs.h
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/unix/makepchinput.py
      ${CMAKE_SOURCE_DIR} . ${pyroot_legacy} ${__cling_pch}
    COMMAND
      ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/etc/dictpch/makepch.py etc/allDict.cxx.pch
      ${__allIncludes} -I${CMAKE_BINARY_DIR}/include -I${CMAKE_SOURCE_DIR}/core
    DEPENDS
      rootcling ${__pch_dependencies} ${__pch_dictionaries}
      ${CMAKE_SOURCE_DIR}/cmake/unix/makepchinput.py
      ${CMAKE_SOURCE_DIR}/etc/dictpch/makepch.py
  )
  add_custom_target(onepcm ALL DEPENDS etc/allDict.cxx.pch)
  install(FILES ${CMAKE_BINARY_DIR}/etc/allDict.cxx.pch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})
  install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/dictpch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})
endif()

# FIXME: move installation of PCMS in ROOT_GENERATE_DICTIONARY().
# We are excluding directories, which are accidentaly copied via unxpected behaviour of install(DIRECTORY ..)
install(
   DIRECTORY ${CMAKE_BINARY_DIR}/lib/
   DESTINATION ${CMAKE_INSTALL_LIBDIR}
   FILES_MATCHING
   PATTERN "*.pcm"
   PATTERN "modules.idx"
   PATTERN "JupyROOT" EXCLUDE
   PATTERN "JsMVA" EXCLUDE
   PATTERN "python*" EXCLUDE
   PATTERN "cmake" EXCLUDE
   PATTERN "pkgconfig" EXCLUDE
)

if(Vc_INCLUDE_DIR)
  set(MODULES_ROOT_INCPATH "ROOT_INCLUDE_PATH=${Vc_INCLUDE_DIR}:${ROOT_INCLUDE_PATH}")
endif()

# modules.idx
if(runtime_cxxmodules)
  ROOT_GET_LIBRARY_OUTPUT_DIR(library_output_dir)
  get_property(modules_idx_deps GLOBAL PROPERTY modules_idx_deps_property)
  if(WIN32)
    set(modules_idx_cmd COMMAND ${CMAKE_COMMAND} -E env PATH="${library_output_dir}\\\;%PATH%"
                                ROOTIGNOREPREFIX=1 ROOT_HIST=0 $<TARGET_FILE:root.exe> -l -q -b)
  else()
    set(modules_idx_cmd COMMAND ${ld_library_path}=${library_output_dir}:$ENV{${ld_library_path}}
                                ROOTIGNOREPREFIX=1 ROOT_HIST=0 $<TARGET_FILE:root.exe> -l -q -b)
  endif()
  add_custom_command(OUTPUT ${library_output_dir}/modules.idx
                     COMMAND ${CMAKE_COMMAND} -E remove -f modules.idx modules.timestamp
                     ${modules_idx_cmd}
                     WORKING_DIRECTORY ${library_output_dir}
                     DEPENDS $<TARGET_FILE:root.exe> Cling Hist Tree Gpad Graf HistPainter move_artifacts
                             ${modules_idx_deps})
  add_custom_target(modules_idx ALL DEPENDS ${library_output_dir}/modules.idx)
  add_dependencies(modules_idx ${modules_idx_deps})
  set_property(TARGET modules_idx PROPERTY modules_idx_file ${library_output_dir}/modules.idx)
  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${library_output_dir}/modules.timestamp")
endif()

#---hsimple.root---------(use the executable for clearer dependencies and proper return code)---
add_custom_target(hsimple ALL DEPENDS tutorials/hsimple.root)
add_dependencies(hsimple onepcm)
if(WIN32)
  set(hsimple_cmd COMMAND ${CMAKE_COMMAND} -E env PATH="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\\\;%PATH%"
                          ROOTIGNOREPREFIX=1 ROOT_HIST=0 $<TARGET_FILE:root.exe> -l -q -b -n -x hsimple.C -e return)
else()
  set(hsimple_cmd COMMAND ${MODULES_ROOT_INCPATH} ${ld_library_path}=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{${ld_library_path}}
                          ROOTIGNOREPREFIX=1 ROOT_HIST=0 $<TARGET_FILE:root.exe> -l -q -b -n -x hsimple.C -e return)
endif()
add_custom_command(OUTPUT tutorials/hsimple.root
                   ${hsimple_cmd}
                   WORKING_DIRECTORY tutorials
                   DEPENDS $<TARGET_FILE:root.exe> Cling Hist Tree Gpad Graf HistPainter move_artifacts)
install(FILES ${CMAKE_BINARY_DIR}/tutorials/hsimple.root DESTINATION ${CMAKE_INSTALL_TUTDIR} COMPONENT tests)

if(runtime_cxxmodules)
  add_dependencies(hsimple modules_idx)
endif()

#---copy special headers required for building on Windows----------------------------------------
if(WIN32)
  file(COPY cmake/win/w32pragma.h DESTINATION ${CMAKE_BINARY_DIR}/include/)
  file(COPY cmake/win/sehmap.h DESTINATION ${CMAKE_BINARY_DIR}/include/)
endif()

#---version--------------------------------------------------------------------------------------
if(NOT WIN32)
add_custom_target(version COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/unix/makeversion.py
                          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

#---distribution commands------------------------------------------------------------------------
add_custom_target(distsrc COMMAND ${CMAKE_SOURCE_DIR}/cmake/unix/makedistsrc.sh "${ROOT_FULL_VERSION}" "${CMAKE_SOURCE_DIR}")
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
                         ${DIR_PERMISSIONS}
                         PATTERN "system.rootrc" EXCLUDE
                         PATTERN "system.rootauthrc" EXCLUDE
                         PATTERN "system.rootdaemonrc" EXCLUDE
                         PATTERN "root.mimes" EXCLUDE
                         PATTERN "*.in" EXCLUDE)
  install(DIRECTORY fonts/  DESTINATION ${CMAKE_INSTALL_FONTDIR} ${DIR_PERMISSIONS})
  install(DIRECTORY icons/  DESTINATION ${CMAKE_INSTALL_ICONDIR} ${DIR_PERMISSIONS})
  install(DIRECTORY macros/ DESTINATION ${CMAKE_INSTALL_MACRODIR} ${DIR_PERMISSIONS})
  if(http)
     install(DIRECTORY js/     DESTINATION ${CMAKE_INSTALL_JSROOTDIR} ${DIR_PERMISSIONS})
  endif()
  if(webgui)
     install(DIRECTORY ui5/    DESTINATION ${CMAKE_INSTALL_OPENUI5DIR} ${DIR_PERMISSIONS})
  endif()
  set(MAN_PATT_EXCL)
  if(NOT fortran OR NOT CMAKE_Fortran_COMPILER)
    list(APPEND MAN_PATT_EXCL PATTERN h2root.1 EXCLUDE)
    list(APPEND MAN_PATT_EXCL PATTERN g2root.1 EXCLUDE)
  endif()
  install(DIRECTORY man/    DESTINATION ${CMAKE_INSTALL_MANDIR} ${DIR_PERMISSIONS} ${MAN_PATT_EXCL})
  install(DIRECTORY tutorials/ DESTINATION ${CMAKE_INSTALL_TUTDIR} ${DIR_PERMISSIONS} COMPONENT tests)
  install(FILES
    cmake/modules/RootMacros.cmake
    cmake/modules/RootTestDriver.cmake
    DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
  install(FILES
    "cmake/modules/FindVdt.cmake"
    DESTINATION "${CMAKE_INSTALL_CMAKEDIR}/modules")
endif()

#---Add configuration files for kernel and jupyter----------------------------------------------
# Make sure the Jupyter ROOT C++ kernel runs with the same Python version as ROOT
set(root_jupyter_dir notebook)
set(root_jupyter_config jupyter_notebook_config.py)
configure_file(etc/${root_jupyter_dir}/${root_jupyter_config}.in etc/${root_jupyter_dir}/${root_jupyter_config})
install(FILES ${CMAKE_BINARY_DIR}/etc/${root_jupyter_dir}/${root_jupyter_config} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${root_jupyter_dir})

set(root_kernel_dir  ${root_jupyter_dir}/kernels/root)
set(root_kernel_file kernel.json)
configure_file(etc/${root_kernel_dir}/${root_kernel_file}.in etc/${root_kernel_dir}/${root_kernel_file})
install(FILES ${CMAKE_BINARY_DIR}/etc/${root_kernel_dir}/${root_kernel_file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${root_kernel_dir})

#---install clad header files-------------------------------------------------------------------
if(clad)
   install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/plugins/
      DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/plugins)
endif()

#---Set flag for PyROOT tests that are expected to fail
if(pyroot)
  set(PYTESTS_WILLFAIL WILLFAIL)
endif()

#---Configure Testing using CTest----------------------------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/CTestCustom.cmake ${CMAKE_BINARY_DIR} COPYONLY)
if(testing)
  include(RootCTest)
  set(upstreamprefix https://github.com/root-project)

  if(roottest)
    find_package(Git REQUIRED)

    # Check whether the repository exists in the source directory or its parent
    get_filename_component(source_dir ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
    if(IS_DIRECTORY ${source_dir}/roottest/.git)
      set(repo_dir ${source_dir}/roottest)
    elseif(IS_DIRECTORY ${source_dir}/../roottest/.git)
      set(repo_dir ${source_dir}/../roottest)
    endif()
    if(DEFINED repo_dir)
      execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${repo_dir}/.git
                      remote get-url origin OUTPUT_VARIABLE originurl OUTPUT_STRIP_TRAILING_WHITESPACE
                      RESULT_VARIABLE query_result
                      ERROR_VARIABLE query_error)
      if(NOT query_result EQUAL 0)
        message(STATUS "Searching for \"origin\" repo of roottest: ${query_error}")
      endif()
    else()
       # The fetch URL of the 'origin' remote is used to determine the prefix for other repositories by
       # removing the `/root(\.git)?` part.  If `GITHUB_PR_ORIGIN` is defined in the environment, its
       # value is used instead.
       if(DEFINED ENV{GITHUB_PR_ORIGIN})
         set(originurl $ENV{GITHUB_PR_ORIGIN})
       else()
         execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git
                         remote get-url origin OUTPUT_VARIABLE originurl OUTPUT_STRIP_TRAILING_WHITESPACE)
       endif()
    endif()
    string(REGEX REPLACE "/root(test)?(\.git)?$" "" originprefix "${originurl}")
    relatedrepo_GetClosestMatch(REPO_NAME roottest
                                ORIGIN_PREFIX ${originprefix} UPSTREAM_PREFIX ${upstreamprefix}
                                FETCHURL_VARIABLE roottest_url FETCHREF_VARIABLE roottest_ref)
    # Use `-Droottest_force_checkout=ON` to force fetch and checkout in an existing repository
    if(roottest_force_checkout)
       set(roottest_opts FORCE)
    endif()
    relatedrepo_Checkout(REPO_NAME roottest FETCHURL ${roottest_url} FETCHREF "${roottest_ref}"
                         REPO_DIR_VARIABLE roottest_dir ${roottest_opts})
    if(NOT IS_DIRECTORY ${roottest_dir})
      message(FATAL_ERROR "Expected roottest at '${roottest_dir}' (not a directory?)")
    endif()
    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/roottest)
    add_subdirectory(${roottest_dir} roottest)
  endif()

  if(rootbench)
    find_package(Git REQUIRED)
    if(rootbench_force_checkout)
       set(rootbench_opts FORCE)
    endif()
    relatedrepo_Checkout(REPO_NAME rootbench FETCHURL ${upstreamprefix} FETCHREF master
                         REPO_DIR_VARIABLE rootbench_dir ${rootbench_opts})
    if(NOT IS_DIRECTORY ${rootbench_dir})
      message(FATAL_ERROR "Expected rootbench at '${rootbench_dir}' (not a directory?)")
    endif()
    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/rootbench)
    add_subdirectory(${rootbench_dir} rootbench)
  endif()
endif()

if(LLVM_LINKER_IS_MOLD)
  message(FATAL_ERROR "The mold linker is not supported by ROOT. Please use a different linker")
endif()

cmake_host_system_information(RESULT PROCESSOR QUERY PROCESSOR_DESCRIPTION)

message(STATUS "ROOT Configuration \n
System:            ${CMAKE_SYSTEM}
ROOT Platform:     ${ROOT_PLATFORM}
ROOT Architecture: ${ROOT_ARCHITECTURE}
Processor:         ${PROCESSOR} (${CMAKE_SYSTEM_PROCESSOR})
Build type:        ${CMAKE_BUILD_TYPE}
Install path:      ${CMAKE_INSTALL_PREFIX}
Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
C++ standard:      ${CMAKE_CXX_STANDARD}
Compiler flags:")
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
message(STATUS "
 - C:                         ${CMAKE_C_FLAGS}
 - C (build type specific):
     Debug:                   ${CMAKE_C_FLAGS_DEBUG}
     Release:                 ${CMAKE_C_FLAGS_RELEASE}
     MinSizeRel:              ${CMAKE_C_FLAGS_MINSIZEREL}
     RelWithDebInfo:          ${CMAKE_C_FLAGS_RELWITHDEBINFO}
 - C++:                       ${CMAKE_CXX_FLAGS}
 - C++ (build type specific):
     Debug:                   ${CMAKE_CXX_FLAGS_DEBUG}
     Release:                 ${CMAKE_CXX_FLAGS_RELEASE}
     MinSizeRel:              ${CMAKE_CXX_FLAGS_MINSIZEREL}
     RelWithDebInfo:          ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
message(STATUS "
 - C:                         ${CMAKE_C_FLAGS}
 - C (build type specific):   ${CMAKE_C_FLAGS_${_BUILD_TYPE_UPPER}}
 - C++:                       ${CMAKE_CXX_FLAGS}
 - C++ (build type specific): ${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}")
endif()
message(STATUS "Linker flags:
 - Executable:     ${CMAKE_EXE_LINKER_FLAGS}
 - Module:         ${CMAKE_MODULE_LINKER_FLAGS}
 - Shared:         ${CMAKE_SHARED_LINKER_FLAGS}\n")


ROOT_SHOW_ENABLED_OPTIONS()

#---Packaging-------------------------------------------------------------------------------------
include(RootCPack)
