# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#---------------------------------------------------------------------------------------------------
#  RootMacros.cmake
#---------------------------------------------------------------------------------------------------

if(WIN32)
  set(libprefix lib)
  set(ld_library_path PATH)
  set(libsuffix .dll)
  set(localruntimedir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(runtimedir ${CMAKE_INSTALL_BINDIR})
elseif(APPLE)
  set(ld_library_path DYLD_LIBRARY_PATH)
  set(ld_preload DYLD_INSERT_LIBRARIES)
  set(libprefix ${CMAKE_SHARED_LIBRARY_PREFIX})
  if(CMAKE_PROJECT_NAME STREQUAL ROOT)
    set(libsuffix .so)
  else()
    set(libsuffix ${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(localruntimedir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(runtimedir ${CMAKE_INSTALL_PYTHONDIR})
else()
  set(ld_library_path LD_LIBRARY_PATH)
  set(ld_preload LD_PRELOAD)
  set(libprefix ${CMAKE_SHARED_LIBRARY_PREFIX})
  set(libsuffix ${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(localruntimedir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(runtimedir ${CMAKE_INSTALL_PYTHONDIR})
endif()

set(ROOT_LIBRARY_PROPERTIES_NO_VERSION ${ROOT_LIBRARY_PROPERTIES_NO_VERSION}
    SUFFIX ${libsuffix}
    PREFIX ${libprefix} )
if(soversion)
  set(ROOT_LIBRARY_PROPERTIES ${ROOT_LIBRARY_PROPERTIES} ${ROOT_LIBRARY_PROPERTIES_NO_VERSION}
      VERSION ${ROOT_VERSION}
      SOVERSION ${ROOT_MAJOR_VERSION}.${ROOT_MINOR_VERSION} )
else()
  set(ROOT_LIBRARY_PROPERTIES ${ROOT_LIBRARY_PROPERTIES} ${ROOT_LIBRARY_PROPERTIES_NO_VERSION} )
endif()

include(CMakeParseArguments)

#---------------------------------------------------------------------------------------------------
#---ROOT_GLOB_FILES( <variable> [REALTIVE path] [FILTER regexp] <sources> ...)
#---------------------------------------------------------------------------------------------------
function(ROOT_GLOB_FILES variable)
  CMAKE_PARSE_ARGUMENTS(ARG "RECURSE" "RELATIVE;FILTER" "" ${ARGN})
  set(_possibly_recurse "")
  if (ARG_RECURSE)
    set(_possibly_recurse "_RECURSE")
  endif()
  if(ARG_RELATIVE)
    file(GLOB${_possibly_recurse} _sources RELATIVE ${ARG_RELATIVE} ${ARG_UNPARSED_ARGUMENTS})
  else()
    file(GLOB${_possibly_recurse} _sources ${ARG_UNPARSED_ARGUMENTS})
  endif()
  if(ARG_FILTER)
    foreach(s ${_sources})
      if(s MATCHES ${ARG_FILTER})
        list(REMOVE_ITEM _sources ${s})
      endif()
    endforeach()
  endif()
  set(${variable} ${_sources} PARENT_SCOPE)
endfunction()

function(ROOT_GLOB_SOURCES variable)
  ROOT_GLOB_FILES(_sources FILTER "(^|/)G__" ${ARGN})
  set(${variable} ${_sources} PARENT_SCOPE)
endfunction()

function(ROOT_GLOB_HEADERS variable)
  ROOT_GLOB_FILES(_sources FILTER "LinkDef" ${ARGN})
  set(${variable} ${_sources} PARENT_SCOPE)
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_GET_SOURCES( <variable> cwd <sources> ...)
#---------------------------------------------------------------------------------------------------
function(ROOT_GET_SOURCES variable cwd )
  set(sources)
  foreach( fp ${ARGN})
    if( IS_ABSOLUTE ${fp})
      file(GLOB files ${fp})
    else()
      if(root7)
        set(root7glob v7/src/${fp})
      endif()
      file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${cwd}/${fp} ${root7glob})
    endif()
    if(files)
      foreach(s ${files})
        if(fp MATCHES "[*]" AND s MATCHES "(^|/)G__") # Eliminate G__* files
        elseif(s MATCHES "${cwd}/G__")
          set(sources ${fp} ${sources})
        else()
          set(sources ${sources} ${s})
        endif()
      endforeach()
    else()
      if(fp MATCHES "(^|/)G__")
        set(sources ${fp} ${sources})
      else()
        set(sources ${sources} ${fp})
      endif()
    endif()
  endforeach()
  set(${variable} ${sources} PARENT_SCOPE)
endfunction()

#---------------------------------------------------------------------------------------------------
#---REFLEX_GENERATE_DICTIONARY( dictionary headerfiles SELECTION selectionfile OPTIONS opt1 opt2 ...
#                               DEPENDS dependency1 dependency2 ...
#                             )
# if dictionary is a TARGET (e.g., created with add_library), we inherit the INCLUDE_DIRECTORES and
# COMPILE_DEFINITIONS properties
#
#---------------------------------------------------------------------------------------------------
function(REFLEX_GENERATE_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "" "SELECTION" "OPTIONS;DEPENDS" ${ARGN})
  #---Get List of header files---------------
  set(headerfiles)
  foreach(fp ${ARG_UNPARSED_ARGUMENTS})
    file(GLOB files inc/${fp})
    if(files)
      foreach(f ${files})
        if(NOT f MATCHES LinkDef)
          set(headerfiles ${headerfiles} ${f})
        endif()
      endforeach()
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${fp})
      set(headerfiles ${headerfiles} ${CMAKE_CURRENT_SOURCE_DIR}/${fp})
    else()
      set(headerfiles ${headerfiles} ${fp})
    endif()
  endforeach()
  #---Get Selection file------------------------------------
  if(IS_ABSOLUTE ${ARG_SELECTION})
    set(selectionfile ${ARG_SELECTION})
  else()
    set(selectionfile ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SELECTION})
  endif()

  set(gensrcdict ${dictionary}.cxx)

  #---roottest compability---------------------------------
  if(CMAKE_ROOTTEST_NOROOTMAP)
    set(rootmapname )
    set(rootmapopts )
  elseif(DEFINED CMAKE_ROOTTEST_NOROOTMAP)  # Follow the roottest dictionary library naming
    set(rootmapname ${dictionary}.rootmap)
    set(rootmapopts --rootmap=${rootmapname} --rootmap-lib=${libprefix}${dictionary}_dictrflx)
  else()
    set(rootmapname ${dictionary}Dict.rootmap)
    set(rootmapopts --rootmap=${rootmapname} --rootmap-lib=${libprefix}${dictionary}Dict)
  endif()

  set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  foreach(d ${incdirs})
    if(NOT "${d}" MATCHES "^(AFTER|BEFORE|INTERFACE|PRIVATE|PUBLIC|SYSTEM)$")
      list(APPEND include_dirs ${d})
    endif()
  endforeach()

  get_directory_property(defs COMPILE_DEFINITIONS)
  foreach( d ${defs})
   list(APPEND definitions ${d})
  endforeach()

  IF(TARGET ${dictionary})
    LIST(APPEND include_dirs $<TARGET_PROPERTY:${dictionary},INCLUDE_DIRECTORIES>)
    # The COMPILE_DEFINITIONS list might contain empty elements. These are
    # removed with the FILTER generator expression, excluding elements that
    # match the ^$ regexp (only matches empty strings).
    LIST(APPEND definitions "$<FILTER:$<TARGET_PROPERTY:${dictionary},COMPILE_DEFINITIONS>,EXCLUDE,^$>")
  ENDIF()

  add_custom_command(
    OUTPUT ${gensrcdict} ${rootmapname}
    COMMAND ${ROOT_genreflex_CMD}
    ARGS ${headerfiles} -o ${gensrcdict} ${rootmapopts} --select=${selectionfile}
         --gccxmlpath=${GCCXML_home}/bin ${ARG_OPTIONS}
         "-I$<JOIN:$<REMOVE_DUPLICATES:$<FILTER:${include_dirs},EXCLUDE,^$>>,;-I>"
         "$<$<BOOL:$<JOIN:${definitions},>>:-D$<JOIN:${definitions},;-D>>"
    DEPENDS ${headerfiles} ${selectionfile} ${ARG_DEPENDS}

    COMMAND_EXPAND_LISTS
    )
  IF(TARGET ${dictionary})
    target_sources(${dictionary} PRIVATE ${gensrcdict})
  ENDIF()

  #---roottest compability---------------------------------
  if(CMAKE_ROOTTEST_DICT)
    ROOTTEST_TARGETNAME_FROM_FILE(targetname ${dictionary})

    set(targetname "${targetname}-dictgen")

    add_custom_target(${targetname} DEPENDS ${gensrcdict} ${ROOT_LIBRARIES})
  else()
    set(targetname "${dictionary}-dictgen")
    # Creating this target at ALL level enables the possibility to generate dictionaries (genreflex step)
    # well before the dependent libraries of the dictionary are build
    add_custom_target(${targetname} ALL DEPENDS ${gensrcdict})
  endif()

  # FIXME: Do not set gensrcdict variable to the outer scope but use an argument to
  # REFLEX_GENERATE_DICTIONARY passed from the outside. Note this would be a
  # breaking change for roottest and other external users.
  set(gensrcdict ${dictionary}.cxx PARENT_SCOPE)

endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_GET_LIBRARY_OUTPUT_DIR( result_var )
# Returns the path to the .so file or .dll file. In the latter case Windows defines the dll files as
# executables and puts them in the $ROOTSYS/bin folder.
function(ROOT_GET_LIBRARY_OUTPUT_DIR result)
  set(library_output_dir)
  if(MSVC)
    if(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY STREQUAL "")
      set(library_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
      set(library_output_dir ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  else()
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY AND NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY STREQUAL "")
      set(library_output_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    else()
      set(library_output_dir ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  endif()
  SET(${result} "${library_output_dir}" PARENT_SCOPE)
endfunction(ROOT_GET_LIBRARY_OUTPUT_DIR)

#---------------------------------------------------------------------------------------------------
#---ROOT_GET_INSTALL_DIR( result_var )
# Returns the path to the shared libraries installation directory. On Windows the pcms and rootmap
# files must go in the $ROOTSYS/bin folder.
function(ROOT_GET_INSTALL_DIR result)
  set(shared_lib_install_dir)
  if(MSVC)
    set(shared_lib_install_dir ${CMAKE_INSTALL_BINDIR})
  else()
    set(shared_lib_install_dir ${CMAKE_INSTALL_LIBDIR})
  endif()
  SET(${result} "${shared_lib_install_dir}" PARENT_SCOPE)
endfunction(ROOT_GET_INSTALL_DIR)

#---------------------------------------------------------------------------------------------------
#---ROOT_REPLACE_BUILD_INTERFACE( include_dir_var include_dir )
# Update the `include_dir` variable after resolve the BUILD_INTERFACE
function(ROOT_REPLACE_BUILD_INTERFACE include_dir_var include_dir)
  string(REGEX REPLACE "^[$]<BUILD_INTERFACE:(.+)>" "\\1" include_dir ${include_dir})
  # BUILD_INTERFACE might contain space-separated paths. They are split by
  # foreach, leaving a trailing 'include/something>'. Remove the trailing '>'.
  string(REGEX REPLACE ">$" "" include_dir ${include_dir})
  set(${include_dir_var} ${include_dir} PARENT_SCOPE)
endfunction(ROOT_REPLACE_BUILD_INTERFACE)

#---------------------------------------------------------------------------------------------------
#---ROOT_GENERATE_DICTIONARY( dictionary headerfiles NODEPHEADERS ghdr1 ghdr2 ...
#                                                    MODULE module DEPENDENCIES dep1 dep2
#                                                    BUILTINS dep1 dep2
#                                                    STAGE1 LINKDEF linkdef OPTIONS opt1 opt2 ...)
#
# <dictionary> is the dictionary stem; the macro creates (among other files) the dictionary source as
#   <dictionary>.cxx
# <headerfiles> are "as included"; set appropriate INCLUDE_DIRECTORIES property on the directory.
#   The dictionary target depends on these headers. These files must exist.
# <NODEPHEADERS> same as <headerfiles>. If these files are not found (given the target include path)
#   no error is emitted. The dictionary does not depend on these headers.
#---------------------------------------------------------------------------------------------------
function(ROOT_GENERATE_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "STAGE1;MULTIDICT;NOINSTALL;NO_CXXMODULE"
    "MODULE;LINKDEF" "NODEPHEADERS;OPTIONS;DEPENDENCIES;EXTRA_DEPENDENCIES;BUILTINS" ${ARGN})

  # Check if OPTIONS start with a dash.
  if (ARG_OPTIONS)
    foreach(ARG_O ${ARG_OPTIONS})
      if (NOT ARG_O MATCHES "^-*")
        message(FATAL_ERROR "Wrong rootcling option: ${ARG_OPTIONS}")
      endif()
    endforeach()
  endif(ARG_OPTIONS)

  #---roottest compability---------------------------------
  if(CMAKE_ROOTTEST_DICT)
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_CURRENT_BINARY_DIR})
    set(libprefix "")
  endif()

  # list of include directories for dictionary generation
  set(incdirs)

  if((CMAKE_PROJECT_NAME STREQUAL ROOT) AND (TARGET ${ARG_MODULE}))
    set(headerdirs)

    get_target_property(target_incdirs ${ARG_MODULE} INCLUDE_DIRECTORIES)
    if(target_incdirs)
       foreach(dir ${target_incdirs})
          ROOT_REPLACE_BUILD_INTERFACE(dir ${dir})
          # check that dir not a empty dir like $<BUILD_INTERFACE:>
          if(NOT ${dir} MATCHES "^[$]")
            list(APPEND incdirs ${dir})
            string(FIND ${dir} "${CMAKE_SOURCE_DIR}" src_dir_in_dir)
            if(${src_dir_in_dir} EQUAL 0)
              list(APPEND headerdirs ${dir})
            endif()
          endif()
       endforeach()
    endif()

    # Comments from Vassil:
    # FIXME: We prepend ROOTSYS/include because if we have built a module
    # and try to resolve the 'same' header from a different location we will
    # get a redefinition error.
    # We should remove these lines when the fallback include is removed. Then
    # we will need a module.modulemap file per `inc` directory.
    # Comments from Sergey:
    # Remove all source dirs also while they preserved in root dictionaries and
    # ends in the gInterpreter->GetIncludePath()

    list(FILTER incdirs EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}")
    list(FILTER incdirs EXCLUDE REGEX "^${CMAKE_BINARY_DIR}/ginclude")
    list(FILTER incdirs EXCLUDE REGEX "^${CMAKE_BINARY_DIR}/externals")
    list(FILTER incdirs EXCLUDE REGEX "^${CMAKE_BINARY_DIR}/builtins")
    list(INSERT incdirs 0 ${CMAKE_BINARY_DIR}/include)

    # this instruct rootcling do not store such paths in dictionary
    set(excludepaths ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ginclude ${CMAKE_BINARY_DIR}/externals ${CMAKE_BINARY_DIR}/builtins)

    set(headerfiles)
    set(_list_of_header_dependencies)
    foreach(fp ${ARG_UNPARSED_ARGUMENTS})
       if(IS_ABSOLUTE ${fp})
          set(headerFile ${fp})
       else()
          find_file(headerFile ${fp}
                  HINTS ${headerdirs}
                  NO_DEFAULT_PATH
                  NO_SYSTEM_ENVIRONMENT_PATH
                  NO_CMAKE_FIND_ROOT_PATH)
       endif()
       if(NOT headerFile)
          message(FATAL_ERROR "Cannot find header ${fp} to generate dictionary ${dictionary} for. Did you forget to set the INCLUDE_DIRECTORIES property for the current directory?")
       endif()
       list(APPEND headerfiles ${fp})
       list(APPEND _list_of_header_dependencies ${headerFile})
       unset(headerFile CACHE) # find_file, forget headerFile!
    endforeach()

    foreach(fp ${ARG_NODEPHEADERS})
      list(APPEND headerfiles ${fp})
      # no dependency - think "vector" etc.
    endforeach()

    if(NOT (headerfiles OR ARG_LINKDEF))
      message(FATAL_ERROR "No headers nor LinkDef.h supplied / found for dictionary ${dictionary}!")
    endif()

  else()

    ####################### old-style includes/headers generation - starts ##################

    #---Get the list of include directories------------------
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    # rootcling invoked on foo.h should find foo.h in the current source dir,
    # no matter what.
    list(APPEND incdirs ${CMAKE_CURRENT_SOURCE_DIR})

    if(TARGET ${ARG_MODULE})
      get_target_property(target_incdirs ${ARG_MODULE} INCLUDE_DIRECTORIES)
      if(target_incdirs)
        foreach(dir ${target_incdirs})
          ROOT_REPLACE_BUILD_INTERFACE(dir ${dir})
          if(NOT ${dir} MATCHES "^[$]")
            list(APPEND incdirs ${dir})
          endif()
        endforeach()
      endif()
    endif()

    set(headerdirs_dflt)

    if(CMAKE_PROJECT_NAME STREQUAL ROOT)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inc)
        list(APPEND headerdirs_dflt ${CMAKE_CURRENT_SOURCE_DIR}/inc)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc)
        list(APPEND headerdirs_dflt ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc)
      endif()
    endif()

    #---Get the list of header files-------------------------
    # CMake needs dependencies from ${CMAKE_CURRENT_SOURCE_DIR} while rootcling wants
    # header files "as included" (and thus as passed as argument to this CMake function).
    set(headerfiles)
    set(_list_of_header_dependencies)
    foreach(fp ${ARG_UNPARSED_ARGUMENTS})
      if(${fp} MATCHES "[*?]") # Is this header a globbing expression?
        file(GLOB files inc/${fp} ${fp}) # Elements of ${fp} have the complete path.
        foreach(f ${files})
          if(NOT f MATCHES LinkDef) # skip LinkDefs from globbing result
            set(add_inc_as_include On)
            string(REGEX REPLACE "^${CMAKE_CURRENT_SOURCE_DIR}/inc/" "" f_no_inc ${f})
            list(APPEND headerfiles ${f_no_inc})
            list(APPEND _list_of_header_dependencies ${f})
          endif()
        endforeach()
      else()
        if(IS_ABSOLUTE ${fp})
          set(headerFile ${fp})
        else()
          set(incdirs_in_build)
          set(incdirs_in_prefix ${headerdirs_dflt})
          foreach(incdir ${incdirs})
            string(FIND ${incdir} "${CMAKE_SOURCE_DIR}" src_dir_in_dir)
            string(FIND ${incdir} "${CMAKE_BINARY_DIR}" bin_dir_in_dir)
            string(FIND ${incdir} "${CMAKE_CURRENT_BINARY_DIR}" cur_dir_in_dir)
            if(NOT IS_ABSOLUTE ${incdir}
               OR ${src_dir_in_dir} EQUAL 0
               OR ${bin_dir_in_dir} EQUAL 0
               OR ${cur_dir_in_dir} EQUAL 0)
              list(APPEND incdirs_in_build ${incdir})
            else()
              list(APPEND incdirs_in_prefix ${incdir})
            endif()
          endforeach()
          if(incdirs_in_build)
            find_file(headerFile ${fp}
              HINTS ${incdirs_in_build}
              NO_DEFAULT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_FIND_ROOT_PATH)
          endif()
          # Try this even if NOT incdirs_in_prefix: might not need a HINT.
          if(NOT headerFile)
            find_file(headerFile ${fp}
              HINTS ${incdirs_in_prefix}
              NO_DEFAULT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH)
          endif()
        endif()
        if(NOT headerFile)
          message(FATAL_ERROR "Cannot find header ${fp} to generate dictionary ${dictionary} for. Did you forget to set the INCLUDE_DIRECTORIES property for the current directory?")
        endif()
        list(APPEND headerfiles ${fp})
        list(APPEND _list_of_header_dependencies ${headerFile})
        unset(headerFile CACHE) # find_file, forget headerFile!
      endif()
    endforeach()

    foreach(fp ${ARG_NODEPHEADERS})
      list(APPEND headerfiles ${fp})
      # no dependency - think "vector" etc.
    endforeach()

    if(NOT (headerfiles OR ARG_LINKDEF))
      message(FATAL_ERROR "No headers nor LinkDef.h supplied / found for dictionary ${dictionary}!")
    endif()

    if(CMAKE_PROJECT_NAME STREQUAL ROOT)
      list(APPEND incdirs ${CMAKE_BINARY_DIR}/include)
      list(APPEND incdirs ${CMAKE_BINARY_DIR}/etc/cling) # This is for the RuntimeUniverse
      # list(APPEND incdirs ${CMAKE_SOURCE_DIR})
      set(excludepaths ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inc)
      list(APPEND incdirs ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    endif()

    foreach(dep ${ARG_DEPENDENCIES})
      if(TARGET ${dep})
        get_target_property(dep_include_dirs ${dep} INTERFACE_INCLUDE_DIRECTORIES)
        if (NOT dep_include_dirs)
          get_target_property(dep_include_dirs ${dep} INCLUDE_DIRECTORIES)
        endif()
        if (dep_include_dirs)
          foreach(d ${dep_include_dirs})
            list(APPEND incdirs ${d})
          endforeach()
        endif()
      endif()
    endforeach()

    ####################### old-style includes/headers generation - end  ##################
  endif()

  #---Get the list of definitions---------------------------
  get_directory_property(defs COMPILE_DEFINITIONS)
  foreach( d ${defs})
   if((NOT d MATCHES "=") AND (NOT d MATCHES "^[$]<.*>$")) # avoid generator expressions
     set(definitions ${definitions} -D${d})
   endif()
  endforeach()
  #---Get LinkDef.h file------------------------------------
  foreach( f ${ARG_LINKDEF})
    if( IS_ABSOLUTE ${f})
      set(_linkdef ${_linkdef} ${f})
    else()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inc/${f})
        set(_linkdef ${_linkdef} ${CMAKE_CURRENT_SOURCE_DIR}/inc/${f})
      else()
        set(_linkdef ${_linkdef} ${CMAKE_CURRENT_SOURCE_DIR}/${f})
      endif()
    endif()
  endforeach()

  #---Build the names for library, pcm and rootmap file ----
  set(library_target_name)
  if(dictionary MATCHES "^G__")
    string(REGEX REPLACE "^G__(.*)" "\\1"  library_target_name ${dictionary})
    if (ARG_MULTIDICT)
      string(REGEX REPLACE "(.*)32$" "\\1"  library_target_name ${library_target_name})
    endif (ARG_MULTIDICT)
  else()
    get_filename_component(library_target_name ${dictionary} NAME_WE)
  endif()
  if (ARG_MODULE)
    if (NOT ${ARG_MODULE} STREQUAL ${library_target_name})
#      message(AUTHOR_WARNING "The MODULE argument ${ARG_MODULE} and the deduced library name "
#        "${library_target_name} mismatch. Deduction stem: ${dictionary}.")
      set(library_target_name ${ARG_MODULE})
    endif()
  endif(ARG_MODULE)

  #---Set the library output directory-----------------------
  ROOT_GET_LIBRARY_OUTPUT_DIR(library_output_dir)
  set(runtime_cxxmodule_dependencies )
  set(cpp_module)
  set(library_name ${libprefix}${library_target_name}${libsuffix})
  set(newargs -s ${library_output_dir}/${library_name})
  set(rootmap_name ${library_output_dir}/${libprefix}${library_target_name}.rootmap)
  set(pcm_name ${library_output_dir}/${libprefix}${library_target_name}_rdict.pcm)
  if(ARG_MODULE)
    if(ARG_MULTIDICT)
      set(newargs ${newargs} -multiDict)
      set(pcm_name ${library_output_dir}/${libprefix}${library_target_name}_${dictionary}_rdict.pcm)
      set(rootmap_name ${library_output_dir}/${libprefix}${library_target_name}32.rootmap)
    else()
      set(cpp_module ${library_target_name})
    endif(ARG_MULTIDICT)

    if(runtime_cxxmodules)
      # If we specify NO_CXXMODULE we should be able to still install the produced _rdict.pcm file.
      if(NOT ARG_NO_CXXMODULE)
        set(pcm_name)
      endif()
      if(cpp_module)
        set(cpp_module_file ${library_output_dir}/${cpp_module}.pcm)
        # The module depends on its modulemap file.
        if (cpp_module_file AND CMAKE_PROJECT_NAME STREQUAL ROOT)
		set (runtime_cxxmodule_dependencies copymodulemap "${CMAKE_BINARY_DIR}/include/ROOT.modulemap")
        endif()
      endif(cpp_module)
    endif()
  endif()

  # modules.idx deps
  get_property(local_modules_idx_deps GLOBAL PROPERTY modules_idx_deps_property)
  get_property(local_no_cxxmodules GLOBAL PROPERTY no_cxxmodules_property)
  if (ARG_NO_CXXMODULE)
    list(APPEND local_no_cxxmodules ${cpp_module})
    set_property(GLOBAL PROPERTY no_cxxmodules_property "${local_no_cxxmodules}")
    unset(cpp_module)
    unset(cpp_module_file)
  else()
    list(APPEND local_modules_idx_deps ${cpp_module})
    set_property(GLOBAL PROPERTY modules_idx_deps_property "${local_modules_idx_deps}")
  endif(ARG_NO_CXXMODULE)



  if(CMAKE_ROOTTEST_NOROOTMAP OR cpp_module_file)
    set(rootmap_name)
    set(rootmapargs)
  else()
    set(rootmapargs -rml ${library_name} -rmf ${rootmap_name})
  endif()

  #---Get the library and module dependencies-----------------
  if(ARG_DEPENDENCIES)
    foreach(dep ${ARG_DEPENDENCIES})
      if(NOT TARGET G__${dep})
        # This is a library that doesn't come with dictionary/pcm
        continue()
      endif()

      set(dependent_pcm ${libprefix}${dep}_rdict.pcm)
      if (runtime_cxxmodules AND NOT dep IN_LIST local_no_cxxmodules)
        set(dependent_pcm ${dep}.pcm)
        if(TARGET ${dep})
          get_target_property(_dep_pcm_filename ${dep} ROOT_PCM_FILENAME)
          if(_dep_pcm_filename)
            list(APPEND pcm_dependencies ${_dep_pcm_filename})
          endif()
        endif()
      endif()
      set(newargs ${newargs} -m  ${dependent_pcm})
    endforeach()
  endif()

  if(cpp_module_file)
    set(newargs -cxxmodule ${newargs})
  endif()

  #---what rootcling command to use--------------------------
  if(ARG_STAGE1)
    set(command $<TARGET_FILE:rootcling_stage1>)
    set(ROOTCINTDEP rconfigure)
    set(pcm_name)
  else()
    if(CMAKE_PROJECT_NAME STREQUAL ROOT)
      if(MSVC AND CMAKE_ROOTTEST_DICT)
        set(command ${CMAKE_COMMAND} -E env "ROOTIGNOREPREFIX=1" ${CMAKE_BINARY_DIR}/bin/rootcling.exe -rootbuild)
      else()
        set(command ${CMAKE_COMMAND} -E env "ROOTIGNOREPREFIX=1" $<TARGET_FILE:rootcling> -rootbuild)
        # Modules need RConfigure.h copied into include/.
        set(ROOTCINTDEP rootcling rconfigure)
      endif()
    elseif(TARGET ROOT::rootcling)
      set(command $<TARGET_FILE:ROOT::rootcling>)
    else()
      set(command rootcling)
    endif()
  endif()

  #---build the path exclusion switches----------------------
  set(excludepathsargs "")
  foreach(excludepath ${excludepaths})
    set(excludepathsargs ${excludepathsargs} -excludePath ${excludepath})
  endforeach()

  #---build the implicit dependencies arguments
  # NOTE: only the Makefile generator respects this!
  foreach(_dep ${_linkdef} ${_list_of_header_dependencies})
    list(APPEND _implicitdeps CXX ${_dep})
  endforeach()

  if(ARG_MODULE)
    set(MODULE_LIB_DEPENDENCY ${ARG_DEPENDENCIES})

    # get target properties added after call to ROOT_GENERATE_DICTIONARY()
    if(TARGET ${ARG_MODULE})
      # NOTE that module_sysincs is already part of ${module_sysincs}. But -isystem "wins",
      # and list exclusion for generator expressions is too complex.
      set(module_incs $<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${ARG_MODULE},INCLUDE_DIRECTORIES>>)
      set(module_sysincs $<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${ARG_MODULE},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>>)
      # The COMPILE_DEFINITIONS list might contain empty elements. These are
      # removed with the FILTER generator expression, excluding elements that
      # match the ^$ regexp (only matches empty strings).
      set(module_defs "$<FILTER:$<TARGET_PROPERTY:${ARG_MODULE},COMPILE_DEFINITIONS>,EXCLUDE,^$>")
    endif()
  endif()

  # provide list of includes for dictionary
  set(includedirs)
  if(incdirs)
     list(REMOVE_DUPLICATES incdirs)
     foreach(dir ${incdirs})
        if (NOT ${dir} MATCHES "^\\$<INSTALL_INTERFACE:")
          list(APPEND includedirs -I${dir})
        endif()
     endforeach()
  endif()

  set(compIncPaths)
  foreach(implinc IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    list(APPEND compIncPaths "-compilerI${implinc}")
  endforeach()

  if(cpp_module_file AND TARGET ${ARG_MODULE})
    set_target_properties(${ARG_MODULE} PROPERTIES
      ROOT_PCM_FILENAME "${cpp_module_file}")
  endif()

  #---call rootcint------------------------------------------
  add_custom_command(OUTPUT ${dictionary}.cxx ${pcm_name} ${rootmap_name} ${cpp_module_file}
                     COMMAND ${command} -v2 -f  ${dictionary}.cxx ${newargs} ${excludepathsargs} ${rootmapargs}
                                        ${ARG_OPTIONS}
                                        ${definitions} "$<$<BOOL:${module_defs}>:-D$<JOIN:${module_defs},;-D>>"
                                        ${compIncPaths}
                                        "$<$<BOOL:${module_sysincs}>:-isystem;$<JOIN:${module_sysincs},;-isystem;>>"
                                        ${includedirs} "$<$<BOOL:${module_incs}>:-I$<JOIN:${module_incs},;-I>>"
                                        ${headerfiles} ${_linkdef}
                     IMPLICIT_DEPENDS ${_implicitdeps}
                     DEPENDS ${_list_of_header_dependencies} ${_linkdef} ${ROOTCINTDEP}
                             ${pcm_dependencies}
                             ${MODULE_LIB_DEPENDENCY} ${ARG_EXTRA_DEPENDENCIES}
                             ${runtime_cxxmodule_dependencies}
                     COMMAND_EXPAND_LISTS)

  # If we are adding to an existing target and it's not the dictionary itself,
  # we make an object library and add its output object file as source to the target.
  # This works around bug https://cmake.org/Bug/view.php?id=14633 in CMake by keeping
  # the generated source at the same scope level as its owning target, something that
  # would not happen if we used target_sources() directly with the dictionary source.
  if(TARGET "${ARG_MODULE}" AND NOT "${ARG_MODULE}" STREQUAL "${dictionary}")
    add_library(${dictionary} OBJECT ${dictionary}.cxx)
    set_target_properties(${dictionary} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    target_link_libraries(${ARG_MODULE} PRIVATE ${dictionary})

    target_compile_options(${dictionary} PRIVATE
      $<TARGET_PROPERTY:${ARG_MODULE},COMPILE_OPTIONS>)

    target_compile_definitions(${dictionary} PRIVATE
      ${definitions} $<TARGET_PROPERTY:${ARG_MODULE},COMPILE_DEFINITIONS>)

    target_compile_features(${dictionary} PRIVATE
      $<TARGET_PROPERTY:${ARG_MODULE},COMPILE_FEATURES>)

    target_include_directories(${dictionary} PRIVATE ${incdirs} $<TARGET_PROPERTY:${ARG_MODULE},INCLUDE_DIRECTORIES>)

    # Above we are copying all include directories of the module, irrespective of whether they are system includes.
    # CMake copies them as -I even when they should be -isystem.
    # We can fix this for INTERFACE includes by also copying the respective property.
    # For PRIVATE includes this doesn't work. In that case, one needs to link both the library as well as the dictionary explicitly:
    #   target_link_libraries(MODULE PRIVATE dependency)
    #   target_link_libraries(G__MODULE PRIVATE dependency)
    set_property(TARGET ${dictionary} APPEND PROPERTY
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${ARG_MODULE},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
  else()
    get_filename_component(dictionary_name ${dictionary} NAME)
    add_custom_target(${dictionary_name} DEPENDS ${dictionary}.cxx ${pcm_name} ${rootmap_name} ${cpp_module_file})
  endif()

  if(PROJECT_NAME STREQUAL "ROOT")
    set_property(GLOBAL APPEND PROPERTY ROOT_PCH_DEPENDENCIES ${dictionary})
    set_property(GLOBAL APPEND PROPERTY ROOT_PCH_DICTIONARIES ${CMAKE_CURRENT_BINARY_DIR}/${dictionary}.cxx)
  endif()

  if(ARG_MULTIDICT)
    if(NOT TARGET "G__${ARG_MODULE}")
      message(FATAL_ERROR
        " Target G__${ARG_MODULE} not found!\n"
        " Please create target G__${ARG_MODULE} before using MULTIDICT.")
    endif()
    add_dependencies(G__${ARG_MODULE} ${dictionary})
  endif()

  if(NOT ARG_NOINSTALL AND NOT CMAKE_ROOTTEST_DICT AND DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    ROOT_GET_INSTALL_DIR(shared_lib_install_dir)
    # Install the C++ module if we generated one.
    if (cpp_module_file)
      install(FILES ${cpp_module_file}
                    DESTINATION ${shared_lib_install_dir} COMPONENT libraries)
    endif()

    if(ARG_STAGE1)
      install(FILES ${rootmap_name}
                    DESTINATION ${shared_lib_install_dir} COMPONENT libraries)
    else()
      install(FILES ${pcm_name} ${rootmap_name}
                    DESTINATION ${shared_lib_install_dir} COMPONENT libraries)
    endif()
  endif()

  if(ARG_BUILTINS)
    foreach(arg1 ${ARG_BUILTINS})
      if(TARGET ${${arg1}_TARGET})
        add_dependencies(${dictionary} ${${arg1}_TARGET})
      endif()
    endforeach()
  endif()

  # FIXME: Support mulptiple dictionaries. In some cases (libSMatrix and
  # libGenVector) we have to have two or more dictionaries (eg. for math,
  # we need the two for double vs Double32_t template specializations).
  # In some other cases, eg. libTreePlayer.so we add in a separate dictionary
  # files which for some reason (temporarily?) cannot be put in the PCH. Eg.
  # all rest of the first dict is in the PCH but this file is not and it
  # cannot be present in the original dictionary.
  if(cpp_module)
    ROOT_CXXMODULES_APPEND_TO_MODULEMAP("${cpp_module}" "${headerfiles}")
  endif()
endfunction(ROOT_GENERATE_DICTIONARY)

#---------------------------------------------------------------------------------------------------
#---ROOT_CXXMODULES_APPEND_TO_MODULEMAP( library library_headers )
#---------------------------------------------------------------------------------------------------
function (ROOT_CXXMODULES_APPEND_TO_MODULEMAP library library_headers)
  ROOT_FIND_DIRS_WITH_HEADERS(dirs)

  set(found_headers "")
  set(dir_headers "")
  foreach(d ${dirs})
    ROOT_GLOB_FILES(dir_headers
                    RECURSE
                    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${d}
                    FILTER "LinkDef" ${d}/*)
    list(APPEND found_headers "${dir_headers}")
  endforeach()

  set(excluded_headers RConfig.h RVersion.h core/foundation/inc/ROOT/RVersion.hxx RtypesImp.h
                        RtypesCore.h TClassEdit.h
                        TIsAProxy.h TVirtualIsAProxy.h
                        DllImport.h ESTLType.h Varargs.h
                        ThreadLocalStorage.h
                        TBranchProxyTemplate.h TGLWSIncludes.h
                        snprintf.h strlcpy.h)

   # Deprecated header files.
  set (excluded_headers "${excluded_headers}")

  set(modulemap_entry "module \"${library}\" {")

  # Add a `use` directive to Core/Thread to signal that they use some
  # split out submodules and we pass the rootcling integrity check.
  if ("${library}" STREQUAL Core)
    set (modulemap_entry "${modulemap_entry}\n  use ROOT_Foundation_Stage1_NoRTTI\n")
    set (modulemap_entry "${modulemap_entry}\n  use ROOT_Foundation_C\n")
  elseif ("${library}" STREQUAL Thread)
    set (modulemap_entry "${modulemap_entry}\n  use ROOT_Foundation_C\n")
  endif()

  # For modules GCocoa and GQuartz we need objc and cplusplus context.
  if (NOT ${library} MATCHES "GCocoa")
    set (modulemap_entry "${modulemap_entry}\n  requires cplusplus\n")
  endif()
  if (library_headers)
    set(found_headers ${library_headers})
  endif()
  foreach(header ${found_headers})
    set(textual_header "")
    if (${header} MATCHES ".*\\.icc$")
      set(textual_header "textual ")
    endif()
    # Check if header is in included header list
    set(is_excluded NO)
    foreach(excluded_header ${excluded_headers})
      if(${header} MATCHES ${excluded_header})
        set(is_excluded YES)
        break()
      endif()
    endforeach()
    if(NOT is_excluded)
      set(modulemap_entry "${modulemap_entry}  module \"${header}\" { ${textual_header}header \"${header}\" export * }\n")
    endif()
  endforeach()
  set(modulemap_entry "${modulemap_entry}  link \"${libprefix}${library}${libsuffix}\"\n")
  set(modulemap_entry "${modulemap_entry}  export *\n}\n\n")
  # Non ROOT projects need a modulemap generated for them in the current
  # directory. The same happens with test dictionaries in ROOT which are not
  # exposed via the main modulemap. This is exposed by setting the
  # ROOT_CXXMODULES_WRITE_TO_CURRENT_DIR.
  if (NOT "${CMAKE_PROJECT_NAME}" STREQUAL ROOT OR ROOT_CXXMODULES_WRITE_TO_CURRENT_DIR)
    set(modulemap_output_file "${CMAKE_CURRENT_BINARY_DIR}/module.modulemap")

    # It's possible that multiple modulemaps are needed in the current
    # directory and we need to merge them. As we don't want to have multiple
    # modules in the same moduluemap when rerunning CMake, we do a quick
    # check if the current module is already in the modulemap (in which case
    # we know we rerun CMake at the moment and start writing a new modulemap
    # instead of appending new modules).

    # The string we use to identify if the current module is already in the
    # modulemap.
    set(modulemap_needle "module \"${library}\"")
    # Check if the needle is in the modulemap. If the file doesn't exist
    # we just pretend we didn't found the string in the modulemap.
    set(match_result -1)
    if (EXISTS "${modulemap_output_file}")
      file(READ "${modulemap_output_file}" existing_contents)
      string(FIND "${existing_contents}" "${modulemap_needle}" match_result)
    endif()
    # Append our new module to the existing modulemap containing other modules.
    if(${match_result} EQUAL -1)
      file(APPEND "${modulemap_output_file}" "${modulemap_entry}")
    else()
      file(WRITE "${modulemap_output_file}" "${modulemap_entry}")
    endif()

    # Sanity check that the string we're looking for is actually in the content
    # we're writing to this file.
    string(FIND "${modulemap_entry}" "${modulemap_needle}" match_result)
    if(${match_result} EQUAL -1)
      message(AUTHOR_WARNING "Couldn't find module declaration in modulemap file."
                             "This would break the modulemap generation when "
                             " rerunning CMake. Module needle was "
                             "'${modulemap_needle}' and the content was '${modulemap_entry}'")
    endif()

  else()
    set_property(GLOBAL APPEND PROPERTY ROOT_CXXMODULES_EXTRA_MODULEMAP_CONTENT ${modulemap_entry})
  endif()
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_LINKER_LIBRARY( <name> source1 source2 ...[TYPE STATIC|SHARED] [DLLEXPORT]
#                        [NOINSTALL]
#                        LIBRARIES library1 library2 ... # PRIVATE link dependencies
#                        DEPENDENCIES dep1 dep2          # PUBLIC link dependencies
#                        BUILTINS dep1 dep2              # dependencies to builtins)
#---------------------------------------------------------------------------------------------------
function(ROOT_LINKER_LIBRARY library)
  CMAKE_PARSE_ARGUMENTS(ARG "DLLEXPORT;CMAKENOEXPORT;TEST;NOINSTALL" "TYPE" "LIBRARIES;DEPENDENCIES;BUILTINS"  ${ARGN})
  ROOT_GET_SOURCES(lib_srcs src ${ARG_UNPARSED_ARGUMENTS})
  if(NOT ARG_TYPE)
    set(ARG_TYPE SHARED)
  endif()
  if(ARG_TEST) # we are building a test, so add EXCLUDE_FROM_ALL
    set(_all EXCLUDE_FROM_ALL)
  endif()
  if(TARGET ${library})
    message(FATAL_ERROR "Target ${library} already exists.")
  endif()
  if(WIN32 AND ARG_TYPE STREQUAL SHARED AND NOT ARG_DLLEXPORT)
    #---create a shared library with the .def file------------------------
    add_library(${library} ${_all} SHARED ${lib_srcs})
  else()
    add_library( ${library} ${_all} ${ARG_TYPE} ${lib_srcs})
    if(ARG_TYPE STREQUAL SHARED)
      set_target_properties(${library} PROPERTIES  ${ROOT_LIBRARY_PROPERTIES} )
    endif()
  endif()
  set_target_properties(${library} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

  if(DEFINED CMAKE_CXX_STANDARD)
    target_compile_features(${library} INTERFACE cxx_std_${CMAKE_CXX_STANDARD})
  endif()

  # Add dependencies passed via LIBRARIES or DEPENDENCIES argument:
  target_link_libraries(${library} PUBLIC ${ARG_DEPENDENCIES})
  target_link_libraries(${library} PRIVATE ${ARG_LIBRARIES})

  if(TARGET G__${library})
    add_dependencies(${library} G__${library})
  endif()
  set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS ${library})

  ROOT_ADD_INCLUDE_DIRECTORIES(${library})

  if(PROJECT_NAME STREQUAL "ROOT")
    add_dependencies(${library} move_headers)
    if(NOT TARGET ROOT::${library})
      add_library(ROOT::${library} ALIAS ${library})
    endif()
  endif()

  set_target_properties(${library} PROPERTIES
      PREFIX ${libprefix}
      IMPORT_PREFIX ${libprefix} # affects the .lib import library (MSVC)
  )
  target_include_directories(${library} INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  # Do not add -Dname_EXPORTS to the command-line when building files in this
  # target. Doing so is actively harmful for the modules build because it
  # creates extra module variants, and not useful because we don't use these
  # macros.
  set_target_properties(${library} PROPERTIES DEFINE_SYMBOL "")
  if(ARG_BUILTINS)
    foreach(arg1 ${ARG_BUILTINS})
      if(${arg1}_TARGET)
        add_dependencies(${library} ${${arg1}_TARGET})
      endif()
    endforeach()
  endif()

  #----Installation details-------------------------------------------------------
  if(NOT ARG_TEST AND NOT ARG_NOINSTALL AND CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    if(NOT MSVC)
      ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(${library} ${CMAKE_INSTALL_LIBDIR})
    endif()
    if(ARG_CMAKENOEXPORT)
      install(TARGETS ${library} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
                                 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
                                 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
    else()
      install(TARGETS ${library} EXPORT ${CMAKE_PROJECT_NAME}Exports
                                 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
                                 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
                                 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
    endif()
    if(WIN32 AND ARG_TYPE STREQUAL SHARED)
      install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${library}.pdb
                    CONFIGURATIONS Debug RelWithDebInfo
                    DESTINATION ${CMAKE_INSTALL_BINDIR}
                    COMPONENT libraries)
    endif()
  else()
    # If the target is not installed, it doesn't make sense to build it with the INSTALL_RPATH
    set_property(TARGET ${library} PROPERTY BUILD_WITH_INSTALL_RPATH OFF)
  endif()
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_ADD_INCLUDE_DIRECTORIES(library)
#---------------------------------------------------------------------------------------------------
function(ROOT_ADD_INCLUDE_DIRECTORIES library)

  if(PROJECT_NAME STREQUAL "ROOT")

      if(TARGET Core)
        get_target_property(lib_incdirs Core INCLUDE_DIRECTORIES)
        if(lib_incdirs)
          foreach(dir ${lib_incdirs})
            ROOT_REPLACE_BUILD_INTERFACE(dir ${dir})
            if(NOT ${dir} MATCHES "^[$]")
              target_include_directories(${library} BEFORE PRIVATE ${dir})
            endif()
          endforeach()
        endif()
      endif()

      if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/res)
        target_include_directories(${library} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/res)
      endif()

      if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc)
        target_include_directories(${library} BEFORE PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)
      endif()

      if(root7 AND (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc))
        target_include_directories(${library} BEFORE PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/v7/inc>)
      endif()

  endif()

endfunction(ROOT_ADD_INCLUDE_DIRECTORIES)

#---------------------------------------------------------------------------------------------------
#---ROOT_OBJECT_LIBRARY( <name> source1 source2 ... BUILTINS dep1 dep2 ...)
#---------------------------------------------------------------------------------------------------
function(ROOT_OBJECT_LIBRARY library)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "BUILTINS"  ${ARGN})
  ROOT_GET_SOURCES(lib_srcs src ${ARG_UNPARSED_ARGUMENTS})
  add_library( ${library} OBJECT ${lib_srcs})
  if(lib_srcs MATCHES "(^|/)(G__[^.]*)[.]cxx.*")
     add_dependencies(${library} ${CMAKE_MATCH_2})
  endif()
  add_dependencies(${library} move_headers)

  ROOT_ADD_INCLUDE_DIRECTORIES(${library})

  #--- Only for building shared libraries
  set_property(TARGET ${library} PROPERTY POSITION_INDEPENDENT_CODE 1)
  # Do not add -Dname_EXPORTS to the command-line when building files in this
  # target. Doing so is actively harmful for the modules build because it
  # creates extra module variants, and not useful because we don't use these
  # macros.
  set_target_properties(${library} PROPERTIES DEFINE_SYMBOL "")
  set_target_properties(${library} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

  if(ARG_BUILTINS)
    foreach(arg1 ${ARG_BUILTINS})
      if(${arg1}_TARGET)
        add_dependencies(${library} ${${arg1}_TARGET})
      endif()
    endforeach()
  endif()

  #--- Fill the property OBJECTS with all the object files
  #    This is needed because the generator expression $<TARGET_OBJECTS:target>
  #    does not get expanded when used in custom command dependencies
  get_target_property(sources ${library} SOURCES)
  foreach(s ${sources})
    if(CMAKE_GENERATOR MATCHES Xcode)
      get_filename_component(name ${s} NAME_WE)
      set(obj ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.build/${CMAKE_CFG_INTDIR}/${library}.build/Objects-normal/x86_64/${name}${CMAKE_CXX_OUTPUT_EXTENSION})
    else()
      if(IS_ABSOLUTE ${s})
        string(FIND ${s} "${CMAKE_CURRENT_SOURCE_DIR}" src_dir_in_src)
        string(FIND ${s} "${CMAKE_CURRENT_BINARY_DIR}" bin_dir_in_src)
        if(${src_dir_in_src} EQUAL 0)
          string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${library}.dir src ${s})
        elseif(${bin_dir_in_src} EQUAL 0)
          string(REPLACE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${library}.dir src ${s})
        else()
          #message(WARNING "Unknown location of source ${s} for object library ${library}")
        endif()
      else()
        set(src ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${library}.dir/${s})
      endif()
      set(obj ${src}${CMAKE_CXX_OUTPUT_EXTENSION})
    endif()
    set_property(TARGET ${library} APPEND PROPERTY OBJECTS ${obj})
  endforeach()
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_MODULE_LIBRARY(<library> source1 source2 ... LIBRARIES library1 library2 ...)
#---------------------------------------------------------------------------------------------------
function(ROOT_MODULE_LIBRARY library)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LIBRARIES" ${ARGN})
  ROOT_GET_SOURCES(lib_srcs src ${ARG_UNPARSED_ARGUMENTS})
  add_library(${library} SHARED ${lib_srcs})
  add_dependencies(${library} move_headers)
  set_target_properties(${library}  PROPERTIES ${ROOT_LIBRARY_PROPERTIES})
  # Do not add -Dname_EXPORTS to the command-line when building files in this
  # target. Doing so is actively harmful for the modules build because it
  # creates extra module variants, and not useful because we don't use these
  # macros.
  set_target_properties(${library} PROPERTIES DEFINE_SYMBOL "")

  ROOT_ADD_INCLUDE_DIRECTORIES(${library})

  target_link_libraries(${library} PUBLIC ${ARG_LIBRARIES})
  #----Installation details-------------------------------------------------------
  install(TARGETS ${library} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
                             LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
                             ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_GENERATE_ROOTMAP( library LINKDEF linkdef LIBRRARY lib DEPENDENCIES lib1 lib2 )
#---------------------------------------------------------------------------------------------------
function(ROOT_GENERATE_ROOTMAP library)
  return()   #--- No needed anymore
  CMAKE_PARSE_ARGUMENTS(ARG "" "LIBRARY" "LINKDEF;DEPENDENCIES" ${ARGN})
  get_filename_component(libname ${library} NAME_WE)
  get_filename_component(path ${library} PATH)

  #---Set the library output directory-----------------------
  ROOT_GET_LIBRARY_OUTPUT_DIR(library_output_dir)

  set(outfile ${library_output_dir}/${libprefix}${libname}.rootmap)
  foreach( f ${ARG_LINKDEF})
    if( IS_ABSOLUTE ${f})
      set(_linkdef ${_linkdef} ${f})
    else()
      set(_linkdef ${_linkdef} ${CMAKE_CURRENT_SOURCE_DIR}/inc/${f})
    endif()
  endforeach()
  foreach(d ${ARG_DEPENDENCIES})
    get_filename_component(_ext ${d} EXT)
    if(_ext)
      set(_dependencies ${_dependencies} ${d})
    else()
      set(_dependencies ${_dependencies} ${libprefix}${d}${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
  endforeach()
  if(ARG_LIBRARY)
    set(_library ${ARG_LIBRARY})
  else()
    set(_library ${libprefix}${library}${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  #---Build the rootmap file--------------------------------------
  #add_custom_command(OUTPUT ${outfile}
  #                   COMMAND ${rlibmap_cmd} -o ${outfile} -l ${_library} -d ${_dependencies} -c ${_linkdef}
  #                   DEPENDS ${_linkdef} ${rlibmap_cmd} )
  add_custom_target( ${libprefix}${library}.rootmap ALL DEPENDS  ${outfile})
  set_target_properties(${libprefix}${library}.rootmap PROPERTIES FOLDER RootMaps )
  #---Install the rootmap file------------------------------------
  install(FILES ${outfile} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
endfunction(ROOT_GENERATE_ROOTMAP)

#---------------------------------------------------------------------------------------------------
#---ROOT_FIND_DIRS_WITH_HEADERS([dir1 dir2 ...] OPTIONS [options])
#---------------------------------------------------------------------------------------------------
function(ROOT_FIND_DIRS_WITH_HEADERS result_dirs)
  set(dirs "")
  if(ARGN)
    set(dirs ${ARGN})
  else()
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc)
      set(dirs inc/)
    endif()
    if(root7 AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc)
      set(dirs ${dirs} v7/inc/)
    endif()
  endif()
  set(${result_dirs} ${dirs} PARENT_SCOPE)
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_INSTALL_HEADERS([dir1 dir2 ...] [FILTER <regex>])
# Glob for headers in the folder where this target is defined, and install them in
# <buildDir>/include
#---------------------------------------------------------------------------------------------------
function(ROOT_INSTALL_HEADERS)
  CMAKE_PARSE_ARGUMENTS(ARG "OPTIONS" "" "FILTER" ${ARGN})
  if (${ARG_OPTIONS})
    message(FATAL_ERROR "ROOT_INSTALL_HEADERS no longer supports the OPTIONS argument. Rewrite using the FILTER argument.")
  endif()
  ROOT_FIND_DIRS_WITH_HEADERS(dirs ${ARG_UNPARSED_ARGUMENTS})
  set (filter "LinkDef")
  set (options REGEX "LinkDef" EXCLUDE)
  foreach (f ${ARG_FILTER})
    set (filter "${filter}|${f}")
    set (options ${options} REGEX "${f}" EXCLUDE)
  endforeach()
  set (filter "(${filter})")
  set(include_files "")
  foreach(d ${dirs})
    install(DIRECTORY ${d} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                           COMPONENT headers
                           ${options})
    string(REGEX REPLACE "(.*)/$" "\\1" d ${d})
    ROOT_GLOB_FILES(globbed_files
      RECURSE
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      FILTER ${filter}
      ${d}/*.h ${d}/*.hxx ${d}/*.icc )
    list(APPEND include_files ${globbed_files})
  endforeach()

  string(REPLACE ${CMAKE_SOURCE_DIR} "" target_name ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE / _ target_name "copy_header_${target_name}")
  string(REGEX REPLACE "_$" "" target_name ${target_name})

  # Register the files to be copied for each target directory (e.g. include/ include/ROOT include/v7/inc/ ...)
  list(REMOVE_DUPLICATES include_files)
  list(TRANSFORM include_files REPLACE "(.*)/[^/]*" "\\1/" OUTPUT_VARIABLE subdirs)
  list(REMOVE_DUPLICATES subdirs)
  foreach(subdir ${subdirs})
    set(input_files ${include_files})
    list(FILTER input_files INCLUDE REGEX "^${subdir}[^/]*$")
    set(output_files ${input_files})

    string(REGEX REPLACE ".*/*inc/" "" destination ${subdir})

    list(TRANSFORM input_files  PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    list(TRANSFORM output_files REPLACE ".*/" "${CMAKE_BINARY_DIR}/include/${destination}")

    set(destination destination_${destination})

    set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_COPY_LISTS ${destination})
    set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_INPUT_${destination} ${input_files})
    set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_OUTPUT_${destination} ${output_files})
  endforeach()
endfunction()

#---------------------------------------------------------------------------------------------------
#--- ROOT_CREATE_HEADER_COPY_TARGETS
#    Creates a target to copy all headers that have been registered for copy in ROOT_INSTALL_HEADERS
#---------------------------------------------------------------------------------------------------
macro(ROOT_CREATE_HEADER_COPY_TARGETS)
  get_property(HEADER_COPY_LISTS GLOBAL PROPERTY ROOT_HEADER_COPY_LISTS)
  list(REMOVE_DUPLICATES HEADER_COPY_LISTS)
  foreach(copy_list ${HEADER_COPY_LISTS})
    get_property(inputs  GLOBAL PROPERTY ROOT_HEADER_INPUT_${copy_list})
    get_property(outputs GLOBAL PROPERTY ROOT_HEADER_OUTPUT_${copy_list})

    string(REPLACE "destination_" "${CMAKE_BINARY_DIR}/include/" destination ${copy_list})

    list(LENGTH inputs LIST_LENGTH)
    # Windows doesn't support long command lines, so split them in packs:
    # foreach(.. RANGE start stop) is inclusive for both start and stop, so we
    # need to decrement the LIST_LENGTH to get the desired logic.
    math(EXPR LIST_LENGTH_MINUS_ONE "${LIST_LENGTH}-1")
    foreach(range_start RANGE 0 ${LIST_LENGTH_MINUS_ONE} 100)
      list(SUBLIST outputs ${range_start} 100 sub_outputs)
      list(SUBLIST inputs ${range_start} 100 sub_inputs)
      list(LENGTH sub_outputs SUB_LENGTH)
      if(NOT SUB_LENGTH EQUAL 0)
        add_custom_command(OUTPUT ${sub_outputs}
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sub_inputs} ${destination}
          COMMENT "Copy headers for ${destination} ${range_start}"
          DEPENDS ${sub_inputs})
      endif()
    endforeach()
    file(MAKE_DIRECTORY ${destination})
    set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_TARGETS ${outputs})
  endforeach()
endmacro()

#---------------------------------------------------------------------------------------------------
#---ROOT_STANDARD_LIBRARY_PACKAGE(libname
#                                 [NO_INSTALL_HEADERS]         : don't install headers for this package
#                                 [STAGE1]                     : use rootcling_stage1 for generating
#                                 HEADERS header1 header2      : relative header path as #included; pass -I to find them. If not specified, globbing for *.h is used
#                                 NODEPHEADERS header1 header2 : like HEADERS, but no dependency is generated
#                                 [NO_HEADERS]                 : don't glob to fill HEADERS variable
#                                 SOURCES source1 source2      : if not specified, globbing for *.cxx is used
#                                 [NO_SOURCES]                 : don't glob to fill SOURCES variable
#                                 [OBJECT_LIBRARY]             : use ROOT_OBJECT_LIBRARY to generate object files
#                                                                and then use those for linking.
#                                 LIBRARIES lib1 lib2          : private arguments for target_link_library()
#                                 DEPENDENCIES lib1 lib2       : PUBLIC arguments for target_link_library() such as Core, MathCore
#                                 BUILTINS builtin1 builtin2   : builtins like AFTERIMAGE
#                                 LINKDEF LinkDef.h            : linkdef file, default value is "LinkDef.h"
#                                 DICTIONARY_OPTIONS option    : options passed to rootcling
#                                 INSTALL_OPTIONS option       : options passed to install headers
#                                 NO_CXXMODULE                 : don't generate a C++ module for this package
#                                )
#---------------------------------------------------------------------------------------------------
function(ROOT_STANDARD_LIBRARY_PACKAGE libname)
  set(options NO_INSTALL_HEADERS STAGE1 NO_HEADERS NO_SOURCES OBJECT_LIBRARY NO_CXXMODULE)
  set(oneValueArgs LINKDEF)
  set(multiValueArgs DEPENDENCIES HEADERS NODEPHEADERS SOURCES BUILTINS LIBRARIES DICTIONARY_OPTIONS INSTALL_OPTIONS)
  CMAKE_PARSE_ARGUMENTS(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Check if we have any unparsed arguments
  if(ARG_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING "Unparsed arguments for ROOT_STANDARD_LIBRARY_PACKAGE: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  # Check that the user doesn't pass NO_HEADERS (to disable globbing) and HEADERS at the same time.
  if ((ARG_HEADERS OR ARG_NODEPHEADERS) AND ARG_NO_HEADERS)
    message(AUTHOR_WARNING "HEADERS and NO_HEADERS arguments are mutually exclusive.")
  endif()
  if (ARG_SOURCES AND ARG_NO_SOURCES)
    message(AUTHOR_WARNING "SOURCES and NO_SOURCES arguments are mutually exclusive.")
  endif()

  # Set default values
  # If HEADERS/SOURCES are not parsed, we glob for those files.
  if (NOT (ARG_HEADERS OR ARG_NO_HEADERS OR ARG_NODEPHEADERS))
    set(ARG_HEADERS "*.h")
  endif()
  if (NOT ARG_SOURCES AND NOT ARG_NO_SOURCES)
    set(ARG_SOURCES "*.cxx")
  endif()
  if (NOT ARG_LINKDEF)
    set(ARG_LINKDEF "LinkDef.h")
  endif()

  if (ARG_STAGE1)
    set(STAGE1_FLAG "STAGE1")
  endif()

  if (ARG_NO_CXXMODULE)
    set(NO_CXXMODULE_FLAG "NO_CXXMODULE")
  endif()

  if(ARG_NO_SOURCES)
    # Workaround bug in CMake by adding a dummy source file if all sources are generated, since
    # in that case the initial call to add_library() may not list any sources and CMake complains.
    add_custom_command(OUTPUT dummy.cxx COMMAND ${CMAKE_COMMAND} -E touch dummy.cxx)
  endif()

  if(runtime_cxxmodules)
    # Record ROOT targets to be used as a dependency targets for "onepcm" target.
    set(ROOT_LIBRARY_TARGETS "${ROOT_LIBRARY_TARGETS};${libname}" CACHE STRING "List of ROOT targets generated from ROOT_STANDARD_LIBRARY_PACKAGE()" FORCE)
  endif()

  if (PROJECT_NAME STREQUAL ROOT)
    include_directories(BEFORE "inc")
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/v7/inc")
      include_directories(BEFORE "v7/inc")
    endif()
  endif()

  if (ARG_OBJECT_LIBRARY)
    ROOT_OBJECT_LIBRARY(${libname}Objs ${ARG_SOURCES}
                        $<$<BOOL:${ARG_NO_SOURCES}>:dummy.cxx>)
    ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:${libname}Objs>
                        LIBRARIES ${ARG_LIBRARIES}
                        DEPENDENCIES ${ARG_DEPENDENCIES}
                        BUILTINS ${ARG_BUILTINS}
                       )
  else(ARG_OBJECT_LIBRARY)
    ROOT_LINKER_LIBRARY(${libname} ${ARG_SOURCES}
                        $<$<BOOL:${ARG_NO_SOURCES}>:dummy.cxx>
                        LIBRARIES ${ARG_LIBRARIES}
                        DEPENDENCIES ${ARG_DEPENDENCIES}
                        BUILTINS ${ARG_BUILTINS}
                       )
  endif(ARG_OBJECT_LIBRARY)

  if (NOT (ARG_HEADERS OR ARG_NODEPHEADERS))
    message(AUTHOR_WARNING "Called with no HEADERS and no NODEPHEADER. The generated "
      "dictionary will be empty. Consider using ROOT_LINKER_LIBRARY instead.")
  endif()

  ROOT_GENERATE_DICTIONARY(G__${libname} ${ARG_HEADERS}
                          ${NO_CXXMODULE_FLAG}
                          ${STAGE1_FLAG}
                          MODULE ${libname}
                          LINKDEF ${ARG_LINKDEF}
                          NODEPHEADERS ${ARG_NODEPHEADERS}
                          OPTIONS ${ARG_DICTIONARY_OPTIONS}
                          DEPENDENCIES ${ARG_DEPENDENCIES}
                          BUILTINS ${ARG_BUILTINS}
                          )

  # Dictionary might include things from the current src dir, e.g. tests. Alas
  # there is no way to set the include directory for a source file (except for
  # the generic COMPILE_FLAGS), so this needs to be glued to the target.
  if(NOT (CMAKE_PROJECT_NAME STREQUAL ROOT))
     target_include_directories(${libname} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  # Install headers if we have any headers and if the user didn't explicitly
  # disabled this.
  if (NOT ARG_NO_INSTALL_HEADERS OR ARG_NO_HEADERS)
    ROOT_INSTALL_HEADERS(${ARG_INSTALL_OPTIONS})
  endif()
endfunction()

#---------------------------------------------------------------------------------------------------
#---ROOT_EXECUTABLE( <name> source1 source2 ... LIBRARIES library1 library2 ... BUILTINS dep1 dep2 ...)
#---------------------------------------------------------------------------------------------------
function(ROOT_EXECUTABLE executable)
  CMAKE_PARSE_ARGUMENTS(ARG "CMAKENOEXPORT;NOINSTALL;TEST" "" "LIBRARIES;BUILTINS;ADDITIONAL_COMPILE_FLAGS"  ${ARGN})
  ROOT_GET_SOURCES(exe_srcs src ${ARG_UNPARSED_ARGUMENTS})
  set(executable_name ${executable})
  if(TARGET ${executable})
    message("Target ${executable} already exists. Renaming target name to ${executable}_new")
    set(executable ${executable}_new)
  endif()
  if(ARG_TEST) # we are building a test, so add EXCLUDE_FROM_ALL
    set(_all EXCLUDE_FROM_ALL)
  endif()
  if(NOT (PROJECT_NAME STREQUAL "ROOT"))
    # only for non-ROOT executable use $ROOTSYS/include
    include_directories(BEFORE ${CMAKE_BINARY_DIR}/include)
  elseif(MSVC)
    set(exe_srcs ${exe_srcs} ${ROOT_RC_SCRIPT})
  endif()
  add_executable(${executable} ${_all} ${exe_srcs})
  target_link_libraries(${executable} PRIVATE ${ARG_LIBRARIES})

  if(WIN32 AND ${executable} MATCHES \\.exe)
    set_target_properties(${executable} PROPERTIES SUFFIX "")
  endif()
  set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS ${executable})
  set_target_properties(${executable} PROPERTIES OUTPUT_NAME ${executable_name})
  if (ARG_ADDITIONAL_COMPILE_FLAGS)
    set_target_properties(${executable} PROPERTIES COMPILE_FLAGS ${ARG_ADDITIONAL_COMPILE_FLAGS})
  endif()
  if(CMAKE_PROJECT_NAME STREQUAL ROOT)
    add_dependencies(${executable} move_headers)
  endif()
  if(ARG_BUILTINS)
    foreach(arg1 ${ARG_BUILTINS})
      if(${arg1}_TARGET)
        add_dependencies(${executable} ${${arg1}_TARGET})
      endif()
    endforeach()
  endif()
  if(TARGET ROOT::ROOTStaticSanitizerConfig)
    set_property(TARGET ${executable}
      APPEND PROPERTY LINK_LIBRARIES ROOT::ROOTStaticSanitizerConfig)
  endif()
  #----Installation details------------------------------------------------------
  if(NOT ARG_NOINSTALL AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    if(NOT MSVC)
      ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(${executable} ${CMAKE_INSTALL_BINDIR})

      # The rootcling executable needs to run during build, so even if
      # CMAKE_BUILD_WITH_INSTALL_RPATH=ON was specified at config time, we have to
      # override it. Otherwise, the RPATH of rootcling inside the build tree is wrong
      # if the install tree doesn't mirror the build tree (e.g. for gnuinstall=ON).
      # All the other executables need the correct path in the build tree too, for testing.
      set_property(TARGET ${executable} PROPERTY BUILD_WITH_INSTALL_RPATH OFF)

    endif()
    if(ARG_CMAKENOEXPORT)
      install(TARGETS ${executable} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT applications)
    else()
      install(TARGETS ${executable} EXPORT ${CMAKE_PROJECT_NAME}Exports RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT applications)
    endif()
    if(WIN32)
      install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${executable}.pdb
              CONFIGURATIONS Debug RelWithDebInfo
              DESTINATION ${CMAKE_INSTALL_BINDIR}
              COMPONENT applications)
    endif()
  else()
    # If the target is not installed, it doesn't make sense to build it with the INSTALL_RPATH
    set_property(TARGET ${executable} PROPERTY BUILD_WITH_INSTALL_RPATH OFF)
  endif()
endfunction()

#---------------------------------------------------------------------------------------------------
#---REFLEX_BUILD_DICTIONARY( dictionary headerfiles selectionfile OPTIONS opt1 opt2 ...  LIBRARIES lib1 lib2 ... )
#---------------------------------------------------------------------------------------------------
function(REFLEX_BUILD_DICTIONARY dictionary headerfiles selectionfile )
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LIBRARIES;OPTIONS" ${ARGN})
  REFLEX_GENERATE_DICTIONARY(${dictionary} ${headerfiles} SELECTION ${selectionfile} OPTIONS ${ARG_OPTIONS})
  add_library(${dictionary}Dict MODULE ${gensrcdict})
  target_link_libraries(${dictionary}Dict ${ARG_LIBRARIES} ${ROOT_Reflex_LIBRARY})
  #----Installation details-------------------------------------------------------
  install(TARGETS ${dictionary}Dict LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  set(mergedRootMap ${CMAKE_INSTALL_PREFIX}/${lib}/${CMAKE_PROJECT_NAME}Dict.rootmap)
  set(srcRootMap ${CMAKE_CURRENT_BINARY_DIR}/${rootmapname})
  install(CODE "EXECUTE_PROCESS(COMMAND ${merge_rootmap_cmd} --do-merge --input-file ${srcRootMap} --merged-file ${mergedRootMap})")
endfunction()

# Need to set this outside of the function so that ${CMAKE_CURRENT_LIST_DIR}
# is for RootMacros.cmake and not for the file currently calling the function.
set(ROOT_TEST_DRIVER ${CMAKE_CURRENT_LIST_DIR}/RootTestDriver.cmake)

#----------------------------------------------------------------------------
# function ROOT_ADD_TEST( <name> COMMAND cmd [arg1... ]
#                        [PRECMD cmd [arg1...]] [POSTCMD cmd [arg1...]]
#                        [OUTPUT outfile] [ERROR errfile] [INPUT infile]
#                        [ENVIRONMENT var1=val1 var2=val2 ...
#                        [DEPENDS test1 ...]
#                        [RUN_SERIAL]
#                        [TIMEOUT seconds]
#                        [DEBUG]
#                        [SOURCE_DIR dir] [BINARY_DIR dir]
#                        [WORKING_DIR dir] [COPY_TO_BUILDDIR files]
#                        [BUILD target] [PROJECT project]
#                        [PASSREGEX exp] [FAILREGEX epx]
#                        [PASSRC code]
#                        [RESOURCE_LOCK lock]
#                        [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...]
#                        [LABELS label1 label2]
#                        [PYTHON_DEPS numpy numba keras torch ...] # List of python packages required to run this test.
#                                                              A fixture will be added the tries to import them before the test starts.
#                        [PROPERTIES prop1 value1 prop2 value2...]
#                       )
#
function(ROOT_ADD_TEST test)
  CMAKE_PARSE_ARGUMENTS(ARG "DEBUG;WILLFAIL;CHECKOUT;CHECKERR;RUN_SERIAL"
                            "TIMEOUT;BUILD;INPUT;OUTPUT;ERROR;SOURCE_DIR;BINARY_DIR;WORKING_DIR;PROJECT;PASSRC;RESOURCE_LOCK"
                            "COMMAND;COPY_TO_BUILDDIR;DIFFCMD;OUTCNV;OUTCNVCMD;PRECMD;POSTCMD;ENVIRONMENT;DEPENDS;PASSREGEX;OUTREF;ERRREF;FAILREGEX;LABELS;PYTHON_DEPS;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED;PROPERTIES"
                            ${ARGN})

  #- Handle COMMAND argument
  list(LENGTH ARG_COMMAND _len)
  if(_len LESS 1)
    if(NOT ARG_BUILD)
      message(FATAL_ERROR "ROOT_ADD_TEST: command is mandatory (without build)")
    endif()
  else()
    list(GET ARG_COMMAND 0 _prg)
    list(REMOVE_AT ARG_COMMAND 0)

    if(TARGET ${_prg})                                 # if command is a target, get the actual executable
      set(_prg "$<TARGET_FILE:${_prg}>")
      set(_cmd ${_prg} ${ARG_COMMAND})
    else()
      find_program(_exe ${_prg})
      if(_exe)                                         # if the command is found in the system, use it
        set(_cmd ${_exe} ${ARG_COMMAND})
      elseif(NOT IS_ABSOLUTE ${_prg})                  # if not absolute, assume is found in current binary dir
        set(_prg ${CMAKE_CURRENT_BINARY_DIR}/${_prg})
        set(_cmd ${_prg} ${ARG_COMMAND})
      else()                                           # take as it is
        set(_cmd ${_prg} ${ARG_COMMAND})
      endif()
      unset(_exe CACHE)
    endif()

    string(REPLACE ";" "^" _cmd "${_cmd}")
  endif()

  set(_command ${CMAKE_COMMAND} -DCMD=${_cmd})

  #- Handle PRE and POST commands
  if(ARG_PRECMD)
    string(REPLACE ";" "^" _pre "${ARG_PRECMD}")
    set(_command ${_command} -DPRE=${_pre})
  endif()

  if(ARG_POSTCMD)
    string(REPLACE ";" "^" _post "${ARG_POSTCMD}")
    set(_command ${_command} -DPOST=${_post})
  endif()

  #- Handle INPUT, OUTPUT, ERROR, DEBUG arguments
  if(ARG_INPUT)
    set(_command ${_command} -DIN=${ARG_INPUT})
  endif()

  if(ARG_OUTPUT)
    set(_command ${_command} -DOUT=${ARG_OUTPUT})
  endif()

  if(ARG_OUTREF)
    set(_command ${_command} -DOUTREF=${ARG_OUTREF})
  endif()

  if(ARG_ERRREF)
    set(_command ${_command} -DERRREF=${ARG_ERRREF})
  endif()

  if(ARG_ERROR)
    set(_command ${_command} -DERR=${ARG_ERROR})
  endif()

  if(ARG_WORKING_DIR)
    set(_command ${_command} -DCWD=${ARG_WORKING_DIR})
  endif()

  if(ARG_DEBUG)
    set(_command ${_command} -DDBG=ON)
  endif()

  if(ARG_PASSRC)
    set(_command ${_command} -DRC=${ARG_PASSRC})
  endif()

  if(ARG_OUTCNVCMD)
    string(REPLACE ";" "^" _outcnvcmd "${ARG_OUTCNVCMD}")
    string(REPLACE "=" "@" _outcnvcmd "${_outcnvcmd}")
    set(_command ${_command} -DCNVCMD=${_outcnvcmd})
  endif()

  if(ARG_OUTCNV)
    string(REPLACE ";" "^" _outcnv "${ARG_OUTCNV}")
    set(_command ${_command} -DCNV=${_outcnv})
  endif()

  if(ARG_DIFFCMD)
    string(REPLACE ";" "^" _diff_cmd "${ARG_DIFFCMD}")
    set(_command ${_command} -DDIFFCMD=${_diff_cmd})

    if(TARGET ROOT::ROOTStaticSanitizerConfig)
      # We have to set up leak sanitizer such that it doesn't report on suppressed
      # leaks. Otherwise, all diffs will fail.
      set(LSAN_OPT ARG_ENVIRONMENT)
      list(FILTER LSAN_OPT INCLUDE REGEX LSAN_OPTIONS=[^;]+)
      if(NOT LSAN_OPT MATCHES LSAN_OPTIONS=.*)
        set(LSAN_OPT LSAN_OPTIONS=)
      endif()
      string(APPEND LSAN_OPT ":print_suppressions=0")
      list(FILTER ARG_ENVIRONMENT EXCLUDE REGEX LSAN_OPTIONS.*)
      list(APPEND ARG_ENVIRONMENT ${LSAN_OPT})
    endif()
  endif()

  if(ARG_CHECKOUT)
    set(_command ${_command} -DCHECKOUT=true)
  endif()

  if(ARG_CHECKERR)
    set(_command ${_command} -DCHECKERR=true)
  endif()

  set(_command ${_command} -DSYS=${ROOTSYS})

  #- Handle ENVIRONMENT argument
  if(ASAN_EXTRA_LD_PRELOAD)
    # Address sanitizer runtime needs to be preloaded in all python tests
    # Check now if the -DCMD= contains "python[0-9.] ", but exclude helper
    # scripts such as roottest/root/meta/genreflex/XMLParsing/parseXMLs.py
    # and roottest/root/rint/driveTabCom.py
    set(theCommand ${_command})
    list(FILTER theCommand INCLUDE REGEX "^-DCMD=.*python[0-9.]*[\\^]")
    if((theCommand AND
        NOT (_command MATCHES XMLParsing/parseXMLs.py OR
             _command MATCHES roottest/root/rint/driveTabCom.py))
       OR (_command MATCHES roottest/python/cmdLineUtils AND
           NOT _command MATCHES MakeNameCyclesRootmvInput))
      set(_command ${_command} -DCMD_ENV=${ld_preload}=${ASAN_EXTRA_LD_PRELOAD})
    endif()
  endif()

  if(ARG_ENVIRONMENT)
    string(REPLACE ";" "#" _env "${ARG_ENVIRONMENT}")
    set(_command ${_command} -DENV=${_env})
  endif()

  #- Copy files to the build directory.
  if(ARG_COPY_TO_BUILDDIR)
    string(REPLACE ";" "^" _copy_files "${ARG_COPY_TO_BUILDDIR}")
    set(_command ${_command} -DCOPY=${_copy_files})
  endif()

  set(_command ${_command} -P ${ROOT_TEST_DRIVER})

  if(ARG_WILLFAIL)
    set(test ${test}_WILL_FAIL)
  endif()

  #- Now we can actually add the test
  if(ARG_BUILD)
    if(NOT ARG_SOURCE_DIR)
      set(ARG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    if(NOT ARG_BINARY_DIR)
      set(ARG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    if(NOT ARG_PROJECT)
       if(NOT PROJECT_NAME STREQUAL "ROOT")
         set(ARG_PROJECT ${PROJECT_NAME})
       else()
         set(ARG_PROJECT ${ARG_BUILD})
       endif()
    endif()
    add_test(NAME ${test} COMMAND ${CMAKE_CTEST_COMMAND}
      --build-and-test  ${ARG_SOURCE_DIR} ${ARG_BINARY_DIR}
      --build-generator ${CMAKE_GENERATOR}
      --build-makeprogram ${CMAKE_MAKE_PROGRAM}
      --build-target ${ARG_BUILD}
      --build-project ${ARG_PROJECT}
      --build-config $<CONFIGURATION>
      --build-noclean
      --test-command ${_command} )
    set_property(TEST ${test} PROPERTY ENVIRONMENT ROOT_DIR=${CMAKE_BINARY_DIR})
  else()
    add_test(NAME ${test} COMMAND ${_command})
    if (gnuinstall)
      set_property(TEST ${test} PROPERTY ENVIRONMENT ROOTIGNOREPREFIX=1)
    endif()
  endif()

  #- provided fixtures and resource lock are set here
  if (ARG_FIXTURES_SETUP)
    set_property(TEST ${test} PROPERTY
      FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()

  if (ARG_FIXTURES_CLEANUP)
    set_property(TEST ${test} PROPERTY
      FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set_property(TEST ${test} PROPERTY
      FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()

  if (ARG_RESOURCE_LOCK)
    set_property(TEST ${test} PROPERTY
      RESOURCE_LOCK ${ARG_RESOURCE_LOCK})
  endif()

  set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT ROOT_HIST=0)

  #- Handle TIMEOUT and DEPENDS arguments
  if(ARG_TIMEOUT)
    set_property(TEST ${test} PROPERTY TIMEOUT ${ARG_TIMEOUT})
  endif()

  if(ARG_DEPENDS)
    set_property(TEST ${test} PROPERTY DEPENDS ${ARG_DEPENDS})
  endif()

  if(ARG_PASSREGEX)
    set_property(TEST ${test} PROPERTY PASS_REGULAR_EXPRESSION ${ARG_PASSREGEX})
  endif()

  if(ARG_FAILREGEX)
    set_property(TEST ${test} PROPERTY FAIL_REGULAR_EXPRESSION ${ARG_FAILREGEX})
  endif()

  if(ARG_WILLFAIL)
    set_property(TEST ${test} PROPERTY WILL_FAIL true)
  endif()

  if(ARG_LABELS)
    set_tests_properties(${test} PROPERTIES LABELS "${ARG_LABELS}")
  endif()

  foreach(python_dep ${ARG_PYTHON_DEPS})
    ROOT_FIND_PYTHON_MODULE(${python_dep})
    if(NOT ROOT_${python_dep}_FOUND)
      set_property(TEST ${test} PROPERTY DISABLED True)
      continue()
    endif()
  endforeach()

  if(ARG_RUN_SERIAL)
    set_property(TEST ${test} PROPERTY RUN_SERIAL true)
  endif()

  # Pass PROPERTIES argument to the set_tests_properties as-is
  if(ARG_PROPERTIES)
    set_tests_properties(${test} PROPERTIES ${ARG_PROPERTIES})
  endif()

endfunction()

#----------------------------------------------------------------------------
# ROOT_PATH_TO_STRING( <variable> path PATH_SEPARATOR_REPLACEMENT replacement )
#
# Mangle the path to a string.
#----------------------------------------------------------------------------
function(ROOT_PATH_TO_STRING resultvar path)
  # FIXME: Copied and modified from ROOTTEST_TARGETNAME_FROM_FILE. We should find a common place for that code.
  # FIXME: ROOTTEST_TARGETNAME_FROM_FILE could be replaced by just a call to string(MAKE_C_IDENTIFIER)...
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "PATH_SEPARATOR_REPLACEMENT" ${ARGN})

  set(sep_replacement "")
  if (ARG_PATH_SEPARATOR_REPLACEMENT)
    set(sep_replacement ${ARG_PATH_SEPARATOR_REPLACEMENT})
  endif()

  get_filename_component(realfp ${path} ABSOLUTE)
  get_filename_component(filename_we ${path} NAME_WE)

  string(REPLACE "${CMAKE_SOURCE_DIR}" "" relativepath ${realfp})
  string(REPLACE "${path}" "" relativepath ${relativepath})

  string(MAKE_C_IDENTIFIER ${relativepath}${filename_we} mangledname)
  string(REPLACE "_" "${sep_replacement}" mangledname ${mangledname})

  set(${resultvar} "${mangledname}" PARENT_SCOPE)
endfunction(ROOT_PATH_TO_STRING)

#----------------------------------------------------------------------------
# ROOT_ADD_UNITTEST_DIR(<libraries ...>)
#----------------------------------------------------------------------------
function(ROOT_ADD_UNITTEST_DIR)
  ROOT_GLOB_FILES(test_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
  # Get the component from the path. Eg. core to form coreTests test suite name.
  ROOT_PATH_TO_STRING(test_name ${CMAKE_CURRENT_SOURCE_DIR}/)
  ROOT_ADD_GTEST(${test_name}Unit ${test_files} LIBRARIES ${ARGN})
endfunction()

#----------------------------------------------------------------------------
# function ROOT_ADD_GTEST(<testsuite> source1 source2...
#                        [WILLFAIL] Negate output of test
#                        [TIMEOUT seconds]
#                        [COPY_TO_BUILDDIR file1 file2] Copy listed files when ctest invokes the test.
#                        [LIBRARIES lib1 lib2...] -- Libraries to link against
#                        [LABELS label1 label2...] -- Labels to annotate the test
#                        [INCLUDE_DIRS label1 label2...] -- Extra target include directories
#                        [REPEATS number] -- Repeats testsuite `number` times, stopping at the first failure.
#                        [FAILREGEX ...] Fail test if this regex matches.
#                        [ENVIRONMENT var1=val1 var2=val2 ...
# Creates a new googletest exectuable, and registers it as a test.
#----------------------------------------------------------------------------
function(ROOT_ADD_GTEST test_suite)
  cmake_parse_arguments(ARG
    "WILLFAIL"
    "TIMEOUT;REPEATS;FAILREGEX"
    "COPY_TO_BUILDDIR;LIBRARIES;LABELS;INCLUDE_DIRS;ENVIRONMENT" ${ARGN})

  ROOT_GET_SOURCES(source_files . ${ARG_UNPARSED_ARGUMENTS})
  # Note we cannot use ROOT_EXECUTABLE without user-specified set of LIBRARIES to link with.
  # The test suites should choose this in their specific CMakeLists.txt file.
  # FIXME: For better coherence we could restrict the libraries the test suite could link
  # against. For example, tests in Core should link only against libCore. This could be tricky
  # to implement because some ROOT components create more than one library.
  ROOT_EXECUTABLE(${test_suite} ${source_files} LIBRARIES ${ARG_LIBRARIES})
  target_link_libraries(${test_suite} PRIVATE GTest::gtest GTest::gmock GTest::gtest_main GTest::gmock_main)
  if(TARGET ROOT::TestSupport)
    target_link_libraries(${test_suite} PRIVATE ROOT::TestSupport)
  else()
    message(WARNING "ROOT_ADD_GTEST(${test_suite} ...): The target ROOT::TestSupport is missing. It looks like the test is declared against a ROOT build that is configured with -Dtesting=OFF.
            If this test sends warning or error messages, this will go unnoticed.")
  endif()
  target_include_directories(${test_suite} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  if (ARG_INCLUDE_DIRS)
    target_include_directories(${test_suite} PRIVATE ${ARG_INCLUDE_DIRS})
  endif(ARG_INCLUDE_DIRS)

  if(MSVC)
    set(test_exports "/EXPORT:_Init_thread_abort /EXPORT:_Init_thread_epoch \
        /EXPORT:_Init_thread_footer /EXPORT:_Init_thread_header /EXPORT:_tls_index")
    set_property(TARGET ${test_suite} APPEND_STRING PROPERTY LINK_FLAGS ${test_exports})
  endif()

  if(ARG_WILLFAIL)
    set(willfail WILLFAIL)
  endif()

  if(ARG_REPEATS)
    set(extra_command --gtest_repeat=${ARG_REPEATS} --gtest_break_on_failure)
  endif()

  ROOT_PATH_TO_STRING(name_with_path ${test_suite} PATH_SEPARATOR_REPLACEMENT "-")
  string(REPLACE "-test-" "-" clean_name_with_path ${name_with_path})
  ROOT_ADD_TEST(
    gtest${clean_name_with_path}
    COMMAND ${test_suite} ${extra_command}
    WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
    COPY_TO_BUILDDIR "${ARG_COPY_TO_BUILDDIR}"
    ${willfail}
    TIMEOUT "${ARG_TIMEOUT}"
    LABELS "${ARG_LABELS}"
    FAILREGEX "${ARG_FAILREGEX}"
    ENVIRONMENT "${ARG_ENVIRONMENT}"
  )
endfunction()


#----------------------------------------------------------------------------
# ROOT_ADD_TEST_SUBDIRECTORY( <name> )
#----------------------------------------------------------------------------
function(ROOT_ADD_TEST_SUBDIRECTORY subdir)
  set(fullsubdir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
  cmake_path(RELATIVE_PATH fullsubdir BASE_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE subdir)
  set_property(GLOBAL APPEND PROPERTY ROOT_TEST_SUBDIRS ${subdir})
endfunction()

#----------------------------------------------------------------------------
# ROOT_ADD_PYUNITTESTS( <name> )
#----------------------------------------------------------------------------
function(ROOT_ADD_PYUNITTESTS name)
  if(MSVC)
    set(ROOT_ENV ROOTSYS=${ROOTSYS}
        PYTHONPATH=${ROOTSYS}/bin;$ENV{PYTHONPATH})
  else()
    set(ROOT_ENV ROOTSYS=${ROOTSYS}
        PATH=${ROOTSYS}/bin:$ENV{PATH}
        ${ld_library_path}=${ROOTSYS}/lib:$ENV{${ld_library_path}}
        PYTHONPATH=${ROOTSYS}/lib:$ENV{PYTHONPATH})
  endif()
  string(REGEX REPLACE "[_]" "-" good_name "${name}")
  ROOT_ADD_TEST(pyunittests-${good_name}
                COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "*.py" -v
                ENVIRONMENT ${ROOT_ENV})
endfunction()

#----------------------------------------------------------------------------
# ROOT_ADD_PYUNITTEST( <name> <file>
#                     [WILLFAIL]
#                     [GENERIC] # Run a generic Python command `python <file>` to run the test.
#                     [COPY_TO_BUILDDIR copy_file1 copy_file1 ...]
#                     [ENVIRONMENT var1=val1 var2=val2 ...]
#                     [PYTHON_DEPS dep_x dep_y ...] # Communicate that this test requires python packages. A fixture checking for these will be run before the test.)
#                     [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...]
#----------------------------------------------------------------------------
function(ROOT_ADD_PYUNITTEST name file)
  CMAKE_PARSE_ARGUMENTS(ARG "WILLFAIL;GENERIC" "" "COPY_TO_BUILDDIR;ENVIRONMENT;PYTHON_DEPS;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED" ${ARGN})
  if(MSVC)
    set(ROOT_ENV ROOTSYS=${ROOTSYS}
        PYTHONPATH=${ROOTSYS}/bin;$ENV{PYTHONPATH})
  else()
    set(ROOT_ENV ROOTSYS=${ROOTSYS}
        PATH=${ROOTSYS}/bin:$ENV{PATH}
        ${ld_library_path}=${ROOTSYS}/lib:$ENV{${ld_library_path}}
        PYTHONPATH=${ROOTSYS}/lib:$ENV{PYTHONPATH})
  endif()
  string(REGEX REPLACE "[_]" "-" good_name "${name}")
  get_filename_component(file_name ${file} NAME)
  get_filename_component(file_dir ${file} DIRECTORY)

  if(ARG_COPY_TO_BUILDDIR)
    foreach(copy_file ${ARG_COPY_TO_BUILDDIR})
      get_filename_component(abs_path ${copy_file} ABSOLUTE)
      set(copy_files ${copy_files} ${abs_path})
    endforeach()
    set(copy_to_builddir COPY_TO_BUILDDIR ${copy_files})
  endif()

  if(ARG_WILLFAIL)
    set(will_fail WILLFAIL)
  endif()

  list(APPEND labels python)
  if(ARG_PYTHON_DEPS)
    list(APPEND labels python_runtime_deps)
  endif()

  ROOT_PATH_TO_STRING(name_with_path ${good_name} PATH_SEPARATOR_REPLACEMENT "-")
  string(REPLACE "-test-" "-" clean_name_with_path ${name_with_path})

  if(ARG_GENERIC)
    set(test_cmd COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${file_name})
  else()
    set(test_cmd COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/${file_dir} -p ${file_name} -v)
  endif()

  set(test_name pyunittests${clean_name_with_path})
  ROOT_ADD_TEST(${test_name}
              ${test_cmd}
              ENVIRONMENT ${ROOT_ENV} ${ARG_ENVIRONMENT}
              LABELS ${labels}
              ${copy_to_builddir}
              ${will_fail}
              PYTHON_DEPS ${ARG_PYTHON_DEPS})

  if (ARG_FIXTURES_SETUP)
    set_property(TEST ${test_name} PROPERTY
      FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()

  if (ARG_FIXTURES_CLEANUP)
    set_property(TEST ${test_name} PROPERTY
      FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set_property(TEST ${test_name} PROPERTY
      FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()

endfunction()

#----------------------------------------------------------------------------
# ROOT_ADD_CXX_FLAG(var flag)
#----------------------------------------------------------------------------
function(ROOT_ADD_CXX_FLAG var flag)
  string(REGEX REPLACE "[-.+/:= ]" "_" flag_esc "${flag}")
  CHECK_CXX_COMPILER_FLAG("-Werror ${flag}" CXX_HAS${flag_esc})
  if(CXX_HAS${flag_esc})
    set(${var} "${${var}} ${flag}" PARENT_SCOPE)
  endif()
endfunction()
#----------------------------------------------------------------------------
# ROOT_ADD_C_FLAG(var flag)
#----------------------------------------------------------------------------
function(ROOT_ADD_C_FLAG var flag)
  string(REGEX REPLACE "[-.+/:= ]" "_" flag_esc "${flag}")
  CHECK_C_COMPILER_FLAG("-Werror ${flag}" C_HAS${flag_esc})
  if(C_HAS${flag_esc})
    set(${var} "${${var}} ${flag}" PARENT_SCOPE)
  endif()
endfunction()

#----------------------------------------------------------------------------
# ROOT_ADD_COMPILE_OPTIONS(flags)
#----------------------------------------------------------------------------
macro(ROOT_ADD_COMPILE_OPTIONS flags)
  foreach(__flag ${flags})
    check_cxx_compiler_flag("-Werror ${__flag}" __result)
    if(__result)
      add_compile_options(${__flag})
    endif()
  endforeach()
  unset(__flag)
  unset(__result)
endmacro()

#----------------------------------------------------------------------------
# ROOT_FIND_PYTHON_MODULE(module [REQUIRED] [QUIET])
# Try importing the python dependency and cache the result in
# ROOT_TEST_<MODULE> (all upper case).
# Also set ROOT_<MODULE>_FOUND (all upper case) as well as ROOT_<module>_FOUND
# (the original spelling of the argument) in the parent scope of this function
# for convenient testing in subsequent if().
#----------------------------------------------------------------------------
function(ROOT_FIND_PYTHON_MODULE module)
  CMAKE_PARSE_ARGUMENTS(ARG "REQUIRED;QUIET" "" "" ${ARGN})
  string(TOUPPER ${module} module_upper)
  set(CACHE_VAR ROOT_TEST_${module_upper})

  if(NOT DEFINED ${CACHE_VAR})
    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c"
                            "import ${module}; print(getattr(${module}, '__version__', 'unknown'))"
      RESULT_VARIABLE status
      OUTPUT_VARIABLE module_version
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)

    if(${status} EQUAL 0)
      set(${CACHE_VAR} ON CACHE BOOL "Enable tests depending on '${module}'")
    else()
      set(${CACHE_VAR} OFF CACHE BOOL "Enable tests depending on '${module}'")
    endif()

    if(NOT ARG_QUIET)
      if(${CACHE_VAR})
        message(STATUS "Found Python module ${module} (found version \"${module_version}\")")
      else()
        message(STATUS "Could NOT find Python module ${module}. Corresponding tests will be disabled.")
      endif()
    endif()
  endif()

  # Set the ROOT_xxx_FOUND to the (cached) result of the search:
  set(ROOT_${module_upper}_FOUND ${${CACHE_VAR}} PARENT_SCOPE)
  set(ROOT_${module}_FOUND ${${CACHE_VAR}} PARENT_SCOPE)

  if(ARG_REQUIRED AND NOT ${CACHE_VAR})
    message(FATAL_ERROR "Python module ${module} is required.")
  endif()
endfunction()

#----------------------------------------------------------------------------
# generateHeader(target input output)
# Generate a help header file with cmake/scripts/argparse2help.py script
# The 1st argument is the target to which the custom command will be attached
# The 2nd argument is the path to the python argparse input file
# The 3rd argument is the path to the output header file
#----------------------------------------------------------------------------
function(generateHeader target input output)
  add_custom_command(OUTPUT ${output}
    MAIN_DEPENDENCY
      ${input}
    DEPENDS
      ${CMAKE_SOURCE_DIR}/cmake/scripts/argparse2help.py
    COMMAND
      ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/cmake/scripts/argparse2help.py ${input} ${output}
  )
  target_sources(${target} PRIVATE ${output})
endfunction()

#----------------------------------------------------------------------------
# Generate and install manual page with cmake/scripts/argparse2help.py script
# The 1st argument is the name of the manual page
# The 2nd argument is the path to the python argparse input file
# The 3rd argument is the path to the output manual page
#----------------------------------------------------------------------------
function(generateManual name input output)
  add_custom_target(${name} ALL DEPENDS ${output})

  add_custom_command(OUTPUT ${output}
    MAIN_DEPENDENCY
      ${input}
    DEPENDS
      ${CMAKE_SOURCE_DIR}/cmake/scripts/argparse2help.py
    COMMAND
      ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/cmake/scripts/argparse2help.py ${input} ${output}
  )

  install(FILES ${output} DESTINATION ${CMAKE_INSTALL_MANDIR}/man1)
endfunction()

#----------------------------------------------------------------------------
# --- ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(target install_dir)
#
# Sets the INSTALL_RPATH for a given target so that it can find the ROOT shared
# libraries at runtime. The RPATH is set relative to the target's own location
# using $ORIGIN (or @loader_path on macOS).
#
# Arguments:
#   target       - The CMake target (e.g., a shared library or executable)
#   install_dir  - The install subdirectory relative to CMAKE_INSTALL_PREFIX
#----------------------------------------------------------------------------
function(ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH target install_dir)
  cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${install_dir}" OUTPUT_VARIABLE to_libdir)

  # New path
  if(APPLE)
    set(new_rpath "@loader_path/${to_libdir}")
  else()
    set(new_rpath "$ORIGIN/${to_libdir}")
  endif()

  # Append to existing RPATH
  set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "${new_rpath}")
endfunction()


#-------------------------------------------------------------------------------
#
#  Former RoottestMacros.cmake starts here
#
#-------------------------------------------------------------------------------

if(CMAKE_GENERATOR MATCHES Makefiles)
  set(fast /fast)
  set(always-make --always-make)
endif()
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  set(always-make -v:m)
endif()
#-------------------------------------------------------------------------------
#
#  function ROOTTEST_ADD_TESTDIRS([EXCLUDED_DIRS] dir)
#
#  Scans all subdirectories for CMakeLists.txt files. Each subdirectory that
#  contains a CMakeLists.txt file is then added as a subdirectory.
#-------------------------------------------------------------------------------
function(ROOTTEST_ADD_TESTDIRS)

  set(dirs "")
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "EXCLUDED_DIRS" ${ARGN})
  set(curdir ${CMAKE_CURRENT_SOURCE_DIR})

  file(GLOB found_dirs ${curdir} ${curdir}/*)

  # If there are excluded directories through EXCLUDED_DIRS,
  # add_subdirectory() for them will not be applied
  if(ARG_EXCLUDED_DIRS)
    foreach(excluded_dir ${ARG_EXCLUDED_DIRS})
      list(REMOVE_ITEM found_dirs "${CMAKE_CURRENT_SOURCE_DIR}/${excluded_dir}")
    endforeach()
  endif()

  foreach(f ${found_dirs})
    if(IS_DIRECTORY ${f})
      if(EXISTS "${f}/CMakeLists.txt" AND NOT ${f} STREQUAL ${curdir})
        list(APPEND dirs ${f})
      endif()
    endif()
  endforeach()

  list(SORT dirs)

  foreach(d ${dirs})
    string(REPLACE "${curdir}/" "" d ${d})
    add_subdirectory(${d})
    # create .rootrc in binary directory to avoid filling $HOME/.root_hist
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${d}/.rootrc")
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${d}/.rootrc ${CMAKE_CURRENT_BINARY_DIR}/${d} COPYONLY)
    else()
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${d}/.rootrc "
Rint.History:  .root_hist
ACLiC.LinkLibs:  1
")
    endif()
  endforeach()

endfunction()

#-------------------------------------------------------------------------------
#
#  function ROOTTEST_SET_TESTOWNER(owner)
#
#  Specify the owner of the tests in the current directory. Note, that the owner
#  can be specified for each test individually, as well.
#
#-------------------------------------------------------------------------------
function(ROOTTEST_SET_TESTOWNER owner)
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
               PROPERTY ROOTTEST_TEST_OWNER ${owner})
endfunction(ROOTTEST_SET_TESTOWNER)

#-------------------------------------------------------------------------------
#
# function ROOTTEEST_TARGETNAME_FROM_FILE(<resultvar> <filename>)
#
# Construct a target name for a given file <filename> and store its name into
# <resultvar>. The target name is of the form:
#
#   <directorypath>-<filename_WE>
#
#-------------------------------------------------------------------------------
function(ROOTTEST_TARGETNAME_FROM_FILE resultvar filename)

  get_filename_component(realfp ${filename} ABSOLUTE)
  get_filename_component(filename_we ${filename} NAME_WE)

  string(REPLACE "${CMAKE_SOURCE_DIR}/" "" relativepath ${realfp})
  string(REPLACE "${filename}"     "" relativepath ${relativepath})

  string(REPLACE "/" "-" targetname ${relativepath}${filename_we})
  set(${resultvar} "${targetname}" PARENT_SCOPE)

endfunction(ROOTTEST_TARGETNAME_FROM_FILE)

#-------------------------------------------------------------------------------
#
# function ROOTTEST_ADD_AUTOMACROS(DEPENDS [dependencies ...]
#                                  [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...])
#
# Automatically adds all macros in the current source directory to the list of
# tests that follow the naming scheme:
#
#   run*.C, run*.cxx, assert*.C, assert*.cxx, exec*.C, exec*.cxx
#
#-------------------------------------------------------------------------------
function(ROOTTEST_ADD_AUTOMACROS)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "DEPENDS;WILLFAIL;EXCLUDE;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED" ${ARGN})

  file(GLOB automacros run*.C run*.cxx assert*.C assert*.cxx exec*.C exec*.cxx)

  foreach(dep ${ARG_DEPENDS})
    if(${dep} MATCHES "[.]C" OR ${dep} MATCHES "[.]cxx" OR ${dep} MATCHES "[.]h")
      ROOTTEST_COMPILE_MACRO(${dep})
      list(APPEND auto_depends ${COMPILE_MACRO_TEST})
    else()
      list(APPEND auto_depends ${dep})
    endif()
  endforeach()

  foreach(am ${automacros})
    get_filename_component(auto_macro_filename ${am} NAME)
    get_filename_component(auto_macro_name  ${am} NAME_WE)
    if(${auto_macro_name} MATCHES "^run")
      string(REPLACE run "" auto_macro_subname ${auto_macro_name})
    elseif(${auto_macro_name} MATCHES "^exec")
      string(REPLACE exec "" auto_macro_subname ${auto_macro_name})
    else()
      set(auto_macro_subname ${auto_macro_name})
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${auto_macro_name}.ref)
      set(outref OUTREF ${auto_macro_name}.ref)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${auto_macro_subname}.ref)
      set(outref OUTREF ${auto_macro_subname}.ref)
    else()
      set(outref "")
    endif()

    ROOTTEST_TARGETNAME_FROM_FILE(targetname ${auto_macro_filename})

    foreach(wf ${ARG_WILLFAIL})
      if(${auto_macro_name} MATCHES ${wf})
        set(arg_wf WILLFAIL)
      endif()
    endforeach()

    set(selected 1)
    foreach(excl ${ARG_EXCLUDE})
      if(${auto_macro_name} MATCHES ${excl})
        set(selected 0)
        break()
      endif()
    endforeach()

    if (ARG_FIXTURES_SETUP)
      set(fixtures_setup ${ARG_FIXTURES_SETUP})
    endif()

    if (ARG_FIXTURES_CLEANUP)
      set(fixtures_cleanup ${ARG_FIXTURES_CLEANUP})
    endif()

    if (ARG_FIXTURES_REQUIRED)
      set(fixtures_required ${ARG_FIXTURES_REQUIRED})
    endif()

    if(selected)
      ROOTTEST_ADD_TEST(${targetname}-auto
                        MACRO ${auto_macro_filename}${${auto_macro_name}-suffix}
                        ${outref}
                        ${arg_wf}
                        FIXTURES_SETUP ${fixtures_setup}
                        FIXTURES_CLEANUP ${fixtures_cleanup}
                        FIXTURES_REQUIRED ${fixtures_required}
                        DEPENDS ${auto_depends})
    endif()
  endforeach()

endfunction(ROOTTEST_ADD_AUTOMACROS)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_COMPILE_MACRO(<filename> [BUILDOBJ object] [BUILDLIB lib]
#                                         [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...]
#                                         [DEPENDS dependencies...])
#
# This macro creates and loads a shared library containing the code from
# the file <filename>. A test that performs the compilation is created.
# The target name of the created test is stored in the variable
# COMPILE_MACRO_TEST which can be accessed by the calling CMakeLists.txt in
# order to manage dependencies.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_COMPILE_MACRO filename)
  CMAKE_PARSE_ARGUMENTS(ARG "" "BUILDOBJ;BUILDLIB" "DEPENDS;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED"  ${ARGN})

  # Add defines to root_compile_macro, in order to have out-of-source builds
  # when using the scripts/build.C macro.
  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    if(d MATCHES "_WIN32" OR d MATCHES "_XKEYCHECK_H" OR d MATCHES "NOMINMAX")
      continue()
    endif()
    list(APPEND RootMacroDirDefines "-e;#define ${d}")
  endforeach()

  set(RootMacroBuildDefines
        -e "#define CMakeEnvironment"
        -e "#define CMakeBuildDir \"${CMAKE_CURRENT_BINARY_DIR}\""
        -e "gSystem->AddDynamicPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
        -e "gROOT->SetMacroPath(\"${CMAKE_CURRENT_SOURCE_DIR}\")"
        -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")"
        -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")"
        -e "gSystem->SetBuildDir(\"${CMAKE_CURRENT_BINARY_DIR}\", true)"
        ${RootMacroDirDefines})

  set(root_compile_macro ${ROOT_root_CMD} ${RootMacroBuildDefines} -q -l -b)

  get_filename_component(realfp ${filename} ABSOLUTE)
  if(MSVC)
    string(REPLACE "/" "\\\\" realfp ${realfp})
  endif()

  set(BuildScriptFile ${ROOT_SOURCE_DIR}/roottest/scripts/build.C)

  set(BuildScriptArg \(\"${realfp}\",\"${ARG_BUILDLIB}\",\"${ARG_BUILDOBJ}\"\))

  set(compile_macro_command ${root_compile_macro}
                            ${BuildScriptFile}${BuildScriptArg}
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(ARG_DEPENDS)
    set(deps ${ARG_DEPENDS})
  endif()

  ROOTTEST_TARGETNAME_FROM_FILE(COMPILE_MACRO_TEST ${filename})

  set(compile_target ${COMPILE_MACRO_TEST}-compile-macro)

  add_custom_target(${compile_target}
                    COMMAND ${compile_macro_command}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    VERBATIM)

  if(ARG_DEPENDS)
    add_dependencies(${compile_target} ${deps})
  endif()

  set(COMPILE_MACRO_TEST ${COMPILE_MACRO_TEST}-build)

  add_test(NAME ${COMPILE_MACRO_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    ${build_config}
                                    --target ${compile_target}${fast}
                                    -- ${always-make})
  if(NOT MSVC OR win_broken_tests)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY FAIL_REGULAR_EXPRESSION "Warning in")
  endif()
  set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})
  if(CMAKE_GENERATOR MATCHES Ninja AND NOT MSVC)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY RUN_SERIAL true)
  endif()
  if (ARG_FIXTURES_SETUP)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY
      FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()
  if (ARG_FIXTURES_CLEANUP)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY
      FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY
      FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()

endmacro(ROOTTEST_COMPILE_MACRO)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_GENERATE_DICTIONARY(<dictname>
#                                    [LINKDEF linkdef]
#                                    [DEPENDS deps]
#                                    [OPTIONS opts]
#                                    [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...]
#                                    [files ...]      )
#
# This macro generates a dictionary <dictname> from the provided <files>.
# A test that performs the dictionary generation is created.  The target name of
# the created test is stored in the variable GENERATE_DICTIONARY_TEST which can
# be accessed by the calling CMakeLists.txt in order to manage dependencies.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_DICTIONARY dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "NO_ROOTMAP;NO_CXXMODULE" "FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED" "LINKDEF;SOURCES;DEPENDS;OPTIONS;COMPILE_OPTIONS" ${ARGN})

  set(CMAKE_ROOTTEST_DICT ON)

  if(ARG_NO_ROOTMAP)
    set(CMAKE_ROOTTEST_NOROOTMAP ON)
  endif()
  if(ARG_NO_CXXMODULE)
    set(EXTRA_ARGS NO_CXXMODULE)
  endif()

  # roottest dictionaries do not need to be relocatable. Instead, allow
  # dictionaries to find the input headers even from the source directory
  # - without ROOT_INCLUDE_PATH - by passing the full path to rootcling:
  set(FULL_PATH_HEADERS )
  foreach(hdr ${ARG_UNPARSED_ARGUMENTS})
    if(IS_ABSOLUTE ${hdr})
      list(APPEND FULL_PATH_HEADERS ${hdr})
    else()
      list(APPEND FULL_PATH_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
    endif()
  endforeach()

  ROOT_GENERATE_DICTIONARY(${dictname} ${FULL_PATH_HEADERS}
                           ${EXTRA_ARGS}
                           MODULE ${dictname}
                           LINKDEF ${ARG_LINKDEF}
                           OPTIONS ${ARG_OPTIONS}
                           DEPENDENCIES ${ARG_DEPENDS})

  ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_DICTIONARY_TEST ${dictname})

  set(GENERATE_DICTIONARY_TEST ${GENERATE_DICTIONARY_TEST}-build)

  set(targetname_libgen ${dictname}libgen)

  add_library(${targetname_libgen} EXCLUDE_FROM_ALL SHARED ${dictname}.cxx)
  set_property(TARGET ${executable} PROPERTY BUILD_WITH_INSTALL_RPATH OFF) # will never be installed anyway

  if(ARG_SOURCES)
    target_sources(${targetname_libgen} PUBLIC ${ARG_SOURCES})
  endif()

  if(ARG_COMPILE_OPTIONS)
    target_compile_options(${targetname_libgen} PRIVATE ${ARG_COMPILE_OPTIONS})
  endif()

  set_target_properties(${targetname_libgen} PROPERTIES ${ROOT_LIBRARY_PROPERTIES})
  set_target_properties(${targetname_libgen} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

  target_link_libraries(${targetname_libgen} ${ROOT_LIBRARIES})

  set_target_properties(${targetname_libgen} PROPERTIES PREFIX "")

  set_target_properties(${targetname_libgen} PROPERTIES OUTPUT_NAME ${dictname})

  set_property(TARGET ${targetname_libgen} APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

  add_dependencies(${targetname_libgen} ${dictname})

  # We use the /fast variant of targetname_libgen, so we won't automatically
  # build dependencies. Still, the dictname target is a clear dependency (see
  # line above), so we have to explicilty build it too.
  add_test(NAME ${GENERATE_DICTIONARY_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    ${build_config}
                                    --target ${dictname}${fast} ${targetname_libgen}${fast}
                                    -- ${always-make})

  set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})
  if(CMAKE_GENERATOR MATCHES Ninja AND NOT MSVC)
    set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if (ARG_FIXTURES_SETUP)
    set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY
      FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()

  if (ARG_FIXTURES_CLEANUP)
    set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY
      FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY
      FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()

  if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
    add_custom_command(TARGET ${targetname_libgen} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${dictname}_rdict.pcm
                                       ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${dictname}_rdict.pcm
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${dictname}.dll
                                       ${CMAKE_CURRENT_BINARY_DIR}/${dictname}.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${dictname}.lib
                                       ${CMAKE_CURRENT_BINARY_DIR}/${dictname}.lib)
  endif()

endmacro(ROOTTEST_GENERATE_DICTIONARY)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_GENERATE_REFLEX_DICTIONARY(<targetname> <dictionary>
#                                              [SELECTION sel...]
#                                              [headerfiles...]
#                                              [LIBNAME lib...]
#                                              [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...]
#                                              [LIBRARIES lib1 lib2 ...]
#                                              [OPTIONS opt1 opt2 ...])
#
# This macro generates a reflexion dictionary and creates a shared library.
# A test that performs the dictionary generation is created.  The target name of
# the created test is stored in the variable GENERATE_REFLEX_TEST which can
# be accessed by the calling CMakeLists.txt in order to manage dependencies.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_REFLEX_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "NO_ROOTMAP" "SELECTION;LIBNAME;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED" "LIBRARIES;OPTIONS;COMPILE_OPTIONS" ${ARGN})

  set(CMAKE_ROOTTEST_DICT ON)

  if(ARG_NO_ROOTMAP)
    set(CMAKE_ROOTTEST_NOROOTMAP ON)
  else()
    set(CMAKE_ROOTTEST_NOROOTMAP OFF)
  endif()

  set(ROOT_genreflex_cmd ${ROOT_BINDIR}/genreflex)

  ROOTTEST_TARGETNAME_FROM_FILE(targetname ${dictionary})

  set(targetname_libgen ${targetname}-libgen)

  # targetname_dictgen is the targetname constructed by the
  # REFLEX_GENERATE_DICTIONARY macro and is used as a dependency.
  set(targetname_dictgen ${targetname}-dictgen)

  if(ARG_OPTIONS)
    set(reflex_pass_options OPTIONS ${ARG_OPTIONS})
  endif()

  REFLEX_GENERATE_DICTIONARY(${dictionary} ${ARG_UNPARSED_ARGUMENTS}
                             SELECTION ${ARG_SELECTION}
                             ${reflex_pass_options})

  add_library(${targetname_libgen} EXCLUDE_FROM_ALL SHARED ${dictionary}.cxx)
  set_target_properties(${targetname_libgen} PROPERTIES  ${ROOT_LIBRARY_PROPERTIES} )
  set_property(TARGET ${executable} PROPERTY BUILD_WITH_INSTALL_RPATH OFF) # will never be installed anyway
  set_target_properties(${targetname_libgen} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

  if(ARG_LIBNAME)
    set_target_properties(${targetname_libgen} PROPERTIES PREFIX "")
    set_property(TARGET ${targetname_libgen}
                 PROPERTY OUTPUT_NAME ${ARG_LIBNAME})
  else()
    set_property(TARGET ${targetname_libgen}
                 PROPERTY OUTPUT_NAME ${dictionary}_dictrflx)
  endif()

  if(ARG_COMPILE_OPTIONS)
    target_compile_options(${targetname_libgen} PRIVATE ${ARG_COMPILE_OPTIONS})
  endif()

  add_dependencies(${targetname_libgen}
                   ${targetname_dictgen})

  target_link_libraries(${targetname_libgen}
                        ${ARG_LIBRARIES}
                        ${ROOT_LIBRARIES})

  set_property(TARGET ${targetname_libgen}
               APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

  set(GENERATE_REFLEX_TEST ${targetname_libgen}-build)

  # We use the /fast variant of targetname_libgen, so we won't automatically
  # build dependencies. Still, the targetname_dictgen is a clear dependency
  # (see line above), so we have to explicilty build it too.
  add_test(NAME ${GENERATE_REFLEX_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    ${build_config}
                                    --target ${targetname_dictgen}${fast} ${targetname_libgen}${fast}
                                    -- ${always-make})

  set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})
  if(CMAKE_GENERATOR MATCHES Ninja AND NOT MSVC)
    set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if (ARG_FIXTURES_SETUP)
    set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY
      FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()

  if (ARG_FIXTURES_CLEANUP)
    set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY
      FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY
      FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()

  if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
    if(ARG_LIBNAME)
      add_custom_command(TARGET ${targetname_libgen} POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${ARG_LIBNAME}.dll
                                          ${CMAKE_CURRENT_BINARY_DIR}/${ARG_LIBNAME}.dll)
    else()
      add_custom_command(TARGET ${targetname_libgen} POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/lib${dictionary}_dictrflx.dll
                                          ${CMAKE_CURRENT_BINARY_DIR}/lib${dictionary}_dictrflx.dll)
    endif()
  endif()

endmacro(ROOTTEST_GENERATE_REFLEX_DICTIONARY)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_GENERATE_EXECUTABLE(<executable>
#                                    [LIBRARIES lib1 lib2 ...]
#                                    [COMPILE_FLAGS flag1 flag2 ...]
#                                    [DEPENDS ...]
#                                    [RESOURCE_LOCK lock]
#                                    [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...])
# This macro generates an executable the the building of it becames a test
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_EXECUTABLE executable)
  CMAKE_PARSE_ARGUMENTS(ARG "" "RESOURCE_LOCK" "LIBRARIES;COMPILE_FLAGS;DEPENDS;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED" ${ARGN})

  add_executable(${executable} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  set_target_properties(${executable} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TARGET ${executable} PROPERTY BUILD_WITH_INSTALL_RPATH OFF) # will never be installed anyway

  set_property(TARGET ${executable}
               APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

  if(ARG_DEPENDS)
    add_dependencies(${executable} ${ARG_DEPENDS})
  endif()

  if(ARG_LIBRARIES)
    if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
      foreach(library ${ARG_LIBRARIES})
        if("${library}" MATCHES "ROOT::")
          string(REPLACE "ROOT::" "" library ${library})
        endif()
        if(${library} MATCHES "[::]")
          set(libraries ${libraries} ${library})
        elseif(NOT ${library} MATCHES "^lib" AND NOT ${library} MATCHES "^gtest" AND NOT ${library} MATCHES "^gmock")
          set(libraries ${libraries} lib${library})
        else()
          set(libraries ${libraries} ${library})
        endif()
      endforeach()
      target_link_libraries(${executable} ${libraries})
    else()
      target_link_libraries(${executable} ${ARG_LIBRARIES})
    endif()
  endif()
  if(MSVC AND DEFINED ROOT_SOURCE_DIR)
    if(TARGET ROOTStaticSanitizerConfig)
      target_link_libraries(${executable} ROOTStaticSanitizerConfig)
    endif()
  else()
    if(TARGET ROOT::ROOTStaticSanitizerConfig)
      target_link_libraries(${executable} ROOT::ROOTStaticSanitizerConfig)
    endif()
  endif()

  if(ARG_COMPILE_FLAGS)
    set_target_properties(${executable} PROPERTIES COMPILE_FLAGS ${ARG_COMPILE_FLAGS})
  endif()

  ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_EXECUTABLE_TEST ${executable})

  set(GENERATE_EXECUTABLE_TEST ${GENERATE_EXECUTABLE_TEST}-build)

  add_test(NAME ${GENERATE_EXECUTABLE_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    ${build_config}
                                    --target ${executable}${fast}
                                    -- ${always-make})
  set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})

  #- provided fixtures and resource lock are set here
  if (ARG_FIXTURES_SETUP)
    set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY
      FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()

  if (ARG_FIXTURES_CLEANUP)
    set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY
      FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY
      FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()

  if (ARG_RESOURCE_LOCK)
    set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY
      RESOURCE_LOCK ${ARG_RESOURCE_LOCK})
  endif()

  if(CMAKE_GENERATOR MATCHES Ninja AND NOT MSVC)
    set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
    add_custom_command(TARGET ${executable} POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${executable}.exe
                                        ${CMAKE_CURRENT_BINARY_DIR}/${executable}.exe)
  endif()

endmacro()

#-------------------------------------------------------------------------------
#
# function ROOTTEST_ADD_OLDTEST()
#
# This function defines a single tests in the current directory that calls the legacy
# make system to run the defined tests.
#
#-------------------------------------------------------------------------------

find_program(ROOT_GMAKE_PROGRAM gmake)
if (${ROOT_GMAKE_PROGRAM} MATCHES NOTFOUND)
  set(ROOT_GMAKE_PROGRAM make)
endif()

function(ROOTTEST_ADD_OLDTEST)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LABELS;TIMEOUT" ${ARGN})

  ROOTTEST_ADD_TEST( make
                     COMMAND ${ROOT_GMAKE_PROGRAM} cleantest ${ROOTTEST_PARALLEL_MAKE}
                     WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                     DEPENDS roottest-root-io-event
                     FIXTURES_REQUIRED UtilsLibraryBuild
                     LABELS ${ARG_LABELS} TIMEOUT ${ARG_TIMEOUT})
  if(MSVC)
    ROOTTEST_TARGETNAME_FROM_FILE(testprefix .)
    set(fulltestname "${testprefix}-make")
    set_property(TEST ${fulltestname} PROPERTY DISABLED true)
  endif()
endfunction()

#-------------------------------------------------------------------------------
# macro ROOTTEST_SETUP_MACROTEST()
#
# A helper macro to define the command to run a ROOT macro (.C, .C+ or .py)
#-------------------------------------------------------------------------------
macro(ROOTTEST_SETUP_MACROTEST)

  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    if(d MATCHES "_WIN32" OR d MATCHES "_XKEYCHECK_H" OR d MATCHES "NOMINMAX")
      continue()
    endif()
    list(APPEND RootExeDefines "-e;#define ${d}")
  endforeach()

  set(root_cmd ${ROOT_root_CMD} ${RootExeDefines}
               -e "gSystem->SetBuildDir(\"${CMAKE_CURRENT_BINARY_DIR}\",true)"
               -e "gSystem->AddDynamicPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
               -e "gROOT->SetMacroPath(\"${CMAKE_CURRENT_SOURCE_DIR}\")"
               -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")"
               -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")"
               ${ARG_ROOTEXE_OPTS}
               -q -l -b)

  set(root_buildcmd ${ROOT_root_CMD} ${RootExeDefines} -q -l -b)

  # Compile macro, then add to CTest.
  if(ARG_MACRO MATCHES "[.]C\\+" OR ARG_MACRO MATCHES "[.]cxx\\+" OR ARG_MACRO MATCHES "[.]cpp\\+" OR ARG_MACRO MATCHES "[.]cc\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${compile_name} REALPATH)

    if(DEFINED ARG_MACROARG)
      set(command ${root_cmd} "${realfp}+(${ARG_MACROARG})")
    else()
      set(command ${root_cmd} "${realfp}+")
    endif()

  # Add interpreted macro to CTest.
  elseif(ARG_MACRO MATCHES "[.]C" OR ARG_MACRO MATCHES "[.]cxx" OR ARG_MACRO MATCHES "[.]cpp" OR ARG_MACRO MATCHES "[.]cc")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)
    if(DEFINED ARG_MACROARG)
      set(realfp "${realfp}(${ARG_MACROARG})")
    endif()

    set(command ${root_cmd} ${realfp})

  # Add python script to CTest.
  elseif(ARG_MACRO MATCHES "[.]py")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)
    set(command ${Python3_EXECUTABLE} ${realfp} ${PYROOT_EXTRAFLAGS})

  elseif(DEFINED ARG_MACRO)
    set(command ${root_cmd} ${ARG_MACRO})
  endif()

  # Check for assert prefix -- only log stderr.
  if(ARG_MACRO MATCHES "^assert")
    set(checkstdout "")
    set(checkstderr CHECKERR)
  else()
    set(checkstdout CHECKOUT)
    set(checkstderr CHECKERR)
  endif()

endmacro(ROOTTEST_SETUP_MACROTEST)

#-------------------------------------------------------------------------------
# macro ROOTTEST_SETUP_EXECTEST()
#
# A helper macro to define the command to run an executable
#-------------------------------------------------------------------------------
macro(ROOTTEST_SETUP_EXECTEST)

  find_program(realexec ${ARG_EXEC}
               HINTS $ENV{PATH}
               PATH ${CMAKE_CURRENT_BINARY_DIR}
               PATH ${CMAKE_CURRENT_SOURCE_DIR})

  # If no program was found, take it as is.
  if(NOT realexec)
    set(realexec ${ARG_EXEC})
  endif()

  if(MSVC)
    if(${realexec} MATCHES "[.]py" AND NOT ${realexec} MATCHES "[.]exe")
      set(realexec ${Python3_EXECUTABLE} ${realexec})
    else()
      set(realexec ${realexec})
    endif()
  endif()

  set(command ${realexec})

  unset(realexec CACHE)

  set(checkstdout CHECKOUT)
  set(checkstderr CHECKERR)

endmacro(ROOTTEST_SETUP_EXECTEST)

#-------------------------------------------------------------------------------
#
# function ROOTTEST_ADD_TEST(testname
#                            MACRO|EXEC macro_or_command
#                            [MACROARG args1 arg2 ...]
#                            [ROOTEXE_OPTS opt1 opt2 ...]
#                            [INPUT infile]
#                            [ENABLE_IF root-feature]
#                            [DISABLE_IF root-feature]
#                            [WILLFAIL]
#                            [OUTREF stdout_reference]
#                            [ERRREF stderr_reference]
#                            [WORKING_DIR dir]
#                            [TIMEOUT tmout]
#                            [RESOURCE_LOCK lock]
#                            [FIXTURES_SETUP ...] [FIXTURES_CLEANUP ...] [FIXTURES_REQUIRED ...]
#                            [COPY_TO_BUILDDIR file1 file2 ...])
#                            [ENVIRONMENT ENV_VAR1=value1;ENV_VAR2=value2; ...]
#                            [PROPERTIES prop1 value1 prop2 value2...]
#                           )
#
# This function defines a roottest test. It adds a number of additional
# options on top of the ROOT defined ROOT_ADD_TEST.
#
#-------------------------------------------------------------------------------
function(ROOTTEST_ADD_TEST testname)
  CMAKE_PARSE_ARGUMENTS(ARG "WILLFAIL;RUN_SERIAL;STOREOUT"
                            "OUTREF;ERRREF;OUTREF_CINTSPECIFIC;OUTCNV;PASSRC;MACROARG;WORKING_DIR;INPUT;ENABLE_IF;DISABLE_IF;TIMEOUT;RESOURCE_LOCK"
                            "TESTOWNER;COPY_TO_BUILDDIR;MACRO;ROOTEXE_OPTS;EXEC;COMMAND;PRECMD;POSTCMD;OUTCNVCMD;FAILREGEX;PASSREGEX;DEPENDS;OPTS;LABELS;ENVIRONMENT;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED;PROPERTIES;PYTHON_DEPS"
                            ${ARGN})

  # Test name
  ROOTTEST_TARGETNAME_FROM_FILE(testprefix .)
  if(testname MATCHES "^roottest-")
    set(fulltestname ${testname})
  else()
    set(fulltestname ${testprefix}-${testname})
  endif()

  if (ARG_ENABLE_IF OR ARG_DISABLE_IF)
    # Turn the output into a cmake list which is easier to work with.
    set(ROOT_ENABLED_FEATURES ${_root_enabled_options})
    set(ROOT_ALL_FEATURES ${_root_all_options})
    if ("${ARG_ENABLE_IF}" STREQUAL "" AND "${ARG_DISABLE_IF}" STREQUAL "")
      message(FATAL_ERROR "ENABLE_IF/DISABLE_IF switch requires a feature.")
    endif()
    if(ARG_ENABLE_IF)
      if(NOT "${ARG_ENABLE_IF}" IN_LIST ROOT_ENABLED_FEATURES)
        list(APPEND CTEST_CUSTOM_TESTS_IGNORE ${fulltestname})
        return()
      endif()
      if(NOT "${ARG_ENABLE_IF}" IN_LIST ROOT_ALL_FEATURES)
        message(FATAL_ERROR "Specified feature ${ARG_ENABLE_IF} not found.")
      endif()
    elseif(ARG_DISABLE_IF)
      if("${ARG_DISABLE_IF}" IN_LIST ROOT_ENABLED_FEATURES)
        list(APPEND CTEST_CUSTOM_TESTS_IGNORE ${fulltestname})
        return()
      endif()
      if(NOT "${ARG_DISABLE_IF}" IN_LIST ROOT_ALL_FEATURES)
        message(FATAL_ERROR "Specified feature ${ARG_DISABLE_IF} not found.")
      endif()
    endif()
  endif()

  # Setup macro test.
  if(ARG_MACRO)
   ROOTTEST_SETUP_MACROTEST()
  endif()

  # Setup executable test.
  if(ARG_EXEC)
    ROOTTEST_SETUP_EXECTEST()
  endif()

  if(ARG_COMMAND)
    set(command ${ARG_COMMAND})
    if(ARG_OUTREF)
      set(checkstdout CHECKOUT)
      set(checkstderr CHECKERR)
    endif()
  endif()
  if(ARG_STOREOUT)
    set(checkstdout CHECKOUT)
    set(checkstderr CHECKERR)
  endif()

  # Reference output given?
  if(ARG_OUTREF_CINTSPECIFIC)
    set(ARG_OUTREF ${ARG_OUTREF_CINTSPECIFIC})
  endif()

  if(ARG_OUTREF)
    get_filename_component(OUTREF_PATH ${ARG_OUTREF} ABSOLUTE)

    if(DEFINED 64BIT)
      set(ROOTBITS 64)
    elseif(DEFINED 32BIT)
      set(ROOTBITS 32)
    else()
      set(ROOTBITS "")
    endif()

    if(ARG_OUTREF_CINTSPECIFIC)
      if(EXISTS ${OUTREF_PATH}${ROOTBITS}-${CINT_VERSION})
        set(OUTREF_PATH ${OUTREF_PATH}${ROOTBITS}-${CINT_VERSION})
      elseif(EXISTS ${OUTREF_PATH}-${CINT_VERSION})
        set(OUTREF_PATH ${OUTREF_PATH}-${CINT_VERSION})
      elseif(EXISTS ${OUTREF_PATH}${ROOTBITS})
        set(OUTREF_PATH ${OUTREF_PATH}${ROOTBITS})
      endif()
    else()
      if(EXISTS ${OUTREF_PATH}${ROOTBITS})
        set(OUTREF_PATH ${OUTREF_PATH}${ROOTBITS})
      endif()
    endif()
    set(outref OUTREF ${OUTREF_PATH})
  endif()

  if(ARG_ERRREF)
    get_filename_component(ERRREF_PATH ${ARG_ERRREF} ABSOLUTE)
    set(errref ERRREF ${ERRREF_PATH})
  endif()

  # Get the real path to the output conversion script.
  if(ARG_OUTCNV)
    get_filename_component(OUTCNV ${ARG_OUTCNV} ABSOLUTE)
    set(outcnv OUTCNV ${OUTCNV})
  endif()

  # Setup the output conversion command.
  if(ARG_OUTCNVCMD)
    set(outcnvcmd OUTCNVCMD ${ARG_OUTCNVCMD})
  endif()

  # Mark the test as known to fail.
  if(ARG_WILLFAIL)
    set(willfail WILLFAIL)
  endif()

  # List of python packages required to run this test.
  if(ARG_PYTHON_DEPS)
    set(pythondeps ${ARG_PYTHON_DEPS})
  endif()

  # Add ownership and test labels.
  get_property(testowner DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                         PROPERTY ROOTTEST_TEST_OWNER)

  if(ARG_TESTOWNER)
    set(testowner ${ARG_TESTOWNER})
  endif()

  if(ARG_LABELS)
    set(labels LABELS ${ARG_LABELS})
    if(testowner)
      set(labels ${labels} ${testowner})
    endif()
  else()
    if(testowner)
      set(labels LABELS ${testowner})
    endif()
  endif()

  # Test will pass for a custom return value.
  if(ARG_PASSRC)
    set(passrc PASSRC ${ARG_PASSRC})
  endif()

  # Pass options to the command.
  if(ARG_OPTS)
    set(command ${command} ${ARG_OPTS})
  endif()

  # Execute a custom command before executing the test.
  if(ARG_PRECMD)
    set(precmd PRECMD ${ARG_PRECMD})
  endif()

  # Copy files into the build directory first.
  if(ARG_COPY_TO_BUILDDIR)
    foreach(copyfile ${ARG_COPY_TO_BUILDDIR})
      get_filename_component(absfilep ${copyfile} ABSOLUTE)
      set(copy_files ${copy_files} ${absfilep})
    endforeach()
    set(copy_to_builddir COPY_TO_BUILDDIR ${copy_files})
  endif()

  # Execute a custom command after executing the test.
  if(ARG_POSTCMD)
    set(postcmd POSTCMD ${ARG_POSTCMD})
  endif()

  if(MSVC)
    if(ARG_MACRO)
      if(ARG_MACRO MATCHES "[.]C\\+" OR ARG_MACRO MATCHES "[.]cxx\\+")
        string(REPLACE "+" "" macro_name "${ARG_MACRO}")
        get_filename_component(fpath ${macro_name} REALPATH)
        get_filename_component(fext ${fpath} EXT)
        string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} fpath ${fpath})
        string(REPLACE ${fext} "" fpath ${fpath})
        string(REPLACE "." "" fext ${fext})
        cmake_path(CONVERT "${fpath}" TO_NATIVE_PATH_LIST fpath)
        set(postcmd POSTCMD cmd /c if exist ${fpath}_${fext}.rootmap del ${fpath}_${fext}.rootmap)
      endif()
    endif()
  endif()

  # Add dependencies. If the test depends on a macro file, the macro
  # will be compiled and the dependencies are set accordingly.
  if(ARG_DEPENDS)
    foreach(dep ${ARG_DEPENDS})
      if(${dep} MATCHES "[.]C" OR ${dep} MATCHES "[.]cxx" OR ${dep} MATCHES "[.]h")
        ROOTTEST_COMPILE_MACRO(${dep})
        list(APPEND deplist ${COMPILE_MACRO_TEST})
      elseif(NOT ${dep} MATCHES "^roottest-")
        list(APPEND deplist ${testprefix}-${dep})
      else()
        list(APPEND deplist ${dep})
      endif()
    endforeach()
  endif(ARG_DEPENDS)

  if(ARG_FAILREGEX)
    set(failregex FAILREGEX ${ARG_FAILREGEX})
  endif()

  if(ARG_PASSREGEX)
    set(passregex PASSREGEX ${ARG_PASSREGEX})
  endif()

  if(ARG_RUN_SERIAL)
    set(run_serial RUN_SERIAL ${ARG_RUN_SERIAL})
  endif()

  if(MSVC)
    set(environment ENVIRONMENT
                    ${ROOTTEST_ENV_EXTRA}
                    ${ARG_ENVIRONMENT}
                    ROOTSYS=${ROOTSYS}
                    PYTHONPATH=${ROOTTEST_ENV_PYTHONPATH})
  else()
    string(REPLACE ";" ":" _path "${ROOTTEST_ENV_PATH}")
    string(REPLACE ";" ":" _pythonpath "${ROOTTEST_ENV_PYTHONPATH}")
    string(REPLACE ";" ":" _librarypath "${ROOTTEST_ENV_LIBRARYPATH}")


    set(environment ENVIRONMENT
                    ${ROOTTEST_ENV_EXTRA}
                    ${ARG_ENVIRONMENT}
                    ROOTSYS=${ROOTSYS}
                    PATH=${_path}:$ENV{PATH}
                    PYTHONPATH=${_pythonpath}:$ENV{PYTHONPATH}
                    ${ld_library_path}=${_librarypath}:$ENV{${ld_library_path}})
  endif()

  if(ARG_WORKING_DIR)
    get_filename_component(test_working_dir ${ARG_WORKING_DIR} ABSOLUTE)
  else()
    get_filename_component(test_working_dir ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
  endif()

  get_filename_component(logfile "${CMAKE_CURRENT_BINARY_DIR}/${testname}.log" ABSOLUTE)
  if(ARG_ERRREF)
    get_filename_component(errfile "${CMAKE_CURRENT_BINARY_DIR}/${testname}.err" ABSOLUTE)
    set(errfile ERROR ${errfile})
  endif()

  if(ARG_INPUT)
    get_filename_component(infile_path ${ARG_INPUT} ABSOLUTE)
    set(infile INPUT ${infile_path})
  endif()

  if(ARG_TIMEOUT)
    set(timeout ${ARG_TIMEOUT})
  else()
    if("${ARG_LABELS}" MATCHES "longtest")
      set(timeout 1800)
    else()
      set(timeout 300)
    endif()
  endif()

  if(TIMEOUT_BINARY AND NOT MSVC)
    # It takes up to 30seconds to get the back trace!
    # And we want the backtrace before CTest sends kill -9.
    math(EXPR timeoutTimeout "${timeout}-30")
    set(command "${TIMEOUT_BINARY}^-s^USR2^${timeoutTimeout}s^${command}")
  endif()

  if (ARG_FIXTURES_SETUP)
    set(fixtures_setup ${ARG_FIXTURES_SETUP})
  endif()

  if (ARG_FIXTURES_CLEANUP)
    set(fixtures_cleanup ${ARG_FIXTURES_CLEANUP})
  endif()

  if (ARG_FIXTURES_REQUIRED)
    set(fixtures_required ${ARG_FIXTURES_REQUIRED})
  endif()

  if (ARG_RESOURCE_LOCK)
    set(resource_lock ${ARG_RESOURCE_LOCK})
  endif()

  if (ARG_PROPERTIES)
    set(properties ${ARG_PROPERTIES})
  endif()

  ROOT_ADD_TEST(${fulltestname} COMMAND ${command}
                        OUTPUT ${logfile}
                        ${infile}
                        ${errfile}
                        ${outcnv}
                        ${outcnvcmd}
                        ${outref}
                        ${errref}
                        WORKING_DIR ${test_working_dir}
                        DIFFCMD ${Python3_EXECUTABLE} ${ROOT_SOURCE_DIR}/roottest/scripts/custom_diff.py
                        TIMEOUT ${timeout}
                        ${environment}
                        ${build}
                        ${checkstdout}
                        ${checkstderr}
                        ${willfail}
                        ${compile_macros}
                        ${labels}
                        ${passrc}
                        ${precmd}
                        ${postcmd}
                        ${run_serial}
                        ${failregex}
                        ${passregex}
                        ${copy_to_builddir}
                        PYTHON_DEPS ${pythondeps}
                        DEPENDS ${deplist}
                        FIXTURES_SETUP ${fixtures_setup}
                        FIXTURES_CLEANUP ${fixtures_cleanup}
                        FIXTURES_REQUIRED ${fixtures_required}
                        RESOURCE_LOCK ${resource_lock}
                        PROPERTIES ${properties})

  if(MSVC)
    if (ARG_OUTCNV OR ARG_OUTCNVCMD)
      set_property(TEST ${fulltestname} PROPERTY DISABLED true)
    endif()
    if(ARG_COMMAND)
      string(FIND "${ARG_COMMAND}" ".sh" APOS)
      if( NOT ("${APOS}" STREQUAL "-1") )
        set_property(TEST ${fulltestname} PROPERTY DISABLED true)
      endif()
      string(FIND "${ARG_COMMAND}" "grep " APOS)
      if( NOT ("${APOS}" STREQUAL "-1") )
        set_property(TEST ${fulltestname} PROPERTY DISABLED true)
      endif()
      string(FIND "${ARG_COMMAND}" "make " APOS)
      if( NOT ("${APOS}" STREQUAL "-1") )
        set_property(TEST ${fulltestname} PROPERTY DISABLED true)
      endif()
    endif()
    if(ARG_PRECMD)
      string(FIND "${ARG_PRECMD}" "sh " APOS)
      if( NOT ("${APOS}" STREQUAL "-1") )
        set_property(TEST ${fulltestname} PROPERTY DISABLED true)
      endif()
      string(FIND "${ARG_PRECMD}" ".sh" APOS)
      if( NOT ("${APOS}" STREQUAL "-1") )
        set_property(TEST ${fulltestname} PROPERTY DISABLED true)
      endif()
    endif()
  endif()

endfunction(ROOTTEST_ADD_TEST)

#-------------------------------------------------------------------------------
#
# function ROOTTEST_ADD_UNITTEST_DIR(libraries...)
#
# This function defines a roottest unit test using Google Test.
# All files in this directory will end up in a unit test binary and run as a
# single test.
#
#-------------------------------------------------------------------------------

function(ROOTTEST_ADD_UNITTEST_DIR)
  CMAKE_PARSE_ARGUMENTS(ARG
    "WILLFAIL"
    ""
    "COPY_TO_BUILDDIR;DEPENDS;OPTS;LABELS;ENVIRONMENT"
    ${ARGN})

  # Test name
  ROOTTEST_TARGETNAME_FROM_FILE(testprefix .)
  set(fulltestname ${testprefix}_unittests)
  set(binary ${testprefix}_exe)
  file(GLOB unittests_SRC
    "*.h"
    "*.hh"
    "*.hpp"
    "*.hxx"
    "*.cpp"
    "*.cxx"
    "*.cc"
    "*.C"
    )

  if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
    foreach(library ${ARG_UNPARSED_ARGUMENTS})
      if(${library} MATCHES "[::]")
        set(libraries ${libraries} ${library})
      else()
        set(libraries ${libraries} lib${library})
      endif()
    endforeach()
  else()
    set (libraries ${ARG_UNPARSED_ARGUMENTS})
  endif()

  add_executable(${binary} ${unittests_SRC})
  target_link_libraries(${binary} PRIVATE GTest::gtest GTest::gtest_main ${libraries})
  set_property(TARGET ${binary} PROPERTY BUILD_WITH_INSTALL_RPATH OFF) # will never be installed anyway

  if(MSVC AND DEFINED ROOT_SOURCE_DIR)
    if(TARGET ROOTStaticSanitizerConfig)
      target_link_libraries(${binary} ROOTStaticSanitizerConfig)
    endif()
  else()
    if(TARGET ROOT::ROOTStaticSanitizerConfig)
      target_link_libraries(${binary} PRIVATE ROOT::ROOTStaticSanitizerConfig)
    endif()
  endif()

  # Mark the test as known to fail.
  if(ARG_WILLFAIL)
    set(willfail WILLFAIL)
  endif()

  if(ARG_LABELS)
    set(labels LABELS ${ARG_LABELS})
    if(testowner)
      set(labels ${labels} ${testowner})
    endif()
  else()
    if(testowner)
      set(labels LABELS ${testowner})
    endif()
  endif()

  # Add ownership and test labels.
  get_property(testowner DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                         PROPERTY ROOTTEST_TEST_OWNER)

  if(ARG_TESTOWNER)
    set(testowner ${ARG_TESTOWNER})
  endif()

  if(ARG_LABELS)
    set(labels LABELS ${ARG_LABELS})
    if(testowner)
      set(labels ${labels} ${testowner})
    endif()
  else()
    if(testowner)
      set(labels LABELS ${testowner})
    endif()
  endif()

  # Copy files into the build directory first.
  if(ARG_COPY_TO_BUILDDIR)
    foreach(copyfile ${ARG_COPY_TO_BUILDDIR})
      get_filename_component(absfilep ${copyfile} ABSOLUTE)
      set(copy_files ${copy_files} ${absfilep})
    endforeach()
    set(copy_to_builddir COPY_TO_BUILDDIR ${copy_files})
  endif()

  # Add dependencies. If the test depends on a macro file, the macro
  # will be compiled and the dependencies are set accordingly.
  if(ARG_DEPENDS)
    foreach(dep ${ARG_DEPENDS})
      if(${dep} MATCHES "[.]C" OR ${dep} MATCHES "[.]cxx" OR ${dep} MATCHES "[.]h")
        ROOTTEST_COMPILE_MACRO(${dep})
        list(APPEND deplist ${COMPILE_MACRO_TEST})
      elseif(NOT ${dep} MATCHES "^roottest-")
        list(APPEND deplist ${testprefix}-${dep})
      else()
        list(APPEND deplist ${dep})
      endif()
    endforeach()
  endif(ARG_DEPENDS)

  if(MSVC)
    set(environment ENVIRONMENT
                    ROOTSYS=${ROOTSYS}
                    PYTHONPATH=${ROOTTEST_ENV_PYTHONPATH})
  else()
    string(REPLACE ";" ":" _path "${ROOTTEST_ENV_PATH}")
    string(REPLACE ";" ":" _pythonpath "${ROOTTEST_ENV_PYTHONPATH}")
    string(REPLACE ";" ":" _librarypath "${ROOTTEST_ENV_LIBRARYPATH}")


    set(environment ENVIRONMENT
                    ${ROOTTEST_ENV_EXTRA}
                    ${ARG_ENVIRONMENT}
                    ROOTSYS=${ROOTSYS}
                    PATH=${_path}:$ENV{PATH}
                    PYTHONPATH=${_pythonpath}:$ENV{PYTHONPATH}
                    ${ld_library_path}=${_librarypath}:$ENV{${ld_library_path}})
  endif()

  ROOT_ADD_TEST(${fulltestname} COMMAND ${binary}
    ${environment}
    ${willfail}
    ${labels}
    ${copy_to_builddir}
    TIMEOUT 600
    DEPENDS ${deplist}
    )
endfunction(ROOTTEST_ADD_UNITTEST_DIR)

#----------------------------------------------------------------------------
# find_python_module(module [REQUIRED] [QUIET])
#----------------------------------------------------------------------------
function(find_python_module module)
   CMAKE_PARSE_ARGUMENTS(ARG "REQUIRED;QUIET" "" "" ${ARGN})
   string(TOUPPER ${module} module_upper)
   if(NOT PY_${module_upper})
      if(ARG_REQUIRED)
         set(py_${module}_FIND_REQUIRED TRUE)
      endif()
      if(ARG_QUIET)
         set(py_${module}_FIND_QUIETLY TRUE)
      endif()
      # A module's location is usually a directory, but for binary modules
      # it's a .so file.
      execute_process(COMMAND "${Python3_EXECUTABLE}" "-c"
         "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
         RESULT_VARIABLE _${module}_status
         OUTPUT_VARIABLE _${module}_location
         ERROR_VARIABLE _${module}_error
         OUTPUT_STRIP_TRAILING_WHITESPACE
         ERROR_STRIP_TRAILING_WHITESPACE)
      if(NOT _${module}_status)
         set(PY_${module_upper} ${_${module}_location} CACHE STRING "Location of Python module ${module}")
         mark_as_advanced(PY_${module_upper})
      else()
         if(NOT ARG_QUIET)
            message(STATUS "Failed to find Python module ${module}: ${_${module}_error}")
          endif()
      endif()
   endif()
   find_package_handle_standard_args(py_${module} DEFAULT_MSG PY_${module_upper})
   set(PY_${module_upper}_FOUND ${PY_${module_upper}_FOUND} PARENT_SCOPE)
endfunction()

#---------------------------------------------------------------------------------------------------
# function ROOTTEST_LINKER_LIBRARY( <name> source1 source2 ...[TYPE STATIC|SHARED] [DLLEXPORT]
#                                   [NOINSTALL] LIBRARIES library1 library2 ...
#                                   DEPENDENCIES dep1 dep2
#                                   BUILTINS dep1 dep2)
#
# this function simply calls the ROOT function ROOT_LINKER_LIBRARY, and add a POST_BUILD custom
# command to copy the .dll and .lib from the standard config directory (Debug/Release) to its
# parent directory (CMAKE_CURRENT_BINARY_DIR) on Windows
#
#---------------------------------------------------------------------------------------------------
function(ROOTTEST_LINKER_LIBRARY library)
   ROOT_LINKER_LIBRARY(${ARGV})
   if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
      add_custom_command(TARGET ${library} POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/lib${library}.dll
                                          ${CMAKE_CURRENT_BINARY_DIR}/lib${library}.dll
         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/lib${library}.lib
                                           ${CMAKE_CURRENT_BINARY_DIR}/lib${library}.lib)
   endif()
endfunction()

#---------------------------------------------------------------------------------------------------
# ROOT_GET_CLANG_LIBRARIES( clang_libraries )
#
# this function is used to collect the required libraries when building ROOT with external
# LLVM & Clang, like in Conda for example.
#---------------------------------------------------------------------------------------------------
function (ROOT_GET_CLANG_LIBRARIES clang_libraries)
  set(found_libraries "")
  FILE(GLOB clangLibs ${LLVM_LIBRARY_DIR}/clang*.lib)
  foreach(lib_path IN LISTS clangLibs)
    get_filename_component(lib_name ${lib_path} NAME)
    if (NOT ${lib_name} IN_LIST found_libraries)
      list(APPEND found_libraries ${lib_name})
    endif()
  endforeach(lib_path)
  foreach(extra_lib "LLVMFrontendDriver.lib" "LLVMFrontendHLSL.lib" "Version.lib")
    if (NOT ${extra_lib} IN_LIST found_libraries)
      list(APPEND found_libraries ${extra_lib})
    endif()
  endforeach(extra_lib)
  SET(${clang_libraries} "${found_libraries}" PARENT_SCOPE)
endfunction(ROOT_GET_CLANG_LIBRARIES)
