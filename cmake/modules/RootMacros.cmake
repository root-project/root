#-------------------------------------------------------------------------------
#
#  RootMacros.cmake
#
#  Macros and definitions regarding ROOT components. 
#
#-------------------------------------------------------------------------------

# Wrap some platform dependencies.
set(lib lib)
set(bin bin)
if(WIN32)
  set(ssuffix .bat)
  set(scomment rem)
  set(libprefix lib)
  set(ld_library_path PATH)
  set(libsuffix .dll)
  set(runtimedir ${CMAKE_INSTALL_BINDIR})
elseif(APPLE)
  set(ld_library_path DYLD_LIBRARY_PATH)
  set(ssuffix .csh)
  set(scomment \#)
  set(libprefix lib)
  set(libsuffix .so)
  set(runtimedir ${CMAKE_INSTALL_LIBDIR})
else()
  set(ld_library_path LD_LIBRARY_PATH)
  set(ssuffix .csh)
  set(scomment \#)
  set(libprefix lib)
  set(libsuffix .so) 
  set(runtimedir ${CMAKE_INSTALL_LIBDIR})
endif()

if(soversion)
  set(ROOT_LIBRARY_PROPERTIES ${ROOT_LIBRARY_PROPERTIES}
      VERSION ${ROOT_VERSION}
      SOVERSION ${ROOT_MAJOR_VERSION}
      SUFFIX ${libsuffix}
      PREFIX ${libprefix} )
else()
  set(ROOT_LIBRARY_PROPERTIES ${ROOT_LIBRARY_PROPERTIES}
      SUFFIX ${libsuffix}
      PREFIX ${libprefix}
      IMPORT_PREFIX ${libprefix} )
endif()

if(APPLE)
  if(gnuinstall)
    set(ROOT_LIBRARY_PROPERTIES ${ROOT_LIBRARY_PROPERTIES}
         INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}"
         BUILD_WITH_INSTALL_RPATH ON)
  else()
    set(ROOT_LIBRARY_PROPERTIES ${ROOT_LIBRARY_PROPERTIES}
         INSTALL_NAME_DIR "@rpath"
         BUILD_WITH_INSTALL_RPATH ON)
  endif()
endif()

#------------------------------------------------------------------------------
#
# macro ROOT_CHECK_OUT_OF_SOURCE_BUILD
#
# Ensures that the project is built out-of-source.
#
#------------------------------------------------------------------------------
macro(ROOT_CHECK_OUT_OF_SOURCE_BUILD)
 string(COMPARE EQUAL ${ROOTTEST_DIR} ${CMAKE_BINARY_DIR} insource)
  if(insource)
     file(REMOVE_RECURSE ${ROOTTEST_DIR}/Testing)
     file(REMOVE ${ROOTTEST_DIR}/DartConfiguration.tcl)

     message(FATAL_ERROR "ROOT should be installed as an out of source build,"
                         " to keep the source directory clean. Please create "
                         "a extra build directory and run the command 'cmake "
                         "<path_to_source_dir>' in this newly created "
                         "directory. You have also to delete the directory "
                         "CMakeFiles and the file CMakeCache.txt in the source "
                         "directory. Otherwise cmake will complain even if you "
                         "run it from an out-of-source directory.") 
  endif()
endmacro()

#-------------------------------------------------------------------------------
#
# function ROOT_COMPILE_MACRO( <filename> [BUILDOBJ object] [BUILDLIB lib] )
#
# This function compiles and loads a shared library containing
# the code from the file <filename>.
#
#-------------------------------------------------------------------------------
function(ROOT_COMPILE_MACRO filename)
  CMAKE_PARSE_ARGUMENTS(ARG "" "BUILDOBJ;BUILDLIB;TARGETNAME;DEPENDS" ""  ${ARGN}) 

  # Add defines to root_cmd, in order to have out-of-source builds
  # when using the scripts/build.C macro.
  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs} )
    list(APPEND RootExeDefines "-e;#define ${d}")
  endforeach()

  set(RootClingDefines
        -e "#define CMakeEnvironment"
        -e "#define CMakeBuildDir \"${CMAKE_CURRENT_BINARY_DIR}\""
        ${RootExeDefines})
  
  set(root_cmd root.exe ${RootClingDefines} -q -l -b) 

  set(_cwd WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  get_filename_component(realfp ${filename} REALPATH)

  set(command ${root_cmd}
              ${ROOTTEST_DIR}/scripts/build.C\(\"${realfp}\",\"${ARG_BUILDLIB}\",\"${ARG_BUILDOBJ}\"\) ${_cwd})

  message("-- Add target to compile macro ${filename}")

  if(ARG_DEPENDS)
    set(deps ${ARG_DEPENDS})
  endif()

  string(REPLACE "/" "-" srcpath "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target("${srcpath}-${filename}-compile-macro" ALL COMMAND ${command} ${_cwd} ${deps} VERBATIM)

endfunction(ROOT_COMPILE_MACRO)

function(ROOTTEST_GENERATE_DICTIONARY dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LINKDEF;DEPENDENCIES" ${ARGN})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(CMAKE_ROOTTEST_DICT ON)
  message("generate dictionary ${dictname}")
  
  ROOT_GENERATE_DICTIONARY(${dictname} ${ARG_UNPARSED_ARGUMENTS} MODULE ${dictname} LINKDEF ${ARG_LINKDEF} DEPENDENCIES ${ARG_DEPENDENCIES}) 
endfunction()

#-------------------------------------------------------------------------------
#
# function ROOT_REFLEX_GENERATE_DICTIONARY( <targetname> [SELECTION sel...] [headerfiles...])
#
# This function generates a reflexion dictionary. 
#
#-------------------------------------------------------------------------------
function(ROOT_REFLEX_GENERATE_DICTIONARY targetname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "SELECTION" ""  ${ARGN})

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

  # Add defines to root_cmd, in order to have out-of-source builds
  # when using the scripts/build.C macro.
  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    list(APPEND GenreflexDefines "-D${d}")
  endforeach()

  set(include_dirs -I${CMAKE_CURRENT_SOURCE_DIR})
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  foreach(d ${incdirs})
   set(include_dirs ${include_dirs} -I${d})
  endforeach()

  set(genreflex_cmd genreflex ${headerfiles} --select=${selectionfile} ${include_dirs} ${GenreflexDefines}) 
  message("genreflex_cmd: ${genreflex_cmd}")

  set(_cwd WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  message("-- Add target to generate reflex dictionary ${targetname}")

  add_custom_target("${targetname}" ALL COMMAND ${genreflex_cmd} ${_cwd} VERBATIM)

endfunction(ROOT_REFLEX_GENERATE_DICTIONARY)

#-------------------------------------------------------------------------------
#
# function ROOT_BUILD_DICT( <dictname> [FILES files...]
#
# Build a simple ROOT dictionary <dictname> from the input <files>.
#
#-------------------------------------------------------------------------------
function(ROOT_BUILD_DICT dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "FILES" ${ARGN}) 

  foreach(f ${ARG_FILES})
    get_filename_component(realpath ${f} REALPATH)
    set(buildfiles ${buildfiles} "${realpath}")
  endforeach()

  set(command ${rootcint_program} -f ${dictname} -c ${buildfiles})

  message("-- Add target to build simple dictionary: ${dictname}")
  add_custom_target("${dictname}-build-dict" ALL COMMAND ${command} ${_cwd} VERBATIM)

endfunction()

#-------------------------------------------------------------------------------
#
# function ROOT_BUILD_COMPILE_DICT( <dictname> [FILES files...]
#
# Build a ROOT dictionary <dictname> and compile it into a shared library
# from the input <files>.
#
#-------------------------------------------------------------------------------
function(ROOT_BUILD_COMPILE_DICT dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "BUILDOBJ;BUILDLIB" "FILES" ${ARGN}) 

  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs} )
    list(APPEND RootExeDefines "-e;#define ${d}")
  endforeach()

  # Setup root.exe command for batch mode.
  set(root_cmd root.exe ${RootExeDefines} -q -l -b) 

  set(_cwd WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  foreach(f ${ARG_FILES})
    get_filename_component(realpath ${f} REALPATH)
    set(buildfiles ${buildfiles} "${realpath}")
  endforeach()

  set(cintcmd ${rootcint_program} -f ${ARG_BUILDOBJ} -c ${buildfiles})

  if(ARG_BUILDLIB)
    get_filename_component(_buildlib ${ARG_BUILDLIB} REALPATH)
  else()
    set(_buildlib "")
  endif()

  if(ARG_BUILDOBJ)
    get_filename_component(_buildobj ${ARG_BUILDOBJ} REALPATH)
  else()
    set(_buildobj "")
  endif()

  STRING(REGEX REPLACE " " ";" _flags ${CMAKE_CXX_FLAGS})
  set(_flags ${_flags} "-std=c++11;-I${ROOT_INCLUDE_DIR}")
  set(createobj_cmd  ${CMAKE_CXX_COMPILER} ${_flags} -c ${ARG_BUILDOBJ} -o ${ARG_BUILDOBJ}.o)

  string(RANDOM _rdm)
  set(_tmp "tmp${_rdm}")

  set(command ${root_cmd} '${ROOTTEST_DIR}/scripts/build.C\(\"${_tmp}.C\",\"${_buildlib}\",\"${ARG_BUILDOBJ}.o\"\)')

  add_custom_command(
    OUTPUT ${dictname}${libsuffix}
    COMMAND ${cintcmd}
    COMMAND ${createobj_cmd}
    COMMAND touch ${_tmp}.C ${_cwd}
    COMMAND ${command}
    COMMAND mv ${_tmp}_C${libsuffix} ${dictname}${libsuffix}
    ${_cwd})
    
  string(REPLACE "/" "-" srcpath "${CMAKE_CURRENT_SOURCE_DIR}")
  message("-- Add target ${dictname}-build-dict-sl")
  add_custom_target(${srcpath}-${dictname}-dict ALL DEPENDS ${dictname}${libsuffix})

endfunction()

function(ROOT_BUILD_REFLEX_LIBRARY libname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "BUILDOBJ;BUILDLIB" ${ARGN})

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

  # Add defines to root_cmd, in order to have out-of-source builds
  # when using the scripts/build.C macro.
  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    list(APPEND GenreflexDefines "-D${d}")
  endforeach()

  set(include_dirs -I${CMAKE_CURRENT_SOURCE_DIR})
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  foreach(d ${incdirs})
   set(include_dirs ${include_dirs} -I${d})
  endforeach()

  set(genreflex_cmd genreflex ${headerfiles} --select=${selectionfile} ${include_dirs} ${GenreflexDefines}) 

  set(_cwd WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  message("-- Add target to generate reflex dictionary ${filename}")
  execute_process(COMMAND ${genreflex_cmd} ${_cwd} RESULT_VARIABLE rc_code)

  if(rc_code)
    message(FATAL_ERROR "error code: ${rc_code}")
  endif()

  # Setup root.exe command for batch mode.
  set(root_cmd root.exe ${ReflexDefines} -q -l -b) 

  execute_process(COMMAND ${command} ${_cwd} RESULT_VARIABLE rc_code)

  if(rc_code)
    message(FATAL_ERROR "error code: ${rc_code}")
  endif()

  STRING(REGEX REPLACE " " ";" _flags ${CMAKE_CXX_FLAGS})
  set(_flags ${_flags} "-std=c++11;-I${ROOT_INCLUDE_DIR}")
  set(createobj_cmd  ${CMAKE_CXX_COMPILER} ${_flags} -c ${ARG_BUILDOBJ} -o ${ARG_BUILDOBJ}.o)
  execute_process(COMMAND ${createobj_cmd} ${_cwd} RESULT_VARIABLE rc_code)

  if(rc_code)
    message(FATAL_ERROR "error code: ${rc_code}")
  endif()

  string(RANDOM _rdm)
  set(_tmp "tmp${_rdm}")
  execute_process(COMMAND touch ${_tmp}.C ${_cwd})

  set(command ${root_cmd} ${ROOTTEST_DIR}/scripts/build.C\(\"${_tmp}.C\",\"${_buildlib}\",\"${ARG_BUILDOBJ}.o\"\) ${_cwd})

  execute_process(COMMAND ${command} ${_cwd} RESULT_VARIABLE rc_code)
  execute_process(COMMAND mv ${_tmp}_C${libsuffix} ${dictname}${libsuffix} ${_cwd})

  if(rc_code)
    message(FATAL_ERROR "error code: ${rc_code}")
  endif()
    
endfunction()
