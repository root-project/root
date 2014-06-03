#-------------------------------------------------------------------------------
#
#  RootMacros.cmake
#
#  Macros and definitions regarding ROOT components.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# function ROOTTEEST_TARGETNAME_FROM_FILE(<resultvar> <filename>)
#
# Construct a target name for a given file <filename> and store its name into
# <resultvar>. The target name is of the form:
#
#   roottest-<directorypath>-<filename_WE> 
#
#-------------------------------------------------------------------------------
function(ROOTTEST_TARGETNAME_FROM_FILE resultvar filename)

  get_filename_component(realfp ${filename} REALPATH)
  get_filename_component(filename_we ${filename} NAME_WE)

  string(REPLACE "${ROOTTEST_DIR}" "" relativepath ${realfp}) 
  string(REPLACE "${filename}"     "" relativepath ${relativepath})

  string(REPLACE "/" "-" targetname ${relativepath}${filename_we})
  set(${resultvar} "roottest${targetname}" PARENT_SCOPE)

endfunction(ROOTTEST_TARGETNAME_FROM_FILE)

#-------------------------------------------------------------------------------
#
# function ROOTTEST_ADD_AUTOMACROS(DEPENDS [dependencies ...])
#
# Automatically adds all macros in the current source directory to the list of
# tests that follow the naming scheme:
#
#   run*.C, run*.cxx, assert*.C, assert*.cxx, exec*.C, exec*.cxx
#
#-------------------------------------------------------------------------------
function(ROOTTEST_ADD_AUTOMACROS)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "DEPENDS" ${ARGN})

  file(GLOB macros ${CMAKE_CURRENT_SOURCE_DIR}/run*.C)
  list(APPEND automacros ${macros})

  file(GLOB macros ${CMAKE_CURRENT_SOURCE_DIR}/run*.cxx)
  list(APPEND automacros ${macros})

  file(GLOB macros ${CMAKE_CURRENT_SOURCE_DIR}/assert*.C)
  list(APPEND automacros ${macros})

  file(GLOB macros ${CMAKE_CURRENT_SOURCE_DIR}/assert*.cxx)
  list(APPEND automacros ${macros})

  file(GLOB macros ${CMAKE_CURRENT_SOURCE_DIR}/exec*.C)
  list(APPEND automacros ${macros})

  file(GLOB macros ${CMAKE_CURRENT_SOURCE_DIR}/exec*.cxx)
  list(APPEND automacros ${macros})

  foreach(dep ${ARG_DEPENDS})
    list(APPEND deplist ${dep})

    if(${dep} MATCHES "[.]C" OR ${dep} MATCHES "[.]cxx" OR ${dep} MATCHES "[.]h")
      ROOTTEST_COMPILE_MACRO(${dep})
      set(add_auto_depends ${add_auto_depends} ${COMPILE_MACRO_TEST})
      
      list(REMOVE_ITEM deplist ${dep})
    endif()
  endforeach()
  set(add_auto_depends ${add_auto_depends} ${deplist})

  foreach(am ${automacros}) 
    get_filename_component(auto_macro_filename ${am} NAME)
    get_filename_component(auto_macro_refname  ${am} NAME_WE)
    set(auto_macro_refname ${auto_macro_refname}.ref)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${auto_macro_refname})
      set(outref OUTREF ${auto_macro_refname})
    else()
      set(outref "")
    endif()

    ROOTTEST_TARGETNAME_FROM_FILE(targetname ${auto_macro_filename})
   
    if(ARG_DEPENDS)
      ROOTTEST_ADD_TEST(${targetname}-auto
                        MACRO ${auto_macro_filename}
                        ${outref}
                        DEPENDS ${add_auto_depends})

    else()
      ROOTTEST_ADD_TEST(${targetname}-auto
                        MACRO ${auto_macro_filename}
                        ${outref})
    endif()
  endforeach()

endfunction(ROOTTEST_ADD_AUTOMACROS)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_COMPILE_MACRO(<filename> [BUILDOBJ object] [BUILDLIB lib]
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
  CMAKE_PARSE_ARGUMENTS(ARG "" "BUILDOBJ;BUILDLIB" "DEPENDS"  ${ARGN})

  # Add defines to root_compile_macro, in order to have out-of-source builds
  # when using the scripts/build.C macro.
  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    list(APPEND RootMacroDirDefines "-e;#define ${d}")
  endforeach()

  set(RootMacroBuildDefines
        -e "#define CMakeEnvironment"
        -e "#define CMakeBuildDir \"${CMAKE_CURRENT_BINARY_DIR}\""
        ${RootMacroDirDefines})

  set(root_compile_macro root.exe ${RootMacroBuildDefines} -q -l -b)

  get_filename_component(realfp ${filename} REALPATH)
  
  set(BuildScriptFile ${ROOTTEST_DIR}/scripts/build.C)

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

  add_dependencies(${compile_target} ${ROOTTEST_LIB_DEPENDS} ${deps})

  set(COMPILE_MACRO_TEST ${COMPILE_MACRO_TEST}-build)

  add_test(NAME ${COMPILE_MACRO_TEST}
           COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR} ${compile_target}/fast)

endmacro(ROOTTEST_COMPILE_MACRO)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_GENERATE_DICTIONARY(<dictname>
#                                    [LINKDEF linkdef]
#                                    [DEPENDS deps]
#                                    [files ...]      )
#
# This macro generates a dictionary <dictname> from the provided <files>.
# A test that performs the dictionary generation is created.  The target name of
# the created test is stored in the variable GENERATE_DICTIONARY_TEST which can
# be accessed by the calling CMakeLists.txt in order to manage dependencies.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_DICTIONARY dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LINKDEF;DEPENDS" ${ARGN})

  include_directories(${ROOT_INCLUDE_DIRS}
                      ${ROOT_INCLUDE_DIR}
                      ${CMAKE_CURRENT_SOURCE_DIR})

  link_directories(${ROOT_LIBRARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(CMAKE_ROOTTEST_DICT ON)

  ROOT_GENERATE_DICTIONARY(${dictname} ${ARG_UNPARSED_ARGUMENTS}
                           MODULE ${dictname}
                           LINKDEF ${ARG_LINKDEF}
                           DEPENDENCIES ${ARG_DEPENDS})
  
  ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_DICTIONARY_TEST ${dictname})

  set(GENERATE_DICTIONARY_TEST ${GENERATE_DICTIONARY_TEST}-build)

  set(targetname_libgen ${dictname}libgen)

  add_library(${targetname_libgen} MODULE EXCLUDE_FROM_ALL ${dictname}.cxx)
  target_link_libraries(${targetname_libgen} ${ROOT_LIBRARIES})

  set_target_properties(${targetname_libgen} PROPERTIES PREFIX "")
  set_property(TARGET ${targetname_libgen} PROPERTY OUTPUT_NAME ${dictname})

  add_dependencies(${targetname_libgen} ${ROOTTEST_LIB_DEPENDS} ${dictname})
  
  add_test(NAME ${GENERATE_DICTIONARY_TEST}
           COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}
                           ${dictname}/fast
                           ${targetname_libgen}/fast)

endmacro(ROOTTEST_GENERATE_DICTIONARY)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_GENERATE_REFLEX_DICTIONARY(<targetname> <dictionary>
#                                              [SELECTION sel...]
#                                              [headerfiles...]     )
#
# This macro generates a reflexion dictionary and creates a shared library.
# A test that performs the dictionary generation is created.  The target name of
# the created test is stored in the variable GENERATE_REFLEX_TEST which can
# be accessed by the calling CMakeLists.txt in order to manage dependencies.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_REFLEX_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "" "SELECTION;ROOTMAPNAME" "LIBRARIES"  ${ARGN})

  include_directories(${ROOT_INCLUDE_DIRS}
                      ${ROOT_INCLUDE_DIR}
                      ${CMAKE_CURRENT_SOURCE_DIR})

  link_directories(${ROOT_LIBRARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(CMAKE_ROOTTEST_DICT ON)

  set(ROOT_genreflex_cmd ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex)

  ROOTTEST_TARGETNAME_FROM_FILE(targetname ${dictionary})

  set(targetname_libgen ${targetname}-libgen)

  # targetname_dictgen is the targetname constructed by the
  # REFLEX_GENERATE_DICTIONARY macro and is used as a dependency.
  set(targetname_dictgen ${targetname}-dictgen)

  REFLEX_GENERATE_DICTIONARY(${dictionary} ${ARG_UNPARSED_ARGUMENTS}
                             SELECTION ${ARG_SELECTION})

  add_library(${targetname_libgen} MODULE ${gensrcdict})

  set_property(TARGET ${targetname_libgen}
              PROPERTY OUTPUT_NAME ${dictionary}_dictrflx)

  add_dependencies(${targetname_libgen}
                   ${ROOTTEST_LIB_DEPENDS}
                   ${targetname_dictgen})

  target_link_libraries(${targetname_libgen}
                        ${ARG_LIBRARIES}
                        ${ROOT_LIBRARIES})

  set(GENERATE_REFLEX_TEST ${targetname_libgen}-build)

  add_test(NAME ${GENERATE_REFLEX_TEST}
           COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}
                           ${targetname_dictgen}/fast
                           ${targetname_libgen}/fast)

endmacro(ROOTTEST_GENERATE_REFLEX_DICTIONARY)
