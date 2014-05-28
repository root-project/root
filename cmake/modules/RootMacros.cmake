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

endfunction()

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_COMPILE_MACRO(<filename>
#                              [BUILDOBJ object] [BUILDLIB lib])
#
# This function creates and loads a shared library containing the code from
# the file <filename>.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_COMPILE_MACRO filename)
  CMAKE_PARSE_ARGUMENTS(ARG "" "BUILDOBJ;BUILDLIB;DEPENDS" ""  ${ARGN})

  # Add defines to root_cmd, in order to have out-of-source builds
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

  set(RootCompileMacroWD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  get_filename_component(realfp ${filename} REALPATH)

  set(command ${root_compile_macro}
              ${ROOTTEST_DIR}/scripts/build.C\(\"${realfp}\",\"${ARG_BUILDLIB}\",\"${ARG_BUILDOBJ}\"\) ${RootCompileMacroWD})
  
  if(ARG_DEPENDS)
    set(deps ${ARG_DEPENDS})
  endif()

  ROOTTEST_TARGETNAME_FROM_FILE(COMPILE_MACRO_TEST ${filename})
  
  set(compile_target ${COMPILE_MACRO_TEST}-compile-macro)

  add_custom_target(${compile_target} COMMAND ${command} ${RootCompileMacroWD} ${deps} VERBATIM)
  add_dependencies(${compile_target} ${ROOTTEST_LIB_DEPENDS} ${deps})

  set(COMPILE_MACRO_TEST ${COMPILE_MACRO_TEST}-build)

  add_test(NAME ${COMPILE_MACRO_TEST} COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR} ${compile_target}/fast)

endmacro(ROOTTEST_COMPILE_MACRO)

#-------------------------------------------------------------------------------
#
# function ROOTTEST_GENERATE_DICTIONARY(<dictname>
#                                       [LINKDEF linkdef]
#                                       [DEPENDS deps]
#                                       [files ...]      )
#
# This function generates a dictionary <dictname> from the provided <files>.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_DICTIONARY dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LINKDEF;DEPENDS" ${ARGN})

  include_directories(${ROOT_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  link_directories(${ROOT_LIBRARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(CMAKE_ROOTTEST_DICT ON)

  ROOT_GENERATE_DICTIONARY(${dictname} ${ARG_UNPARSED_ARGUMENTS} MODULE ${dictname} LINKDEF ${ARG_LINKDEF} DEPENDENCIES ${ARG_DEPENDS})
  
  ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_DICTIONARY_TEST ${dictname})

  set(GENERATE_DICTIONARY_TEST ${GENERATE_DICTIONARY_TEST}-build)

  set(targetname_libgen ${dictname}libgen)

  add_library(${targetname_libgen} MODULE ${dictname}.cxx)

  set_target_properties(${targetname_libgen} PROPERTIES PREFIX "")
  set_property(TARGET ${targetname_libgen} PROPERTY OUTPUT_NAME ${dictname})

  add_dependencies(${targetname_libgen} ${ROOTTEST_LIB_DEPENDS} ${dictname})
  
  add_test(NAME ${GENERATE_DICTIONARY_TEST} COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR} ${dictname}/fast ${targetname_libgen}/fast)

endmacro(ROOTTEST_GENERATE_DICTIONARY)

#-------------------------------------------------------------------------------
#
# function ROOTTEST_GENERATE_REFLEX_DICTIONARY(<targetname> <dictionary>
#                                              [SELECTION sel...]
#                                              [headerfiles...]     )
#
# This function generates a reflexion dictionary and creates a shared library.
#
#-------------------------------------------------------------------------------
function(ROOTTEST_GENERATE_REFLEX_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "" "SELECTION" ""  ${ARGN})

  include_directories(${ROOT_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  link_directories(${ROOT_LIBRARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(CMAKE_ROOTTEST_DICT ON)

  set(ROOT_genreflex_cmd ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex)

  ROOTTEST_TARGETNAME_FROM_FILE(targetname ${dictionary})

  set(targetname_libgen ${targetname}-libgen)

  # targetname_dictgen is the targetname constructed by the REFLEX_GENERATE_DICTIONARY
  # macro and is used as a dependency.
  set(targetname_dictgen ${targetname}-dictgen)

  REFLEX_GENERATE_DICTIONARY(${dictionary} ${ARG_UNPARSED_ARGUMENTS} SELECTION ${ARG_SELECTION})

  add_library(${targetname_libgen} MODULE ${gensrcdict})

  set_property(TARGET ${targetname_libgen} PROPERTY OUTPUT_NAME ${dictionary}_dictrflx)

  add_dependencies(${targetname_libgen} ${ROOTTEST_LIB_DEPENDS} ${targetname_dictgen})

  target_link_libraries(${targetname_libgen} ${ARG_LIBRARIES} ${ROOT_Reflex_LIBRARY})

  set(GENERATE_REFLEX_TEST ${targetname_libgen}-build)

  add_test(NAME ${GENERATE_REFLEX_TEST}
           COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR} ${targetname_dictgen}/fast
           ${targetname_libgen}/fast)

endfunction(ROOTTEST_GENERATE_REFLEX_DICTIONARY)
