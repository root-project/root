#-------------------------------------------------------------------------------
#
#  RootMacros.cmake
#
#  Macros and definitions regarding ROOT components.
#
#-------------------------------------------------------------------------------

if(CMAKE_GENERATOR MATCHES Makefiles)
  set(fast /fast)
  set(always-make --always-make)
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
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${d}/.rootrc "
Rint.History:  .root_hist
ACLiC.LinkLibs:  1
")
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
#   roottest-<directorypath>-<filename_WE>
#
#-------------------------------------------------------------------------------
function(ROOTTEST_TARGETNAME_FROM_FILE resultvar filename)

  get_filename_component(realfp ${filename} ABSOLUTE)
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
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "DEPENDS;WILLFAIL;EXCLUDE" ${ARGN})

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

    if(selected)
      ROOTTEST_ADD_TEST(${targetname}-auto
                        MACRO ${auto_macro_filename}${${auto_macro_name}-suffix}
                        ${outref}
                        ${arg_wf}
                        DEPENDS ${auto_depends})
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
        ${RootMacroDirDefines})

  set(root_compile_macro ${ROOT_root_CMD} ${RootMacroBuildDefines} -q -l -b)

  get_filename_component(realfp ${filename} ABSOLUTE)
  if(MSVC)
    string(REPLACE "/" "\\\\" realfp ${realfp})
  endif()

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

  if(ARG_DEPENDS)
    add_dependencies(${compile_target} ${deps})
  endif()

  set(COMPILE_MACRO_TEST ${COMPILE_MACRO_TEST}-build)

  add_test(NAME ${COMPILE_MACRO_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    --config $<CONFIG>
                                    --target ${compile_target}${fast}
                                    -- ${always-make})
  if(NOT MSVC OR win_broken_tests)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY FAIL_REGULAR_EXPRESSION "Warning in")
  endif()
  set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})
  if(CMAKE_GENERATOR MATCHES Ninja)
    set_property(TEST ${COMPILE_MACRO_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if(MSVC)
    string(REPLACE "." "_" dll_name ${filename})
    add_custom_command(TARGET ${compile_target} POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${dll_name}.dll
                                        ${CMAKE_CURRENT_BINARY_DIR}/
       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${dll_name}_ACLiC_dict_rdict.pcm
                                        ${CMAKE_CURRENT_BINARY_DIR}/)
  endif()

endmacro(ROOTTEST_COMPILE_MACRO)

#-------------------------------------------------------------------------------
#
# macro ROOTTEST_GENERATE_DICTIONARY(<dictname>
#                                    [LINKDEF linkdef]
#                                    [DEPENDS deps]
#                                    [OPTIONS opts]
#                                    [files ...]      )
#
# This macro generates a dictionary <dictname> from the provided <files>.
# A test that performs the dictionary generation is created.  The target name of
# the created test is stored in the variable GENERATE_DICTIONARY_TEST which can
# be accessed by the calling CMakeLists.txt in order to manage dependencies.
#
#-------------------------------------------------------------------------------
macro(ROOTTEST_GENERATE_DICTIONARY dictname)
  CMAKE_PARSE_ARGUMENTS(ARG "NO_ROOTMAP;NO_CXXMODULE" "" "LINKDEF;DEPENDS;OPTIONS" ${ARGN})

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
  set_target_properties(${targetname_libgen} PROPERTIES  ${ROOT_LIBRARY_PROPERTIES} )
  if(MSVC)
    set_target_properties(${targetname_libgen} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  endif()
  target_link_libraries(${targetname_libgen} ${ROOT_LIBRARIES})

  set_target_properties(${targetname_libgen} PROPERTIES PREFIX "")

  set_property(TARGET ${targetname_libgen}
               PROPERTY OUTPUT_NAME ${dictname})

  set_property(TARGET ${targetname_libgen}
               APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

  add_dependencies(${targetname_libgen} ${dictname})

  add_test(NAME ${GENERATE_DICTIONARY_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    --config $<CONFIG>
                                    --target  ${targetname_libgen}${fast}
                                    -- ${always-make})

  set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})
  if(CMAKE_GENERATOR MATCHES Ninja)
    set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if(MSVC)
    add_custom_command(TARGET ${targetname_libgen} POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${dictname}_rdict.pcm
                                        ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${dictname}_rdict.pcm)
    add_custom_command(TARGET ${targetname_libgen} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${dictname}.dll
                                       ${CMAKE_CURRENT_BINARY_DIR}/${dictname}.dll)
  endif()

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
  CMAKE_PARSE_ARGUMENTS(ARG "NO_ROOTMAP" "SELECTION;LIBNAME" "LIBRARIES;OPTIONS"  ${ARGN})

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
  if(MSVC)
    set_target_properties(${targetname_libgen} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  endif()

  if(ARG_LIBNAME)
    set_target_properties(${targetname_libgen} PROPERTIES PREFIX "")
    set_property(TARGET ${targetname_libgen}
                 PROPERTY OUTPUT_NAME ${ARG_LIBNAME})
  else()
    set_property(TARGET ${targetname_libgen}
                 PROPERTY OUTPUT_NAME ${dictionary}_dictrflx)
  endif()

  add_dependencies(${targetname_libgen}
                   ${targetname_dictgen})

  target_link_libraries(${targetname_libgen}
                        ${ARG_LIBRARIES}
                        ${ROOT_LIBRARIES})

  set_property(TARGET ${targetname_libgen}
               APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

  set(GENERATE_REFLEX_TEST ${targetname_libgen}-build)

  add_test(NAME ${GENERATE_REFLEX_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    --config $<CONFIG>
                                    --target ${targetname_libgen}${fast}
                                    -- ${always-make})

  set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})
  if(CMAKE_GENERATOR MATCHES Ninja)
    set_property(TEST ${GENERATE_REFLEX_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if(MSVC)
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

  set_property(TARGET ${executable}
               APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

  if(ARG_DEPENDS)
    add_dependencies(${executable} ${ARG_DEPENDS})
  endif()

  if(ARG_LIBRARIES)
    if(MSVC)
      foreach(library ${ARG_LIBRARIES})
        if(${library} MATCHES "[::]")
          set(libraries ${libraries} ${library})
        else()
          set(libraries ${libraries} lib${library})
        endif()
      endforeach()
      target_link_libraries(${executable} ${libraries})
    else()
      target_link_libraries(${executable} ${ARG_LIBRARIES})
    endif()
  endif()
  if(TARGET ROOT::ROOTStaticSanitizerConfig)
    target_link_libraries(${executable} ROOT::ROOTStaticSanitizerConfig)
  endif()

  if(ARG_COMPILE_FLAGS)
    set_target_properties(${executable} PROPERTIES COMPILE_FLAGS ${ARG_COMPILE_FLAGS})
  endif()

  ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_EXECUTABLE_TEST ${executable})

  set(GENERATE_EXECUTABLE_TEST ${GENERATE_EXECUTABLE_TEST}-build)

  add_test(NAME ${GENERATE_EXECUTABLE_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    --config $<CONFIG>
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

  if(CMAKE_GENERATOR MATCHES Ninja)
    set_property(TEST ${GENERATE_EXECUTABLE_TEST} PROPERTY RUN_SERIAL true)
  endif()

  if(MSVC)
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
function(ROOTTEST_ADD_OLDTEST)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LABELS;TIMEOUT" ${ARGN})

  ROOTTEST_ADD_TEST( make
                     COMMAND make cleantest
                     WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                     DEPENDS roottest-root-io-event
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
               ${RootExternalIncludes} ${RootExeOptions}
               -q -l -b)

  set(root_buildcmd ${ROOT_root_CMD} ${RootExeDefines} -q -l -b)

  # Compile macro, then add to CTest.
  if(ARG_MACRO MATCHES "[.]C\\+" OR ARG_MACRO MATCHES "[.]cxx\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${compile_name} REALPATH)

    if(DEFINED ARG_MACROARG)
      set(command ${root_cmd} "${realfp}+(${ARG_MACROARG})")
    else()
      set(command ${root_cmd} "${realfp}+")
    endif()

  # Add interpreted macro to CTest.
  elseif(ARG_MACRO MATCHES "[.]C" OR ARG_MACRO MATCHES "[.]cxx")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)
    if(DEFINED ARG_MACROARG)
      set(realfp "${realfp}(${ARG_MACROARG})")
    endif()

    set(command ${root_cmd} ${realfp})

  # Add python script to CTest.
  elseif(ARG_MACRO MATCHES "[.]py")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)
    set(command ${PYTHON_EXECUTABLE} ${realfp} ${PYROOT_EXTRAFLAGS})

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
      set(realexec ${PYTHON_EXECUTABLE} ${realexec})
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
  CMAKE_PARSE_ARGUMENTS(ARG "WILLFAIL;RUN_SERIAL"
                            "OUTREF;ERRREF;OUTREF_CINTSPECIFIC;OUTCNV;PASSRC;MACROARG;WORKING_DIR;INPUT;ENABLE_IF;DISABLE_IF;TIMEOUT;RESOURCE_LOCK"
                            "TESTOWNER;COPY_TO_BUILDDIR;MACRO;EXEC;COMMAND;PRECMD;POSTCMD;OUTCNVCMD;FAILREGEX;PASSREGEX;DEPENDS;OPTS;LABELS;ENVIRONMENT;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED;PROPERTIES"
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
        file(TO_NATIVE_PATH "${fpath}" fpath)
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
                        DIFFCMD ${PYTHON_EXECUTABLE} ${ROOTTEST_DIR}/scripts/custom_diff.py
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

  if(MSVC)
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
  target_include_directories(${binary} PRIVATE ${GTEST_INCLUDE_DIR})
  target_link_libraries(${binary} gtest gtest_main ${libraries})

  if(TARGET ROOT::ROOTStaticSanitizerConfig)
    target_link_libraries(${binary} ROOT::ROOTStaticSanitizerConfig)
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
      execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
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
   if(MSVC)
      add_custom_command(TARGET ${library} POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/lib${library}.dll
                                          ${CMAKE_CURRENT_BINARY_DIR}/lib${library}.dll
         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/lib${library}.lib
                                           ${CMAKE_CURRENT_BINARY_DIR}/lib${library}.lib)
   endif()
endfunction()
