#-------------------------------------------------------------------------------
#
# RootCTestMacros.cmake
#
# Macros for adding tests to CTest.
#
#-------------------------------------------------------------------------------

include(RootMacros)

function(ROOTTEST_ADD_TEST test)
  CMAKE_PARSE_ARGUMENTS(ARG "WILLFAIL"
                            "OUTREF;OUTCNV;PASSRC;MACROARG;WORKING_DIR"
                            "MACRO;OUTCNVCMD;DEPENDS;OPTS;LABELS" ${ARGN})

  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    list(APPEND RootExeDefines "-e;#define ${d}")
  endforeach()

  set(root_cmd root.exe ${RootExeDefines}
               -e "gSystem->SetBuildDir(\"${CMAKE_CURRENT_BINARY_DIR}\",true)"
               -e "gSystem->AddDynamicPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
               -e "gROOT->SetMacroPath(\"${CMAKE_CURRENT_SOURCE_DIR}\")"
               -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")"
               -q -l -b) 

  set(root_buildcmd root.exe ${RootExeDefines} -q -l -b)

  # Reference output given?
  if(ARG_OUTREF)
    get_filename_component(OUTREF_PATH ${ARG_OUTREF} ABSOLUTE)

    if(DEFINED X86_64 AND EXISTS ${OUTREF_PATH}64)
      set(OUTREF_PATH ${OUTREF_PATH}64)
    elseif(DEFINED X86 AND EXISTS ${OUTREF_PATH}32)
      set(OUTREF_PATH ${OUTREF_PATH}32)
    endif()
  else()
    set(OUTREF_PATH, "")
  endif()

  if(ARG_OUTCNV)
    get_filename_component(OUTCNV ${ARG_OUTCNV} ABSOLUTE)
  endif()

  # Compile macro, then add to CTest.
  if(ARG_MACRO MATCHES "[.]C\\+" OR ARG_MACRO MATCHES "[.]cxx\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${compile_name} ABSOLUTE)

    ROOTTEST_COMPILE_MACRO(${compile_name})

    set(depends ${depends} ${COMPILE_MACRO_TEST})

    if(DEFINED ARG_MACROARG)
      set(realfp "${realfp}(${ARG_MACROARG})") 
    endif()

    set(command ${root_cmd} "${realfp}+")

  # Add interpreted macro to CTest.
  elseif(ARG_MACRO MATCHES "[.]C" OR ARG_MACRO MATCHES "[.]cxx")
    get_filename_component(realfp ${ARG_MACRO} ABSOLUTE)

    if(DEFINED ARG_MACROARG)
      set(realfp "${realfp}(${ARG_MACROARG})") 
    endif()

    set(command ${root_cmd} ${realfp})
    
  # Add python script to CTest.
  elseif(ARG_MACRO MATCHES "[.]py")
    get_filename_component(pycmd ${ARG_MACRO} ABSOLUTE)
    set(command ${python_cmd} ${pycmd})

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

  # Add labels to the test.
  if(ARG_LABELS)
    set(labels LABELS ${ARG_LABELS})
  endif()

  # Test will pass for a custom return value.
  if(ARG_PASSRC)
    set(passrc PASSRC ${ARG_PASSRC})
  endif()

  if(ARG_OPTS)
    set(command ${command} ${ARG_OPTS})
  endif()

  # Add dependencies. If the test depends on a macro file, the macro
  # will be compiled and the dependencies are set accordingly.
  if(ARG_DEPENDS)
    foreach(dep ${ARG_DEPENDS})
      list(APPEND deplist ${dep})

      if(${dep} MATCHES "[.]C" OR ${dep} MATCHES "[.]cxx" OR ${dep} MATCHES "[.]h")
        ROOTTEST_COMPILE_MACRO(${dep})

        set(depends ${depends} ${COMPILE_MACRO_TEST})
        
        list(REMOVE_ITEM deplist ${dep})
      endif()
    endforeach()
    set(depends ${depends} ${deplist})
  endif(ARG_DEPENDS)

  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(LIBRARYPATH LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})
  elseif(APPLE)
    set(LIBRARYPATH DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH})
  else()
    set(LIBRARYPATH "") 
  endif()

  set(environment ENVIRONMENT
                  ROOTSYS=${ROOTSYS}
                  PYTHONPATH=$ENV{PYTHONPATH}
                  PATH=$ENV{PATH}
                  ${LIBRARYPATH})

  if(ARG_WORKING_DIR)
    set(test_working_dir ${ARG_WORKING_DIR})
  else()
    set(test_working_dir ${CMAKE_CURRENT_SOURCE_DIR}) 
  endif()

  ROOT_ADD_TEST(${test} COMMAND ${command}
                        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${test}.log
                        ${outcnv}
                        ${outcnvcmd}
                        CMPOUTPUT ${OUTREF_PATH}
                        WORKING_DIR ${test_working_dir}
                        DIFFCMD sh ${ROOTTEST_DIR}/scripts/custom_diff.sh
                        TIMEOUT 3600
                        ${environment}
                        ${build}
                        ${checkstdout}
                        ${checkstderr}
                        ${willfail}
                        ${compile_macros}
                        ${labels}
                        ${passrc}
                        DEPENDS ${depends})

endfunction(ROOTTEST_ADD_TEST)
