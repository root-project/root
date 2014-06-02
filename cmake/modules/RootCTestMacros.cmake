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
                            "MACRO;OUTREF;OUTCNV;PASSRC;MACROARG"
                            "OUTCNVCMD;DEPENDS;LABELS" ${ARGN})

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
    get_filename_component(OUTREF_PATH ${ARG_OUTREF} REALPATH)

    if(DEFINED X86_64 AND EXISTS ${OUTREF_PATH}64)
      set(OUTREF_PATH ${OUTREF_PATH}64)
    elseif(DEFINED X86 AND EXISTS ${OUTREF_PATH}32)
      set(OUTREF_PATH ${OUTREF_PATH}32)
    endif()
  else()
    set(OUTREF_PATH, "")
  endif()

  if(ARG_OUTCNV)
    get_filename_component(OUTCNV ${ARG_OUTCNV} REALPATH)
  endif()

  # Test has dependencies?
  if(ARG_DEPENDS)
    set(depends ${ARG_DEPENDS})
  endif(ARG_DEPENDS)

  # Compile macro, then add to CTest.
  if(ARG_MACRO MATCHES "[.]C\\+" OR ARG_MACRO MATCHES "[.]cxx\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${compile_name} REALPATH)

    ROOTTEST_COMPILE_MACRO(${compile_name})

    set(depends ${depends} ${COMPILE_MACRO_TEST})

    set(command ${root_cmd} "${realfp}+")

  # Add interpreted macro to CTest.
  elseif(ARG_MACRO MATCHES "[.]C" OR ARG_MACRO MATCHES "[.]cxx")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)

    if(DEFINED ARG_MACROARG)
      set(realfp "${realfp}(${ARG_MACROARG})") 
    endif()

    set(command ${root_cmd} ${realfp})
    
  # Add python script to CTest.
  elseif(ARG_MACRO MATCHES "[.]py")
    get_filename_component(pycmd ${ARG_MACRO} REALPATH)
    set(command ${python_cmd} ${pycmd})
  endif()

  # Check for assert prefix -- only log stderr.
  if(ARG_MACRO MATCHES "^assert")
    set(checkstdout "")
    set(checkstderr CHECKERR)
  else()
    set(checkstdout CHECKOUT)
    set(checkstderr CHECKERR)
  endif()

  if(ARG_OUTCNV)
    get_filename_component(OUTCNV ${ARG_OUTCNV} REALPATH)
    set(outcnv OUTCNV ${OUTCNV})
  endif()

  if(ARG_OUTCNVCMD)
    set(outcnvcmd OUTCNVCMD ${ARG_OUTCNVCMD})
  endif()

  if(ARG_WILLFAIL)
    set(willfail WILLFAIL)
  endif()

  if(ARG_LABELS)
    set(labels LABELS ${ARG_LABELS})
  endif()

  if(ARG_PASSRC)
    set(passrc PASSRC ${ARG_PASSRC})
  endif()

  if(ARG_DEPENDS)
    set(depends ${ARG_DEPENDS})
  endif(ARG_DEPENDS)

  set(environment ENVIRONMENT
                  ROOTSYS=${ROOTSYS}
                  PATH=$ENV{PATH}
                  LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
                  PYTHONPATH=$ENV{PYTHONPATH}
                  DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}
                  SHLIB_PATH=$ENV{SHLIB_PATH}
                  LIBPATH=$ENV{LIBPATH}
  )

  ROOT_ADD_TEST(${test} COMMAND ${command}
                        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${test}.log
                        ${outcnv}
                        ${outcnvcmd}
                        CMPOUTPUT ${OUTREF_PATH}
                        WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
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
