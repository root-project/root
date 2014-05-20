#-------------------------------------------------------------------------------
#
# RootCTestMacros.cmake
#
# Macros for adding tests to CTest.
#
#-------------------------------------------------------------------------------

include(RootMacros)

function(ROOT_ADD_ROOTTEST test)
  CMAKE_PARSE_ARGUMENTS(ARG "DEBUG;USEBUILDC;WILLFAIL" "MACRO;OUTREF;OUTCNV" "OUTCNVCMD;COMPILEMACROS;DEPENDS;LABELS" ${ARGN})

  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    list(APPEND RootExeDefines "-e;#define ${d}")
  endforeach()

  set(root_cmd root.exe ${RootExeDefines} -e "gSystem->SetBuildDir(\"${CMAKE_CURRENT_BINARY_DIR}\",true)" -e "gSystem->AddDynamicPath(\"${CMAKE_CURRENT_BINARY_DIR}\")" -e "gROOT->SetMacroPath(\"${CMAKE_CURRENT_SOURCE_DIR}\")" -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")" -q -l -b) 
  set(root_buildcmd root.exe ${RootExeDefines} -q -l -b)

  # Reference output given?
  if(ARG_OUTREF)
    get_filename_component(OUTREF_PATH ${ARG_OUTREF} REALPATH)
  else()
    set(OUTREF_PATH, "")
  endif()

  if(ARG_OUTCNV)
    get_filename_component(OUTCNV ${ARG_OUTCNV} REALPATH)
  endif()

  # Test has dependencies?
  if(ARG_DEPENDS)
    set(depends DEPENDS ${ARG_DEPENDS})
  endif(ARG_DEPENDS)

  # Compile macro, then add to CTest.
  if(ARG_MACRO MATCHES "[.]C\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${compile_name} REALPATH)

    ROOT_COMPILE_MACRO(${compile_name}) 

    set(command ${root_cmd} "${realfp}+")

  elseif(ARG_MACRO MATCHES "[.]cxx\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)

    ROOT_COMPILE_MACRO(${compile_name}) 

    set(command ${root_cmd} "${realfp}")

  elseif(ARG_MACRO MATCHES "[.]cxx")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)
    set(command ${root_cmd} ${realfp})


  # Add interpreted macro to CTest.
  elseif(ARG_MACRO MATCHES "[.]C")
    get_filename_component(realfp ${ARG_MACRO} REALPATH)
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

  ROOT_ADD_TEST(${test} COMMAND ${command}
                        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${test}.log
                        ${outcnv}
                        ${outcnvcmd}
                        CMPOUTPUT ${OUTREF_PATH}
                        WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                        DIFFCMD sh ${ROOTTEST_DIR}/scripts/custom_diff.sh
                        ENVIRONMENT ROOTSYS=${ROOTSYS} PATH=$ENV{PATH} LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} PYTHONPATH=$ENV{PYTHONPATH}
                        TIMEOUT 3600
                        ${build}
                        ${checkstdout}
                        ${checkstderr}
                        ${willfail}
                        ${compile_macros}
                        ${labels}
                        ${depends})

endfunction(ROOT_ADD_ROOTTEST)
