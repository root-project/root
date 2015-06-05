#-------------------------------------------------------------------------------
#
# RootCTestMacros.cmake
#
# Macros for adding tests to CTest.
#
#-------------------------------------------------------------------------------

include(RootMacros)

#-------------------------------------------------------------------------------
# macro ROOTTEST_SETUP_MACROTEST()
#
# A helper macro to define the command to run a ROOT macro (.C, .C+ or .py)
#-------------------------------------------------------------------------------
macro(ROOTTEST_SETUP_MACROTEST)

  get_directory_property(DirDefs COMPILE_DEFINITIONS)

  foreach(d ${DirDefs})
    list(APPEND RootExeDefines "-e;#define ${d}")
  endforeach()

  set(root_cmd root.exe ${RootExeDefines}
               -e "gSystem->SetBuildDir(\"${CMAKE_CURRENT_BINARY_DIR}\",true)"
               -e "gSystem->AddDynamicPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
               -e "gROOT->SetMacroPath(\"${CMAKE_CURRENT_SOURCE_DIR}\")"
               -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_BINARY_DIR}\")"
               ${RootExeOptions}
               -q -l -b) 

  set(root_buildcmd root.exe ${RootExeDefines} -q -l -b)

  # Compile macro, then add to CTest.
  if(ARG_MACRO MATCHES "[.]C\\+" OR ARG_MACRO MATCHES "[.]cxx\\+")
    string(REPLACE "+" "" compile_name "${ARG_MACRO}")
    get_filename_component(realfp ${compile_name} REALPATH)

    #---Do not compile the macro beforehand
    #ROOTTEST_COMPILE_MACRO(${compile_name})
    #set(depends ${depends} ${COMPILE_MACRO_TEST})

    if(DEFINED ARG_MACROARG)
      set(realfp "${realfp}(${ARG_MACROARG})") 
    endif()

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
#                            [WILLFAIL]
#                            [OUTREF stdout_reference]
#                            [ERRREF stderr_reference]
#                            [WORKING_DIR dir]
#                            [COPY_TO_BUILDDIR file1 file2 ...])
#
# This function defines a roottest test. It adds a number of additional
# options on top of the ROOT defined ROOT_ADD_TEST.
#
#-------------------------------------------------------------------------------

function(ROOTTEST_ADD_TEST test)
  CMAKE_PARSE_ARGUMENTS(ARG "WILLFAIL"
                            "OUTREF;ERRREF;OUTREF_CINTSPECIFIC;OUTCNV;PASSRC;MACROARG;WORKING_DIR;INPUT"
                            "TESTOWNER;COPY_TO_BUILDDIR;MACRO;EXEC;COMMAND;PRECMD;POSTCMD;OUTCNVCMD;FAILREGEX;PASSREGEX;DEPENDS;OPTS;LABELS" ${ARGN})

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
    set(postcmd POSTCMD ${ARG_PRECMD})
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

  if(ARG_FAILREGEX)
    set(failregex FAILREGEX ${ARG_FAILREGEX})
  endif()

  if(ARG_PASSREGEX)
    set(passregex PASSREGEX ${ARG_PASSREGEX})
  endif()

  string(REPLACE ";" ":" _path "${ROOTTEST_ENV_PATH}")
  string(REPLACE ";" ":" _pythonpath "${ROOTTEST_ENV_PYTHONPATH}")
  string(REPLACE ";" ":" _librarypath "${ROOTTEST_ENV_LIBRARYPATH}")

  set(environment ENVIRONMENT
                  ROOTSYS=${ROOTSYS}
                  PATH=${_path}:$ENV{PATH}
                  PYTHONPATH=${_pythonpath}:$ENV{PYTHONPATH}
                  ${ld_library_path}=${_librarypath}:$ENV{${ld_library_path}} )

  if(ARG_WORKING_DIR)
    get_filename_component(test_working_dir ${ARG_WORKING_DIR} ABSOLUTE)
  else()
    get_filename_component(test_working_dir ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
  endif()

  get_filename_component(logfile "${CMAKE_CURRENT_BINARY_DIR}/${test}.log" ABSOLUTE)
  if(ARG_ERRREF)
    get_filename_component(errfile "${CMAKE_CURRENT_BINARY_DIR}/${test}.err" ABSOLUTE)
    set(errfile ERROR ${errfile})
  endif()

  if(ARG_INPUT)
    get_filename_component(infile_path ${ARG_INPUT} ABSOLUTE)
    set(infile INPUT ${infile_path})
  endif()

  ROOT_ADD_TEST(${test} COMMAND ${command}
                        OUTPUT ${logfile}
                        ${infile}
                        ${errfile}
                        ${outcnv}
                        ${outcnvcmd}
                        ${outref}
                        ${errref}
                        WORKING_DIR ${test_working_dir}
                        DIFFCMD ${ROOTTEST_DIR}/scripts/custom_diff.py
                        TIMEOUT 3600
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
                        ${failregex}
                        ${passregex}
                        ${copy_to_builddir}
                        DEPENDS ${depends})

endfunction(ROOTTEST_ADD_TEST)
