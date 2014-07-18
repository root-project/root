#-------------------------------------------------------------------------------
#
# RootCTestDriver.cmake
#
# CTest testdriver. Takes arguments via -DARG.
#
# Script arguments:
#
#   CMD   Command to be executed for the test.
#   PRE   Command to be executed before the test command.
#   POST  Command to be executed after the test command.
#   OUT   File to collect stdout and stderr.
#   ENV   Environment VAR1=Value1;VAR2=Value2.
#   CWD   Current working directory.
#   SYS   Value of ROOTSYS
#   DBG   Debug flag.
#   RC    Return code for success.
#
#-------------------------------------------------------------------------------

if(DBG)
  message(STATUS "ENV=${ENV}")
endif()

if(CMD)
  string(REPLACE "^" ";" _cmd ${CMD})
  if(DBG)
    message(STATUS "testdriver:CMD=${_cmd}")
  endif()
endif()

if(PRE)
  string(REPLACE "^" ";" _pre ${PRE})
  if(DBG)
    message(STATUS "testdriver:PRE=${_pre}")
  endif()
endif()

if(POST)
  string(REPLACE "^" ";" _post ${POST})
  if(DBG)
    message(STATUS "testdriver:POST=${_post}")
  endif()
endif()

if(CWD)
  set(_cwd WORKING_DIRECTORY ${CWD})
  if(DBG)
    message(STATUS "testdriver:CWD=${CWD}")
  endif()
endif()

find_program(diff_cmd diff)

if(DIFFCMD)
  string(REPLACE "^" ";" diff_cmd ${DIFFCMD})
endif()

#---Set environment --------------------------------------------------------------------------------
if(ENV)
  string(REPLACE "@" "=" _env ${ENV})
  string(REPLACE "#" ";" _env ${_env})
  foreach(pair ${_env})
    string(REPLACE "=" ";" pair ${pair})
    list(GET pair 0 var)
    list(GET pair 1 val)
    set(ENV{${var}} ${val})
    if(DBG)
      message(STATUS "testdriver[ENV]:${var}==>${val}")
    endif()
  endforeach()
endif()

if(WIN32 AND SYS)
  file(TO_NATIVE_PATH ${SYS}/bin _path)
  set(ENV{PATH} "${_path};$ENV{PATH}")
endif()

#---Execute pre-command-----------------------------------------------------------------------------
if(PRE)
  execute_process(COMMAND ${_pre} ${_cwd} RESULT_VARIABLE _rc)
  if(_rc)
    message(FATAL_ERROR "pre-command error code : ${_rc}")
  endif()
endif()

if(CMD)
  #---Execute the actual test ------------------------------------------------------------------------
  string (REPLACE ";" " " _strcmd "${_cmd}")
  message("Command: ${_strcmd}")
  if(OUT)

    # log stdout
    if(CHECKOUT)
      set(_chkout OUTPUT_VARIABLE _outvar)
    else()
      set(_chkout "")
    endif()

    # log stderr
    if(CHECKERR)
      set(_chkerr ERROR_VARIABLE _outvar)
    else()
      set(_chkerr "")
    endif()

    execute_process(COMMAND ${_cmd} ${_chkout} ${_chkerr} WORKING_DIRECTORY ${CWD} RESULT_VARIABLE _rc)
    
    message("-- BEGIN TEST OUTPUT --")
    message("${_outvar}")
    message("-- END TEST OUTPUT --")

    file(WRITE ${OUT} "${_outvar}")

    if(DEFINED RC AND (NOT _rc EQUAL RC))
      message(FATAL_ERROR "error code: ${_rc}")
    elseif(NOT DEFINED RC AND _rc)
      message(FATAL_ERROR "error code: ${_rc}")
    endif()
    
    if(CNVCMD)
      set(_outvar, "")
      string(REPLACE "^" ";" _outcnvcmd "${CNVCMD}^${OUT}")
      execute_process(COMMAND ${_outcnvcmd} ${_chkout} ${_chkerr} RESULT_VARIABLE _rc)
      file(WRITE ${OUT} ${_outvar})

      if(DEFINED RC AND (NOT _rc EQUAL RC))
        message(FATAL_ERROR "error code: ${_rc}")
      elseif(NOT DEFINED RC AND _rc)
        message(FATAL_ERROR "error code: ${_rc}")
      endif()
    endif()

    if(CNV)
      set(_outvar, "")
      string(REPLACE "^" ";" _outcnv "sh;${CNV}")
      execute_process(COMMAND ${_outcnv} INPUT_FILE "${OUT}" OUTPUT_VARIABLE _outvar RESULT_VARIABLE _rc)
      file(WRITE ${OUT} "${_outvar}")

      if(DEFINED RC AND (NOT _rc EQUAL RC))
        message(FATAL_ERROR "error code: ${_rc}")
      elseif(NOT DEFINED RC AND _rc)
        message(FATAL_ERROR "error code: ${_rc}")
      endif()
    endif()
  else()
    execute_process(COMMAND ${_cmd} ${_out} ${_err} ${_cwd} RESULT_VARIABLE _rc)

    if(DEFINED RC AND (NOT _rc EQUAL RC))
      message(FATAL_ERROR "error code: ${_rc}")
    elseif(NOT DEFINED RC AND _rc)
      message(FATAL_ERROR "error code: ${_rc}")
    endif()
  endif()

endif()

#---Execute post-command-----------------------------------------------------------------------------
if(POST)
  execute_process(COMMAND ${_post} ${_cwd} RESULT_VARIABLE _rc)
  if(_rc)
    message(FATAL_ERROR "post-command error code : ${_rc}")
  endif()
endif()

if(CMPOUTPUT)
  set(command COMMAND ${diff_cmd} ${OUT} ${CMPOUTPUT})

  execute_process(${command} ${OUT} ${CMPOUTPUT} RESULT_VARIABLE _rc)

  if(_rc)
    message(FATAL_ERROR "compare output error: ${_rc}")
  endif()
endif()
