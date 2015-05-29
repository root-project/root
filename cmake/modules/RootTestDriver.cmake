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
#   IN    File to be used as input
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

if(COPY)
  string(REPLACE "^" ";" _copy_files ${COPY})
  if(DBG)
    message(STATUS "files to copy: ${_copy_files}")
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

#---Copy files to current direcotory----------------------------------------------------------------
if(COPY)
  foreach(copyfile ${_copy_files})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${copyfile} ${CMAKE_CURRENT_BINARY_DIR}
                    RESULT_VARIABLE _rc)
    if(_rc)
      message(FATAL_ERROR "Copying file ${copyfile} to ${CMAKE_CURRENT_BINARY_DIR} failed! Error code : ${_rc}")
    endif()
  endforeach()
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
  if(IN)
    set(_input INPUT_FILE ${IN})
  endif()

  if(OUT)
    # log stdout
    if(CHECKOUT)
      set(_chkout OUTPUT_VARIABLE _outvar)
    else()
      set(_chkout OUTPUT_VARIABLE _outvar2)
    endif()

    # log stderr
    if(CHECKERR AND NOT ERRREF)
      set(_chkerr ERROR_VARIABLE _outvar)
    elseif(ERRREF)
      set(_chkerr ERROR_VARIABLE _errvar)
    else()
      set(_chkerr ERROR_VARIABLE _errvar2)
    endif()

    execute_process(COMMAND ${_cmd} ${_input} ${_chkout} ${_chkerr} WORKING_DIRECTORY ${CWD} RESULT_VARIABLE _rc)

    string(REGEX REPLACE "([.]*)[;][-][e][;]([^;]+)([.]*)" "\\1;-e '\\2\\3'" res "${_cmd}")
    string(REPLACE ";" " " res "${res}")
    message("-- TEST COMMAND -- ")
    message("cd ${CWD}")
    message("${res}")
    message("-- BEGIN TEST OUTPUT --")
    message("${_outvar}${_outvar2}")
    message("-- END TEST OUTPUT --")
    if(_errvar OR _errvar2)
      message("-- BEGIN TEST ERROR --")
      message("${_errvar}${_errvar2}")
      message("-- END TEST ERROR --")
    endif()

    file(WRITE ${OUT} "${_outvar}")
    if(ERR)
      file(WRITE ${ERR} "${_errvar}")
    endif()

    if(DEFINED RC AND (NOT _rc EQUAL RC))
      message(FATAL_ERROR "error code: ${_rc}")
    elseif(NOT DEFINED RC AND _rc)
      message(FATAL_ERROR "error code: ${_rc}")
    endif()

    if(CNVCMD)
      string(REPLACE "^" ";" _outcnvcmd "${CNVCMD}^${OUT}")
      execute_process(COMMAND ${_outcnvcmd} ${_chkout} ${_chkerr} RESULT_VARIABLE _rc)
      file(WRITE ${OUT} "${_outvar}")
      if(_rc)
        message(FATAL_ERROR "out conversion error code: ${_rc}")
      endif()
    endif()

    if(CNV)
      string(REPLACE "^" ";" _outcnv "sh;${CNV}")
      execute_process(COMMAND ${_outcnv} INPUT_FILE "${OUT}" OUTPUT_VARIABLE _outvar RESULT_VARIABLE _rc)
      file(WRITE ${OUT} "${_outvar}")
      if(_rc)
        message(FATAL_ERROR "out conversion error code: ${_rc}")
      endif()

      if(ERR)
        execute_process(COMMAND ${_outcnv} INPUT_FILE "${ERR}" OUTPUT_VARIABLE _errvar RESULT_VARIABLE _rc)
        file(WRITE ${ERR} "${_errvar}")
        if(_rc)
          message(FATAL_ERROR "err conversion error code: ${_rc}")
        endif()
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
  execute_process(COMMAND ${_post} ${_cwd} OUTPUT_VARIABLE _outvar ERROR_VARIABLE _outvar RESULT_VARIABLE _rc)
  if(_outvar)
    message("-- BEGIN POST OUTPUT --")
    message("${_outvar}")
    message("-- END POST OUTPUT --")
  endif()
  if(_rc)
    message(FATAL_ERROR "post-command error code : ${_rc}")
  endif()
endif()

if(OUTREF)
  execute_process(COMMAND ${diff_cmd} ${OUT} ${OUTREF} OUTPUT_VARIABLE _outvar ERROR_VARIABLE _outvar RESULT_VARIABLE _rc)
  if(_outvar)
    message("-- BEGIN OUTDIFF OUTPUT --")
    message("${_outvar}")
    message("-- END OUTDIFF OUTPUT --")
  endif()
  if(_rc)
    message(FATAL_ERROR "compare 'stdout' error: ${_rc}")
  endif()
endif()

if(ERRREF)
  execute_process(COMMAND ${diff_cmd} ${ERR} ${ERRREF} OUTPUT_VARIABLE _outvar ERROR_VARIABLE _outvar RESULT_VARIABLE _rc)
  if(_outvar)
    message("-- BEGIN ERRDIFF OUTPUT --")
    message("${_outvar}")
    message("-- END ERRDIFF OUTPUT --")
  endif()
  if(_rc)
    message(FATAL_ERROR "compare 'stderr' error: ${_rc}")
  endif()
endif()

