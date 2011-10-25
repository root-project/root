#---------------------------------------------------------------------------------------------------
# ROOT test driver
#   Script arguments: 
#     CMD command to be executed for the test
#     PRE command to be executed before the test command
#     POST command to be executed after the test command
#     OUT file to collect stdout
#     ERR file to collect stderr
#     ENV evironment VAR1=Value1;VAR2=Value2
#     CWD current working directory
#     DBG debug flag

if(DBG)
  message(STATUS "ENV=${ENV}")
endif()

#---Massage arguments---------------------------------------------------------------------------------
if(CMD)
  string(REPLACE "#" ";" _cmd ${CMD})
  if(DBG)
    message(STATUS "testdriver:CMD=${_cmd}")
  endif()
endif()

if(PRE)
  string(REPLACE "#" ";" _pre ${PRE})
  if(DBG)
    message(STATUS "testdriver:PRE=${_pre}")
  endif()
endif()

if(POST)
  string(REPLACE "#" ";" _post ${POST})
  if(DBG)
    message(STATUS "testdriver:POST=${_post}")
  endif()
endif()

if(OUT)
  set(_out OUTPUT_FILE ${OUT})
  if(DBG)
    message(STATUS "testdriver:OUT=${OUT}")
  endif()
endif()

if(ERR)
  set(_err ERROR_FILE ${ERR})
  if(DBG)
    message(STATUS "testdriver:ERR=${ERR}")
  endif()
#else()
#  set(_err ERROR_VARIABLE _errvar)
endif()

if(CWD)
  set(_cwd WORKING_DIRECTORY ${CWD})
  if(DBG)
    message(STATUS "testdriver:CWD=${CWD}")
  endif()
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

#---Execute pre-command-----------------------------------------------------------------------------
if(PRE)
  execute_process(COMMAND ${_pre} ${_cwd} RESULT_VARIABLE _rc)
  if(_rc)
    message(FATAL_ERROR "pre-command error code : ${_rc}")
  endif()
endif()

if(CMD)
  #---Execute the actual test ------------------------------------------------------------------------
  execute_process(COMMAND ${_cmd} ${_out} ${_err} ${_cwd} RESULT_VARIABLE _rc)

  #---Return error is test returned an error code of write somthing to the stderr---------------------
  #if(_errvar)
  #  message(FATAL_ERROR "output error: ${_errvar}")
  #endif()
  if(_rc)
    message(FATAL_ERROR "error code: ${_rc}")
  endif()
endif()


#---Execute post-command-----------------------------------------------------------------------------
if(POST)
  execute_process(COMMAND ${_post} ${_cwd} RESULT_VARIABLE _rc)
  if(_rc)
    message(FATAL_ERROR "post-command error code : ${_rc}")
  endif()
endif()






