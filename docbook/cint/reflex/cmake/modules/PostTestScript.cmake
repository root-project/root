FILE(GLOB _logs ${CMAKE_CURRENT_SOURCE_DIR}/Testing/Temporary/*.log)
FILE(GLOB _tmp_logs ${CMAKE_CURRENT_SOURCE_DIR}/Testing/Temporary/*.log.tmp)

# format all temporary log file entry extensions to since ctest
# renames them after this script terminates.
FOREACH (_log ${_tmp_logs})
   STRING(REGEX REPLACE "\\.tmp$" "" _log ${_log})
   LIST(APPEND _logs ${_log})
ENDFOREACH (_log ${_tmp_logs})

# sort and remove duplicates that result from the above step
LIST(SORT _logs)
LIST(REMOVE_DUPLICATES _logs)

# print out the message
MESSAGE("\nThe following logs were generated:")
FOREACH (_log ${_logs})
   FILE(TO_NATIVE_PATH ${_log} _log)
   MESSAGE("\t ${_log}")
ENDFOREACH (_log ${_logs})
MESSAGE("")
