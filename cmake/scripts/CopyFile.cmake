#arguments SRC - source file
#          TGT - target file

if(NOT EXISTS "${TGT}")
   message("Create ${TGT}")
   set(do_copy ON)
else()
   # not possible to use file operation IS_NEWER_THAN while by copying of the
   # file sub-seconds of file NOT copied !!!
   # get time in UTC seconds, only they are preserved by cmake file copy
   file(TIMESTAMP ${SRC} src_time "%s")
   file(TIMESTAMP ${TGT} tgt_time "%s")
   if(src_time GREATER tgt_time)
      message("Update ${TGT}")
      set(do_copy ON)
   endif()
endif()

if(do_copy)
   get_filename_component(dir ${TGT} DIRECTORY)
   file(COPY ${SRC} DESTINATION ${dir})
endif()
