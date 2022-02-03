#arguments SRC - source file
#          TGT - target file, use only directory

get_filename_component(dir ${TGT} DIRECTORY)

file(COPY ${SRC} DESTINATION ${dir})
