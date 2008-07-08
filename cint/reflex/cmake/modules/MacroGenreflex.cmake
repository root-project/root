MACRO (GENREFLEX SOURCE TARGET)

   SET(_genreflex_extra_flags)

   GET_FILENAME_COMPONENT(_gccxmlpath ${GCCXML} PATH)
   GET_FILENAME_COMPONENT(_src_path ${SOURCE} ABSOLUTE)
   GET_FILENAME_COMPONENT(_src_name ${_src_path} NAME_WE)
   GET_FILENAME_COMPONENT(_src_dir ${_src_path} PATH)

   # extract the include dirs for resuse by genreflex 
   GET_DIRECTORY_PROPERTY(_cmake_include_directories INCLUDE_DIRECTORIES)
   SET(_genreflex_include_dirs)
   FOREACH (it ${_cmake_include_directories})
      SET(_genreflex_include_dirs ${_genreflex_include_dirs} "-I${it}")
   ENDFOREACH (it)

   # extract the definitions for resuse by genreflex 
   GET_DIRECTORY_PROPERTY(_cmake_definitions COMPILE_DEFINITIONS)
   SET(_genreflex_definitions)
   FOREACH (it ${_cmake_definitions})
      SET(_genreflex_definitions ${_genreflex_definitions} "-D${it}")
   ENDFOREACH (it)

   # mark reflex dictionary files as generated
   SET(_target "${CMAKE_CURRENT_BINARY_DIR}/${_src_name}_rflx.cpp")
   SET_SOURCE_FILES_PROPERTIES(${_target} PROPERTIES GENERATED 1)
   SET(${TARGET} ${_target})

   # link src to target through a genreflex command
   ADD_CUSTOM_COMMAND(OUTPUT ${_target}
                      COMMAND ${PYTHON_EXECUTABLE}
                      ARGS "${CMAKE_SOURCE_DIR}/python/genreflex/genreflex.py" "${_src_path}" -s "${_src_dir}/selection.xml" -o "${_target}" --quiet "--gccxmlpath=${_gccxmlpath}" ${_genreflex_include_dirs} ${_genreflex_definitions}
                      DEPENDS ${_src_path}
                      VERBATIM
   )

ENDMACRO (GENREFLEX SOURCE TARGET)