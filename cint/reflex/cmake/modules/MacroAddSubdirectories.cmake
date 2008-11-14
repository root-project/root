# MACRO_ADD_SUBDIRECTORIES(_source_dir)
#   This macro adds all subdirectories containing top-level CMakeLists.txt files to the build

MACRO (MACRO_ADD_SUBDIRECTORIES _source_dir)

   FILE(GLOB_RECURSE _subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${_source_dir}/CMakeLists.txt)
   LIST(SORT _subdirs)
   SET(_parent "--")

   FOREACH (_it ${_subdirs})

      GET_FILENAME_COMPONENT(_subdir ${_it} PATH)

      IF (NOT ${_subdir} MATCHES "^${_parent}")
         SET(_parent ${_subdir})
         ADD_SUBDIRECTORY(${_subdir})
      ENDIF (NOT ${_subdir} MATCHES "^${_parent}")

   ENDFOREACH (_it ${_subdirs})

ENDMACRO (MACRO_ADD_SUBDIRECTORIES _source_dir)
