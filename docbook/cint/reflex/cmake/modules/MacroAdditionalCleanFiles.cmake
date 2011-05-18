# MACRO_ADDITIONAL_CLEAN_FILES(files...)

# Copyright (c) 2006, Alexander Neundorf, <neundorf@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

MACRO (MACRO_ADDITIONAL_CLEAN_FILES)

   GET_DIRECTORY_PROPERTY(_tmp_DIR_PROPS ADDITIONAL_MAKE_CLEAN_FILES )

   IF (_tmp_DIR_PROPS)
      SET(_tmp_DIR_PROPS ${_tmp_DIR_PROPS} ${ARGN})
   ELSE (_tmp_DIR_PROPS)
      SET(_tmp_DIR_PROPS ${ARGN})
   ENDIF (_tmp_DIR_PROPS)

   SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${_tmp_DIR_PROPS}")

ENDMACRO (MACRO_ADDITIONAL_CLEAN_FILES)
