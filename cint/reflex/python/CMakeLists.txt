FIND_PACKAGE(GCCXML)
FIND_PACKAGE(PythonInterp)

# if no python interpeter found,
# force to 'python' and just hope it works
# maybe this should be removed?
IF (NOT PYTHONINTERP_FOUND)
   SET(PYTHON_EXECUTABLE "python")
ENDIF (NOT PYTHONINTERP_FOUND)

# create gccxmlpath.py file under lib
IF (GCCXML)
   GET_FILENAME_COMPONENT(GCCXMLPATH ${GCCXML} PATH)
   SET(_gccxmlpath_filename ${CMAKE_CURRENT_BINARY_DIR}/install/gccxmlpath.py)
   FILE(WRITE "${_gccxmlpath_filename}" "gccxmlpath='${GCCXMLPATH}'\n")
   INSTALL(FILES ${_gccxmlpath_filename} DESTINATION lib/python/genreflex)
ENDIF (GCCXML)

# create genreflex script under bin
IF (WIN32)
   SET(_genreflex_filename ${CMAKE_CURRENT_BINARY_DIR}/bin/genreflex.bat)
   FILE(WRITE "${_genreflex_filename}" "@echo off\n\"${PYTHON_EXECUTABLE}\" ${CMAKE_INSTALL_PREFIX}\\lib\\python\\genreflex\\genreflex.py %*\n")
ELSE (WIN32)
   SET(_genreflex_filename ${CMAKE_CURRENT_BINARY_DIR}/bin/genreflex)
   FILE(WRITE "${_genreflex_filename}" "#!/bin/sh\n${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/lib/python/genreflex/genreflex.py $*\n")
ENDIF (WIN32)

# install
INSTALL(DIRECTORY genreflex DESTINATION lib/python
	     FILES_MATCHING PATTERN "*.py"
        PATTERN "genreflex-rootcint.py" EXCLUDE)
INSTALL(PROGRAMS ${_genreflex_filename} DESTINATION bin)
