FIND_PACKAGE(GCCXML)
FIND_PACKAGE(PythonInterp)

#####################   python    ####################

# if no python interpeter found, force to 'python' and
# just hope it works (maybe this should be removed?)
IF (NOT PYTHONINTERP_FOUND)
   SET(PYTHON_EXECUTABLE "python")
ENDIF (NOT PYTHONINTERP_FOUND)

#####################   sources   ####################

FILE(GLOB_RECURSE GENREFLEX_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} genreflex/*.py)
SET(GENREFLEX_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/genreflex/genreflex.pyc PARENT_SCOPE)
SET(GENREFLEX_OBJS)

#####################   targets   ####################

# compile modified genreflex sources to the binary dir
# dir (for out-of-source builds) to avoid polluting the source dir
FOREACH (f ${GENREFLEX_SRCS})

   SET(_in ${CMAKE_CURRENT_SOURCE_DIR}/${f})
   SET(_out ${CMAKE_CURRENT_BINARY_DIR}/${f}c)

   ADD_CUSTOM_COMMAND(OUTPUT ${_out}
                      COMMAND ${PYTHON_EXECUTABLE}
                      ARGS -c "import py_compile; py_compile.compile('${_in}', '${_out}')"
                      DEPENDS ${_in}
                      VERBATIM)

   LIST(APPEND GENREFLEX_OBJS ${_out})

ENDFOREACH (f ${GENREFLEX_SRCS})

# create a top-level custom target that links
# the all target to the above custom commands
ADD_CUSTOM_TARGET(genreflex ALL
                  DEPENDS ${GENREFLEX_OBJS})

# set the list of genreflex 'objects' to be used as
# dependencies for the genreflex macro. i.e: changing
# genreflex sources relaunches genreflex
SET(GENREFLEX_OBJS ${GENREFLEX_OBJS} PARENT_SCOPE)

#####################  gccxmlpath ####################

# create gccxmlpath.py file under binary dir
# todo: this should be more dynamic since installer packages
# will have the paths set at configure time.
IF (GCCXML)
   GET_FILENAME_COMPONENT(GCCXMLPATH ${GCCXML} PATH)
   SET(_gccxmlpath_filename ${CMAKE_CURRENT_BINARY_DIR}/genreflex/gccxmlpath.py)
   FILE(WRITE "${_gccxmlpath_filename}" "gccxmlpath='${GCCXMLPATH}'\n")
ENDIF (GCCXML)

#####################  genreflex  ####################

# create genreflex script for later installation.
# todo: this should be more dynamic since installer packages
# will have the paths set at configure time.
IF (WIN32)
   SET(_genreflex_filename ${CMAKE_CURRENT_BINARY_DIR}/bin/genreflex.bat)
   FILE(WRITE "${_genreflex_filename}" "@echo off\n\"${PYTHON_EXECUTABLE}\" ${CMAKE_INSTALL_PREFIX}\\lib\\python\\genreflex\\genreflex.py %*\n")
ELSE (WIN32)
   SET(_genreflex_filename ${CMAKE_CURRENT_BINARY_DIR}/bin/genreflex)
   FILE(WRITE "${_genreflex_filename}" "#!/bin/sh\n${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/lib/python/genreflex/genreflex.py $*\n")
ENDIF (WIN32)

#####################   install   ####################

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/genreflex
        DESTINATION lib/python
        FILES_MATCHING PATTERN "*.py"
        PATTERN "gccxmlpath.py" EXCLUDE)

INSTALL(PROGRAMS ${_genreflex_filename}
        DESTINATION bin)

