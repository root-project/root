FIND_PACKAGE(GCCXML)
FIND_PACKAGE(PythonInterp)

INCLUDE(PythonMacros)

#####################   python    ####################

# if no python interpeter found, force to 'python' and
# just hope it works (maybe this should be removed?)
IF (NOT PYTHONINTERP_FOUND)
   SET(PYTHON_EXECUTABLE "python")
   MESSAGE(STATUS "WARNING: Setting Python executable to: ${PYTHON_EXECUTABLE}")
ENDIF (NOT PYTHONINTERP_FOUND)

#####################   sources   ####################

FILE(GLOB_RECURSE GENREFLEX_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} genreflex/*.py)
SET(GENREFLEX_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/genreflex/genreflex.pyc PARENT_SCOPE)

#####################  gccxmlpath ####################

# create gccxmlpath.py file under binary dir
# todo: this should be more dynamic since installer packages
# will have the paths set at configure time.
IF (GCCXML)
   GET_FILENAME_COMPONENT(GCCXMLPATH ${GCCXML} PATH)
   SET(_gccxmlpath_filename ${CMAKE_CURRENT_BINARY_DIR}/genreflex/gccxmlpath.py)
   FILE(WRITE "${_gccxmlpath_filename}" "gccxmlpath='${GCCXMLPATH}'\n")
   #LIST(APPEND GENREFLEX_SRCS ${_gccxmlpath_filename})
ENDIF (GCCXML)

#####################  genreflex  #################### 

# compile genreflex sources to the binary dir
# dir (for out-of-source builds) to avoid polluting the source dir
PYTHON_COMPILE(GENREFLEX_OBJS ${GENREFLEX_SRCS})

# create a target that links the 'genreflex' target to the
# above custom commands. a fake executable target was used
# since custom targets are always considered out of date
SET(_genreflex_main ${CMAKE_CURRENT_BINARY_DIR}/genreflex/Main.cxx)
ADD_CUSTOM_COMMAND(OUTPUT ${_genreflex_main}
                   COMMAND ${CMAKE_COMMAND} -E copy ${REFLEX_TEMPLATE_DIR}/Main.cxx ${_genreflex_main}
                   DEPENDS ${GENREFLEX_OBJS} ${REFLEX_TEMPLATE_DIR}/Main.cxx)
REFLEX_ADD_EXECUTABLE(genreflex EXCLUDE_FROM_ALL ${_genreflex_main})

# change the executable name since a file called 'genreflex'
# (under UNIX) would conflict with the directory of the same name
SET_TARGET_PROPERTIES(genreflex PROPERTIES OUTPUT_NAME genreflex.bin)

# make the genreflex target depend on the generated python 'objects'
SET_SOURCE_FILES_PROPERTIES(${_genreflex_main} PROPERTIES OBJECT_DEPENDS "${GENREFLEX_OBJS}")

#####################   script    #################### 

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
        DESTINATION ${LIB_INSTALL_DIR}/python
        FILES_MATCHING PATTERN "*.py"
        PATTERN "gccxmlpath.py" EXCLUDE)

INSTALL(PROGRAMS ${_genreflex_filename}
        DESTINATION ${BIN_INSTALL_DIR})

