############################################################################
# CMakeLists.txt file for building ROOT net/monalisa package
# @author Pere Mato, CERN
############################################################################

include_directories(${MONALISA_INCLUDE_DIR})
#set_directory_properties(PROPERTIES COMPILE_DEFINITIONS R__KRB5INIT="${KRB5_INIT}")

ROOT_STANDARD_LIBRARY_PACKAGE(Monalisa
                              LIBRARIES ${MONALISA_LIBRARIES}
                              DEPENDENCIES Net RIO)


