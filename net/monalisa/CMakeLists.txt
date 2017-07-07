############################################################################
# CMakeLists.txt file for building ROOT net/monalisa package
# @author Pere Mato, CERN
############################################################################

include_directories(${MONALISA_INCLUDE_DIR})
#set_directory_properties(PROPERTIES COMPILE_DEFINITIONS R__KRB5INIT="${KRB5_INIT}")


ROOT_GENERATE_DICTIONARY(G__Monalisa *.h MODULE Monalisa LINKDEF LinkDef.h
                                                   DEPENDENCIES Net RIO)

ROOT_LINKER_LIBRARY(Monalisa *.cxx G__Monalisa.cxx LIBRARIES ${MONALISA_LIBRARIES}
                                                   DEPENDENCIES Net RIO )

ROOT_INSTALL_HEADERS()


