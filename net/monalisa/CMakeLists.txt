############################################################################
# CMakeLists.txt file for building ROOT net/monalisa package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(net/net)

include_directories(${MONALISA_INCLUDE_DIR})
#set_directory_properties(PROPERTIES COMPILE_DEFINITIONS R__KRB5INIT="${KRB5_INIT}")


ROOT_GENERATE_DICTIONARY(G__Monalisa *.h MODULE Monalisa LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Monalisa *.cxx G__Monalisa.cxx LIBRARIES ${MONALISA_LIBRARIES}
                                                   DEPENDENCIES Net RIO MathCore )

ROOT_INSTALL_HEADERS()


