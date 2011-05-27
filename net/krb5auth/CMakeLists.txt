############################################################################
# CMakeLists.txt file for building ROOT net/krb5auth package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(net/auth)

include_directories(${KRB5_INCLUDE_DIR})
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS R__KRB5INIT="${KRB5_INIT}")

ROOT_GENERATE_DICTIONARY(G__Krb5Auth TKSocket.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Krb5Auth LINKDEF LinkDef.h DEPENDENCIES RootAuth Net)
ROOT_LINKER_LIBRARY(Krb5Auth *.cxx G__Krb5Auth.cxx  LIBRARIES ${KRB5_LIBRARIES})

ROOT_INSTALL_HEADERS()


