############################################################################
# CMakeLists.txt file for building ROOT io/sql package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

ROOT_GENERATE_DICTIONARY(G__SQL *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(SQLIO LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(SQLIO *.cxx G__SQL.cxx DEPENDENCIES Net RIO  )
ROOT_INSTALL_HEADERS()

