############################################################################
# CMakeLists.txt file for building ROOT io/sql package
############################################################################

ROOT_GENERATE_DICTIONARY(G__SQLIO *.h MODULE SQLIO LINKDEF LinkDef.h DEPENDENCIES Net RIO  )

ROOT_LINKER_LIBRARY(SQLIO *.cxx G__SQLIO.cxx DEPENDENCIES Net RIO  )
ROOT_INSTALL_HEADERS()

