############################################################################
# CMakeLists.txt file for building ROOT cint/cintex package
############################################################################

ROOT_USE_PACKAGE(cint/reflex)
ROOT_USE_PACKAGE(core/metautils)
  
#---Cintex library----------------------------------------------------------
ROOT_LINKER_LIBRARY(Cintex *.cxx LIBRARIES Reflex Cint Core)  
ROOT_GENERATE_ROOTMAP(Cintex LINKDEF LinkDef.h DEPENDENCIES Reflex)

#---Install python modules--------------------------------------------------
install(FILES python/PyCintex.py DESTINATION ${runtimedir})
install(CODE "execute_process(COMMAND python -m py_compile ${CMAKE_INSTALL_PREFIX}/${runtimedir}/PyCintex.py )")
install(CODE "execute_process(COMMAND python -O -m py_compile ${CMAKE_INSTALL_PREFIX}/${runtimedir}/PyCintex.py )")

#---install headers----------------------------------------------------------
ROOT_INSTALL_HEADERS()  
