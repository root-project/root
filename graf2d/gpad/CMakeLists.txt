############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

if(root7)
    set(root7src v7/src/)
    ROOT_GLOB_HEADERS(Gpad_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
endif()

ROOT_GENERATE_DICTIONARY(G__Gpad *.h ${Gpad_v7_dict_headers} MODULE Gpad LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Graf MultiProc Hist)

ROOT_LINKER_LIBRARY(Gpad *.cxx ${root7src} G__Gpad.cxx DEPENDENCIES Graf Hist)
ROOT_INSTALL_HEADERS()
