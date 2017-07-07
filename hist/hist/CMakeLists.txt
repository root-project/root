############################################################################
# CMakeLists.txt file for building ROOT hist/hist package
############################################################################

set(libname Hist)

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
    ROOT_GLOB_HEADERS(Hist_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
endif()

ROOT_GENERATE_DICTIONARY(G__${libname} *.h Math/*.h v5/*.h ${Hist_v7_dict_headers} MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Matrix MathCore RIO)

ROOT_LINKER_LIBRARY(${libname} *.cxx ${root7src} G__${libname}.cxx DEPENDENCIES Matrix MathCore RIO)
ROOT_INSTALL_HEADERS()

if(testing)
  add_subdirectory(test)

if(root7)
    add_subdirectory(v7/test)
  endif()
endif()
