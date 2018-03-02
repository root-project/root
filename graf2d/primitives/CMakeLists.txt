############################################################################
# CMakeLists.txt file for building ROOT graf2d/primitives package
# @author Pere Mato, CERN
############################################################################


if(root7)
   ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
   ROOT_GLOB_HEADERS(Graf2dPrim_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)

   ROOT_GENERATE_DICTIONARY(G__Graf2dPrim_v7
                            ${Graf2dPrim_v7_dict_headers}
                            MODULE ROOTv7GraphicsPrimitives
                            OPTIONS "-writeEmptyRootPCM"
                            LINKDEF LinkDef.h)

   ROOT_LINKER_LIBRARY(ROOTGraphics
                       $<TARGET_OBJECTS:ROOTv7GraphicsGPad>
                       ${root7src}
                       G__Graf2dPrim_v7.cxx
                       DEPENDENCIES RIO)

   ROOT_INSTALL_HEADERS()

   if(testing)
      add_subdirectory(v7/test)
   endif()
endif()
