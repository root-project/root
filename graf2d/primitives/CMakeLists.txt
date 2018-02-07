############################################################################
# CMakeLists.txt file for building ROOT graf2d/primitives package
# @author Pere Mato, CERN
############################################################################


if(root7)
   ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
   ROOT_GLOB_HEADERS(Graf2dPrim_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)

   ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGPrimitives
      HEADERS ${Graf2dPrim_v7_dict_headers}
      SOURCES *.cxx ${root7src}
      DICTIONARY_OPTIONS "-writeEmptyRootPCM")

   if(testing)
      add_subdirectory(v7/test)
   endif()
endif()
