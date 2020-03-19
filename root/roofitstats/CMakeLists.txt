if(ROOT_roofit_FOUND)
  ROOTTEST_ADD_TEST(read-scientificnotation
                    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/read_scientific_notation.py
                    OUTCNVCMD grep -v -e "Wouter"
                    OUTREF read_scientific_notation.ref)

  ROOTTEST_ADD_TEST(RooDataSet_ASCII_in
                    MACRO ${CMAKE_CURRENT_SOURCE_DIR}/ASCII-in-out.C
                    MACROARG "false, \"${CMAKE_CURRENT_SOURCE_DIR}/ASCII-in-out_data.txt\""
                    OUTCNVCMD grep -v -e "Wouter" -e "\.txt" -e "reserved"
                    OUTREF ASCII-in-out.ref)
  ROOTTEST_ADD_TEST(RooDataSet_ASCII_out
                    MACRO ${CMAKE_CURRENT_SOURCE_DIR}/ASCII-in-out.C
                    MACROARG "true, \"./ASCII-in-out_result.txt\""
                    POSTCMD diff ${CMAKE_CURRENT_SOURCE_DIR}/ASCII-in-out_data.txt ./ASCII-in-out_result.txt)

  ROOT_ADD_GTEST(stabilityTests stabilityTests.cxx LIBRARIES Core Tree RooFitCore RooFit
                 COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/stabilityTests_data_1.root)

  add_subdirectory(vectorisedPDFs) 
endif()
