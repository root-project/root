if(ROOT_roofit_FOUND)
  ROOTTEST_ADD_TEST(read-scientificnotation
                    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/read_scientific_notation.py
                    OUTCNVCMD grep -v -e "Wouter"
                    OUTREF read_scientific_notation.ref)
endif()
