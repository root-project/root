ROOTTEST_ADD_TESTDIRS()

# ------------------------------------------------------------------------------
# noStreamer tests
ROOTTEST_ADD_TEST(invalidNoStreamerGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo.h -o ${CMAKE_CURRENT_BINARY_DIR}/invalidNoStreamer_rflx.cpp  --selection_file=invalidNoStreamer_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF invalidNoStreamerGenreflex.ref)

ROOTTEST_ADD_TEST(noStreamerGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo.h -o ${CMAKE_CURRENT_BINARY_DIR}/noStreamer_rflx.cpp  --selection_file=noStreamer_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF emptyGenreflexOutput.ref)

ROOTTEST_ADD_TEST(noStreamerDict
                  COMMAND grep -v "void Foo::Streamer(TBuffer &R__b)" ${CMAKE_CURRENT_BINARY_DIR}/noStreamer_rflx.cpp
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS noStreamerGenreflex)

ROOTTEST_ADD_TEST(noStreamerFalseGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo.h -o ${CMAKE_CURRENT_BINARY_DIR}/noStreamer_false_rflx.cpp  --selection_file=noStreamer_false_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF emptyGenreflexOutput.ref)

ROOTTEST_ADD_TEST(noStreamerFalseDict
                  COMMAND grep "void Foo::Streamer(TBuffer &R__b)" ${CMAKE_CURRENT_BINARY_DIR}/noStreamer_false_rflx.cpp
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS noStreamerFalseGenreflex)

# ------------------------------------------------------------------------------
# noInputOperator tests
ROOTTEST_ADD_TEST(invalidNoInpuOperatorGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo.h -o ${CMAKE_CURRENT_BINARY_DIR}/invalidNoInputOperator_rflx.cpp  --selection_file=invalidNoInputOperator_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF invalidNoInputOperatorGenreflex.ref)

ROOTTEST_ADD_TEST(noInputOperatorGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo_custom_input_operator.h -o ${CMAKE_CURRENT_BINARY_DIR}/noInputOperator_rflx.cpp  --selection_file=noInputOperator_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF emptyGenreflexOutput.ref)

ROOTTEST_ADD_TEST(noInputOperatorFalseGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo_custom_input_operator.h -o ${CMAKE_CURRENT_BINARY_DIR}/noInputOperator_false_rflx.cpp  --selection_file=noInputOperator_false_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF emptyGenreflexOutput.ref)
