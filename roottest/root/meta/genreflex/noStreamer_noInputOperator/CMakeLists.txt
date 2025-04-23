# disable tests which are failing on Windows because of their PATH being too long
# (>260 characters) when run in the CI (Jenkins)
if(NOT MSVC OR NOT DEFINED ENV{JENKINS_HOME} OR win_broken_tests)

ROOTTEST_ADD_TESTDIRS()

# ------------------------------------------------------------------------------
# noStreamer tests

#   invalid noStreamer attribute
ROOTTEST_ADD_TEST(invalidNoStreamerGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo.h -o ${CMAKE_CURRENT_BINARY_DIR}/invalidNoStreamer_rflx.cpp  --selection_file=invalidNoStreamer_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF invalidNoStreamerGenreflex.ref
                  PASSRC 1)

#   custom streamer
ROOTTEST_GENERATE_REFLEX_DICTIONARY(customStreamer_rflx foo_custom_streamer.h SELECTION noStreamer_selection.xml)

ROOTTEST_ADD_TEST(customStreamer
                  MACRO  customStreamer.C
                  OUTREF customStreamer.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

#   generated streamer
ROOTTEST_GENERATE_REFLEX_DICTIONARY(noStreamer_false_rflx foo.h SELECTION noStreamer_false_selection.xml)

file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/noStreamer_false_rflx.cxx" NOSTREAMER_FALSE_RFLX)
ROOTTEST_ADD_TEST(noStreamerFalseDict
                  COMMAND ${grep_cmd} "void Foo::Streamer(TBuffer &R__b)" ${NOSTREAMER_FALSE_RFLX}
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS ${GENERATE_REFLEX_TEST})

# ------------------------------------------------------------------------------
# noInputOperator tests

#   invalid noInputOperator attribute
ROOTTEST_ADD_TEST(invalidNoInpuOperatorGenreflex
                  COMMAND ${ROOT_genreflex_CMD} foo.h -o ${CMAKE_CURRENT_BINARY_DIR}/invalidNoInputOperator_rflx.cpp  --selection_file=invalidNoInputOperator_selection.xml
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF invalidNoInputOperatorGenreflex.ref
                  PASSRC 1)

#   noInputOperator = true
ROOTTEST_GENERATE_REFLEX_DICTIONARY(noInputOperator_rflx foo_custom_input_operator.h SELECTION noInputOperator_selection.xml NO_ROOTMAP)

#   noInputOperator = false
ROOTTEST_GENERATE_REFLEX_DICTIONARY(noInputOperator_false_rflx foo_custom_input_operator.h SELECTION noInputOperator_false_selection.xml NO_ROOTMAP)

endif()
