if(NOT ROOT_root7_FOUND)
  return()
endif()

ROOTTEST_ADD_TEST(test_Basics
                  MACRO basics.C
                  OUTREF test_basics.ref)

ROOT_ADD_GTEST(test_Merge merge.cxx LIBRARIES ROOT::ROOTNTuple)

ROOTTEST_ADD_TEST(test_MergeChangeComp
                  COPY_TO_BUILDDIR merge_gen_input_tuples.C merge_changeComp_check_output.C
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C+(\"test_in1.root\", \"test_in2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f404 test_out.root test_in1.root test_in2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C+(404, \"test_out.root\", \"test_in1.root\", \"test_in2.root\")"
)
