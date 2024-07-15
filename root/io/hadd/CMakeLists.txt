if(ROOT_pyroot_FOUND)
    ROOTTEST_ADD_TEST(compression_settings
                      MACRO test_compression_settings.py)

    ROOTTEST_ADD_TEST(input_validation
                      MACRO input_validation.py)
endif()

ROOTTEST_ADD_TEST(test_Merge
                  COPY_TO_BUILDDIR merge_gen_input_tuples.C merge_check_output.C
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C+(\"test_merge_in1.root\", \"test_merge_in2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f test_merge_out.root test_merge_in1.root test_merge_in2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_check_output.C+(404, \"test_merge_out.root\", \"test_merge_in1.root\", \"test_merge_in2.root\")"
)

ROOTTEST_ADD_TEST(test_MergeChangeComp
                  COPY_TO_BUILDDIR merge_gen_input_tuples.C merge_changeComp_check_output.C
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C+(\"test_comp_in1.root\", \"test_comp_in2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f404 test_comp_out.root test_comp_in1.root test_comp_in2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C+(404, \"test_comp_out.root\", \"test_comp_in1.root\", \"test_comp_in2.root\")"
)
