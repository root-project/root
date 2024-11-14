if(ROOT_pyroot_FOUND)
    ROOTTEST_ADD_TEST(compression_settings
                      MACRO test_compression_settings.py)
    ROOTTEST_ADD_TEST(input_validation
                      MACRO input_validation.py)
endif()

# merge 2 RNTuples passing an explicit compression
ROOTTEST_ADD_TEST(test_MergeChangeComp_explicit
                  COPY_TO_BUILDDIR merge_gen_input_tuples.C merge_changeComp_check_output.C
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C(\"test_comp_in1.root\", \"test_comp_in2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f404 test_comp_out.root test_comp_in1.root test_comp_in2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C(404, 404, \"test_comp_out.root\", \"test_comp_in1.root\", \"test_comp_in2.root\")"
)

# merge 2 RNTuples without passing a compression
ROOTTEST_ADD_TEST(test_MergeChangeComp_default
                  DEPENDS test_MergeChangeComp_explicit
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C(\"test_comp_in_def_1.root\", \"test_comp_in_def_2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f test_comp_out_def.root test_comp_in_def_1.root test_comp_in_def_2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C(505, 101, \"test_comp_out_def.root\", \"test_comp_in_def_1.root\", \"test_comp_in_def_2.root\")"
)

# merge 2 RNTuples using the first tuple's compression
# NOTE: merge_gen_input_tuples.C generates the first tuple as 505 and the second as 101
ROOTTEST_ADD_TEST(test_MergeChangeComp_first
                  DEPENDS test_MergeChangeComp_explicit
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C(\"test_comp_in_ff_1.root\", \"test_comp_in_ff_2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -ff test_comp_out_ff.root test_comp_in_ff_2.root test_comp_in_ff_1.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C(101, 101, \"test_comp_out_ff.root\", \"test_comp_in_ff_1.root\", \"test_comp_in_ff_2.root\")"
)

ROOTTEST_ADD_TEST(test_TreeChangeComp
                  COPY_TO_BUILDDIR hadd_gen_input_tree.C hadd_check_comp_tree.C
                  PRECMD ${ROOT_root_CMD} -q -b -l "hadd_gen_input_tree.C(\"test_comp_tree_in.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f401 test_comp_tree_out.root test_comp_tree_in.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_check_comp_tree.C(\"test_comp_tree_out.root\", \"test_comp_tree_in.root\", 401)"
)

ROOTTEST_ADD_TEST(test_MergeNestedSameName
                  COPY_TO_BUILDDIR hadd_check_nested_same_name.C
                  COMMAND ${ROOT_root_CMD} -q -b -l "hadd_check_nested_same_name.C"
                  OUTREF hadd_check_nested_same_name.ref
)

ROOTTEST_ADD_TEST(test_MergeCMSOpenDataRNTuples
                  COPY_TO_BUILDDIR hadd_check_cms.C cms_opendata_0_100.root cms_opendata_100_200.root cms_opendata_0_200.root
                  COMMAND ${ROOT_hadd_CMD} -f cms_opendata_merged.root cms_opendata_0_100.root cms_opendata_100_200.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_check_cms.C(\"cms_opendata_merged.root\", \"cms_opendata_0_200.root\")"
)
