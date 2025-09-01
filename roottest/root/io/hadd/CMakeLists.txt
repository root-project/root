if(pyroot)
  ROOT_ADD_PYUNITTEST(input_validation input_validation.py GENERIC PYTHON_DEPS pytest)
endif()


################################################################
#                    compression tests                         #
################################################################
if(pyroot)
  ROOT_ADD_PYUNITTEST(compression_settings test_compression_settings.py GENERIC PYTHON_DEPS pytest)
endif()

# merge 2 RNTuples passing an explicit compression
ROOTTEST_ADD_TEST(test_MergeChangeComp_explicit
                  COPY_TO_BUILDDIR merge_gen_input_tuples.C merge_changeComp_check_output.C
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C(\"test_comp_in1.root\", \"test_comp_in2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f404 test_comp_out.root test_comp_in1.root test_comp_in2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C(404, 404, \"test_comp_out.root\", \"test_comp_in1.root\", \"test_comp_in2.root\")"
                  FIXTURES_SETUP root-io-hadd-test_MergeChangeComp_explicit-fixture
)

# merge 2 RNTuples without passing a compression
ROOTTEST_ADD_TEST(test_MergeChangeComp_default
                  FIXTURES_REQUIRED root-io-hadd-test_MergeChangeComp_explicit-fixture
                  PRECMD ${ROOT_root_CMD} -q -b -l "merge_gen_input_tuples.C(\"test_comp_in_def_1.root\", \"test_comp_in_def_2.root\")"
                  COMMAND ${ROOT_hadd_CMD} -f test_comp_out_def.root test_comp_in_def_1.root test_comp_in_def_2.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "merge_changeComp_check_output.C(505, 101, \"test_comp_out_def.root\", \"test_comp_in_def_1.root\", \"test_comp_in_def_2.root\")"
)

# merge 2 RNTuples using the first tuple's compression
# NOTE: merge_gen_input_tuples.C generates the first tuple as 505 and the second as 101
ROOTTEST_ADD_TEST(test_MergeChangeComp_first
                  FIXTURES_REQUIRED root-io-hadd-test_MergeChangeComp_explicit-fixture
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

################################################################
#                 miscellaneous tests                          #
################################################################

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

ROOTTEST_ADD_TEST(test_MergeFilterBlacklist
                  COPY_TO_BUILDDIR hadd_gen_input_files.C hadd_check_filtered.C hadd_filter.txt
                  PRECMD ${ROOT_root_CMD} -q -b -l "hadd_gen_input_files.C(\"hadd_input\")"
                  COMMAND ${ROOT_hadd_CMD} -L hadd_filter.txt -Ltype SkipListed -f filtered_blacklist.root hadd_input0.root hadd_input1.root hadd_input2.root hadd_input3.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_check_filtered.C(\"filtered_blacklist.root\", \"blacklist\")"
                  FIXTURES_SETUP root-io-hadd-test_MergeFilterBlacklist-fixture
)

ROOTTEST_ADD_TEST(test_MergeFilterWhitelist
                  COPY_TO_BUILDDIR hadd_gen_input_files.C hadd_check_filtered.C hadd_filter.txt
                  FIXTURES_REQUIRED root-io-hadd-test_MergeFilterBlacklist-fixture
                  COMMAND ${ROOT_hadd_CMD} -L hadd_filter.txt -Ltype OnlyListed -f filtered_whitelist.root hadd_input0.root hadd_input1.root hadd_input2.root hadd_input3.root
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_check_filtered.C(\"filtered_whitelist.root\", \"whitelist\")"
)

################################################################
#                 argument parsing tests                       #
################################################################
ROOTTEST_ADD_TEST(test_hadd_args_gen_input
                  COPY_TO_BUILDDIR hadd_gen_input_tree.C hadd_args_verify.C
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_in.root
                  COMMAND ${ROOT_root_CMD} -q -b -l "hadd_gen_input_tree.C(\"hadd_args_in.root\")"
                  FIXTURES_SETUP root-io-hadd-test_hadd_args_gen_input-fixture
)

ROOTTEST_ADD_TEST(test_hadd_args_n0
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_n0_out.root
                  COMMAND ${ROOT_hadd_CMD} -n0 hadd_args_n0_out.root hadd_args_in.root
                  OUTREF hadd_args_n0.ref
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"hadd_args_n0_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_na
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_na_out.root
                  COMMAND ${ROOT_hadd_CMD} -na hadd_args_na_out.root hadd_args_in.root
                  PASSREGEX "Error in <hadd>: the argument after '-n' flag was not of the expected type."
)

ROOTTEST_ADD_TEST(test_hadd_args_n_1
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_n_1_out.root
                  COMMAND ${ROOT_hadd_CMD} -n=1 hadd_args_n_1_out.root hadd_args_in.root
                  OUTREF hadd_args_n_1.ref
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"hadd_args_n_1_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_jfa
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_jfa_out.root
                  COMMAND ${ROOT_hadd_CMD} -jfa hadd_args_jfa_out.root hadd_args_in.root
                  PASSREGEX "Error in <hadd>: the argument after '-j' flag was not of the expected type"
)

ROOTTEST_ADD_TEST(test_hadd_args_j_10a
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_j_10a_out.root
                  COMMAND ${ROOT_hadd_CMD} -j 10a hadd_args_j_10a_out.root hadd_args_in.root
                  PASSREGEX "Error in <hadd>: error parsing integer argument '10a'"
)

ROOTTEST_ADD_TEST(test_hadd_args_j
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_j_out.root
                  COMMAND ${ROOT_hadd_CMD} -j hadd_args_j_out.root hadd_args_in.root
                  PASSRC 0
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"hadd_args_j_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_ff_dbg_j_2
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_ff_dbg_j_2_out.root
                  COMMAND ${ROOT_hadd_CMD} -ff hadd_args_ff_dbg_j_2_out.root hadd_args_in.root hadd_args_in.root -dbg -j 2
                  OUTREF hadd_args_ff_dbg_j_2.ref
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"hadd_args_ff_dbg_j_2_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_minusminus
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f -- --hadd_args_out.root
                  COMMAND ${ROOT_hadd_CMD} -f0 -- --hadd_args_out.root hadd_args_in.root
                  OUTREF hadd_args_minusminus.ref
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"--hadd_args_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_minusminus2
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f -- --hadd_args2_out.root
                  COMMAND ${ROOT_hadd_CMD} -f --hadd_args2_out.root -- hadd_args_in.root
                  PASSREGEX "Warning in <hadd>: unknown flag: --hadd_args2_out.root"
                  PASSRC 1
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"--hadd_args2_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_multiple_pos
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_multiple_pos_out.root
                  COMMAND ${ROOT_hadd_CMD} -ff hadd_args_multiple_pos_out.root -k hadd_args_in.root
                  PASSREGEX "Error in <hadd>: seen a positional argument 'hadd_args_in.root' after some flags."
)

ROOTTEST_ADD_TEST(test_hadd_args_cachesize
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_cachesize_out.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_cachesize_out.root hadd_args_in.root -cachesize 100M -f
                  OUTREF hadd_args_cachesize.ref
                  POSTCMD ${ROOT_root_CMD} -q -b -l "hadd_args_verify.C(\"hadd_args_cachesize_out.root\")"
)

ROOTTEST_ADD_TEST(test_hadd_args_missing_arg
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_missing_arg_out.root
                  COMMAND ${ROOT_hadd_CMD} -n hadd_args_missing_arg_out.root hadd_args_in.root
                  PASSREGEX "Error in <hadd>: the argument after '-n' flag was not of the expected type"
)

ROOTTEST_ADD_TEST(test_hadd_args_missing_arg2
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_missing_arg2_out.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_missing_arg2_out.root hadd_args_in.root -n
                  PASSREGEX "Error in <hadd>: expected argument after '-n' flag."
)

ROOTTEST_ADD_TEST(test_hadd_args_wrong_ltype
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_wrong_ltype.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_wrong_ltype.root hadd_args_in.root -L foo -Ltype onlylisted
                  PASSREGEX "Error in <hadd>: invalid argument for -Ltype: 'onlylisted'"
)

ROOTTEST_ADD_TEST(test_hadd_args_missing_ltype
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_missing_ltype.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_missing_ltype.root hadd_args_in.root -L foo
                  PASSREGEX "Error in <hadd>: -L must always be passed along with -Ltype"
)

ROOTTEST_ADD_TEST(test_hadd_args_missing_l
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_missing_l.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_missing_l.root hadd_args_in.root -Ltype SkipListed
                  PASSREGEX "Error in <hadd>: -L must always be passed along with -Ltype"
)

ROOTTEST_ADD_TEST(test_hadd_args_quiet
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_quiet.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_quiet.root hadd_args_in.root -v2
                  OUTREF hadd_args_quiet.ref
)

ROOTTEST_ADD_TEST(test_hadd_args_quieter
                  FIXTURES_REQUIRED root-io-hadd-test_hadd_args_gen_input-fixture
                  PRECMD ${CMAKE_COMMAND} -E rm -f hadd_args_quieter.root
                  COMMAND ${ROOT_hadd_CMD} hadd_args_quieter.root hadd_args_in.root -v1
                  OUTREF hadd_args_quieter.ref
)
