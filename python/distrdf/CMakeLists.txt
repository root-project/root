# Test only if:
# * RDataFrame is available
# * PyROOT is available
# * pyspark is installed on that label for that python version
if (ROOT_dataframe_FOUND AND
    ROOT_pyroot_FOUND AND
    ROOT_test_distrdf_pyspark_FOUND)

file(COPY test_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY test_shared_libraries DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ROOTTEST_ADD_TEST(spark_test_definepersample 
                  MACRO test_definepersample.py)

ROOTTEST_ADD_TEST(spark_test_friend_trees
                  MACRO test_friend_trees.py)

ROOTTEST_ADD_TEST(spark_test_histo_write 
                  MACRO test_histo_write.py)

ROOTTEST_ADD_TEST(spark_test_include_headers 
                  MACRO test_include_headers.py)

ROOTTEST_ADD_TEST(spark_test_inv_mass 
                  MACRO test_inv_mass.py)

ROOTTEST_ADD_TEST(spark_test_reducer_merge 
                  MACRO test_reducer_merge.py)

endif()
