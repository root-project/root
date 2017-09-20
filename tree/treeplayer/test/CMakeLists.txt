ROOT_ADD_UNITTEST_DIR(TreePlayer)
add_custom_command(TARGET treetreeplayertestUnit POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data.h data.h)

ROOT_ADD_GTEST(dataframe_simple dataframe/dataframe_simple.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(dataframe_regression dataframe/dataframe_regression.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(dataframe_interface dataframe/dataframe_interface.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(dataframe_utils dataframe/dataframe_utils.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(dataframe_nodes dataframe/dataframe_nodes.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(dataframe_histomodels dataframe/dataframe_histomodels.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(datasource_trivial dataframe/datasource_trivial.cxx LIBRARIES TreePlayer)
ROOT_ADD_GTEST(datasource_root dataframe/datasource_root.cxx LIBRARIES TreePlayer)
ROOT_ADD_PYUNITTEST(dataframe_histograms dataframe/dataframe_histograms.py)