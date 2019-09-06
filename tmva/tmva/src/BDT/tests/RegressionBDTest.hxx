#include "gtest/gtest.h"
#include "RForestInference.hxx"
#include "TreeHelpers.hxx"

namespace RegressionTests {
//#include <xgboost/c_api.h> // for xgboost
//#include "generated_files/evaluate_forest2.h"
std::string events_file     = "./data/regression_events.csv";
std::string preds_file      = "./data/regression_python_predictions.csv";
std::string scores_file     = "./data/regression_python_scores.csv";
std::string json_model_file = "./data/regression_model.json";
std::string tmp_file        = "./data/regression_tmp.csv";
int         loop_size       = 256;

int tree_number = 1;

template <typename T, typename ForestType>
void test_regression(int loop_size = 1, std::string my_tmp_file = "")
{
   DataStructRegression<T> _data(events_file, preds_file, scores_file);

   ForestType Forest;
   Forest.LoadFromJson("my_key", json_model_file);
   if (loop_size > 1) {
      Forest.inference(_data.events_pointer, _data.rows, _data.cols, _data.scores.data(), loop_size);
   } else {
      Forest.inference(_data.events_pointer, _data.rows, _data.cols, _data.scores.data());
   }
   //_predict<T>(_data.scores.data(), _data.preds.size(), _data.preds);

   if ((!my_tmp_file.empty())) {
      write_csv<T>(my_tmp_file, _data.scores);
   }

   for (size_t i = 0; i < _data.scores.size(); i++) {
      ASSERT_EQ(_data.scores[i], _data.python_scores[i][0]);
   }
}
} // namespace RegressionTests
using namespace RegressionTests;

TEST(RegressionBDT, BranchedPredictionsSingleEvent)
{
   test_regression<float, ForestBranched<float>>(1, RegressionTests::tmp_file);
   // test_predictions<double, ForestBranched<double>>();
   // test_predictions<long double, ForestBranched<long double>>();
}

/*
TEST(forestBDT, BranchedPredictionsBatch)
{
   test_predictions<float, ForestBranched<float>>(loop_size, "./data/tmp4.csv");
}
TEST(forestBDT, BranchedPredictionsBatchDoubles)
{
   test_predictions<double, ForestBranched<double>>(loop_size);
   test_predictions<long double, ForestBranched<long double>>(loop_size);
}

TEST(forestBDT, BranchlessPredictions)
{
   test_predictions<float, ForestBranchless<float>>();
   test_predictions<double, ForestBranchless<double>>();
   test_predictions<long double, ForestBranchless<long double>>();
}
TEST(forestBDT, BranchlessPredictionsBatch)
{
   test_predictions<float, ForestBranchless<float>>(loop_size);
   // test_predictions<double, ForestBranchless<double>>(loop_size);
   // test_predictions<long double, ForestBranchless<long double>>(loop_size);
}

TEST(forestBDT, JitForestPredictions)
{
   test_predictions<float, ForestBranchedJIT<float>>();
   test_predictions<double, ForestBranchedJIT<double>>();
   test_predictions<long double, ForestBranchedJIT<long double>>();

   // test_predictions<float, ForestBranchedJIT<float>>(loop_size);
   // test_predictions<double, ForestBranchedJIT<double>>(loop_size);
   // test_predictions<long double, ForestBranchedJIT<long double>>(loop_size);
}
TEST(forestBDT, JitForestPredictionsBatch)
{
   test_predictions<float, ForestBranchedJIT<float>>(loop_size);
   test_predictions<double, ForestBranchedJIT<double>>(loop_size);
   test_predictions<long double, ForestBranchedJIT<long double>>(loop_size);
}

TEST(forestBDT, JitForestBranchless)
{
   test_predictions<float, ForestBranchlessJIT<float>>();
   test_predictions<double, ForestBranchlessJIT<double>>();
   test_predictions<long double, ForestBranchlessJIT<long double>>();
}
TEST(forestBDT, JitForestBranchlessBatch)
{
   test_predictions<float, ForestBranchlessJIT<float>>(loop_size);
   test_predictions<double, ForestBranchlessJIT<double>>(loop_size);
   test_predictions<long double, ForestBranchlessJIT<long double>>(loop_size);
}
// */
