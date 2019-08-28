#include "gtest/gtest.h"
#include "RForestInference.hxx"
#include "bdt_helpers.hxx"

//#include <xgboost/c_api.h> // for xgboost
//#include "generated_files/evaluate_forest2.h"
std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/python_predictions.csv";
std::string json_model_file = "./data/model.json";
int         loop_size       = 256;

int tree_number = 1;

template <typename T>
struct DataStruct {
   const std::vector<std::vector<T>>    events_vec_vec;
   const std::vector<T>                 events_vector;
   const T *                            events_pointer = nullptr;
   const std::vector<std::vector<bool>> groundtruth;
   std::vector<bool>                    preds;
   const int                            rows, cols;

   DataStruct(const std::string &events_file, const std::string &preds_file)
      : events_vec_vec(read_csv<T>(events_file)), events_vector(convert_VecMatrix2Vec<T>(events_vec_vec)),
        events_pointer(events_vector.data()), groundtruth(read_csv<bool>(preds_file)), rows(events_vec_vec.size()),
        cols(events_vec_vec[0].size())
   {
      preds.reserve(rows);
   }
};

template <typename T, typename ForestType>
void test_predictions()
{
   DataStruct<T> _data(events_file, preds_file);

   ForestType Forest;
   Forest.LoadFromJson("lala", json_model_file);
   Forest.inference(_data.events_pointer, _data.rows, _data.cols, _data.preds);

   for (size_t i = 0; i < _data.preds.size(); i++) {
      ASSERT_EQ(_data.preds[i], _data.groundtruth[i][0]);
   }
}

TEST(forestBDT, BranchedPredictions)
{
   test_predictions<float, ForestBranched<float>>();
   test_predictions<double, ForestBranched<double>>();
   test_predictions<long double, ForestBranched<long double>>();
}

TEST(forestBDT, BranchlessPredictions)
{
   test_predictions<float, ForestBranchless<float>>();
   test_predictions<double, ForestBranchless<double>>();
   test_predictions<long double, ForestBranchless<long double>>();
}

TEST(forestBDT, JitForestPredictions)
{
   test_predictions<float, ForestBranchedJIT<float>>();
   test_predictions<double, ForestBranchedJIT<double>>();
   // test_predictions<long double, ForestBranchedJIT<long double>>();
}

TEST(forestBDT, JitForestBranchless)
{
   test_predictions<float, ForestBranchlessJIT<float>>();
   test_predictions<double, ForestBranchlessJIT<double>>();
   // test_predictions<long double, ForestBranchlessJIT<long double>>();
}

/*
TEST(forestBDT, UniqueBatchPredictions)
{

   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>  groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest(json_model_file);
   Forest.do_predictions_batch(events_vector, preds, loop_size);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}
*/
