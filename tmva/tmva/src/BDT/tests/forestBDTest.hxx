#include "gtest/gtest.h"
#include "forest.hxx"
#include "bdt_helpers.hxx"

//#include <xgboost/c_api.h> // for xgboost
//#include "generated_files/evaluate_forest2.h"
std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/python_predictions.csv";
std::string json_model_file = "./data/model.json";
int         loop_size       = 256;

TEST(forestBDT, UniquePredictions)
{
   const std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>        groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>                     preds;
   preds.reserve(events_vector.size());

   ForestBranched<float> Forest;
   Forest.LoadFromJson("lala", json_model_file);
   int rows = 5;
   int cols = 5;
   Forest.inference(events_vector, rows, cols, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}

TEST(forestBDT, ArrayPredictions)
{
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>  groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   ForestBranchless<float> Forest;
   Forest.LoadFromJson("lala", json_model_file);
   int rows = 5;
   int cols = 5;
   Forest.inference(events_vector, rows, cols, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }

   preds.clear();

   Forest.inference(events_vector.data(), rows, cols, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
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

TEST(forestBDT, UniqueBatch2Predictions)
{
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>  groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest(json_model_file);
   Forest.do_predictions(events_vector, preds, loop_size);

   // std::string tmp_filename = "./data/tmp2.csv";
   // write_csv(tmp_filename, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}



TEST(forestBDT, JitForestPredictions)
{
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>  groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   Forest<std::function<bool(const std::vector<float> &)>> Forest;
   Forest.get_Forest(json_model_file);
   Forest.do_predictions(events_vector, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}

TEST(forestBDT, JitPredictionsBranchless)
{
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>  groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   Forest<std::function<bool(const float *)>> Forest;
   Forest.get_Forest(json_model_file);
   Forest.do_predictions(events_vector, preds);

   std::string tmp_filename = "data/tmp3.csv";
   write_csv(tmp_filename, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}

TEST(forestBDT, JitPredictionsAll)
{
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<std::vector<bool>>  groundtruth   = read_csv<bool>(preds_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>> Forest;
   Forest.get_Forest(json_model_file, events_vector);
   Forest.do_predictions(events_vector, preds);

   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}
*/
