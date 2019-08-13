#include "gtest/gtest.h"
#include "forest.h"

//#include <xgboost/c_api.h> // for xgboost
//#include "generated_files/evaluate_forest2.h"
std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/python_predictions.csv";
std::string json_model_file = "./data/model.json";

TEST(forestBDT, JitPredictions)
{
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>> Forest;
   Forest.get_Forest(json_model_file, events_vector);

   std::vector<bool> preds;
   preds.reserve(events_vector.size());
   Forest.do_predictions(events_vector, preds);

   std::vector<std::vector<bool>> groundtruth = read_csv<bool>(preds_file);
   for (size_t i = 0; i < preds.size(); i++) {
      ASSERT_EQ(preds[i], groundtruth[i][0]);
   }
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
