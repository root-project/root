#include "forest.hxx"
#include "setup.h"

int main()
{
   int rows = events_vector.size();
   int cols = events_vector[0].size();

   ForestBranched<float> ForestTTT;
   ForestTTT.LoadFromJson("lala", json_model_file);

   ForestBranchless<float> Forest;
   Forest.LoadFromJson("lala", json_model_file);
   Forest.branchedForest2BranchlessForest(json_model_file);

   /*
   { // Test 1
      ForestBranched<float> Forest;
      Forest.LoadFromJson("lala", json_model_file);
      preds.reserve(events_vector.size());
      std::cout << "Predicting \n";
      Forest.inference(events_vector, rows, cols, preds);
      std::cout << "writing \n";
      write_csv(preds_file, preds);

      ForestBranchless<float> Forest2;
      Forest.LoadFromJson("lala", json_model_file);
   }
   {
      ForestBranchless<float> Forest;
      Forest.LoadFromJson("lala", json_model_file);
      preds.clear();
      preds.reserve(events_vector.size());
      std::cout << "Predicting \n";
      Forest.inference(events_vector, rows, cols, preds);
      std::cout << "writing \n";
      write_csv(preds_file, preds);
   }
   */

   return 0;
}
