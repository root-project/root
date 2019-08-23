#include "forest2.hxx"
#include "setup.h"

int main()
{
   int rows = events_vector.size();
   int cols = events_vector[0].size();
   { // Test 1
      ForestBranched<float> Forest;
      Forest.LoadFromJson("lala", json_model_file);
      preds.reserve(events_vector.size());
      Forest.inference(events_vector.data()->data(), rows, cols, preds);
      write_csv(preds_file, preds);

      ForestBranchless<float> Forest2;
      Forest.LoadFromJson("lala", json_model_file);
   }
   {
      ForestBranchless<float> Forest;
      Forest.LoadFromJson("lala", json_model_file);
      preds.clear();
      preds.reserve(events_vector.size());
      Forest.inference(events_vector.data()->data(), rows, cols, preds);
      write_csv(preds_file, preds);
   }

   return 0;
}
