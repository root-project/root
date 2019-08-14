#include "forest.h"
#include "setup.h"
#include "evaluate_forest_batch.h"

int main()
{
   // Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>> Forest;
   // Forest.trees.push_back(evaluate_forest_batch);
   // Forest.get_Forest(json_model_file, events_vector);
   preds.reserve(events_vector.size());

   evaluate_forest_batch(events_vector, preds);
   // Forest.do_predictions(events_vector, preds);

   write_csv(preds_file, preds);
   return 0;
}
