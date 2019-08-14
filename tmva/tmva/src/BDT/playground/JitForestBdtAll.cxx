#include "forest.h"
#include "setup.h"

int main()
{
   Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>> Forest;
   Forest.get_Forest(json_model_file, events_vector);
   preds.reserve(events_vector.size());
   Forest.do_predictions(events_vector, preds);
   write_csv(preds_file, preds);
   return 0;
}
