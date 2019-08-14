#include "forest.h"
#include "setup.h"

int main()
{
   Forest<unique_bdt::Tree> Forest;
   preds.reserve(events_vector.size());
   Forest.get_Forest(json_model_file);
   Forest.do_predictions_batch2(events_vector, preds, 256);
   write_csv(preds_file, preds);
   return 0;
}
