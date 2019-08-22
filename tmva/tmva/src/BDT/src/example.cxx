#include "forest2.h"
#include "bdt_helpers.h"

std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/test.csv";
std::string json_model_file = "./data/model.json";

std::vector<std::vector<float>> events_vector  = read_csv<float>(events_file);
float *                         events_vector2 = events_vector.data();

std::vector<bool> preds;

int main()
{
   JittedForest<float> model;
   model.LoadFromJson("model.json");
}
