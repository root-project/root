// File for profiling

#include <iostream>
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>
#include <array>
#include <utility>

#include "TInterpreter.h" // for gInterpreter

#include "bdt_helpers.h"

#include "unique_bdt.h"
#include "array_bdt.h"
#include "forest.h"

//#include <xgboost/c_api.h> // for xgboost

using json = nlohmann::json;

int main()
{
   Forest<std::function<float(std::vector<float>)>> Forest;
   Forest.get_Forest("model.json");
   std::vector<bool> preds;
   std::string       preds_file  = "./data_files/test.csv";
   std::string       events_file = "./data_files/events.csv";

   std::vector<std::vector<float>> events_vector = read_csv(events_file);
   std::vector<bool>               preds2;
   std::vector<float>              preds_tmp;
   for (auto i = 0; i < 100; i++) { // only bench what is inside the loop
      preds.clear();
      preds = Forest.do_predictions_bis(events_vector, preds2, preds_tmp);
   }
   write_csv(preds_file, preds);
   return 0;
} // End main
