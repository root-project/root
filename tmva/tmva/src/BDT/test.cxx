
#include <iostream>
#include <string>
#include <streambuf>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <functional> // for std::fucntion
#include <chrono>     // for date

#include "bdt_helpers.h"
#include "TInterpreter.h" // for gInterpreter
//# include "generated_code.h"
#include "jitted_bdt.h"

int main()
{
   std::cout << "Jitting\n";
   std::string data_folder = "./data_files/";

   std::string events_file      = data_folder + "events.csv";
   std::string test_file        = data_folder + "test.csv";
   std::string scores_file      = data_folder + "cpp_scores.csv";
   std::string predictions_file = data_folder + "cpp_predictions.csv";

   std::vector<std::vector<float>> events_vector = read_csv(events_file);
   write_csv(test_file, events_vector); // to check consistency

   int trees_number = 4;

   // Read functions
   std::function<float(std::vector<float>)>              func;
   std::vector<std::function<float(std::vector<float>)>> function_vector;
   for (int i = 0; i < trees_number; i++) {
      func = jit_function_reader_file(i);
      function_vector.push_back(func);
   }

   float                           prediction = 0; // define used variables
   std::vector<float>              scores_tmp;
   std::vector<std::vector<float>> scores_out;
   std::vector<std::vector<bool>>  prediction_out;
   // std::vector<std::vector<bool>> prediction_out;

   float scores_sum;
   // Predict events
   for (auto &event : events_vector) {
      scores_tmp.clear();
      for (auto &func : function_vector) {
         prediction = func(event);
         scores_tmp.push_back(prediction);
      }
      scores_out.push_back(scores_tmp);
      // scores_sum = std::accumulate(scores_tmp.begin(), scores_tmp.end(), 0.0);
      scores_sum = vec_sum(scores_tmp);
      prediction_out.push_back(std::vector<bool>{binary_logistic(scores_sum)});
   }
   write_csv(scores_file, scores_out);          // write scores
   write_csv(predictions_file, prediction_out); // write predictions

   // write_csv(array_pred_file, array_pred_out); // write predictions

   return 0;
}
