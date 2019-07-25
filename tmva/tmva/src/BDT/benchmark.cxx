#include <benchmark/benchmark.h>
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

using json = nlohmann::json;

static void BM_StringCreation(benchmark::State &state)
{
   for (auto _ : state) std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State &state)
{
   std::string x = "hello";
   for (auto _ : state) std::string copy(x);
}

BENCHMARK(BM_StringCopy);

/// Benchmark unique_bdts
static void BM_UniqueBdt(benchmark::State &state)
{
   std::string my_config       = read_file_string("model.json");
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   unique_bdt::Tree trees[number_of_trees];

   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }

   std::string                     data_folder   = "./data_files/";
   std::string                     events_file   = data_folder + "events.csv";
   std::vector<std::vector<float>> events_vector = read_csv(events_file);

   float                          prediction = 0; // define used variables
   std::vector<float>             preds_tmp;
   std::vector<std::vector<bool>> preds;
   float                          preds_sum;

   for (auto _ : state) { // only bench what is inside the loop
      preds.clear();
      for (auto &event : events_vector) {
         preds_tmp.clear();
         for (auto &tree : trees) {
            prediction = tree.inference(event);
            preds_tmp.push_back(prediction);
         }
         preds_sum = vec_sum(preds_tmp);
         preds.push_back(std::vector<bool>{binary_logistic(preds_sum)});
      }
   }
   std::string preds_unique_file = data_folder + "preds_unique_file.csv";
   write_csv(preds_unique_file, preds); // write predictions
}
BENCHMARK(BM_UniqueBdt);

/// Benchmark eval unique_bdts
static void BM_EvalUniqueBdt(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest("model.json");
   std::vector<bool> preds;
   preds.clear();
   std::string preds_file  = "./data_files/test.csv";
   std::string events_file = "./data_files/events.csv";

   std::vector<std::vector<float>> events_vector = read_csv(events_file);

   for (auto _ : state) { // only bench what is inside the loop
      preds = Forest.do_predictions(events_vector);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalUniqueBdt)->Unit(benchmark::kMillisecond);

/// Benchmark eval array_bdts
static void BM_EvalArrayBdt(benchmark::State &state)
{
   for (auto _ : state) { // only bench what is inside the loop

      Forest<array_bdt::Tree> Forest;

      Forest.get_Forest("model.json");
      std::vector<bool> preds;
      preds.clear();
      std::string events_file = "./data_files/events.csv";
      std::string preds_file  = "./data_files/test.csv";

      std::vector<std::vector<float>> events_vector = read_csv(events_file);

      preds = Forest.do_predictions(events_vector);
      write_csv(preds_file, preds);
   }
}
BENCHMARK(BM_EvalArrayBdt)->Unit(benchmark::kMillisecond);

/// Benchmark eval Jitted_bdts
static void BM_EvalJittedBdt(benchmark::State &state)
{
   Forest<std::function<float(std::vector<float>)>> Forest;
   Forest.get_Forest("model.json");
   std::vector<bool> preds;
   preds.clear();
   std::string preds_file  = "./data_files/test.csv";
   std::string events_file = "./data_files/events.csv";

   std::vector<std::vector<float>> events_vector = read_csv(events_file);

   for (auto _ : state) { // only bench what is inside the loop
      preds = Forest.do_predictions(events_vector);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalJittedBdt)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
