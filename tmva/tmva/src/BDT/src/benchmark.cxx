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

#include <xgboost/c_api.h> // for xgboost
#include "../generated_files/evaluate_forest2.h"
#include "../generated_files/evaluate_forest_batch.h"

using json = nlohmann::json;

/// Global variables
std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/test.csv";
std::string json_model_file = "./data/model.json";

const std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);

#define safe_xgboost(call)                                                                          \
   {                                                                                                \
      int err = (call);                                                                             \
      if (err != 0) {                                                                               \
         fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
         exit(1);                                                                                   \
      }                                                                                             \
   }

/// Benchmark eval unique_bdts
static void BM_EvalUniqueBdt(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;

   Forest.get_Forest(json_model_file);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUniqueBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval unique_bdts
static void BM_EvalUnique_Bdt_batch_128(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest(json_model_file);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());
   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions_batch(events_vector, preds, 128);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUnique_Bdt_batch_128)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval unique_bdts
static void BM_EvalUnique_Bdt_batch2_128(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest(json_model_file);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());
   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions_batch2(events_vector, preds, 128);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUnique_Bdt_batch2_128)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval unique_bdts
static void BM_EvalUnique_Bdt_batch2_128_branch(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;

   Forest.get_Forest(json_model_file);
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   std::vector<bool>               preds;
   preds.reserve(events_vector.size());

   std::sort(events_vector.begin(), events_vector.end(),
             [](const std::vector<float> &a, const std::vector<float> &b) { return a[0] < b[0]; });

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions_batch2(events_vector, preds, 128);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalUnique_Bdt_batch2_128_branch)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval array_bdts
static void BM_EvalArrayBdt(benchmark::State &state)
{

   Forest<array_bdt::Tree> Forest;

   Forest.get_Forest(json_model_file);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalArrayBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */
/*
/// Benchmark eval Jitted_bdts
static void BM_EvalJittedBdt(benchmark::State &state)
{
   Forest<std::function<float(std::vector<float>)>> Forest;
   Forest.get_Forest("model.json");
   std::vector<bool> preds;
   preds.clear();
   std::string preds_file  = "./data_files/test.csv";
   std::string events_file = "./data_files/events.csv";

   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   preds.reserve(events_vector.size());
   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalJittedBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
//   */

// /*
/// Benchmark eval Jitted_bdts
static void BM_EvalJitForestBdt(benchmark::State &state)
{
   Forest<std::function<bool(std::vector<float>)>> Forest;

   Forest.get_Forest(json_model_file);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalJitForestBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval Jitted_bdts
static void BM_EvalJitForestWholeBdt(benchmark::State &state)
{
   Forest<std::function<std::vector<bool>(std::vector<std::vector<float>>)>> Forest;

   Forest.get_Forest(json_model_file);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   std::function<std::vector<bool>(std::vector<std::vector<float>>)> my_func = Forest.trees[0];
   for (auto _ : state) { // only bench what is inside the loop
      preds = my_func(events_vector);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalJitForestWholeBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval Jitted_bdts
static void BM_StaticForestWholeBdt_batch(benchmark::State &state)
{

   Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>> Forest;

   Forest.get_Forest(json_model_file, events_vector);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_StaticForestWholeBdt_batch)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval Jitted_bdts
static void BM_StaticForestWholeBdt_batch_branch(benchmark::State &state)
{
   Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>> Forest;

   Forest.get_Forest(json_model_file, events_vector);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());
   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);

   std::sort(events_vector.begin(), events_vector.end(),
             [](const std::vector<float> &a, const std::vector<float> &b) { return a[0] < b[0]; });
   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_StaticForestWholeBdt_batch_branch)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval Jitted_bdts
static void BM_ForestWholeBdtStatic(benchmark::State &state)
{
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   std::function<std::vector<bool>(std::vector<std::vector<float>>)> my_func = s_f_event_61565384376::evaluate_forest;

   for (auto _ : state) { // only bench what is inside the loop
      preds = my_func(events_vector);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_ForestWholeBdtStatic)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/*
/// Benchmark eval Jitted_bdts batch
static void BM_StaticForestWholeBdt_static_batch(benchmark::State &state)
{
   std::vector<bool> preds;
   preds.clear();
   std::string preds_file  = "./data_files/test.csv";
   std::string events_file = "./data_files/events.csv";

   std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);
   preds.reserve(events_vector.size());

   std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)> my_func =
      s_fa_event_21565601031::evaluate_forest_array;

   // const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds;
   for (auto _ : state) { // only bench what is inside the loop
      my_func(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_StaticForestWholeBdt_static_batch)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

// /*
/// Benchmark eval xgboost_bdt
static void BM_EvalXgboostBdt(benchmark::State &state)
{
   std::string events_fname = "./data/events.csv";
   std::string preds_fname  = "./data/python_predictions.csv";
   const char *model_fname  = "./data/model.rabbit";

   std::vector<std::vector<float>> events_vector;
   std::vector<std::vector<int>>   labels;
   events_vector = read_csv<float>(events_fname);
   labels        = read_csv<int>(preds_fname);

   int cols = events_vector[0].size();
   int rows = events_vector.size();

   std::vector<float> train2;
   train2.reserve(rows * cols);

   float tmp = 0;
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         train2[i * cols + j] = events_vector.at(i).at(j);
      }
   }

   float m_labels[rows];
   // Transform to single vector and pass vector.data();
   DMatrixHandle h_train;
   safe_xgboost(XGDMatrixCreateFromMat((float *)train2.data(), rows, cols, -1, &h_train));

   BoosterHandle boosterHandle;
   safe_xgboost(XGBoosterCreate(0, 0, &boosterHandle));
   // std::cout << "Loading model \n";
   safe_xgboost(XGBoosterLoadModel(boosterHandle, model_fname));
   XGBoosterSetParam(boosterHandle, "objective", "binary:logistic");

   // std::cout << "***** Predicts ***** \n";
   bst_ulong    out_len;
   const float *f;

   for (auto _ : state) { // only bench what is inside the loop
      // for (int i = 0; i < 1000; i++)
      XGBoosterPredict(boosterHandle, h_train, 0, 0, &out_len, &f);
   }

   std::vector<float> preds;
   for (int i = 0; i < out_len; i++) preds.push_back(f[i]);
   std::string preds_file = "data_files/test.csv";
   write_csv(preds_file, preds);

   // free xgboost internal structures
   safe_xgboost(XGBoosterFree(boosterHandle));
}
// /*
BENCHMARK(BM_EvalXgboostBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

BENCHMARK_MAIN();
