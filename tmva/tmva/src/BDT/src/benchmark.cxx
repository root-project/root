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

#include "forest.h"
#include <xgboost/c_api.h> // for xgboost
#include "../generated_files/generated_forest2.h"
#include "../generated_files/generated_ordered_forest2.h"
#include "../generated_files/branchless_generated_forest2.h"
//#include "../generated_files/evaluate_forest2.h"
//#include "../generated_files/evaluate_forest_batch2.h"

using json = nlohmann::json;

/// Global variables
std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/test.csv";
std::string json_model_file = "./data/model.json";
int         loop_size       = 256;

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
static void BM_EvalUniqueBdt_clever(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;

   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUniqueBdt_clever)
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
      Forest.do_predictions_batch(events_vector, preds, loop_size);
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
static void BM_EvalUnique_Bdt_batch_128_clever(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());
   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions_batch(events_vector, preds, loop_size);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUnique_Bdt_batch_128_clever)
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
      Forest.do_predictions(events_vector, preds, loop_size);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUnique_Bdt_batch2_128)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval unique_bdts
static void BM_EvalUnique_Bdt_batch2_128_clever(benchmark::State &state)
{
   Forest<unique_bdt::Tree> Forest;
   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());
   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds, loop_size);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalUnique_Bdt_batch2_128_clever)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

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

/// Benchmark eval array_bdts
static void BM_EvalArrayBdt_ordered(benchmark::State &state)
{

   Forest<array_bdt::Tree> Forest;

   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalArrayBdt_ordered)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

/// Benchmark eval array_bdts
static void BM_EvalArrayBdt_ordered_batch(benchmark::State &state)
{

   Forest<array_bdt::Tree> Forest;

   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds, loop_size);
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_EvalArrayBdt_ordered_batch)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

// /*
/// Benchmark eval Jitted_bdts
static void BM_EvalJitForestBdt(benchmark::State &state)
{
   Forest<std::function<bool(const std::vector<float> &)>> Forest;

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
static void BM_EvalJitForestBdt_clever(benchmark::State &state)
{
   Forest<std::function<bool(const std::vector<float> &)>> Forest;

   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalJitForestBdt_clever)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval Jitted_bdts
static void BM_EvalJitForestBdt_branchless(benchmark::State &state)
{
   Forest<std::function<bool(const float *)>> Forest;

   Forest.get_Forest(json_model_file, true);
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   for (auto _ : state) { // only bench what is inside the loop
      Forest.do_predictions(events_vector, preds);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalJitForestBdt_branchless)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval Jitted_bdts
static void BM_ForestWholeBdt_batch(benchmark::State &state)
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
BENCHMARK(BM_ForestWholeBdt_batch)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

///////////////////////////  STATIC  ////////////////////////
///*
/// Benchmark eval Jitted_bdts
static void BM_ForestBdtStatic(benchmark::State &state)
{
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   std::function<bool(const std::vector<float> &)> my_func = s_f_71566403686::generated_forest;

   for (auto _ : state) { // only bench what is inside the loop
      for (size_t i = 0; i < events_vector.size(); i++) {
         preds.push_back(my_func(events_vector[i]));
      }
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_ForestBdtStatic)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

// /*
/// Benchmark eval Jitted_bdts
static void BM_ForestBdtOrderedStatic(benchmark::State &state)
{
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   std::function<bool(const std::vector<float> &)> my_func = s_f_141566403691::generated_forest;

   for (auto _ : state) { // only bench what is inside the loop
      for (size_t i = 0; i < events_vector.size(); i++) {
         preds.push_back(my_func(events_vector[i]));
      }
   }
   write_csv(preds_file, preds);
}
// /*
BENCHMARK(BM_ForestBdtOrderedStatic)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

// /*
/// Benchmark eval Jitted_bdts
static void BM_ForestBranchlessBdtStatic(benchmark::State &state)
{
   std::vector<bool> preds;
   preds.reserve(events_vector.size());

   std::function<bool(const float *)> my_func = branchless_81566403515::branchless_generated_forest;

   for (auto _ : state) { // only bench what is inside the loop
      for (size_t i = 0; i < events_vector.size(); i++) {
         preds.push_back(my_func(events_vector[i].data()));
      }
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_ForestBranchlessBdtStatic)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
// */

//////////////////////////   xgboost   ///////////////////////////
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

/// -----------------------------------------   Test pointers
static volatile int global_var = 0;

void my_int_func(int x)
{
   global_var = x + x + 3;
   benchmark::DoNotOptimize(global_var);
   benchmark::DoNotOptimize(x);
}

static void AAA_RawFunctionPointer(benchmark::State &state)
{
   void (*bar)(int) = &my_int_func;
   srand(time(nullptr));
   for (auto _ : state) {
      bar(rand());
      benchmark::DoNotOptimize(my_int_func);
      benchmark::DoNotOptimize(bar);
   }
}

static void AAA_StdFunction(benchmark::State &state)
{
   std::function<void(int)> bar = my_int_func;
   srand(time(nullptr));
   for (auto _ : state) {
      bar(rand());
      benchmark::DoNotOptimize(my_int_func);
      benchmark::DoNotOptimize(bar);
   }
}

static void AAA_StdBind(benchmark::State &state)
{
   auto bar = std::bind(my_int_func, std::placeholders::_1);
   srand(time(nullptr));
   for (auto _ : state) {
      bar(rand());
      benchmark::DoNotOptimize(my_int_func);
      benchmark::DoNotOptimize(bar);
   }
}

static void AAA_Lambda(benchmark::State &state)
{
   auto bar = [](int x) {
      global_var = x + x + 3;
      benchmark::DoNotOptimize(global_var);
      benchmark::DoNotOptimize(x);
   };
   srand(time(nullptr));
   for (auto _ : state) {
      bar(rand());
      benchmark::DoNotOptimize(my_int_func);
      benchmark::DoNotOptimize(bar);
   }
}
BENCHMARK(AAA_RawFunctionPointer);
BENCHMARK(AAA_StdBind);
BENCHMARK(AAA_StdFunction);
BENCHMARK(AAA_Lambda);

BENCHMARK_MAIN();
