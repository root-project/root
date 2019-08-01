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

using json = nlohmann::json;

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
BENCHMARK(BM_EvalUniqueBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval array_bdts
static void BM_EvalArrayBdt(benchmark::State &state)
{

   Forest<array_bdt::Tree> Forest;

   Forest.get_Forest("model.json");
   std::vector<bool> preds;
   preds.clear();
   std::string events_file = "./data_files/events.csv";
   std::string preds_file  = "./data_files/test.csv";

   std::vector<std::vector<float>> events_vector = read_csv(events_file);
   for (auto _ : state) { // only bench what is inside the loop
      // for (int i = 0; i < 1000; i++)
      preds = Forest.do_predictions(events_vector);
   }
   write_csv(preds_file, preds);
}
BENCHMARK(BM_EvalArrayBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

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

BENCHMARK(BM_EvalJittedBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });

/// Benchmark eval Jitted_bdts
static void BM_EvalJitForestBdt(benchmark::State &state)
{
   Forest<std::function<bool(std::vector<float>)>> Forest;
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
BENCHMARK(BM_EvalJitForestBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
//

/// Benchmark eval xgboost_bdt
static void BM_EvalXgboostBdt(benchmark::State &state)
{
   std::string events_fname = "data_files/events.csv";
   std::string preds_fname  = "data_files/python_predictions.csv";
   const char *model_fname  = "./data/model.rabbit";

   std::vector<std::vector<float>> events;
   std::vector<std::vector<float>> labels;
   events = read_csv(events_fname);
   labels = read_csv(preds_fname);

   int cols = events[0].size();
   int rows = events.size();
   // std::cout << rows << std::endl;

   float train[rows][cols];
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) train[i][j] = events[i][j];

   float m_labels[rows];
   // for (int i = 0; i < rows; i++) m_labels[i] = labels[i][0];

   DMatrixHandle h_train;
   XGDMatrixCreateFromMat((float *)train, rows, cols, -1, &h_train);

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
BENCHMARK(BM_EvalXgboostBdt)
   ->Unit(benchmark::kMillisecond)
   ->ComputeStatistics("min", [](const std::vector<double> &v) -> double {
      return *(std::min_element(std::begin(v), std::end(v)));
   });
;

BENCHMARK_MAIN();
