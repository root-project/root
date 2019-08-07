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

   // --------------------------------------------------------------------
   std::string events_fname = "data_files/events.csv";
   std::string preds_fname  = "data_files/python_predictions.csv";
   const char *model_fname  = "./data/model.rabbit";

   std::vector<std::vector<float>> events;
   std::vector<std::vector<float>> labels;
   events = read_csv(events_fname);
   labels = read_csv(preds_fname);

   size_t cols = events[0].size();
   size_t rows = events.size();
   // std::cout << "r" << rows << std::endl;
   // std::cout << cols << std::endl;

   std::vector<std::vector<float>> events2 = events;

   float train[rows][cols];
   float train3[rows * cols];

   std::vector<float> train2;
   train2.reserve(rows * cols);

   // for (size_t i = 0; i < rows; i++)
   //  for (size_t j = 0; j < cols; j++) events2[i][j] = events.at(i).at(j);

   for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++) train2[i * cols + j] = events.at(i).at(j);

   for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++) train3[i * cols + j] = events.at(i).at(j);

   // for (size_t i = 0; i < rows; i++)
   //  for (size_t j = 0; j < cols; j++) train[i][j] = events.at(i).at(j);

   float m_labels[rows];
   // for (int i = 0; i < rows; i++) m_labels[i] = labels[i][0];
   // Transform to single vector and pass vector.data();
   DMatrixHandle h_train;
   XGDMatrixCreateFromMat((float *)train2.data(), rows, cols, -1, &h_train);
   // XGDMatrixCreateFromMat((float *)train, rows, cols, -1, &h_train);

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
   return 0;
} // End main
