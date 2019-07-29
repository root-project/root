
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
#include <chrono>
#include <ctime>      // for date
#include <functional> // for std::fucntion

#include "bdt_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

#define safe_xgboost(call)                                                                          \
   {                                                                                                \
      int err = (call);                                                                             \
      if (err != 0) {                                                                               \
         fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
         exit(1);                                                                                   \
      }                                                                                             \
   }

int main()
{
   int silent  = 0;
   int use_gpu = 0; // set to 1 to use the GPU for training
   //   xgboost ------------------------------------------------
   // create the train data
   std::vector<std::vector<float>> events;
   std::vector<std::vector<float>> labels;

   std::string events_fname = "data_files/events.csv";
   events                   = read_csv(events_fname);
   std::string preds_fname  = "data_files/python_predictions.csv";
   labels                   = read_csv(preds_fname);

   int rows2 = events.size();
   int cols  = events[0].size();

   std::cout << "c: " << cols << "   r: " << rows2 << std::endl;

   int rows = 5;
   // read events in arrays
   float train[rows][cols];
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) train[i][j] = events[i][j];

   float train_labels[rows];
   for (int i = 0; i < rows; i++) train_labels[i] = labels[i][0];

   // convert to DMatrix
   DMatrixHandle h_train[1];
   XGDMatrixCreateFromMat((float *)train, rows, cols, -1, &h_train[0]);

   // load the labels
   XGDMatrixSetFloatInfo(h_train[0], "label", train_labels, rows);

   // read back the labels, just a sanity check
   bst_ulong    bst_result;
   const float *out_floats;
   XGDMatrixGetFloatInfo(h_train[0], "label", &bst_result, &out_floats);
   for (unsigned int i = 0; i < bst_result; i++) std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;

   BoosterHandle h_booster2;
   char *        fname2 = "model2.json";
   // XGBoosterLoadModel(&h_booster2, &fname2);
   // create the booster and load some parameters
   BoosterHandle h_booster;
   XGBoosterCreate(h_train, 1, &h_booster);
   XGBoosterSetParam(h_booster, "booster", "gbtree");
   XGBoosterSetParam(h_booster, "objective", "reg:linear");
   XGBoosterSetParam(h_booster, "max_depth", "3");
   XGBoosterSetParam(h_booster, "eta", "0.1");
   XGBoosterSetParam(h_booster, "min_child_weight", "1");
   XGBoosterSetParam(h_booster, "subsample", "0.5");
   XGBoosterSetParam(h_booster, "colsample_bytree", "1");
   XGBoosterSetParam(h_booster, "num_parallel_tree", "1");

   // perform 200 learning iterations
   for (int iter = 0; iter < 200; iter++) XGBoosterUpdateOneIter(h_booster, iter, h_train[0]);

   // predict
   const int sample_rows = 5;
   float     test[sample_rows][cols];
   for (int i = 0; i < sample_rows; i++)
      for (int j = 0; j < cols; j++) test[i][j] = (i + 1) * (j + 1);
   DMatrixHandle h_test;
   XGDMatrixCreateFromMat((float *)test, sample_rows, cols, -1, &h_test);
   bst_ulong    out_len;
   const float *f;
   XGBoosterPredict(h_booster, h_test, 0, 0, &out_len, &f);

   for (unsigned int i = 0; i < out_len; i++) std::cout << "prediction[" << i << "]=" << f[i] << std::endl;

   int results = 2;
   std::cout << "Dumping 1 \n";
   // const char **out_dump_array;
   //           results = XGBoosterDumpModel(&h_booster, "", 3, &out_len, &out_dump_array);
   std::cout << "Dump Moldel: " << results << std::endl;
   // std::cout << "BB " << **out_dump_array << std::endl;
   // delete out_dump_array;

   // results = XGBoosterSaveModel(&h_booster, fname2);
   std::cout << "Save Moldel: " << results << std::endl;

   // free xgboost internal structures
   XGDMatrixFree(h_train[0]);
   XGDMatrixFree(h_test);
   XGBoosterFree(h_booster);

   // ------------------ example 2 -----------------------------------------------
   // load the data
   DMatrixHandle dtrain, dtest;
   safe_xgboost(XGDMatrixCreateFromFile("./data/agaricus.txt.train", silent, &dtrain));
   safe_xgboost(XGDMatrixCreateFromFile("./data/agaricus.txt.test", silent, &dtest));

   // create the booster
   BoosterHandle booster;
   DMatrixHandle eval_dmats[2] = {dtrain, dtest};
   safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

   // configure the training
   // available parameters are described here:
   //   https://xgboost.readthedocs.io/en/latest/parameter.html
   safe_xgboost(XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "hist"));
   if (use_gpu) {
      // set the number of GPUs and the first GPU to use;
      // this is not necessary, but provided here as an illustration
      safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "1"));
      safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "0"));
   } else {
      // avoid evaluating objective and metric on a GPU
      safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "0"));
   }

   safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
   safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "1"));
   safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
   safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
   safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1"));

   // train and evaluate for 10 iterations
   int         n_trees       = 10;
   const char *eval_names[2] = {"train", "test"};
   const char *eval_result   = NULL;
   for (int i = 0; i < n_trees; ++i) {
      safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
      safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
      printf("%s\n", eval_result);
   }

   // predict
   bst_ulong    out_len2   = 0;
   const float *out_result = NULL;
   int          n_print    = 10;

   const char *out_dptr = NULL;
   safe_xgboost(XGBoosterGetModelRaw(booster, &out_len2, &out_dptr));
   std::cout << "CC " << out_dptr << std::endl;

   safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, &out_len2, &out_result));
   printf("y_pred: ");
   for (int i = 0; i < n_print; ++i) {
      printf("%1.4f ", out_result[i]);
   }
   printf("\n");

   // print true labels
   safe_xgboost(XGDMatrixGetFloatInfo(dtest, "label", &out_len2, &out_result));
   printf("y_test: ");
   for (int i = 0; i < n_print; ++i) {
      printf("%1.4f ", out_result[i]);
   }
   printf("\n");
   // safe_xgboost(XGDMatrixCreateFromFile("./data/agaricus.txt.test", silent, &dtest));
   safe_xgboost(XGBoosterSaveModel(booster, "./data/model.txt"));

   const char **out_dump_array;
   safe_xgboost(XGBoosterDumpModel(booster, "", 0, &out_len2, &out_dump_array));
   std::cout << "BB " << **out_dump_array << std::endl;

   // free everything
   safe_xgboost(XGBoosterFree(booster));
   safe_xgboost(XGDMatrixFree(dtrain));
   safe_xgboost(XGDMatrixFree(dtest));

   return 0;
}
