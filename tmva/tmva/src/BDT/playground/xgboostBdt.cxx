#include <xgboost/c_api.h> // for xgboost
#include <functional>
#include "setup.h"

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

   XGBoosterPredict(boosterHandle, h_train, 0, 0, &out_len, &f);

   std::vector<float> preds;
   for (int i = 0; i < out_len; i++) preds.push_back(f[i]);
   std::string preds_file = "./data/test.csv";
   write_csv(preds_file, preds);

   // free xgboost internal structures
   safe_xgboost(XGBoosterFree(boosterHandle));
   return 0;
}
