/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Stefan Wunsch (stefan.wunsch@cern.ch)                                     *
 *                                                                                *
 * Copyright (c) 2019:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_RBDT
#define TMVA_RBDT

#include "TMVA/RTensor.hxx"
#include "TMVA/TreeInference/Forest.hxx"
#include "TFile.h"

#include <vector>
#include <string>
#include <sstream> // std::stringstream
#include <memory>

namespace TMVA {
namespace Experimental {

/// Fast boosted decision tree inference
template <typename Backend = BranchlessJittedForest<float>>
class RBDT {
public:
   using Value_t = typename Backend::Value_t;
   using Backend_t = Backend;

private:
   int fNumOutputs;
   bool fNormalizeOutputs;
   std::vector<Backend_t> fBackends;

public:
   /// Construct backends from model in ROOT file
   RBDT(const std::string &key, const std::string &filename)
   {
      // Get number of output nodes of the forest
      std::unique_ptr<TFile> file{TFile::Open(filename.c_str(),"READ")};
      if (!file || file->IsZombie()) {
         throw std::runtime_error("Failed to open input file " + filename);
      }
      auto numOutputs = Internal::GetObjectSafe<std::vector<int>>(file.get(), filename, key + "/num_outputs");
      fNumOutputs = numOutputs->at(0);
      delete numOutputs;

      // Get objective and decide whether to normalize output nodes for example in the multiclass case
      auto objective = Internal::GetObjectSafe<std::string>(file.get(), filename, key + "/objective");
      if (objective->compare("softmax") == 0)
         fNormalizeOutputs = true;
      else
         fNormalizeOutputs = false;
      delete objective;
      file->Close();

      // Initialize backends
      fBackends = std::vector<Backend_t>(fNumOutputs);
      for (int i = 0; i < fNumOutputs; i++)
         fBackends[i].Load(key, filename, i);
   }

   /// Compute model prediction on a single event
   ///
   /// The method is intended to be used with std::vectors-like containers,
   /// for example RVecs.
   template <typename Vector>
   Vector Compute(const Vector &x)
   {
      Vector y;
      y.resize(fNumOutputs);
      for (int i = 0; i < fNumOutputs; i++)
         fBackends[i].Inference(&x[0], 1, true, &y[i]);
      if (fNormalizeOutputs) {
         Value_t s = 0.0;
         for (int i = 0; i < fNumOutputs; i++)
            s += y[i];
         for (int i = 0; i < fNumOutputs; i++)
            y[i] /= s;
      }
      return y;
   }

   /// Compute model prediction on a single event
   std::vector<Value_t> Compute(const std::vector<Value_t> &x) { return this->Compute<std::vector<Value_t>>(x); }

   /// Compute model prediction on input RTensor
   RTensor<Value_t> Compute(const RTensor<Value_t> &x)
   {
      const auto rows = x.GetShape()[0];
      RTensor<Value_t> y({rows, static_cast<std::size_t>(fNumOutputs)}, MemoryLayout::ColumnMajor);
      const bool layout = x.GetMemoryLayout() == MemoryLayout::ColumnMajor ? false : true;
      for (int i = 0; i < fNumOutputs; i++)
         fBackends[i].Inference(x.GetData(), rows, layout, &y(0, i));
      if (fNormalizeOutputs) {
         Value_t s;
         for (int i = 0; i < static_cast<int>(rows); i++) {
            s = 0.0;
            for (int j = 0; j < fNumOutputs; j++)
               s += y(i, j);
            for (int j = 0; j < fNumOutputs; j++)
               y(i, j) /= s;
         }
      }
      return y;
   }
};

extern template class TMVA::Experimental::RBDT<TMVA::Experimental::BranchlessForest<float>>;
extern template class TMVA::Experimental::RBDT<TMVA::Experimental::BranchlessJittedForest<float>>;

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBDT
