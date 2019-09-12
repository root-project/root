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

#include <sstream> // std::stringstream

namespace TMVA {
namespace Experimental {

/// Fast boosted decision tree inference
template <typename Backend = BranchlessForest<float>>
class RBDT {
public:
   using Value_t = typename Backend::Value_t;
   using Backend_t = Backend;

private:
   int fNumClasses;
   Backend_t fBackend;

public:
   /// Construct backend from model in ROOT file
   RBDT(const std::string &key, const std::string &filename)
   {
      fBackend = Backend_t();
      fBackend.Load(key, filename);
   }

   /// Compute model prediction on a single event
   ///
   /// The method is intended to be used with std::vectors-like containers,
   /// for example RVecs.
   template <typename Vector>
   Vector Compute(const Vector &x)
   {
      // TODO: fNumClasses -> fNumOutputs?
      // TODO: Implement multi-class
      Vector y;
      y.resize(1);
      fBackend.Inference(&x[0], 1, &y[0]);
      return y;
   }

   /// Compute model prediction on a single event
   std::vector<Value_t> Compute(const std::vector<Value_t> &x)
   {
      // TODO: fNumClasses -> fNumOutputs?
      // TODO: Implement multi-class
      std::vector<Value_t> y;
      y.resize(1);
      fBackend.Inference(&x[0], 1, &y[0]);
      return y;
   }

   /// Compute model prediction on input RTensor
   RTensor<Value_t> Compute(const RTensor<Value_t> &x)
   {
      // TODO: Add inference for a batch of events
      // TODO: Check that input tensor is row major
      const auto rows = x.GetShape()[0];
      RTensor<Value_t> y({rows, 1});
      fBackend.Inference(x.GetData(), rows, y.GetData());
      return y;
   }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBDT
