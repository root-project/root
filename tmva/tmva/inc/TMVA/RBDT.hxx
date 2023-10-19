/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 *                                                                                *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Stefan Wunsch (stefan.wunsch@cern.ch)                                     *
 *      Jonas Rembser (jonas.rembser@cern.ch)                                     *
 *                                                                                *
 * Copyright (c) 2024:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_RBDT
#define TMVA_RBDT

#include <Rtypes.h>
#include <ROOT/RSpan.hxx>
#include <TMVA/RTensor.hxx>

#include <array>
#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace TMVA {

namespace Experimental {

class RBDT final {
public:
   typedef float Value_t;

   /// IO constructor (both for ROOT IO and LoadText()).
   RBDT() = default;

   /// Construct backends from model in ROOT file.
   RBDT(const std::string &key, const std::string &filename);

   /// Compute model prediction on a single event.
   ///
   /// The method is intended to be used with std::vectors-like containers,
   /// for example RVecs.
   template <typename Vector>
   Vector Compute(const Vector &x) const
   {
      std::size_t nOut = fBaseResponses.size() > 2 ? fBaseResponses.size() : 1;
      Vector y(nOut);
      ComputeImpl(x.data(), y.data());
      return y;
   }

   /// Compute model prediction on a single event.
   inline std::vector<Value_t> Compute(std::vector<Value_t> const &x) const { return Compute<std::vector<Value_t>>(x); }

   RTensor<Value_t> Compute(RTensor<Value_t> const &x) const;

   static RBDT LoadText(std::string const &txtpath, std::vector<std::string> &features, int nClasses, bool logistic,
                        Value_t baseScore);

private:
   /// Map from XGBoost to RBDT indices.
   using IndexMap = std::unordered_map<int, int>;

   void Softmax(const Value_t *array, Value_t *out) const;
   void ComputeImpl(const Value_t *array, Value_t *out) const;
   Value_t EvaluateBinary(const Value_t *array) const;
   static void correctIndices(std::span<int> indices, IndexMap const &nodeIndices, IndexMap const &leafIndices);
   static void terminateTree(TMVA::Experimental::RBDT &ff, int &nPreviousNodes, int &nPreviousLeaves,
                             IndexMap &nodeIndices, IndexMap &leafIndices, int &treesSkipped);
   static RBDT
   LoadText(std::istream &is, std::vector<std::string> &features, int nClasses, bool logistic, Value_t baseScore);

   std::vector<int> fRootIndices;
   std::vector<unsigned int> fCutIndices;
   std::vector<Value_t> fCutValues;
   std::vector<int> fLeftIndices;
   std::vector<int> fRightIndices;
   std::vector<Value_t> fResponses;
   std::vector<int> fTreeNumbers;
   std::vector<Value_t> fBaseResponses;
   Value_t fBaseScore = 0.0;
   bool fLogistic = false;

   ClassDefNV(RBDT, 1);
};

} // namespace Experimental

} // namespace TMVA

#endif // TMVA_RBDT
