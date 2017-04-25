// @(#)root/tmva $Id$
// Author: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer, Simon Pfreundschuh and Kim Albertsson

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ROCCurve                                                              *
 *                                                                                *
 * Description:                                                                   *
 *      This is class to compute ROC Integral (AUC)                               *
 *                                                                                *
 * Authors :                                                                      *
 *      Omar Zapata     <Omar.Zapata@cern.ch>    - UdeA/ITM Colombia              *
 *      Lorenzo Moneta  <Lorenzo.Moneta@cern.ch> - CERN, Switzerland              *
 *      Sergei Gleyzer  <Sergei.Gleyzer@cern.ch> - U of Florida & CERN            *
 *      Kim Albertsson  <kim.albertsson@cern.ch> - LTU & CERN                     *
 *                                                                                *
 * Copyright (c) 2015:                                                            *
 *      CERN, Switzerland                                                         *
 *      UdeA/ITM, Colombia                                                        *
 *      U. of Florida, USA                                                        *
 **********************************************************************************/

/*! \class TMVA::ROCCurve
\ingroup TMVA

*/
#include "TMVA/Tools.h"
#include "TMVA/ROCCurve.h"
#include "TMVA/Config.h"
#include "TMVA/Version.h"
#include "TMVA/MsgLogger.h"
#include "TGraph.h"

#include <vector>
#include <cassert>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
///

TMVA::ROCCurve::ROCCurve(const std::vector<Float_t> &mvaValues, const std::vector<Bool_t> &mvaTargets,
                         const std::vector<Float_t> &mvaWeights)
   : fLogger(new TMVA::MsgLogger("ROCCurve")), fGraph(NULL)
{
   assert(mvaValues.size() == mvaTargets.size());
   assert(mvaValues.size() == mvaWeights.size());

   for (UInt_t i = 0; i < mvaValues.size(); i++) {
      if (mvaTargets.at(i)) {
         fMvaSignal.push_back(mvaValues.at(i));
         fMvaSignalWeights.push_back(mvaWeights.at(i));
      } else {
         fMvaBackground.push_back(mvaValues.at(i));
         fMvaBackgroundWeights.push_back(mvaWeights.at(i));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::ROCCurve::ROCCurve(const std::vector<Float_t> &mvaValues, const std::vector<Bool_t> &mvaTargets)
   : fLogger(new TMVA::MsgLogger("ROCCurve")), fGraph(NULL)
{
   assert(mvaValues.size() == mvaTargets.size());

   for (UInt_t i = 0; i < mvaValues.size(); i++) {
      if (mvaTargets.at(i)) {
         fMvaSignal.push_back(mvaValues.at(i));
      } else {
         fMvaBackground.push_back(mvaValues.at(i));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::ROCCurve::ROCCurve(const std::vector<Float_t> &mvaSignal, const std::vector<Float_t> &mvaBackground)
   : fLogger(new TMVA::MsgLogger("ROCCurve")), fGraph(NULL)
{
   fMvaSignal = mvaSignal;
   fMvaBackground = mvaBackground;
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::ROCCurve::ROCCurve(const std::vector<Float_t> &mvaSignal, const std::vector<Float_t> &mvaBackground,
                         const std::vector<Float_t> &mvaSignalWeights, const std::vector<Float_t> &mvaBackgroundWeights)
   : ROCCurve(mvaSignal, mvaBackground)
{
   assert(mvaSignal.size() == mvaSignalWeights.size());
   assert(mvaBackground.size() == mvaBackgroundWeights.size());

   fMvaSignalWeights = mvaSignalWeights;
   fMvaBackgroundWeights = mvaBackgroundWeights;
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::ROCCurve::~ROCCurve() {
   delete fLogger;
   if(fGraph) delete fGraph;
}

////////////////////////////////////////////////////////////////////////////////
///

std::vector<Float_t> TMVA::ROCCurve::ComputeSpecificity(const UInt_t num_points)
{
   if (num_points == 0) {
      return {0.0, 1.0};
   }

   UInt_t num_divisions = num_points - 1;
   std::vector<Float_t> specificity_vector;
   specificity_vector.push_back(0.0);

   for (Float_t threshold = -1.0; threshold < 1.0; threshold += (1.0 / num_divisions)) {
      Float_t false_positives = 0.0;
      Float_t true_negatives = 0.0;

      for (size_t i = 0; i < fMvaBackground.size(); ++i) {
         auto value = fMvaBackground.at(i);
         auto weight = fMvaBackgroundWeights.empty() ? (1.0) : fMvaBackgroundWeights.at(i);

         if (value > threshold) {
            false_positives += weight;
         } else {
            true_negatives += weight;
         }
      }

      Float_t total_background = false_positives + true_negatives;
      Float_t specificity =
         (total_background <= std::numeric_limits<Float_t>::min()) ? (0.0) : (true_negatives / total_background);

      specificity_vector.push_back(specificity);
   }

   specificity_vector.push_back(1.0);
   return specificity_vector;
}

////////////////////////////////////////////////////////////////////////////////
///

std::vector<Float_t> TMVA::ROCCurve::ComputeSensitivity(const UInt_t num_points)
{
   if (num_points == 0) {
      return {1.0, 0.0};
   }

   UInt_t num_divisions = num_points - 1;
   std::vector<Float_t> sensitivity_vector;
   sensitivity_vector.push_back(1.0);

   for (Float_t threshold = -1.0; threshold < 1.0; threshold += (1.0 / num_divisions)) {
      Float_t true_positives = 0.0;
      Float_t false_negatives = 0.0;

      for (size_t i = 0; i < fMvaSignal.size(); ++i) {
         auto value = fMvaSignal.at(i);
         auto weight = fMvaSignalWeights.empty() ? (1.0) : fMvaSignalWeights.at(i);

         if (value > threshold) {
            true_positives += weight;
         } else {
            false_negatives += weight;
         }
      }

      Float_t total_signal = true_positives + false_negatives;
      Float_t sensitivity =
         (total_signal <= std::numeric_limits<Float_t>::min()) ? (0.0) : (true_positives / total_signal);
      sensitivity_vector.push_back(sensitivity);
   }

   sensitivity_vector.push_back(0.0);
   return sensitivity_vector;
}

////////////////////////////////////////////////////////////////////////////////
/// ROC Integral (AUC)

Double_t TMVA::ROCCurve::GetROCIntegral(const UInt_t num_points)
{
   auto sensitivity = ComputeSensitivity(num_points);
   auto specificity = ComputeSpecificity(num_points);

   Float_t integral = 0;
   for (UInt_t i = 0; i < sensitivity.size() - 1; i++) {
      // FNR, false negatigve rate = 1 - Sensitivity
      Float_t fnrCurr = 1 - sensitivity[i];
      Float_t fnrNext = 1 - sensitivity[i + 1];
      integral += 0.5 * (fnrNext - fnrCurr) * (specificity[i] + specificity[i + 1]);
   }

   return integral;
}

////////////////////////////////////////////////////////////////////////////////
///

TGraph *TMVA::ROCCurve::GetROCCurve(const UInt_t num_points)
{
   if (fGraph != nullptr) {
      delete fGraph;
   }

   auto sensitivity = ComputeSensitivity(num_points);
   auto specificity = ComputeSpecificity(num_points);

   fGraph = new TGraph(sensitivity.size(), &specificity[0], &sensitivity[0]);

   return fGraph;
}
