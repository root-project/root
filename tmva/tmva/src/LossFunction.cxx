// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : LossFunction                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#include "TMVA/LossFunction.h"
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
/// huber constructor

TMVA::HuberLossFunction::HuberLossFunction(){
    fTransitionPoint = -9999;
    fSumOfWeights = -9999;
    fQuantile = 0.7;      // the quantile value determines the bulk of the data, e.g. 0.7 defines
                          // the core as the first 70% and the tails as the last 30%
}

TMVA::HuberLossFunction::HuberLossFunction(Double_t quantile){
    fSumOfWeights = -9999;
    fTransitionPoint = -9999;
    fQuantile = quantile;
                          
}

////////////////////////////////////////////////////////////////////////////////
/// huber destructor

TMVA::HuberLossFunction::~HuberLossFunction(){

}

////////////////////////////////////////////////////////////////////////////////
/// figure out the residual that determines the separation between the 
/// "core" and the "tails" of the residuals distribution

void TMVA::HuberLossFunction::Init(std::vector<LossFunctionEventInfo>& evs){

   // Calculate the residual that separates the core and the tails
   SetSumOfWeights(evs);
   SetTransitionPoint(evs);
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the quantile for a given input

Double_t TMVA::HuberLossFunction::CalculateSumOfWeights(std::vector<LossFunctionEventInfo>& evs){

   // Calculate the sum of the weights
   Double_t sumOfWeights = 0;
   for(UInt_t i = 0; i<evs.size(); i++)
      sumOfWeights+=evs[i].weight;

   return sumOfWeights;
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the quantile for a given input

Double_t TMVA::HuberLossFunction::CalculateQuantile(std::vector<LossFunctionEventInfo>& evs, Double_t whichQuantile, Double_t sumOfWeights){
   Double_t temp = 0.0;

   // use a lambda function to tell the vector how to sort the LossFunctionEventInfo data structures
   // (sort them in ascending order of residual magnitude)
   std::sort(evs.begin(), evs.end(), [](LossFunctionEventInfo a, LossFunctionEventInfo b){ 
                                        return TMath::Abs(a.trueValue-a.predictedValue) < TMath::Abs(b.trueValue-b.predictedValue); });
   UInt_t i = 0;
   while(i<evs.size() && temp <= sumOfWeights*whichQuantile){
      temp += evs[i].weight;
      i++;
   }
   if (i >= evs.size()) return 0.; // prevent uncontrolled memory access in return value calculation 
   return TMath::Abs(evs[i].trueValue-evs[i].predictedValue);
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the transition point using the values for fQuantile and fSumOfWeights
/// which presumably have already been set

void TMVA::HuberLossFunction::SetTransitionPoint(std::vector<LossFunctionEventInfo>& evs){
   fTransitionPoint = CalculateQuantile(evs, fQuantile, fSumOfWeights);
}

////////////////////////////////////////////////////////////////////////////////
/// huber, set the sum of weights given a collection of events

void TMVA::HuberLossFunction::SetSumOfWeights(std::vector<LossFunctionEventInfo>& evs){
   fSumOfWeights = CalculateSumOfWeights(evs);
}

////////////////////////////////////////////////////////////////////////////////
/// huber,  determine the loss for a single event

Double_t TMVA::HuberLossFunction::CalculateLoss(LossFunctionEventInfo& e){
   Double_t residual = TMath::Abs(e.trueValue - e.predictedValue);
   Double_t loss = 0;
   // Quadratic loss in terms of the residual for small residuals
   if(residual <= fTransitionPoint) loss = 0.5*residual*residual; 
   // Linear loss for large residuals, so that the tails don't dominate the net loss calculation
   else loss = fQuantile*residual - 0.5*fQuantile*fQuantile;  
   return e.weight*loss;
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the loss for a collection of events

Double_t TMVA::HuberLossFunction::CalculateLoss(std::vector<LossFunctionEventInfo>& evs){
   Double_t netloss = 0;
   for(UInt_t i=0; i<evs.size(); i++)
       netloss+=CalculateLoss(evs[i]);
   return netloss;
   // should get a function to return the average loss as well
   // return netloss/fSumOfWeights
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, initialize the targets and prepare for the regression

void TMVA::HuberLossFunctionBDT::Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights){
// Should only need to run this once before building the forest
   std::vector<LossFunctionEventInfo> evinfovec;
   for (auto &e: evinfomap){
      evinfovec.push_back(LossFunctionEventInfo(e.second.trueValue, e.second.predictedValue, e.first->GetWeight()));
   }

   // Calculates fSumOfWeights and fTransitionPoint with the current residuals
   SetSumOfWeights(evinfovec);
   Double_t weightedMedian = CalculateQuantile(evinfovec, 0.5, fSumOfWeights);

   //Store the weighted median as a first boosweight for later use
   boostWeights.push_back(weightedMedian);
   for (auto &e: evinfomap ) {
      // set the initial prediction for all events to the median
      e.second.predictedValue += weightedMedian;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, set the targets for a collection of events

void TMVA::HuberLossFunctionBDT::SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap){
   std::vector<LossFunctionEventInfo> events;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++){
      events.push_back(LossFunctionEventInfo(evinfomap[*e].trueValue, evinfomap[*e].predictedValue, (*e)->GetWeight()));
   }

   // Recalculate the residual that separates the "core" of the data and the "tails"
   // This residual is the quantile given by fQuantile, defaulted to 0.7
   // the quantile corresponding to 0.5 would be the usual median
   SetSumOfWeights(evinfovec); // This was already set in init, but may change if there is subsampling for each tree
   SetTransitionPoint(events);

   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++) {
         const_cast<TMVA::Event*>(*e)->SetTarget(0,Target(evinfomap[*e]));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, set the target for a single event

Double_t TMVA::HuberLossFunctionBDT::Target(LossFunctionEventInfo& e){
    Double_t residual = e.trueValue - e.predictedValue;
    if(TMath::Abs(residual) <= fTransitionPoint) return residual;
    else return fTransitionPoint*(residual<0?-1.0:1.0);
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, determine the fit value for the terminal node based upon the 
/// events in the terminal node

Double_t TMVA::HuberLossFunctionBDT::Fit(std::vector<LossFunctionEventInfo>& evs){
// The fit in the terminal node for huber is basically the median of the residuals.
// Then you add the average difference from the median to that.
// The tails are discounted. If a residual is in the tails then we just use the
// cutoff residual that sets the "core" and the "tails" instead of the large residual. 
// So we get something between least squares (mean as fit) and absolute deviation (median as fit).
      Double_t sumOfWeights = CalculateSumOfWeights(evs);
      Double_t shift=0,diff= 0;
      Double_t residualMedian = CalculateQuantile(evs,0.5,sumOfWeights);
      for(UInt_t j=0;j<evs.size();j++){
         Double_t residual = evs[j].trueValue - evs[j].predictedValue;
         diff = residual-residualMedian;
         // if we are using weights then I'm not sure why this isn't weighted
         shift+=1.0/evs.size()*((diff<0)?-1.0:1.0)*TMath::Min(fTransitionPoint,fabs(diff));
         // I think this should be 
         // shift+=evs[j].weight/sumOfWeights*((diff<0)?-1.0:1.0)*TMath::Min(fTransitionPoint,fabs(diff));
      }
      return residualMedian + shift;

}
