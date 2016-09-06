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
//-----------------------------------------------------------------------------
// Huber Loss Function
//-----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/// huber constructor

TMVA::HuberLossFunction::HuberLossFunction(){
    fTransitionPoint = -9999;
    fSumOfWeights = -9999;
    fQuantile = 0.7;      // the quantile value determines the bulk of the data, e.g. 0.7 defines
                          // the core as the first 70% and the tails as the last 30%
                          
   std::cout << "HuberLossFunction::HuberLossFunction" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::cout << "fSumOfWeights, fTransitionPoint, fQuantile" << std::endl;
   std::cout << fSumOfWeights << ", " << fTransitionPoint << ", " << fQuantile << std::endl;
}

TMVA::HuberLossFunction::HuberLossFunction(Double_t quantile){
    fSumOfWeights = -9999;
    fTransitionPoint = -9999;
    fQuantile = quantile;
                          
   std::cout << "HuberLossFunction::HuberLossFunction" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::cout << "fSumOfWeights, fTransitionPoint, fQuantile" << std::endl;
   std::cout << fSumOfWeights << ", " << fTransitionPoint << ", " << fQuantile << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// huber destructor

TMVA::HuberLossFunction::~HuberLossFunction(){

}

////////////////////////////////////////////////////////////////////////////////
/// figure out the residual that determines the separation between the 
/// "core" and the "tails" of the residuals distribution

void TMVA::HuberLossFunction::Init(std::vector<LossFunctionEventInfo>& evs){

   std::cout << "HuberLossFunction::Init" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   // Calculate the residual that separates the core and the tails
   SetSumOfWeights(evs);
   SetTransitionPoint(evs);

   std::cout << "fSumOfWeights, fTransitionPoint, fQuantile" << std::endl;
   std::cout << fSumOfWeights << ", " << fTransitionPoint << ", " << fQuantile << std::endl;
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

Double_t TMVA::HuberLossFunction::CalculateQuantile(std::vector<LossFunctionEventInfo>& evs, Double_t whichQuantile, Double_t sumOfWeights, bool abs){

   // use a lambda function to tell the vector how to sort the LossFunctionEventInfo data structures
   // (sort them in ascending order of residual magnitude) if abs is true
   // otherwise sort them in ascending order of residual
   if(abs)
      std::sort(evs.begin(), evs.end(), [](LossFunctionEventInfo a, LossFunctionEventInfo b){ 
                                           return TMath::Abs(a.trueValue-a.predictedValue) < TMath::Abs(b.trueValue-b.predictedValue); });
   else
      std::sort(evs.begin(), evs.end(), [](LossFunctionEventInfo a, LossFunctionEventInfo b){ 
                                           return (a.trueValue-a.predictedValue) < (b.trueValue-b.predictedValue); });
   UInt_t i = 0;
   Double_t temp = 0.0;
   while(i<evs.size()-1 && temp <= sumOfWeights*whichQuantile){
      temp += evs[i].weight;
      i++;
   }
   // edge cases
   if(whichQuantile == 0) i=0;             // assume 0th quantile to mean the 0th entry in the ordered series

   // usual returns
   if(abs) return TMath::Abs(evs[i].trueValue-evs[i].predictedValue);
   else return evs[i].trueValue-evs[i].predictedValue; 
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the transition point using the values for fQuantile and fSumOfWeights
/// which presumably have already been set

void TMVA::HuberLossFunction::SetTransitionPoint(std::vector<LossFunctionEventInfo>& evs){
   fTransitionPoint = CalculateQuantile(evs, fQuantile, fSumOfWeights, true);
}

////////////////////////////////////////////////////////////////////////////////
/// huber, set the sum of weights given a collection of events

void TMVA::HuberLossFunction::SetSumOfWeights(std::vector<LossFunctionEventInfo>& evs){
   fSumOfWeights = CalculateSumOfWeights(evs);
}

////////////////////////////////////////////////////////////////////////////////
/// huber,  determine the loss for a single event

Double_t TMVA::HuberLossFunction::CalculateLoss(LossFunctionEventInfo& e){
   // If the huber loss function is uninitialized then assume a group of one
   // and initialize the transition point and weights for this single event
   if(fSumOfWeights == -9999){
      std::vector<LossFunctionEventInfo> evs;
      evs.push_back(e);
       
      SetSumOfWeights(evs);
      SetTransitionPoint(evs);
   }

   Double_t residual = TMath::Abs(e.trueValue - e.predictedValue);
   Double_t loss = 0;
   // Quadratic loss in terms of the residual for small residuals
   if(residual <= fTransitionPoint) loss = 0.5*residual*residual; 
   // Linear loss for large residuals, so that the tails don't dominate the net loss calculation
   else loss = fQuantile*residual - 0.5*fQuantile*fQuantile;  
   return e.weight*loss;
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the net loss for a collection of events

Double_t TMVA::HuberLossFunction::CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs){
   // Initialize the Huber Loss Function so that we can calculate the loss.
   // The loss for each event depends on the other events in the group
   // that define the cutoff quantile (fTransitionPoint).
   SetSumOfWeights(evs);
   SetTransitionPoint(evs);

   Double_t netloss = 0;
   for(UInt_t i=0; i<evs.size(); i++)
       netloss+=CalculateLoss(evs[i]);
   return netloss;
   // should get a function to return the average loss as well
   // return netloss/fSumOfWeights
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the mean loss for a collection of events

Double_t TMVA::HuberLossFunction::CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs){
   // Initialize the Huber Loss Function so that we can calculate the loss.
   // The loss for each event depends on the other events in the group
   // that define the cutoff quantile (fTransitionPoint).
   SetSumOfWeights(evs);
   SetTransitionPoint(evs);

   Double_t netloss = 0;
   for(UInt_t i=0; i<evs.size(); i++)
       netloss+=CalculateLoss(evs[i]);
   return netloss/fSumOfWeights;
}

////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Huber BDT Loss Function
//-----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////

TMVA::HuberLossFunctionBDT::HuberLossFunctionBDT(){
   std::cout << "HuberLossFunctionBDT::HuberLossFunctionBDT" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::cout << "fSumOfWeights, fTransitionPoint, fQuantile" << std::endl;
   std::cout << fSumOfWeights << ", " << fTransitionPoint << ", " << fQuantile << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, initialize the targets and prepare for the regression

void TMVA::HuberLossFunctionBDT::Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights){
// Run this once before building the forest. Set initial prediction to weightedMedian.

   std::cout << "HuberLossFunctionBDT::Init" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::vector<LossFunctionEventInfo> evinfovec;
   for (auto &e: evinfomap){
      evinfovec.push_back(LossFunctionEventInfo(e.second.trueValue, e.second.predictedValue, e.first->GetWeight()));
   }

   // Calculates fSumOfWeights and fTransitionPoint with the current residuals
   SetSumOfWeights(evinfovec);
   Double_t weightedMedian = CalculateQuantile(evinfovec, 0.5, fSumOfWeights, false);

   std::cout << "fSumOfWeights, fTransitionPoint, fQuantile, weightedMedian" << std::endl;
   std::cout << fSumOfWeights << ", " << fTransitionPoint << ", " << fQuantile << ", " << weightedMedian << std::endl;

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

   std::cout << "HuberLossFunctionBDT::SetTargets" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::vector<LossFunctionEventInfo> eventvec;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++){
      eventvec.push_back(LossFunctionEventInfo(evinfomap[*e].trueValue, evinfomap[*e].predictedValue, (*e)->GetWeight()));
   }

   // Recalculate the residual that separates the "core" of the data and the "tails"
   // This residual is the quantile given by fQuantile, defaulted to 0.7
   // the quantile corresponding to 0.5 would be the usual median
   SetSumOfWeights(eventvec); // This was already set in init, but may change if there is subsampling for each tree
   SetTransitionPoint(eventvec);

   std::cout << "fSumOfWeights, fTransitionPoint, fQuantile" << std::endl;
   std::cout << fSumOfWeights << ", " << fTransitionPoint << ", " << fQuantile << std::endl;
   std::cout << std::endl;

   Int_t i=0;
   std::cout << "i: trueValue, predictedValue, target, weight" << std::endl;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++) {
         const_cast<TMVA::Event*>(*e)->SetTarget(0,Target(evinfomap[*e]));
         if(i<=10)
            std::cout << i << ": " << evinfomap[*e].trueValue << ", " << evinfomap[*e].predictedValue << (*e)->GetTarget(0) <<", " << (*e)->GetWeight() << std::endl;
         i++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, set the target for a single event

Double_t TMVA::HuberLossFunctionBDT::Target(LossFunctionEventInfo& e){
    Double_t residual = e.trueValue - e.predictedValue;
    // The weight/target relationships are taken care of in the tmva decision tree operations so we don't need to worry about that here
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
   Double_t residualMedian = CalculateQuantile(evs,0.5,sumOfWeights, false);
   std::cout << sumOfWeights << ", " << residualMedian << ", " << evs.size() << std::endl;
   std::cout << "   j: residual, diff, shift" << std::endl;
   for(UInt_t j=0;j<evs.size();j++){
      Double_t residual = evs[j].trueValue - evs[j].predictedValue;
      diff = residual-residualMedian;
      // if we are using weights then I'm not sure why this isn't weighted
      shift+=1.0/evs.size()*((diff<0)?-1.0:1.0)*TMath::Min(fTransitionPoint,fabs(diff));
      if(j<=10) 
      {
          std::cout << "   " << j << ": " << residual << ", " << diff << ", " << shift << std::endl; 
      }
      // I think this should be 
      // shift+=evs[j].weight/sumOfWeights*((diff<0)?-1.0:1.0)*TMath::Min(fTransitionPoint,fabs(diff));
   }
   return (residualMedian + shift);

}

////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Least Squares Loss Function
//-----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////

// Constructor and destructor are in header file. They don't do anything.

////////////////////////////////////////////////////////////////////////////////
/// least squares ,  determine the loss for a single event

Double_t TMVA::LeastSquaresLossFunction::CalculateLoss(LossFunctionEventInfo& e){
   Double_t residual = (e.trueValue - e.predictedValue);
   Double_t loss = 0;
   loss = residual*residual;  
   return e.weight*loss;
}

////////////////////////////////////////////////////////////////////////////////
/// least squares , determine the net loss for a collection of events

Double_t TMVA::LeastSquaresLossFunction::CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs){
   Double_t netloss = 0;
   for(UInt_t i=0; i<evs.size(); i++)
       netloss+=CalculateLoss(evs[i]);
   return netloss;
   // should get a function to return the average loss as well
   // return netloss/fSumOfWeights
}

////////////////////////////////////////////////////////////////////////////////
/// least squares , determine the mean loss for a collection of events

Double_t TMVA::LeastSquaresLossFunction::CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs){
   Double_t netloss = 0;
   Double_t sumOfWeights = 0;
   for(UInt_t i=0; i<evs.size(); i++){
       sumOfWeights+=evs[i].weight;
       netloss+=CalculateLoss(evs[i]);
   }
   // return the weighted mean
   return netloss/sumOfWeights;
}

////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Least Squares BDT Loss Function
//-----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////

// Constructor and destructor defined in header. They don't do anything.

////////////////////////////////////////////////////////////////////////////////
/// least squares BDT, initialize the targets and prepare for the regression

void TMVA::LeastSquaresLossFunctionBDT::Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights){
// Run this once before building the foresut. Set initial prediction to the weightedMean

   std::cout << "LeastSquaresLossFunctionBDT::Init" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::vector<LossFunctionEventInfo> evinfovec;
   for (auto &e: evinfomap){
      evinfovec.push_back(LossFunctionEventInfo(e.second.trueValue, e.second.predictedValue, e.first->GetWeight()));
   }

   // Initial prediction for least squares is the weighted mean
   Double_t weightedMean = Fit(evinfovec);

   std::cout << "weightedMean" << std::endl;
   std::cout << weightedMean << std::endl;

   //Store the weighted median as a first boosweight for later use
   boostWeights.push_back(weightedMean);
   for (auto &e: evinfomap ) {
      // set the initial prediction for all events to the median
      e.second.predictedValue += weightedMean;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// least squares BDT, set the targets for a collection of events

void TMVA::LeastSquaresLossFunctionBDT::SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap){

   std::cout << "LeastSquaresLossFunctionBDT::SetTargets" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::vector<LossFunctionEventInfo> eventvec;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++){
      eventvec.push_back(LossFunctionEventInfo(evinfomap[*e].trueValue, evinfomap[*e].predictedValue, (*e)->GetWeight()));
   }

   Int_t i=0;
   std::cout << "i: trueValue, predictedValue, target" << std::endl;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++) {
         const_cast<TMVA::Event*>(*e)->SetTarget(0,Target(evinfomap[*e]));
         if(i<=100)
            std::cout << i << ": " << evinfomap[*e].trueValue << ", " << evinfomap[*e].predictedValue << ", " << (*e)->GetTarget(0) << std::endl;
         i++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// least squares BDT, set the target for a single event

Double_t TMVA::LeastSquaresLossFunctionBDT::Target(LossFunctionEventInfo& e){
    Double_t residual = e.trueValue - e.predictedValue;
    // The weight/target relationships are taken care of in the tmva decision tree operations. We don't need to worry about that here
    // and we return the residual instead of the weight*residual.
    return residual;
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, determine the fit value for the terminal node based upon the 
/// events in the terminal node

Double_t TMVA::LeastSquaresLossFunctionBDT::Fit(std::vector<LossFunctionEventInfo>& evs){
// The fit in the terminal node for least squares is the weighted average of the residuals.
   Double_t sumOfWeights = 0;
   Double_t weightedResidualSum = 0;
   std::cout << "i: trueValue, predictedValue, residual" << std::endl;
   for(UInt_t j=0;j<evs.size();j++){
      sumOfWeights += evs[j].weight;
      Double_t residual = evs[j].trueValue - evs[j].predictedValue;
      weightedResidualSum += evs[j].weight*residual;
      if(j<=100) 
      {
          std::cout << "   " << j << ": " << evs[j].trueValue << ", " << evs[j].predictedValue << ", " << residual << std::endl; 
      }
   }
   Double_t weightedMean = weightedResidualSum/sumOfWeights;

   // return the weighted mean
   return weightedMean;
}

////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Absolute Deviation Loss Function
//-----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////

// Constructors in the header. They don't do anything.

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation,  determine the loss for a single event

Double_t TMVA::AbsoluteDeviationLossFunction::CalculateLoss(LossFunctionEventInfo& e){
   Double_t residual = e.trueValue - e.predictedValue;
   return e.weight*TMath::Abs(residual);
}

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation, determine the net loss for a collection of events

Double_t TMVA::AbsoluteDeviationLossFunction::CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs){

   Double_t netloss = 0;
   for(UInt_t i=0; i<evs.size(); i++)
       netloss+=CalculateLoss(evs[i]);
   return netloss;
}

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation, determine the mean loss for a collection of events

Double_t TMVA::AbsoluteDeviationLossFunction::CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs){
   Double_t sumOfWeights = 0;
   Double_t netloss = 0;
   for(UInt_t i=0; i<evs.size(); i++){
       sumOfWeights+=evs[i].weight;
       netloss+=CalculateLoss(evs[i]);
   }
   return netloss/sumOfWeights;
}

////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Absolute Deviation BDT Loss Function
//-----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation BDT, initialize the targets and prepare for the regression

void TMVA::AbsoluteDeviationLossFunctionBDT::Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights){
// Run this once before building the forest. Set initial prediction to weightedMedian.

   std::cout << "AbsoluteDeviationLossFunctionBDT::Init" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::vector<LossFunctionEventInfo> evinfovec;
   for (auto &e: evinfomap){
      evinfovec.push_back(LossFunctionEventInfo(e.second.trueValue, e.second.predictedValue, e.first->GetWeight()));
   }

   Double_t weightedMedian = Fit(evinfovec);

   std::cout << "weightedMedian" << std::endl;
   std::cout << weightedMedian << std::endl;

   //Store the weighted median as a first boostweight for later use
   boostWeights.push_back(weightedMedian);
   for (auto &e: evinfomap ) {
      // set the initial prediction for all events to the median
      e.second.predictedValue += weightedMedian;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation BDT, set the targets for a collection of events

void TMVA::AbsoluteDeviationLossFunctionBDT::SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap){

   std::cout << "AbsoluteDeviationLossFunctionBDT::SetTargets" << std::endl;
   std::cout << "=======================================================" << std::endl << std::endl;

   std::vector<LossFunctionEventInfo> eventvec;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++){
      eventvec.push_back(LossFunctionEventInfo(evinfomap[*e].trueValue, evinfomap[*e].predictedValue, (*e)->GetWeight()));
   }

   Int_t i=0;
   std::cout << "i: trueValue, predictedValue, target, weight" << std::endl;
   for (std::vector<const TMVA::Event*>::const_iterator e=evs.begin(); e!=evs.end();e++) {
         const_cast<TMVA::Event*>(*e)->SetTarget(0,Target(evinfomap[*e]));
         if(i<=10)
            std::cout << i << ": " << evinfomap[*e].trueValue << ", " << evinfomap[*e].predictedValue << (*e)->GetTarget(0) <<", " << (*e)->GetWeight() << std::endl;
         i++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation BDT, set the target for a single event

Double_t TMVA::AbsoluteDeviationLossFunctionBDT::Target(LossFunctionEventInfo& e){
// The target is the sign of the residual.
    Double_t residual = e.trueValue - e.predictedValue;
    // The weight/target relationships are taken care of in the tmva decision tree operations so we don't need to worry about that here
    return (residual<0?-1.0:1.0);
}

////////////////////////////////////////////////////////////////////////////////
/// absolute deviation BDT, determine the fit value for the terminal node based upon the 
/// events in the terminal node

Double_t TMVA::AbsoluteDeviationLossFunctionBDT::Fit(std::vector<LossFunctionEventInfo>& evs){
// For Absolute Deviation, the fit in each terminal node is the weighted residual median.

   // use a lambda function to tell the vector how to sort the LossFunctionEventInfo data structures
   // sort in ascending order of residual value
   std::sort(evs.begin(), evs.end(), [](LossFunctionEventInfo a, LossFunctionEventInfo b){ 
                                        return (a.trueValue-a.predictedValue) < (b.trueValue-b.predictedValue); });

   // calculate the sum of weights, used in the weighted median calculation
   Double_t sumOfWeights = 0;
   for(UInt_t j=0; j<evs.size(); j++)
      sumOfWeights+=evs[j].weight;

   // get the index of the weighted median
   UInt_t i = 0;
   Double_t temp = 0.0;
   while(i<evs.size() && temp <= sumOfWeights*0.5){
      temp += evs[i].weight;
      i++;
   }
   if (i >= evs.size()) return 0.; // prevent uncontrolled memory access in return value calculation 

   // return the median residual
   return evs[i].trueValue-evs[i].predictedValue; 
}

