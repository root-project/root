// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
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
    fQuantile = 0.7;      // the quantile value determines the bulk of the data, e.g. 0.7 defines
                          // the core as the first 70% and the tails as the last 30%
}

TMVA::HuberLossFunction::HuberLossFunction(Double_t quantile){
    fQuantile = quantile;
                          
}

////////////////////////////////////////////////////////////////////////////////
/// huber destructor

TMVA::HuberLossFunction::~HuberLossFunction(){

}

////////////////////////////////////////////////////////////////////////////////
/// huber,  determine the loss for a single event

Double_t TMVA::HuberLossFunction::CalculateLoss(const LossFunctionEventInfo e){
   return 1.0;
}

////////////////////////////////////////////////////////////////////////////////
/// huber, determine the loss for a collection of events

Double_t TMVA::HuberLossFunction::CalculateLoss(std::vector<const LossFunctionEventInfo>& evs){
   return 1.0;
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, initialize the targets and prepare for the regression

void TMVA::HuberLossFunctionBDT::Init(std::map<const TMVA::Event*, LossFunctionEventInfo> evinfomap){
   
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, set the targets for a collection of events

void TMVA::HuberLossFunctionBDT::SetTargets(std::map<const TMVA::Event*, LossFunctionEventInfo> evinfomap){
   
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, set the target for a single event

Double_t TMVA::HuberLossFunctionBDT::Target(const LossFunctionEventInfo e){
   return 1.0; 
}

////////////////////////////////////////////////////////////////////////////////
/// huber BDT, determine the fit value for the terminal node based upon the 
/// events in the terminal node

Double_t TMVA::HuberLossFunctionBDT::Fit(std::vector<const LossFunctionEventInfo>& evs){
   
}
