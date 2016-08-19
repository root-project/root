// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      LossFunction abstract class                                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
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

#ifndef ROOT_TMVA_LossFunction
#define ROOT_TMVA_LossFunction

//#include <iosfwd>
#include <vector>
#include <map>
#include "TMVA/Event.h"
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {
   
   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Data Structure  used by LossFunction and LossFunctionBDT to calculate errors, targets, etc
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   class LossFunctionEventInfo{

   public:
      LossFunctionEventInfo();
      LossFunctionEventInfo(Double_t trueValue_, Double_t predictedValue_, Double_t weight_){
         trueValue = trueValue_;
         predictedValue = predictedValue_;
         weight = weight_;
      }
      ~LossFunctionEventInfo();

      Double_t trueValue; 
      Double_t predictedValue;
      Double_t weight;
   };


   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Loss Function base class for general error calculations in regression/classification
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   class LossFunction {

   public:

      // constructors
      LossFunction(){};
      ~LossFunction(){};

      // abstract methods that need to be implemented
      virtual Double_t CalculateLoss(const LossFunctionEventInfo e) = 0;
      virtual Double_t CalculateLoss(std::vector<const LossFunctionEventInfo>& evs) = 0;

      virtual TString Name() = 0;
      virtual Int_t Id() = 0;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Loss Function base class for boosted decision trees. Inherits from LossFunction
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   // The HuberLossFunctionBDT class implements the LossFunctionBDT interface
   // while also deriving from the HuberLossFunction itself
   // LossFunctionBDT 
   class LossFunctionBDT : public virtual LossFunction{

   public:

      // constructors
      LossFunctionBDT(){};
      ~LossFunctionBDT(){};

      // abstract methods that need to be implemented
      virtual void Init(std::map<const TMVA::Event*, LossFunctionEventInfo> evinfomap) = 0;
      virtual void SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap) = 0;
      virtual Double_t Target(const LossFunctionEventInfo e) = 0;
      virtual Double_t Fit(std::vector<const LossFunctionEventInfo>& evs) = 0;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Huber loss function for regression error calculations
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   class HuberLossFunction : public virtual LossFunction{

   public:
      HuberLossFunction();
      HuberLossFunction(Double_t quantile);
      ~HuberLossFunction();

      // The LossFunction methods
      Double_t CalculateLoss(const LossFunctionEventInfo e);
      Double_t CalculateLoss(std::vector<const LossFunctionEventInfo>& evs);

      // We go ahead and implement the simple ones
      TString Name(){ return TString("Huber_Loss_Function"); };
      Int_t Id(){ return 0; } ;

   protected:
      Double_t fQuantile;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Huber loss function with boosted decision tree functionality
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   class HuberLossFunctionBDT : public LossFunctionBDT, public HuberLossFunction{
   
   public:
      HuberLossFunctionBDT(){};
      HuberLossFunctionBDT(Double_t quantile):HuberLossFunction(quantile){};
      ~HuberLossFunctionBDT(){};

      // LossFunction methods taken from HuberLossFunction
      // Give the BDT implementation a different name though.
      TString Name(){ return TString("Huber_Loss_Function_BDT"); };
      
      // The LossFunctionBDT methods
      void Init(std::map<const TMVA::Event*, LossFunctionEventInfo> evinfomap);
      void SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap);
      Double_t Target(const LossFunctionEventInfo e);
      Double_t Fit(std::vector<const LossFunctionEventInfo>& evs);
      

   private:
      // some data fields
      Double_t fSumOfWeights;
      Double_t fTransitionPoint;
   };

   
}

#endif
