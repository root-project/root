// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      LossFunction and associated classes                                       *
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

#include "TMVA/Types.h"


namespace TMVA {

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Data Structure  used by LossFunction and LossFunctionBDT to calculate errors, targets, etc
   ///////////////////////////////////////////////////////////////////////////////////////////////

   class LossFunctionEventInfo{

   public:
      LossFunctionEventInfo(){
          trueValue = 0.;
          predictedValue = 0.;
          weight = 0.;
      };
      LossFunctionEventInfo(Double_t trueValue_, Double_t predictedValue_, Double_t weight_){
         trueValue = trueValue_;
         predictedValue = predictedValue_;
         weight = weight_;
      }
      ~LossFunctionEventInfo(){};

      Double_t trueValue;
      Double_t predictedValue;
      Double_t weight;
   };


   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Loss Function interface defining base class for general error calculations in
   // regression/classification
   ///////////////////////////////////////////////////////////////////////////////////////////////

   class LossFunction {

   public:

      // constructors
      LossFunction(){};
      virtual ~LossFunction(){};

      // abstract methods that need to be implemented
      virtual Double_t CalculateLoss(LossFunctionEventInfo& e) = 0;
      virtual Double_t CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs) = 0;
      virtual Double_t CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs) = 0;

      virtual TString Name() = 0;
      virtual Int_t Id() = 0;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Loss Function interface for boosted decision trees. Inherits from LossFunction
   ///////////////////////////////////////////////////////////////////////////////////////////////

   /* Must inherit LossFunction with the virtual keyword so that we only have to implement
   * the LossFunction interface once.
   *
   *       LossFunction
   *      /            \
   *SomeLossFunction  LossFunctionBDT
   *      \            /
   *       \          /
   *    SomeLossFunctionBDT
   *
   * Without the virtual keyword the two would point to their own LossFunction objects
   * and SomeLossFunctionBDT would have to implement the virtual functions of LossFunction twice, once
   * for each object. See diagram below.
   *
   * LossFunction  LossFunction
   *     |             |
   *SomeLossFunction  LossFunctionBDT
   *      \            /
   *       \          /
   *     SomeLossFunctionBDT
   *
   * Multiple inheritance is often frowned upon. To avoid this, We could make LossFunctionBDT separate
   * from LossFunction but it really is a type of loss function.
   * We could also put LossFunction into LossFunctionBDT. In either of these scenarios, if you are doing
   * different regression methods and want to compare the Loss this makes it more convoluted.
   * I think that multiple inheritance seems justified in this case, but we could change it if it's a problem.
   * Usually it isn't a big deal with interfaces and this results in the simplest code in this case.
   */

   class LossFunctionBDT : public virtual LossFunction{

   public:

      // constructors
      LossFunctionBDT(){};
      virtual ~LossFunctionBDT(){};

      // abstract methods that need to be implemented
      virtual void Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights) = 0;
      virtual void SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap) = 0;
      virtual Double_t Target(LossFunctionEventInfo& e) = 0;
      virtual Double_t Fit(std::vector<LossFunctionEventInfo>& evs) = 0;

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
      Double_t CalculateLoss(LossFunctionEventInfo& e);
      Double_t CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs);
      Double_t CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs);

      // We go ahead and implement the simple ones
      TString Name(){ return TString("Huber"); };
      Int_t Id(){ return 0; } ;

      // Functions needed beyond the interface
      void Init(std::vector<LossFunctionEventInfo>& evs);
      Double_t CalculateQuantile(std::vector<LossFunctionEventInfo>& evs, Double_t whichQuantile, Double_t sumOfWeights, bool abs);
      Double_t CalculateSumOfWeights(const std::vector<LossFunctionEventInfo>& evs);
      void SetTransitionPoint(std::vector<LossFunctionEventInfo>& evs);
      void SetSumOfWeights(std::vector<LossFunctionEventInfo>& evs);

   protected:
      Double_t fQuantile;
      Double_t fTransitionPoint;
      Double_t fSumOfWeights;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Huber loss function with boosted decision tree functionality
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // The bdt loss function implements the LossFunctionBDT interface and inherits the HuberLossFunction
   // functionality.
   class HuberLossFunctionBDT : public LossFunctionBDT, public HuberLossFunction{

   public:
      HuberLossFunctionBDT();
      HuberLossFunctionBDT(Double_t quantile):HuberLossFunction(quantile){};
      ~HuberLossFunctionBDT(){};

      // The LossFunctionBDT methods
      void Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights);
      void SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap);
      Double_t Target(LossFunctionEventInfo& e);
      Double_t Fit(std::vector<LossFunctionEventInfo>& evs);

   private:
      // some data fields
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // LeastSquares loss function for regression error calculations
   ///////////////////////////////////////////////////////////////////////////////////////////////

   class LeastSquaresLossFunction : public virtual LossFunction{

   public:
      LeastSquaresLossFunction(){};
      ~LeastSquaresLossFunction(){};

      // The LossFunction methods
      Double_t CalculateLoss(LossFunctionEventInfo& e);
      Double_t CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs);
      Double_t CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs);

      // We go ahead and implement the simple ones
      TString Name(){ return TString("LeastSquares"); };
      Int_t Id(){ return 1; } ;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Least Squares loss function with boosted decision tree functionality
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // The bdt loss function implements the LossFunctionBDT interface and inherits the LeastSquaresLossFunction
   // functionality.
   class LeastSquaresLossFunctionBDT : public LossFunctionBDT, public LeastSquaresLossFunction{

   public:
      LeastSquaresLossFunctionBDT(){};
      ~LeastSquaresLossFunctionBDT(){};

      // The LossFunctionBDT methods
      void Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights);
      void SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap);
      Double_t Target(LossFunctionEventInfo& e);
      Double_t Fit(std::vector<LossFunctionEventInfo>& evs);
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Absolute Deviation loss function for regression error calculations
   ///////////////////////////////////////////////////////////////////////////////////////////////

   class AbsoluteDeviationLossFunction : public virtual LossFunction{

   public:
      AbsoluteDeviationLossFunction(){};
      ~AbsoluteDeviationLossFunction(){};

      // The LossFunction methods
      Double_t CalculateLoss(LossFunctionEventInfo& e);
      Double_t CalculateNetLoss(std::vector<LossFunctionEventInfo>& evs);
      Double_t CalculateMeanLoss(std::vector<LossFunctionEventInfo>& evs);

      // We go ahead and implement the simple ones
      TString Name(){ return TString("AbsoluteDeviation"); };
      Int_t Id(){ return 2; } ;
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Absolute Deviation loss function with boosted decision tree functionality
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // The bdt loss function implements the LossFunctionBDT interface and inherits the AbsoluteDeviationLossFunction
   // functionality.
   class AbsoluteDeviationLossFunctionBDT : public LossFunctionBDT, public AbsoluteDeviationLossFunction{

   public:
      AbsoluteDeviationLossFunctionBDT(){};
      ~AbsoluteDeviationLossFunctionBDT(){};

      // The LossFunctionBDT methods
      void Init(std::map<const TMVA::Event*, LossFunctionEventInfo>& evinfomap, std::vector<double>& boostWeights);
      void SetTargets(std::vector<const TMVA::Event*>& evs, std::map< const TMVA::Event*, LossFunctionEventInfo >& evinfomap);
      Double_t Target(LossFunctionEventInfo& e);
      Double_t Fit(std::vector<LossFunctionEventInfo>& evs);
   };
}

#endif
