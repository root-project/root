// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSeedDistance                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005-2006:                                                       *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodSeedDistance
#define ROOT_TMVA_MethodSeedDistance

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodSeedDistance                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_IFitterTarget
#include "TMVA/IFitterTarget.h"
#endif

class TFormula;

namespace TMVA {

   class Interval;
   class Event;
   class FitterBase;
   class SeedDistance;
   class IMetric;

   class MethodSeedDistance : public MethodBase, public IFitterTarget {

   public:

      MethodSeedDistance( TString jobName, 
                 TString methodTitle, 
                 DataSet& theData,
                 TString theOption = "",
                 TDirectory* theTargetDir = 0 );
      
      MethodSeedDistance( DataSet& theData, 
                 TString theWeightFile,  
                 TDirectory* theTargetDir = NULL );
      
      virtual ~MethodSeedDistance( void );
    
      // training method
      virtual void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      void InitSeedDistance( void );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

      Double_t EstimatorFunction( std::vector<Double_t>& );

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

      void MakeListFromStructure( std::vector<Double_t>& linear, 
                                  std::vector< std::vector< Double_t > >& seeds,
                                  std::vector<Double_t>& metricParams );

      void MakeStructureFromList( std::vector<Double_t>& linear, 
                                  std::vector< std::vector< Double_t > >& seeds,
                                  std::vector<Double_t>& metricParams );

   private:

      // interpret formula expression and compute estimator
      Double_t InterpretFormula( const Event&, std::vector<Double_t>& pars );

      // clean up 
      void ClearAll();

      // print fit results
      void PrintResults( const TString&, std::vector<Double_t>&, const Double_t ) const;

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      TString                fSeedRangeStringP;    // string with ranges of parameters      
      TString                fSeedRangeStringT;    // string with ranges of parameters      
      Bool_t                 fScalingFactor;

      IMetric*               fMetric;
      SeedDistance*          fSeedDistance;
      std::vector< std::vector< Double_t > > fSeeds;    // the pars that optimise (minimise) the estimator
      std::vector<Double_t>  fMetricPars;         // 
      std::vector<Double_t>  fPars;           // the pars that optimise (minimise) the estimator

      Int_t                  fDataSeeds;
      Int_t                  fBackSeeds;
      TString                fMetricType;

      Bool_t                 fPow2Estimator;

      Int_t                  fNPars;              // number of parameters
      std::vector<Interval*> fParRange;           // ranges of parameters
      TString                fFitMethod;          // estimator optimisation method
      TString                fConverger;          // fitmethod uses fConverger as intermediate step to converge into local minimas
      FitterBase*            fFitter;             // the fitter used in the training
      IFitterTarget*         fIntermediateFitter; // intermediate fitter


      // speed up access to training events by caching
      std::vector<const Event*>    fEventsSig;          // event cache (signal)
      std::vector<const Event*>    fEventsBkg;          // event cache (background)

      // sum of weights (this should become centrally available through the dataset)
      Double_t               fSumOfWeightsSig;    // sum of weights (signal)
      Double_t               fSumOfWeightsBkg;    // sum of weights (background)

      ClassDef(MethodSeedDistance,0)  // Function Discriminant Analysis
   };

} // namespace TMVA

#endif // MethodSeedDistance_H
