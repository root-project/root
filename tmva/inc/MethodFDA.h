// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodFDA                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Function discriminant analysis (FDA). This simple classifier              *
 *      fits any user-defined TFormula (via option configuration string) to       *
 *      the training data by requiring a formula response of 1 (0) to signal      *
 *      (background) events. The parameter fitting is done via the abstract       *
 *      class FitterBase, featuring Monte Carlo sampling, Genetic                 *
 *      Algorithm, Simulated Annealing, MINUIT and combinations of these.         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
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

#ifndef ROOT_TMVA_MethodFDA
#define ROOT_TMVA_MethodFDA

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodFDA                                                            //
//                                                                      //
// Function discriminant analysis (FDA). This simple classifier         //
// fits any user-defined TFormula (via option configuration string) to  //
// the training data by requiring a formula response of 1 (0) to signal //
// (background) events. The parameter fitting is done via the abstract  //
// class FitterBase, featuring Monte Carlo sampling, Genetic            //
// Algorithm, Simulated Annealing, MINUIT and combinations of these.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#include "TMVA/IFitterTarget.h"
#endif

class TFormula;

namespace TMVA {

   class Interval;
   class Event;
   class FitterBase;

   class MethodFDA : public MethodBase, public IFitterTarget {

   public:

      MethodFDA( TString jobName, 
                 TString methodTitle, 
                 DataSet& theData,
                 TString theOption = "",
                 TDirectory* theTargetDir = 0 );
      
      MethodFDA( DataSet& theData, 
                 TString theWeightFile,  
                 TDirectory* theTargetDir = NULL );
      
      virtual ~MethodFDA( void );
    
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

      void InitFDA( void );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

      Double_t EstimatorFunction( std::vector<Double_t>& );

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

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

      TString                fFormulaStringP;     // string with function
      TString                fParRangeStringP;    // string with ranges of parameters      
      TString                fFormulaStringT;     // string with function
      TString                fParRangeStringT;    // string with ranges of parameters      

      TFormula*              fFormula;            // the discrimination function
      Int_t                  fNPars;              // number of parameters
      std::vector<Interval*> fParRange;           // ranges of parameters
      std::vector<Double_t>  fBestPars;           // the pars that optimise (minimise) the estimator
      TString                fFitMethod;          // estimator optimisation method
      TString                fConverger;          // fitmethod uses fConverger as intermediate step to converge into local minimas
      FitterBase*            fFitter;             // the fitter used in the training
      IFitterTarget*         fConvergerFitter;    // intermediate fitter


      // speed up access to training events by caching
      std::vector<const Event*>    fEventsSig;          // event cache (signal)
      std::vector<const Event*>    fEventsBkg;          // event cache (background)

      // sum of weights (this should become centrally available through the dataset)
      Double_t               fSumOfWeightsSig;    // sum of weights (signal)
      Double_t               fSumOfWeightsBkg;    // sum of weights (background)

      ClassDef(MethodFDA,0)  // Function Discriminant Analysis
   };

} // namespace TMVA

#endif // MethodFDA_H
