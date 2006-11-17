// @(#)root/tmva $Id: MethodCuts.h,v 1.36 2006/11/17 14:59:24 stelzer Exp $
// Author: Andreas Hoecker, Matt Jachowski, Peter Speckmayer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCuts                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Multivariate optimisation of signal efficiency for given background       *
 *      efficiency, using rectangular minimum and maximum requirements on         *
 *      input variables                                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Matt Jachowski   <jachowski@stanford.edu> - Stanford University, USA      *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *      Helge Voss       <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany     *
 *      Kai Voss         <Kai.Voss@cern.ch>       - U. of Victoria, Canada        *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-K Heidelberg, Germany ,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCuts
#define ROOT_TMVA_MethodCuts

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCuts                                                           //
//                                                                      //
// Multivariate optimisation of signal efficiency for given background  //
// efficiency, using rectangular minimum and maximum requirements on    //
// input variables                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif
#ifndef ROOT_TMVA_GeneticBase
#include "TMVA/GeneticBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif

class TRandom;

namespace TMVA {

   class MethodCuts : public MethodBase {

   public:

      MethodCuts( TString jobName,
                  TString methodTitle, 
                  DataSet& theData,
                  TString theOption = "MC:150:10000:",
                  TDirectory* theTargetFile = 0 );

      MethodCuts( DataSet& theData,
                  TString theWeightFile,
                  TDirectory* theTargetDir = NULL );

      virtual ~MethodCuts( void );

      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value (for CUTs this is just a dummy)
      virtual Double_t GetMvaValue();

      // write method specific histos to target file
      virtual void WriteMonitoringHistosToFile( void ) const;

      // test the method
      virtual void Test( TTree* theTestTree );
     
      // also overwrite:
      virtual Double_t GetSignificance( void )   { return 0; }
      virtual Double_t GetSeparation  ( void )   { return 0; }
      virtual Double_t GetmuTransform ( TTree *) { return 0; }
      virtual Double_t GetEfficiency  ( TString, TTree *);
      virtual Double_t GetTrainingEfficiency  ( TString );

      // accessors for Minuit
      Double_t ComputeEstimator( const std::vector<Double_t> & );

      void SetTestSignalEfficiency( Double_t eff ) { fTestSignalEff = eff; }

      // static pointer to this object
      static MethodCuts* ThisCuts( void ) { return fgThisCuts; } // original version of above

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

      virtual void DeclareOptions();
      virtual void ProcessOptions();

   protected:

   private:

      // determines type of data to be optimised
      enum EConstrainType { kConstrainEffS = 0,
                            kConstrainEffB } fConstrainType;

      // optimisation method
      enum EFitMethodType { kUseMonteCarlo = 0,
                            kUseGeneticAlgorithm,
                            kUseSimulatedAnnealing };

      // efficiency calculation method
      // - kUseEventSelection: computes efficiencies from given data sample
      // - kUsePDFs          : creates smoothed PDFs from data samples, and 
      //                       uses this to compute efficiencies
      enum EEffMethod     { kUseEventSelection = 0,
                            kUsePDFs };

      // improve the Monte Carlo by providing some additional information
      enum EFitParameters { kNotEnforced = 0,
                            kForceMin,
                            kForceMax,
                            kForceSmart,
                            kForceVerySmart };

      // general
      TString                 fFitMethodS;    // chosen fit method (string)
      EFitMethodType          fFitMethod;     // chosen fit method
      TString                 fEffMethodS;    // chosen efficiency calculation method (string)
      EEffMethod              fEffMethod;     // chosen efficiency calculation method
      vector<EFitParameters>* fFitParams;     // vector for series of fit methods
      Double_t                fTestSignalEff; // used to test optimized signal efficiency
      Double_t                fEffSMin;       // used to test optimized signal efficiency
      Double_t                fEffSMax;       // used to test optimized signal efficiency

      // for the use of the binary tree method
      BinarySearchTree*  fBinaryTreeS;
      BinarySearchTree*  fBinaryTreeB;

      // GA (genetic algorithm) options
      Int_t              fGA_nsteps;          // GA settings: number of steps
      Int_t              fGA_cycles;          // GA settings: number of pre-calc steps
      Int_t              fGA_popSize;         // GA settings: population size
      Int_t              fGA_SC_steps;        // GA settings: SC_steps
      Int_t              fGA_SC_offsteps;     // GA settings: SC_offsteps
      Double_t           fGA_SC_factor;       // GA settings: SC_factor
 
      // SA (simulated annealing) options
      Int_t              fSA_MaxCalls;                // max number of FCN calls
      Double_t           fSA_TemperatureGradient;     // starting value for temperature gradient
      Bool_t             fSA_UseAdaptiveTemperature;  // compute temperature steps on the fly
      Double_t           fSA_InitialTemperature;      // initial temperature (depends on FCN)
      Double_t           fSA_MinTemperature;          // minimum temperature before SA quit
      Double_t           fSA_Eps;                     // relative required FCN accuracy at minimum
      Int_t              fSA_NFunLoops;               // number of FCN loops
      Int_t              fSA_NEps;                    // test parameter

      // MC method
      Int_t              fNRandCuts;          // number of random cut samplings
      Double_t**         fCutMin;             // minimum requirement
      Double_t**         fCutMax;             // maximum requirement
      Double_t*          fTmpCutMin;          // temporary minimum requirement
      Double_t*          fTmpCutMax;          // temporary maximum requirement

      TString            fAllVars;
      TString            fAllVarsI[10];


      // relevant for all methods
      Int_t              fNpar;               // number of parameters in fit (default: 2*Nvar)
      Double_t           fEffRef;             // reference efficiency
      vector<Int_t>*     fRangeSign;          // used to match cuts to fit parameters (and vice versa)
      TRandom*           fRandom;             // random generator for MC optimisation method

      // basic statistics
      vector<Double_t>*  fMeanS;              // means of variables (signal)
      vector<Double_t>*  fMeanB;              // means of variables (background)
      vector<Double_t>*  fRmsS;               // RMSs of variables (signal)
      vector<Double_t>*  fRmsB;               // RMSs of variables (background)
      vector<Double_t>*  fXmin;               // minimum values of variables
      vector<Double_t>*  fXmax;               // maximum values of variables 

      TH1*               fEffBvsSLocal;       // intermediate eff. background versus eff signal histo

      // PDF section
      vector<TH1*>*      fVarHistS;           // reference histograms (signal)
      vector<TH1*>*      fVarHistB;           // reference histograms (background)
      vector<TH1*>*      fVarHistS_smooth;    // smoothed reference histograms (signal)        
      vector<TH1*>*      fVarHistB_smooth;    // smoothed reference histograms (background)
      vector<PDF*>*      fVarPdfS;            // reference PDFs (signal)
      vector<PDF*>*      fVarPdfB;            // reference PDFs (background)

      static MethodCuts* fgThisCuts;          // used for function reference (GA)


      // the definition of fit parameters can be different from the actual 
      // cut requirements; these functions provide the matching
      void     MatchParsToCuts( const std::vector<Double_t> &, Double_t*, Double_t* );
      void     MatchParsToCuts( Double_t*, Double_t*, Double_t* );

      void     MatchCutsToPars( Double_t*, Double_t*, Double_t* );

      // creates PDFs in case these are used to compute efficiencies 
      // (corresponds to: EffMethod == kUsePDFs)
      void     CreateVariablePDFs( void );

      // checks ordering of variables in vectors
      Bool_t   SanityChecks( void );

      // returns signal and background efficiencies for given cuts - using event counting
      void     GetEffsfromSelection(  Double_t* cutMin, Double_t* cutMax,
                                      Double_t& effS, Double_t& effB);
      // returns signal and background efficiencies for given cuts - using PDFs
      void     GetEffsfromPDFs(  Double_t* cutMin, Double_t* cutMax,
                                 Double_t& effS, Double_t& effB );

      // default initialisation method called by all constructors
      void     InitCuts( void );

      ClassDef(MethodCuts,0)  // Multivariate optimisation of signal efficiency
         ;
	};

} // namespace TMVA

#endif
