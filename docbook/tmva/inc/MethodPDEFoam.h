// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPDEFoam                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      The PDEFoam method is an extension of the PDERS method, which divides     *
 *      the multi-dimensional phase space in a finite number of hyper-rectangles  *
 *      (cells) of constant event density. This "foam" of cells is filled with    *
 *      averaged probability-density information sampled from a training event    *
 *      sample.                                                                   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Peter Speckmayer <peter.speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *                                                                                *
 * Original author of the TFoam implementation:                                   *
 *      S. Jadach - Institute of Nuclear Physics, Cracow, Poland                  *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPDEFoam
#define ROOT_TMVA_MethodPDEFoam

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// MethodPDEFoam                                                            //
//                                                                          //
// The PDEFoam method is an                                                 //
// extension of the PDERS method, which divides the multi-dimensional       //
// phase space in a finite number of hyper-rectangles (cells) of constant   //
// event density.                                                           //
// This "foam" of cells is filled with averaged probability-density         //
// information sampled from a training event sample.                        //
//                                                                          //
// For a given number of cells, the binning algorithm adjusts the size      //
// and position of the cells inside the multidimensional phase space        //
// based on a binary-split algorithm, minimizing the variance of the        //
// event density in the cell.                                               //
// The binned event density information of the final foam is stored in      //
// binary trees, allowing for a fast and memory-efficient classification    //
// of events.                                                               //
//                                                                          //
// The implementation of PDEFoam is based on the Monte-Carlo integration    //
// package TFoam included in the analysis package ROOT.                     //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRandom3
#include "TRandom3.h"
#endif

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

#ifndef ROOT_TMVA_PDEFoam
#include "TMVA/PDEFoam.h"
#endif

namespace TMVA {

   class MethodPDEFoam : public MethodBase {

   public:

      MethodPDEFoam( const TString& jobName,
                     const TString& methodTitle,
                     DataSetInfo& dsi,
                     const TString& theOption = "PDEFoam",
                     TDirectory* theTargetDir = 0 );

      MethodPDEFoam( DataSetInfo& dsi,
                     const TString& theWeightFile,
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodPDEFoam( void );

      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );

      // training methods
      void Train( void );
      void TrainMonoTargetRegression( void );    // Regression output: one value
      void TrainMultiTargetRegression( void );   // Regression output: any number of values
      void TrainSeparatedClassification( void ); // Classification: one foam for Sig, one for Bg
      void TrainUnifiedClassification( void );   // Classification: one foam for Signal and Bg

      using MethodBase::ReadWeightsFromStream;

      // write weights to stream
      void AddWeightsXMLTo( void* parent ) const;

      // read weights from stream
      void ReadWeightsFromStream( std::istream & i );
      void ReadWeightsFromXML   ( void* wghtnode );

      // write/read pure foams to/from file
      void WriteFoamsToFile() const;
      void ReadFoamsFromFile();

      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );

      // regression procedure
      virtual const std::vector<Float_t>& GetRegressionValues();

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

      // helper functions to convert enum types to UInt_t and back
      EKernel GetKernel( void ) { return fKernel; }
      UInt_t KernelToUInt(EKernel ker) const { return UInt_t(ker); }
      EKernel UIntToKernel(UInt_t iker);
      UInt_t TargetSelectionToUInt(ETargetSelection ts) const { return UInt_t(ts); }
      ETargetSelection UIntToTargetSelection(UInt_t its);

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

      // calculate Xmin and Xmax for Foam
      void CalcXminXmax();

      // Set Xmin, Xmax in foam with index 'foam_index'
      void SetXminXmax(TMVA::PDEFoam*);

      // Set foam options
      void InitFoam(TMVA::PDEFoam*, EFoamType);

      // fill variable names into foam
      void FillVariableNamesToFoam() const;

   private:

      // the option handling methods
      void DeclareOptions();
      void DeclareCompatibilityOptions();
      void ProcessOptions();

      // nice output
      void PrintCoefficients( void );

      // Square function (fastest implementation)
      template<typename T> T Sqr(T x) const { return x*x; }

      // options to be used
      Bool_t        fSigBgSeparated;  // Separate Sig and Bg, or not
      Double_t      fFrac;            // Fraction used for calc of Xmin, Xmax
      Double_t      fDiscrErrCut;     // cut on discrimant error
      Float_t       fVolFrac;         // inverse volume fraction (used for density calculation during buildup)
      Float_t       fVolFracInv;      // volume fraction (used for density calculation during buildup)
      Int_t         fnCells;          // Number of Cells  (1000)
      Int_t         fnActiveCells;    // Number of active cells
      Int_t         fnSampl;          // Number of MC events per cell in build-up (1000)
      Int_t         fnBin;            // Number of bins in build-up (100)
      Int_t         fEvPerBin;        // Maximum events (equiv.) per bin in buid-up (1000)

      Bool_t        fCompress;        // compress foam output file
      Bool_t        fMultiTargetRegression; // do regression on multible targets
      UInt_t        fNmin;            // minimal number of events in cell necessary to split cell"
      Bool_t        fCutNmin;         // Keep for bw compatibility: Grabbing cell with maximal RMS to split next (TFoam default)
      UInt_t        fMaxDepth;        // maximum depth of cell tree

      TString       fKernelStr;       // Kernel for GetMvaValue() (option string)
      EKernel       fKernel;          // Kernel for GetMvaValue()
      TString       fTargetSelectionStr; // method of selecting the target (only mulit target regr.)
      ETargetSelection fTargetSelection; // method of selecting the target (only mulit target regr.)
      Bool_t        fFillFoamWithOrigWeights; // fill the foam with boost weights
      Bool_t        fUseYesNoCell;    // return -1 or 1 for bg or signal like event
      TString       fDTLogic;         // use DT algorithm to split cells
      EDTSeparation fDTSeparation;    // enum which specifies the separation to use for the DT logic
      Bool_t        fPeekMax;         // peek up cell with max. driver integral for split

      std::vector<Double_t> fXmin, fXmax; // range for histograms and foams

      // foams and densities
      // foam[0]=signal, if Sig and BG are Seperated; else foam[0]=signal/bg
      // foam[1]=background, if Sig and BG are Seperated; else it is not used
      std::vector<PDEFoam*> fFoam;

      // default initialisation called by all constructors
      void Init( void );

      ClassDef(MethodPDEFoam,0) // Analysis of PDEFoam discriminant (PDEFoam or Mahalanobis approach)
   };

} // namespace TMVA

#endif // MethodPDEFoam_H
