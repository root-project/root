// @(#)root/tmva $Id$ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodLikelihood                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Likelihood analysis ("non-parametric approach")                           *
 *      Also implemented is a "diagonalized likelihood approach",                 *
 *      which improves over the uncorrelated likelihood ansatz by transforming    *
 *      linearly the input variables into a diagonal space, using the square-root *
 *      of the covariance matrix. This approach can be chosen by inserting        *
 *      the letter "D" into the option string.                                    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodLikelihood
#define ROOT_TMVA_MethodLikelihood

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodLikelihood                                                     //
//                                                                      //
// Likelihood analysis ("non-parametric approach")                      //
// Also implemented is a "diagonalized likelihood approach",            //
// which improves over the uncorrelated likelihood ansatz by            //
// transforming linearly the input variables into a diagonal space,     //
// using the square-root of the covariance matrix                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif

class TH1D;

namespace TMVA {

   class MethodLikelihood : public MethodBase {

   public:

      MethodLikelihood( const TString& jobName, 
                        const TString& methodTitle, 
                        DataSetInfo& theData,
                        const TString& theOption = "",
                        TDirectory* theTargetDir = 0 );
  
      MethodLikelihood( DataSetInfo& theData, 
                        const TString& theWeightFile,  
                        TDirectory* theTargetDir = NULL );

      virtual ~MethodLikelihood();
    
      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, 
                                      UInt_t numberClasses, UInt_t numberTargets );

      // training method
      void Train();

      // write weights to file
      void WriteWeightsToStream( TFile& rf ) const;
      void AddWeightsXMLTo( void* parent ) const;

      // read weights from file
      void ReadWeightsFromStream( istream& istr );
      void ReadWeightsFromStream( TFile& istr );
      void ReadWeightsFromXML( void* wghtnode );
      // calculate the MVA value
      // the argument is used for internal ranking tests
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );

      // write method specific histos to target file
      void WriteMonitoringHistosToFile() const;

      // ranking of input variables
      const Ranking* CreateRanking();

      virtual void WriteOptionsToStream ( ostream& o, const TString& prefix ) const;

   protected:

      void DeclareCompatibilityOptions();

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // header and auxiliary classes
      void MakeClassSpecificHeader( std::ostream&, const TString& = "" ) const;

      // get help message text
      void GetHelpMessage() const;

   private:

      // returns transformed or non-transformed output
      Double_t TransformLikelihoodOutput( Double_t ps, Double_t pb ) const;

      // the option handling methods
      void Init();
      void DeclareOptions();
      void ProcessOptions();
      
      // options
      Double_t             fEpsilon;                   // minimum number of likelihood (to avoid zero)
      Bool_t               fTransformLikelihoodOutput; // likelihood output is sigmoid-transformed

      Int_t                fDropVariable;              //  for ranking test
      
      std::vector<TH1*>*   fHistSig;                   // signal PDFs (histograms)
      std::vector<TH1*>*   fHistBgd;                   // background PDFs (histograms)
      std::vector<TH1*>*   fHistSig_smooth;            // signal PDFs (smoothed histograms)
      std::vector<TH1*>*   fHistBgd_smooth;            // background PDFs (smoothed histograms)
  
      PDF*                 fDefaultPDFLik;             // pdf that contains default definitions
      std::vector<PDF*>*   fPDFSig;                    // list of PDFs (signal)    
      std::vector<PDF*>*   fPDFBgd;                    // list of PDFs (background)

      // default initialisation called by all constructors

      // obsolete variables kept for backward combatibility
      Int_t                fNsmooth;                   // number of smooth passes
      Int_t*               fNsmoothVarS;               // number of smooth passes
      Int_t*               fNsmoothVarB;               // number of smooth passes
      Int_t                fAverageEvtPerBin;          // average events per bin; used to calculate fNbins
      Int_t*               fAverageEvtPerBinVarS;      // average events per bin; used to calculate fNbins
      Int_t*               fAverageEvtPerBinVarB;      // average events per bin; used to calculate fNbins
      TString              fBorderMethodString;        // the method to take care about "border" effects (string)
      Float_t              fKDEfineFactor;             // fine tuning factor for Adaptive KDE
      TString              fKDEiterString;             // Number of iterations (string)
      TString              fKDEtypeString;             // Kernel type to use for KDE (string)
      TString*             fInterpolateString;         // which interpolation method used for reference histograms (individual for each variable)

      ClassDef(MethodLikelihood,0) // Likelihood analysis ("non-parametric approach") 
   };

} // namespace TMVA

#endif // MethodLikelihood_H
