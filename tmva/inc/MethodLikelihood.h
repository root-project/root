// @(#)root/tmva $Id: MethodLikelihood.h,v 1.11 2007/04/19 06:53:01 brun Exp $ 
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
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
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

      MethodLikelihood( TString jobName, 
                        TString methodTitle, 
                        DataSet& theData,
                        TString theOption = "",
                        TDirectory* theTargetDir = 0 );
  
      MethodLikelihood( DataSet& theData, 
                        TString theWeightFile,  
                        TDirectory* theTargetDir = NULL );

      virtual ~MethodLikelihood( void );
    
      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;
      virtual void WriteWeightsToStream( TFile& rf ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );
      virtual void ReadWeightsFromStream( TFile& istr );

      // calculate the MVA value
      // the argument is used for internal ranking tests
      virtual Double_t GetMvaValue( void );

      // write method specific histos to target file
      virtual void WriteMonitoringHistosToFile( void ) const;

      // ranking of input variables
      const Ranking* CreateRanking();

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;

      // header and auxiliary classes
      virtual void MakeClassSpecificHeader( std::ostream&, const TString& = "" ) const;

      // get help message text
      void GetHelpMessage() const;

   private:

      // returns transformed or non-transformed output
      Double_t TransformLikelihoodOutput( Double_t ps, Double_t pb ) const;

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();
      
      // options
      Int_t     fAverageEvtPerBin;        // average events per bin; used to calculate fNbins
      Int_t*    fAverageEvtPerBinVarS;    // average events per bin; used to calculate fNbins
      Int_t*    fAverageEvtPerBinVarB;    // average events per bin; used to calculate fNbins

      Int_t            fNsmooth;        // number of smooth passes
      Int_t*           fNsmoothVarS;    // number of smooth passes
      Int_t*           fNsmoothVarB;    // number of smooth passes
      Double_t         fEpsilon;        // minimum number of likelihood (to avoid zero)
      Bool_t           fTransformLikelihoodOutput; // likelihood output is sigmoid-transformed


      // type of Splines (or Kernel) used to smooth PDFs
      TString*           fInterpolateString;  // which interpolation method used for reference histograms (individual for each variable)
      PDF::EInterpolateMethod *fInterpolateMethod; //enumerators encoding the interpolation method

      Int_t                    fSpline;        // Spline order to smooth histograms (if spline is selected for interpolation)

      TString                  fKDEtypeString; // Kernel type to use for KDE (string) (if KDE is selected for interpolation) 
      TString                  fKDEiterString; // Number of iterations (string)
      KDEKernel::EKernelType   fKDEtype;       // Kernel type to use for KDE
      KDEKernel::EKernelIter   fKDEiter;       // Number of iterations
      Float_t                  fKDEfineFactor; // fine tuning factor for Adaptive KDE: factor to multiply the "width" of the Kernel function
      KDEKernel::EKernelBorder fBorderMethod;  // the method to take care about "border" effects
      TString                  fBorderMethodString; // the method to take care about "border" effects (string)
      Int_t                    fDropVariable;  // for ranking test
      
      std::vector<TH1*>* fHistSig;        // signal PDFs (histograms)
      std::vector<TH1*>* fHistBgd;        // background PDFs (histograms)
      std::vector<TH1*>* fHistSig_smooth; // signal PDFs (smoothed histograms)
      std::vector<TH1*>* fHistBgd_smooth; // background PDFs (smoothed histograms)
  
      std::vector<PDF*>* fPDFSig;  // list of PDFs (signal)    
      std::vector<PDF*>* fPDFBgd;  // list of PDFs (background)

      // default initialisation called by all constructors
      void InitLik( void );
   
      ClassDef(MethodLikelihood,0) // Likelihood analysis ("non-parametric approach") 
   };

} // namespace TMVA

#endif // MethodLikelihood_H
