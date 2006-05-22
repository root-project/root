// @(#)root/tmva $Id: MethodLikelihood.h,v 1.3 2006/05/22 08:04:39 andreas.hoecker Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodLikelihood                                                      *
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
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
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif

class TH1D;

namespace TMVA {

  class MethodLikelihood : public MethodBase {

  public:

    MethodLikelihood( TString jobName, 
		      vector<TString>* theVariables, 
		      TTree* theTree = 0,
		      TString theOption = "",
		      TDirectory* theTargetDir = 0 );
  
    MethodLikelihood( vector<TString> *theVariables, 
		      TString theWeightFile,  
		      TDirectory* theTargetDir = NULL );

    virtual ~MethodLikelihood( void );
    
    // training method
    virtual void Train( void );

    // write weights to file
    virtual void WriteWeightsToFile( void );
  
    // read weights from file
    virtual void ReadWeightsFromFile( void );

    // calculate the MVA value
    virtual Double_t GetMvaValue( Event *e );

    // write method specific histos to target file
    virtual void WriteHistosToFile( void ) ;

    // additional accessor
    Bool_t DecorrVarSpace( void ) { return fDecorrVarSpace; }

  protected:

  private:

    // weight file
    TFile* fFin;

    // type of Splines used to smooth PDFs
    PDF::SmoothMethod fSmoothMethod;

    Int_t            fNevt;    // total number of events in sample
    Int_t            fNsig;    // number of signal events in sample
    Int_t            fNbgd;    // number of background events in sample

    Int_t            fNsmooth; // naumber of smooth passes
    Double_t         fEpsilon; // minimum number of likelihood (to avoid zero)
    TMatrixD*        fSqS;     // square-root matrix for signal
    TMatrixD*        fSqB;     // square-root matrix for background

    vector<TH1*>*    fHistSig; // signal PDFs (histograms)
    vector<TH1*>*    fHistBgd; // background PDFs (histograms)
    vector<TH1*>*    fHistSig_smooth; // signal PDFs (smoothed histograms)
    vector<TH1*>*    fHistBgd_smooth; // background PDFs (smoothed histograms)
  
    TList* fSigPDFHist;        // list of PDF histograms (signal)
    TList* fBgdPDFHist;        // list of PDF histograms (background)

    vector<PDF*>*  fPDFSig; // list of PDFs (signal)    
    vector<PDF*>*  fPDFBgd; // list of PDFs (background)

    Int_t     fNbins;            // number of bins in reference histograms
    Int_t     fAverageEvtPerBin; // average events per bin; used to calculate fNbins

    Bool_t    fDecorrVarSpace;   // flag for decorrelation method

    // computes square-root-matrices
    void GetSQRMats( void );     

    // default initialisation called by all constructors
    void InitLik( void );
   
    ClassDef(MethodLikelihood,0) //Likelihood analysis ("non-parametric approach") 
  };

} // namespace TMVA

#endif // MethodLikelihood_H
