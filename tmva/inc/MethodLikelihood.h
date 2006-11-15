// @(#)root/tmva $Id: MethodLikelihood.h,v 1.21 2006/11/02 15:44:50 andreas.hoecker Exp $ 
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
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
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

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      // write method specific histos to target file
      virtual void WriteMonitoringHistosToFile( void ) const;

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

      // overload test event reading
      virtual Bool_t ReadTestEvent(UInt_t ievt, Types::SBType type = Types::kSignal) { 
         return Data().ReadTestEvent( ievt, Types::kNone, type ); 
      }

   protected:

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();
      
      // options
      Int_t     fSpline;           // Spline order to smooth histograms
      Int_t     fAverageEvtPerBin; // average events per bin; used to calculate fNbins

      // type of Splines used to smooth PDFs
      PDF::SmoothMethod fSmoothMethod;

      // global weight file -- (needed !)
      TFile*             fFin;

      Int_t            fNsmooth; // naumber of smooth passes
      Double_t         fEpsilon; // minimum number of likelihood (to avoid zero)
      Bool_t           fTransformLikelihoodOutput; // likelihood output is sigmoid-transformed

      std::vector<TH1*>* fHistSig;        // signal PDFs (histograms)
      std::vector<TH1*>* fHistBgd;        // background PDFs (histograms)
      std::vector<TH1*>* fHistSig_smooth; // signal PDFs (smoothed histograms)
      std::vector<TH1*>* fHistBgd_smooth; // background PDFs (smoothed histograms)
  
      TList* fSigPDFHist;         // list of PDF histograms (signal)
      TList* fBgdPDFHist;         // list of PDF histograms (background)

      std::vector<UInt_t>* fIndexSig; // used for caching in GetMvaValue
      std::vector<UInt_t>* fIndexBgd; // used for caching in GetMvaValue

      std::vector<PDF*>* fPDFSig; // list of PDFs (signal)    
      std::vector<PDF*>* fPDFBgd; // list of PDFs (background)

      // default initialisation called by all constructors
      void InitLik( void );
   
      ClassDef(MethodLikelihood,0) //Likelihood analysis ("non-parametric approach") 
         ;
   };

} // namespace TMVA

#endif // MethodLikelihood_H
