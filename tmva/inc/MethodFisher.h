// @(#)root/tmva $Id: MethodFisher.h,v 1.5 2006/05/23 09:53:10 stelzer Exp $
// Author: Andreas Hoecker, Xavier Prudent, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodFisher                                                          *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Fisher discriminant (Fisher or Mahalanobis approach)          *
 *                                                                                *
 * Original author of this Fisher-Discriminant implementation:                    *
 *      Andre Gaidot, CEA-France;                                                 *
 *      (Translation from FORTRAN)                                                *
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
 * File and Version Information:                                                  *
 * $Id: MethodFisher.h,v 1.5 2006/05/23 09:53:10 stelzer Exp $          
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodFisher
#define ROOT_TMVA_MethodFisher

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodFisher                                                         //
//                                                                      //
// Analysis of Fisher discriminant (Fisher or Mahalanobis approach)     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrix
#include "TMatrix.h"
#endif
#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

class TH1D;

namespace TMVA {

   class MethodFisher : public MethodBase {

   public:

      MethodFisher( TString jobName, 
                    vector<TString>* theVariables, 
                    TTree* theTree = 0, 
                    TString theOption = "Fisher",
                    TDirectory* theTargetDir = 0 );

      MethodFisher( vector<TString> *theVariables, 
                    TString theWeightFile,  
                    TDirectory* theTargetDir = NULL );

      virtual ~MethodFisher( void );
    
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

      enum FisherMethod { kFisher, kMahalanobis };
      virtual FisherMethod GetMethod( void ) { return fFisherMethod; }

   protected:

   private:
  
      Int_t fNevt; // total number of events 
      Int_t fNsig; // number of signal events 
      Int_t fNbgd; // number of background events

      // event matrices: (first index: variable, second index: event)
      TMatrixF *fSig; // variables for signal 
      TMatrixF *fBgd; // variables for background

      // Initialization and allocation
      void Init( void );

      // get mean value of variables
      void GetMean( void );

      // get matrix of covariance within class
      void GetCov_WithinClass( void );

      // get matrix of covariance between class
      void GetCov_BetweenClass( void );

      // and the full covariance matrix
      void GetCov_Full( void );

      // get discriminating power
      void GetDiscrimPower( void );

      // nice output
      void PrintCoefficients( void );

      // get Fisher coefficients
      void GetFisherCoeff( void );

      // matrix of variables means: S, B, S+B vs. variables
      TMatrixD *fMeanMatx;

      // covariance matrices
      TMatrixD *fBetw;  // between-class matrix
      TMatrixD *fWith;  // within-class matrix
      TMatrixD *fCov;   // full covariance matrix

      // discriminating power
      vector<Double_t> *fDiscrimPow;

      // Fisher coefficients
      vector<Double_t> *fFisherCoeff;
      Double_t fF0;

      // method to be used (Fisher or Mahalanobis)
      FisherMethod fFisherMethod;

      // default initialisation called by all constructors
      void InitFisher( void );

      ClassDef(MethodFisher,0) //Analysis of Fisher discriminant (Fisher or Mahalanobis approach) 
         };

} // namespace TMVA

#endif // MethodFisher_H
