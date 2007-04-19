// @(#)root/tmva $Id: MethodFisher.h,v 1.10 2006/11/20 15:35:28 brun Exp $
// Author: Andreas Hoecker, Xavier Prudent, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodFisher                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
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
 *                                                                                *
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

#include <vector>
#include <map>

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
                    TString methodTitle, 
                    DataSet& theData,
                    TString theOption = "Fisher",
                    TDirectory* theTargetDir = 0 );

      MethodFisher( DataSet& theData, 
                    TString theWeightFile,  
                    TDirectory* theTargetDir = NULL );

      virtual ~MethodFisher( void );
    
      // training method
      virtual void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to stream
      virtual void WriteWeightsToStream( std::ostream & o) const;

      // read weights from stream
      virtual void ReadWeightsFromStream( std::istream & i );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      enum EFisherMethod { kFisher, kMahalanobis };
      virtual EFisherMethod GetFisherMethod( void ) { return fFisherMethod; }

      // ranking of input variables
      const Ranking* CreateRanking();

   protected:

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();
  
      // Initialization and allocation of matrices
      void InitMatrices( void );

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
      
      // method to be used
      TString       fTheMethod;       // Fisher or Mahalanobis
      EFisherMethod fFisherMethod;    // Fisher or Mahalanobis 

      // covariance matrices
      TMatrixD *fBetw;                // between-class matrix
      TMatrixD *fWith;                // within-class matrix
      TMatrixD *fCov;                 // full covariance matrix

      // number of events (sumOfWeights)
      Double_t fSumOfWeightsS;        // sum-of-weights for signal training events
      Double_t fSumOfWeightsB;        // sum-of-weights for background training events
      
      vector<Double_t> *fDiscrimPow;  // discriminating power
      vector<Double_t> *fFisherCoeff; // Fisher coefficients
      Double_t fF0;                   // offset


      // default initialisation called by all constructors
      void InitFisher( void );

      ClassDef(MethodFisher,0) // Analysis of Fisher discriminant (Fisher or Mahalanobis approach) 
   };

} // namespace TMVA

#endif // MethodFisher_H
