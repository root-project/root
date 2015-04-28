// @(#)root/tmva $Id$
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
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
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

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif

class TH1D;

namespace TMVA {

   class MethodFisher : public MethodBase {

   public:

      MethodFisher( const TString& jobName,
                    const TString& methodTitle,
                    DataSetInfo& dsi,
                    const TString& theOption = "Fisher",
                    TDirectory* theTargetDir = 0 );

      MethodFisher( DataSetInfo& dsi,
                    const TString& theWeightFile,
                    TDirectory* theTargetDir = NULL );

      virtual ~MethodFisher( void );

      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );


      // training method
      void Train( void );

      using MethodBase::ReadWeightsFromStream;

      // write weights to stream
      void AddWeightsXMLTo     ( void* parent ) const;

      // read weights from stream
      void ReadWeightsFromStream( std::istream & i );
      void ReadWeightsFromXML   ( void* wghtnode );

      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );

      enum EFisherMethod { kFisher, kMahalanobis };
      EFisherMethod GetFisherMethod( void ) { return fFisherMethod; }

      // ranking of input variables
      const Ranking* CreateRanking();

      // nice output
      void PrintCoefficients( void );


   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

   private:

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();

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
      
      std::vector<Double_t>* fDiscrimPow;  // discriminating power
      std::vector<Double_t>* fFisherCoeff; // Fisher coefficients
      Double_t fF0;                   // offset

      // default initialisation called by all constructors
      void Init( void );

      ClassDef(MethodFisher,0) // Analysis of Fisher discriminant (Fisher or Mahalanobis approach) 
   };

} // namespace TMVA

#endif // MethodFisher_H
