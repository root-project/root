// @(#)root/tmva $Id$    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodHMatrix                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      H-Matrix method, which is implemented as a simple comparison of           *
 *      chi-squared estimators for signal and background, taking into account     *
 *      the linear correlations between the input variables.                      *
 *      Method is (also) used by D0 Collaboration (FNAL) for electron             *
 *      identification; for more information, see, eg,                            *
 *      http://www-d0.fnal.gov/d0dist/dist/packages/tau_hmchisq/devel/doc/        *
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

#ifndef ROOT_TMVA_MethodHMatrix
#define ROOT_TMVA_MethodHMatrix

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodHMatrix                                                        //
//                                                                      //
// H-Matrix method, which is implemented as a simple comparison of      // 
// chi-squared estimators for signal and background, taking into        //
// account the linear correlations between the input variables          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixDfwd
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif
#endif
#ifndef ROOT_TMVA_TVectorD
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif
#endif

namespace TMVA {

   class MethodHMatrix : public MethodBase {

   public:

      MethodHMatrix( const TString& jobName, 
                     const TString& methodTitle, 
                     DataSetInfo& theData,
                     const TString& theOption = "",
                     TDirectory* theTargetDir = 0 );

      MethodHMatrix( DataSetInfo& theData, 
                     const TString& theWeightFile,  
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodHMatrix();
    
      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );

      // training method
      void Train();

      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void AddWeightsXMLTo( void* parent ) const;

      // read weights from file
      void ReadWeightsFromStream( istream& istr );
      void ReadWeightsFromXML( void* wghtnode );
      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

   private:

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();

      // returns chi2 estimator for given type (signal or background)
      Double_t GetChi2( Event* e, Types::ESBType ) const;
      Double_t GetChi2( Types::ESBType ) const;

      // compute correlation matrices
      void     ComputeCovariance( Bool_t, TMatrixD* );

      // arrays of input evt vs. variable 
      TMatrixD* fInvHMatrixS; // inverse H-matrix (signal)
      TMatrixD* fInvHMatrixB; // inverse H-matrix (background)
      TVectorD* fVecMeanS;    // vector of mean values (signal)
      TVectorD* fVecMeanB;    // vector of mean values (background)

      // default initialisation method called by all constructors
      void Init(); 

      ClassDef(MethodHMatrix,0) // H-Matrix method, a simple comparison of chi-squared estimators for signal and background
   }; 

} // namespace TMVA

#endif
