// @(#)root/tmva $Id: TMVA_MethodHMatrix.h,v 1.5 2006/05/02 23:27:40 helgevoss Exp $    
// Author: Andreas Hoecker, Xavier Prudent, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodHMatrix                                                    *
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
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodHMatrix.h,v 1.5 2006/05/02 23:27:40 helgevoss Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodHMatrix
#define ROOT_TMVA_MethodHMatrix

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodHMatrix                                                   //
//                                                                      //
// H-Matrix method, which is implemented as a simple comparison of      // 
// chi-squared estimators for signal and background, taking into        //
// account the linear correlations between the input variables          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif

class TMVA_MethodHMatrix : public TMVA_MethodBase {

 public:

  TMVA_MethodHMatrix( TString jobName, 
		     vector<TString>* theVariables, 
		     TTree* theTree = 0, 
		     TString theOption = "",
		     TDirectory* theTargetDir = 0 );

  TMVA_MethodHMatrix( vector<TString> *theVariables, 
		      TString theWeightFile,  
		      TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodHMatrix( void );
    
  // training method
  virtual void Train( void );

  // write weights to file
  virtual void WriteWeightsToFile( void );
  
  // read weights from file
  virtual void ReadWeightsFromFile( void );

  // calculate the MVA value
  virtual Double_t GetMvaValue( TMVA_Event *e );

  // write method specific histos to target file
  virtual void WriteHistosToFile( void );

 protected:

 private:

  Double_t GetChi2( TMVA_Event *e, Type ) const;

  // arrays of input evt vs. variable 
  TMatrixD* fInvHMatrixS;
  TMatrixD* fInvHMatrixB;
  TVectorD* fVecMeanS;
  TVectorD* fVecMeanB;

  Bool_t    fNormaliseInputVars;

  void InitHMatrix( void );

  ClassDef(TMVA_MethodHMatrix,0) //H-Matrix method, a simple comparison of chi-squared estimators for signal and background
};

#endif
