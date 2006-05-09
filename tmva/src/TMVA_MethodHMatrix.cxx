// @(#)root/tmva $Id: TMVA_MethodHMatrix.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $    
// Author: Andreas Hoecker, Xavier Prudent, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodHMatrix                                                    *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
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
 * $Id: TMVA_MethodHMatrix.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $        
 **********************************************************************************/

#include "TMVA_MethodHMatrix.h"
#include "TMVA_Tools.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

#define DEBUG_TMVA_MethodHMatrix kFALSE

ClassImp(TMVA_MethodHMatrix)

//_______________________________________________________________________
//                                                                      
// H-Matrix method, which is implemented as a simple comparison of      
// chi-squared estimators for signal and background, taking into        
// account the linear correlations between the input variables          
//                                                                      
//_______________________________________________________________________
 

//_______________________________________________________________________
TMVA_MethodHMatrix::TMVA_MethodHMatrix( TString jobName, vector<TString>* theVariables,  
					TTree* theTree, TString theOption, 
					TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  InitHMatrix();
}

//_______________________________________________________________________
TMVA_MethodHMatrix::TMVA_MethodHMatrix( vector<TString> *theVariables, 
					TString theWeightFile,  
					TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  InitHMatrix();
}

//_______________________________________________________________________
void TMVA_MethodHMatrix::InitHMatrix( void )
{
  fMethodName         = "HMatrix";
  fMethod             = TMVA_Types::HMatrix;
  fTestvar            = fTestvarPrefix+GetMethodName();
  fNormaliseInputVars = kTRUE;

  fInvHMatrixS = new TMatrixD( fNvar, fNvar );
  fInvHMatrixB = new TMatrixD( fNvar, fNvar );
  fVecMeanS    = new TVectorD( fNvar );
  fVecMeanB    = new TVectorD( fNvar );
}

//_______________________________________________________________________
TMVA_MethodHMatrix::~TMVA_MethodHMatrix( void )
{
  if (NULL != fInvHMatrixS) delete fInvHMatrixS;
  if (NULL != fInvHMatrixB) delete fInvHMatrixB;
  if (NULL != fVecMeanS   ) delete fVecMeanS;
  if (NULL != fVecMeanB   ) delete fVecMeanB;
}

//_______________________________________________________________________
void TMVA_MethodHMatrix::Train( void )
{
  //--------------------------------------------------------------

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }

  // get mean values 
  Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    TMVA_Tools::ComputeStat( fTrainingTree, (*fInputVars)[ivar], 
			     meanS, meanB, rmsS, rmsB, xmin, xmax, 
			     fNormaliseInputVars );
    (*fVecMeanS)(ivar) = meanS;
    (*fVecMeanB)(ivar) = meanB;
  }

  // compute covariance matrix
  TMVA_Tools::GetCovarianceMatrix( fTrainingTree, fInvHMatrixS, fInputVars, 1, 
				   fNormaliseInputVars );
  TMVA_Tools::GetCovarianceMatrix( fTrainingTree, fInvHMatrixB, fInputVars, 0, 
				   fNormaliseInputVars );

  // invert matrix
  fInvHMatrixS->Invert();
  fInvHMatrixB->Invert();

  // write weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
Double_t TMVA_MethodHMatrix::GetMvaValue( TMVA_Event *e )
{
  Double_t myMVA = 0;

  Double_t s = GetChi2( e, kSignal     );
  Double_t b = GetChi2( e, kBackground );
  
  if ((s + b) > 0) myMVA = (b - s)/(s + b);
  else {
    cout << "--- " << GetName() << ": Big trouble: s+b: " << s+b << " ==> abort"
	 << endl;
    exit(1);
  }
  return myMVA;
}

//_______________________________________________________________________
Double_t TMVA_MethodHMatrix::GetChi2( TMVA_Event *e,  Type type ) const
{
  
  // loop over variables
  Int_t ivar,jvar;
  vector<Double_t> val( fNvar );
  for (ivar=0; ivar<fNvar; ivar++) {
    val[ivar] = e->GetData(ivar);
    if (fNormaliseInputVars) 
      val[ivar] = __N__( val[ivar], GetXminNorm( ivar ), GetXmaxNorm( ivar ) );    
  }

  Double_t chi2 = 0;
  for (ivar=0; ivar<fNvar; ivar++) {
    for (jvar=0; jvar<fNvar; jvar++) {
      if (type == kSignal) 
  	chi2 += ( (val[ivar] - (*fVecMeanS)(ivar))*(val[jvar] - (*fVecMeanS)(jvar))
  		  * (*fInvHMatrixS)(ivar,jvar) );
      else
  	chi2 += ( (val[ivar] - (*fVecMeanB)(ivar))*(val[jvar] - (*fVecMeanB)(jvar))
  		  * (*fInvHMatrixB)(ivar,jvar) );
    }
  }

  // sanity check
  if (chi2 < 0) {
    cout << "--- " << GetName() << ": Error in ::GetChi2: negative chi2 ==> abort"
	 << chi2 << endl;
    exit(1);
  }

  return chi2;
}

//_______________________________________________________________________
void  TMVA_MethodHMatrix::WriteWeightsToFile( void )
{  
   // write coefficients to file
   
  Int_t ivar,jvar;
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": creating weight file: " << fname << endl;
  ofstream fout( fname );
  if (!fout.good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::WriteWeightsToFile: "
         << "unable to open output  weight file: " << fname << endl;
    exit(1);
  }

  // write variable names and min/max 
  // NOTE: the latter values are mandatory for the normalisation 
  // in the reader application !!!
  fout << this->GetMethodName() <<endl;
  fout << "NVars= " << fNvar <<endl; 
  for (ivar=0; ivar<fNvar; ivar++) {
    TString var = (*fInputVars)[ivar];
    fout << var << "  " << GetXminNorm( var ) << "  " << GetXmaxNorm( var ) << endl;
  }

  // mean vectors
  for (ivar=0; ivar<fNvar; ivar++) {
    fout << (*fVecMeanS)(ivar) << "  " << (*fVecMeanB)(ivar) << endl;
  }

  // inverse covariance matrices (signal)
  for (ivar=0; ivar<fNvar; ivar++) {
    for (jvar=0; jvar<fNvar; jvar++) {
      fout << (*fInvHMatrixS)(ivar,jvar) << "  ";
    }
    fout << endl;
  }

  // inverse covariance matrices (background)
  for (ivar=0; ivar<fNvar; ivar++) {
    for (jvar=0; jvar<fNvar; jvar++) {
      fout << (*fInvHMatrixB)(ivar,jvar) << "  ";
    }
    fout << endl;
  }

  fout.close();    
}
  
//_______________________________________________________________________
void  TMVA_MethodHMatrix::ReadWeightsFromFile( void )
{
   // read coefficients from file
   
  Int_t ivar,jvar;
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
  ifstream fin( fname );

  if (!fin.good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
         << "unable to open input file: " << fname << endl;
    exit(1);
  }

  // read variable names and min/max
  // NOTE: the latter values are mandatory for the normalisation 
  // in the reader application !!!
  TString var, dummy;
  Double_t xmin, xmax;
  fin >> dummy;
  this->SetMethodName(dummy);
  fin >> dummy >> fNvar;

  for (ivar=0; ivar<fNvar; ivar++) {
    fin >> var >> xmin >> xmax;

    // sanity check
    if (var != (*fInputVars)[ivar]) {
      cout << "--- " << GetName() << ": Error while reading weight file; "
	   << "unknown variable: " << var << " at position: " << ivar << ". "
	   << "Expected variable: " << (*fInputVars)[ivar] << " ==> abort" << endl;
      exit(1);
    }
    
    // set min/max
    this->SetXminNorm( ivar, xmin );
    this->SetXmaxNorm( ivar, xmax );
  }    

  // mean vectors
  for (ivar=0; ivar<fNvar; ivar++) 
    fin >> (*fVecMeanS)(ivar) >> (*fVecMeanB)(ivar);

  // inverse covariance matrices (signal)
  for (ivar=0; ivar<fNvar; ivar++) 
    for (jvar=0; jvar<fNvar; jvar++) 
      fin >> (*fInvHMatrixS)(ivar,jvar);

  // inverse covariance matrices (background)
  for (ivar=0; ivar<fNvar; ivar++) 
    for (jvar=0; jvar<fNvar; jvar++) 
      fin >> (*fInvHMatrixB)(ivar,jvar);

  fin.close();    
}

//_______________________________________________________________________
void  TMVA_MethodHMatrix::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName() 
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}
