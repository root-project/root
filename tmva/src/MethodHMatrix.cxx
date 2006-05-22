// @(#)root/tmva $Id: MethodHMatrix.cxx,v 1.1 2006/05/22 17:36:01 brun Exp $    
// Author: Andreas Hoecker, Xavier Prudent, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodHMatrix                                                   *
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
 **********************************************************************************/

#include "TMVA/MethodHMatrix.h"
#include "TMVA/Tools.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

ClassImp(TMVA::MethodHMatrix)

//_______________________________________________________________________
//Begin_Html                                                                      
/*
H-Matrix method, which is implemented as a simple comparison of      
chi-squared estimators for signal and background, taking into        
account the linear correlations between the input variables          

This MVA approach is used by the D&#216; collaboration (FNAL) for the 
purpose of electron identification (see, eg., 
<a href="http://arxiv.org/abs/hep-ex/9507007">hep-ex/9507007</a>). 
As it is implemented in TMVA, it is usually equivalent or worse than
the Fisher-Mahalanobis discriminant, and it has only been added for 
the purpose of completeness.
Two &chi;<sup>2</sup> estimators are computed for an event, each one
for signal and background, using the estimates for the means and 
covariance matrices obtained from the training sample:<br>
<center>
<img vspace=6 src="gif/tmva_chi2.gif" align="bottom" > 
</center>
TMVA then uses as normalised analyser for event (<i>i</i>) the ratio:
(<i>&chi;<sub>S</sub>(i)<sup>2</sup> &minus; &chi;<sub>B</sub><sup>2</sup>(i)</i>)
(<i>&chi;<sub>S</sub><sup>2</sup>(i) + &chi;<sub>B</sub><sup>2</sup>(i)</i>).
*/
//End_Html
//_______________________________________________________________________
 

//_______________________________________________________________________
TMVA::MethodHMatrix::MethodHMatrix( TString jobName, vector<TString>* theVariables,  
                                        TTree* theTree, TString theOption, 
                                        TDirectory* theTargetDir )
  : TMVA::MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  // standard constructor for the H-Matrix method
  //
  // HMatrix options: none
  InitHMatrix();
}

//_______________________________________________________________________
TMVA::MethodHMatrix::MethodHMatrix( vector<TString> *theVariables, 
                                        TString theWeightFile,  
                                        TDirectory* theTargetDir )
  : TMVA::MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  // constructor to calculate the H-Matrix from the weight file
  InitHMatrix();
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::InitHMatrix( void )
{
  // default initialisation called by all constructors
  fMethodName         = "HMatrix";
  fMethod             = TMVA::Types::HMatrix;
  fTestvar            = fTestvarPrefix+GetMethodName();
  fNormaliseInputVars = kTRUE;

  fInvHMatrixS = new TMatrixD( fNvar, fNvar );
  fInvHMatrixB = new TMatrixD( fNvar, fNvar );
  fVecMeanS    = new TVectorD( fNvar );
  fVecMeanB    = new TVectorD( fNvar );
}

//_______________________________________________________________________
TMVA::MethodHMatrix::~MethodHMatrix( void )
{
  // destructor
  if (NULL != fInvHMatrixS) delete fInvHMatrixS;
  if (NULL != fInvHMatrixB) delete fInvHMatrixB;
  if (NULL != fVecMeanS   ) delete fVecMeanS;
  if (NULL != fVecMeanB   ) delete fVecMeanB;
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::Train( void )
{
  // computes H-matrices for signal and background samples

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }

  // get mean values 
  Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    TMVA::Tools::ComputeStat( fTrainingTree, (*fInputVars)[ivar], 
                             meanS, meanB, rmsS, rmsB, xmin, xmax, 
                             fNormaliseInputVars );
    (*fVecMeanS)(ivar) = meanS;
    (*fVecMeanB)(ivar) = meanB;
  }

  // compute covariance matrix
  TMVA::Tools::GetCovarianceMatrix( fTrainingTree, fInvHMatrixS, fInputVars, 1, 
                                   fNormaliseInputVars );
  TMVA::Tools::GetCovarianceMatrix( fTrainingTree, fInvHMatrixB, fInputVars, 0, 
                                   fNormaliseInputVars );

  // invert matrix
  fInvHMatrixS->Invert();
  fInvHMatrixB->Invert();

  // write weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
Double_t TMVA::MethodHMatrix::GetMvaValue( TMVA::Event *e )
{
  // returns the H-matrix signal estimator
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
Double_t TMVA::MethodHMatrix::GetChi2( TMVA::Event *e,  Type type ) const
{
  // compute chi2-estimator for event according to type (signal/background)

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
void  TMVA::MethodHMatrix::WriteWeightsToFile( void )
{  
  // write matrices and mean vectors to file
   
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
void  TMVA::MethodHMatrix::ReadWeightsFromFile( void )
{
  // read matrices and mean vectors from file
   
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
void  TMVA::MethodHMatrix::WriteHistosToFile( void )
{
  // write special monitoring histograms to file - not implemented for H-Matrix
  cout << "--- " << GetName() << ": write " << GetName() 
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}
