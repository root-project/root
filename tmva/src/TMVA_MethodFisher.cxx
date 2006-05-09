// @(#)root/tmva $Id: TMVA_MethodFisher.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $
// Author: Andreas Hoecker, Xavier Prudent, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodFisher                                                     *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
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
 **********************************************************************************/
 
//_______________________________________________________________________
//                                                                      
// Analysis of Fisher discriminant (Fisher or Mahalanobis approach)     
//                                                                      
//_______________________________________________________________________

#include "TMVA_MethodFisher.h"
#include "TMVA_Tools.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

#define DEBUG_TMVA_MethodFisher kFALSE

ClassImp(TMVA_MethodFisher)

//_______________________________________________________________________
TMVA_MethodFisher::TMVA_MethodFisher( TString jobName, vector<TString>* theVariables,  
				      TTree* theTree, TString theOption, TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  InitFisher();

  if (fOptions.Sizeof()<2) {
    fOptions = "Fisher";
    fOptions = "Fisher ";
    cout << "--- " << GetName() << ": using default options= "<< fOptions <<endl;
  }
  // option string defines "Method" (Fisher, Mahalanobis)
  // add to instance name 
  fOptions.ToLower();
  if      (fOptions.Contains( "fi" )) fFisherMethod = kFisher;
  else if (fOptions.Contains( "ma" )) fFisherMethod = kMahalanobis;
  else {
    cout << "--- " << GetName() << ": Error: unrecognized option string: " 
	 << GetOptions() << " | " << fOptions
	 << " --> exit(1)" << endl;
    exit(1);
  }

  // note that one variable is type
  if (0 != fTrainingTree) {

    // trainingTree should only contain those variables that are used in the MVA
    if (fTrainingTree->GetListOfBranches()->GetEntries() - 1 != fNvar) {
      cout << "--- " << GetName() << ": Error: mismatch in number of variables: " 
	   << fTrainingTree->GetListOfBranches()->GetEntries() << " " << fNvar
	   << " --> exit(1)" << endl;
      exit(1);
    }

    // count number of signal and background events
    fNevt = fTrainingTree->GetEntries();
    fNsig = 0;
    fNbgd = 0;
    for (Int_t ievt = 0; ievt < fNevt; ievt++) {
      if ((Int_t)TMVA_Tools::GetValue( fTrainingTree, ievt, "type" ) == 1) 
	++fNsig;
      else       
	++fNbgd;
    }        

    // numbers of events should match
    if (fNsig + fNbgd != fNevt) {
      cout << "--- " << GetName() << ": Error: mismatch in number of events" 
	   << " --> exit(1)" << endl;
      exit(1);
    }

    if (Verbose())
      cout << "--- " << GetName() << " <verbose>: num of events for training (signal, background): "
	   << " (" << fNsig << ", " << fNbgd << ")" << endl;

    // Fisher wants same number of events in each species
    if (fNsig != fNbgd) {
      cout << "--- " << GetName() << ":\t--------------------------------------------------"
	   << endl;
      cout << "--- " << GetName() << ":\tWarning: different number of signal and background\n"
	   << "--- " << GetName() << " \tevents: Fisher training will not be optimal :-("
	   << endl;
      cout << "--- " << GetName() << ":\t--------------------------------------------------"
	   << endl;
    }      

    // allocate arrays 
    Init();
  }
  else {    
    fNevt = 0;
    fNsig = 0;
    fNbgd = 0;
  }
}

//_______________________________________________________________________
TMVA_MethodFisher::TMVA_MethodFisher( vector<TString> *theVariables, 
				      TString theWeightFile,  
				      TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  InitFisher();
}

//_______________________________________________________________________
void TMVA_MethodFisher::InitFisher( void )
{
  fMethodName  = "Fisher";
  fMethod      = TMVA_Types::Fisher;  
  fTestvar     = fTestvarPrefix+GetMethodName();
  fMeanMatx    = 0; 
  fBetw        = 0;
  fWith        = 0;
  fCov         = 0;

  fNevt        = 0;
  fNsig        = 0;
  fNbgd        = 0;

  // allocate Fisher coefficients
  fF0          = 0;
  fFisherCoeff = new vector<Double_t>( fNvar );
}

//_______________________________________________________________________
TMVA_MethodFisher::~TMVA_MethodFisher( void )
{
  delete fSig;
  delete fBgd;
  delete fBetw;
  delete fWith;
  delete fCov;
  delete fDiscrimPow;
  delete fFisherCoeff;
}

//_______________________________________________________________________
void TMVA_MethodFisher::Train( void )
{
  //--------------------------------------------------------------

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }
  // get mean value of each variables for signal, backgd and signal+backgd
  GetMean();

  // get the matrix of covariance 'within class'
  GetCov_WithinClass();

  // get the matrix of covariance 'between class'
  GetCov_BetweenClass();

  // get the matrix of covariance 'between class'
  GetCov_Full();
 
  //--------------------------------------------------------------

  // get the Fisher coefficients
  GetFisherCoeff();

  // get the discriminating power of each variables
  GetDiscrimPower();

  // nice output
  PrintCoefficients();

  // write weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
Double_t TMVA_MethodFisher::GetMvaValue( TMVA_Event *e )
{
  Double_t result = fF0;
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    result += (*fFisherCoeff)[ivar]*__N__( e->GetData(ivar) , GetXminNorm( ivar ), GetXmaxNorm( ivar ) );
  }
  
  return result;
}


//_______________________________________________________________________
void TMVA_MethodFisher::Init( void )
{
  // should never be called without existing trainingTree
  if (0 == fTrainingTree) {
    cout << "--- " << GetName() << ": Error in ::Init(): fTrainingTree is zero pointer"
	 << " --> exit(1)" << endl;
    exit(1);
  }

  // signal and background LUTs
  fSig = new TMatrix( fNvar, fNsig );
  fBgd = new TMatrix( fNvar, fNbgd );
   
  // average value of each variables for S, B, S+B
  fMeanMatx = new TMatrix( fNvar, 3 );

  // the covariance 'within class' and 'between class' matrices
  fBetw = new TMatrix( fNvar, fNvar );
  fWith = new TMatrix( fNvar, fNvar );
  fCov  = new TMatrix( fNvar, fNvar );

  // discriminating power
  fDiscrimPow = new vector<Double_t>( fNvar );

  // ---- fill LUTs

  Int_t isig = 0, ibgd = 0, ivar;
  for (Int_t ievt=0; ievt<fNevt; ievt++) {

    // separate signal and background events  
    if ((Int_t)TMVA_Tools::GetValue( fTrainingTree, ievt, "type" ) == 1) {
      for (ivar=0; ivar<fNvar; ivar++) {
	Double_t x = TMVA_Tools::GetValue( fTrainingTree, ievt, (*fInputVars)[ivar] );
	(*fSig)(ivar, isig) = __N__( x, GetXminNorm( ivar ), GetXmaxNorm( ivar ) );
      }
      ++isig;
    }
    else {
      for (ivar=0; ivar<fNvar; ivar++) {
	Double_t x = TMVA_Tools::GetValue( fTrainingTree, ievt, (*fInputVars)[ivar] );
	(*fBgd)(ivar, ibgd) = __N__( x, GetXminNorm( ivar ), GetXmaxNorm( ivar ) );
      }
      ++ibgd;
    }
  }  
}

//_______________________________________________________________________
void TMVA_MethodFisher::GetMean( void )
{
  for(Int_t ivar=0; ivar<fNvar; ivar++) {   

    // signal
    Double_t sum = 0;
    for (Int_t ievt=0; ievt<fNsig; ievt++) sum += (*fSig)(ivar, ievt);
    (*fMeanMatx)( ivar, 2 ) = sum;
    (*fMeanMatx)( ivar, 0 ) = sum/fNsig;
    
    // background
    sum = 0;
    for (Int_t ievt=0; ievt<fNbgd; ievt++) sum += (*fBgd)(ivar, ievt);
    (*fMeanMatx)( ivar, 2 ) += sum;
    (*fMeanMatx)( ivar, 1 ) = sum/fNbgd;       

    // signal + background
    (*fMeanMatx)( ivar, 2 ) /= (fNsig + fNbgd);
  }  
}

//_______________________________________________________________________
void TMVA_MethodFisher::GetCov_WithinClass( void )
{
  // the matrix of covariance 'within class' reflects the dispersion of the
  // events relative to the center of gravity of their own class  

  // products matrix's (x-<x>)(y-<y>) where x;y are variables
  Double_t prodSig, prodBgd;
  Int_t    ievt;

  // 'within class' covariance
  for (Int_t x=0; x<fNvar; x++) {
    for (Int_t y=0; y<fNvar; y++) {

      Double_t sumSig = 0;
      Double_t sumBgd = 0;

      for (ievt=0; ievt<fNsig; ievt++) {
	prodSig = ( ((*fSig)(x, ievt) - (*fMeanMatx)(x, 0))* 
		    ((*fSig)(y, ievt) - (*fMeanMatx)(y, 0)) );
	sumSig += prodSig;
      }

      for (ievt=0; ievt<fNbgd; ievt++) {
	prodBgd = ( ((*fBgd)(x, ievt) - (*fMeanMatx)(x, 1))* 
		    ((*fBgd)(y, ievt) - (*fMeanMatx)(y, 1)) );
	sumBgd += prodBgd;
      }

      (*fWith)(x, y) = (sumSig + sumBgd)/fNevt;
    }
  }
}

//_______________________________________________________________________
void TMVA_MethodFisher::GetCov_BetweenClass( void )
{
  // the matrix of covariance 'between class' reflects the dispersion of the
  // events of a class relative to the global center of gravity of all the class
  // hence the separation between classes

  Double_t prodSig, prodBgd;

  for (Int_t x=0; x<fNvar; x++) {
    for (Int_t y=0; y<fNvar; y++) {

      prodSig = ( ((*fMeanMatx)(x, 0) - (*fMeanMatx)(x, 2))*
		  ((*fMeanMatx)(y, 0) - (*fMeanMatx)(y, 2)) );
      prodBgd = ( ((*fMeanMatx)(x, 1) - (*fMeanMatx)(x, 2))*
		  ((*fMeanMatx)(y, 1) - (*fMeanMatx)(y, 2)) );

      (*fBetw)(x, y) = (fNsig*prodSig + fNbgd*prodBgd)/Double_t(fNevt);
    }
  }
}

//_______________________________________________________________________
void TMVA_MethodFisher::GetCov_Full( void )
{
  for (Int_t x=0; x<fNvar; x++) 
    for (Int_t y=0; y<fNvar; y++) 
      (*fCov)(x, y) = (*fWith)(x, y) + (*fBetw)(x, y);
}

//_______________________________________________________________________
void TMVA_MethodFisher::GetFisherCoeff( void )
{
  // Fisher = Sum { [coeff]*[variables] }
  //
  // let Xs be the array of the mean values of variables for signal evts
  // let Xb be the array of the mean values of variables for backgd evts
  // let InvWith be the inverse matrix of the 'within class' correlation matrix
  //
  // then the array of Fisher coefficients is 
  // [coeff] =sqrt(fNsig*fNbgd)/fNevt*transpose{Xs-Xb}*InvWith

  // invert covariance matrix
  TMatrix* theMat = 0;
  switch (fFisherMethod) {
  case kFisher:
    theMat = fWith;
    break;
  case kMahalanobis:
    theMat = fCov;
    break;
  default:
    cout << "--- " << GetName() << ": ERROR: undefined method ==> exit(1)" << endl;
    exit(1);
  }

  TMatrix invCov( *theMat );
  invCov.Invert();

  // apply rescaling factor
  Double_t xfact = sqrt(Double_t(fNsig*fNbgd))/Double_t(fNsig + fNbgd);

  // compute difference of mean values
  vector<Double_t> diffMeans( fNvar );
  Int_t ivar, jvar;
  for (ivar=0; ivar<fNvar; ivar++) {
    (*fFisherCoeff)[ivar] = 0;

    for(jvar=0; jvar<fNvar; jvar++) {
      Double_t d = (*fMeanMatx)(jvar, 0) - (*fMeanMatx)(jvar, 1);
      (*fFisherCoeff)[ivar] += invCov(ivar, jvar)*d;
    }    
    
    // rescale
    (*fFisherCoeff)[ivar] *= xfact;
  }

  // offset correction
  fF0 = 0.0;
  for(ivar=0; ivar<fNvar; ivar++) 
    fF0 += (*fFisherCoeff)[ivar]*((*fMeanMatx)(ivar, 0) + (*fMeanMatx)(ivar, 1));
  fF0 /= -2.0;  
}

//_______________________________________________________________________
void TMVA_MethodFisher::GetDiscrimPower( void )
{
  //small values of "fWith" indicates little compactness of sig & of backgd
  //big values of "fBetw" indicates large separation between sig & backgd
  //
  //we want signal & backgd classes as compact and separated as possible
  //the discriminating power is then defined as the ration "fBetw/fWith"
  for (Int_t ivar=0; ivar<fNvar; ivar++)
    if ((*fCov)(ivar, ivar) != 0) 
      (*fDiscrimPow)[ivar] = (*fBetw)(ivar, ivar)/(*fCov)(ivar, ivar);
    else
      (*fDiscrimPow)[ivar] = 0;
}

//_______________________________________________________________________
void TMVA_MethodFisher::PrintCoefficients( void ) 
{
  // display Fisher coefficients and discriminating power for each variable
  // check maximum length of variable name
  Int_t maxL = 0; 
  vector<Double_t> dp( fNvar );
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    if ((*fInputVars)[ivar].Length() > maxL) maxL = (*fInputVars)[ivar].Length();
    dp[ivar] = (*fDiscrimPow)[ivar];
  }

  // sort according to rank (descending)
  sort   ( dp.begin(), dp.end() );
  reverse( dp.begin(), dp.end() );

  cout << "--- " << endl;
  cout << "--- " << GetName() << ": ranked output (top variable is best ranked)" << endl;
  cout << "----------------------------------------------------------------" << endl;
  cout << "--- " << setiosflags(ios::left) << setw(maxL+5) << "Variable   :"
       << resetiosflags(ios::right) 
       << setw(12) << "  Coefficient:"
       << "   Discr. power:" << endl;
  cout << "----------------------------------------------------------------" << endl;
  for (Int_t ivar=0; ivar<fNvar; ivar++) 
    for (Int_t jvar=0; jvar<fNvar; jvar++) 
      if (dp[ivar] == (*fDiscrimPow)[jvar])
	printf( "--- %-11s:   %+.3f         %.4f\n", 
		(const char*)(*fInputVars)[jvar], (*fFisherCoeff)[jvar], (*fDiscrimPow)[jvar]);
  printf( "--- %-11s:   %+.3f         %i\n", "(offset)", fF0, 0 );
  cout << "----------------------------------------------------------------" << endl;
  cout << "--- " << endl;
}

//_______________________________________________________________________
void  TMVA_MethodFisher::WriteWeightsToFile( void )
{  
  // write coefficients to file
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
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    TString var = (*fInputVars)[ivar];
    fout << var << "  " << GetXminNorm( var ) << "  " << GetXmaxNorm( var ) << endl;
  }

  // and save the weights
  fout << fF0 << endl;
  for (Int_t ivar=0; ivar<fNvar; ivar++) fout << (*fFisherCoeff)[ivar] << endl;
  fout.close();    
}
  
//_______________________________________________________________________
void  TMVA_MethodFisher::ReadWeightsFromFile( void )
{
  // read coefficients from file
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
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
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

  // and read the weights (Fisher coefficients)
  fin >> fF0;
  for (Int_t ivar=0; ivar<fNvar; ivar++) fin >> (*fFisherCoeff)[ivar];
  fin.close();    
}

//_______________________________________________________________________
void  TMVA_MethodFisher::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName() 
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}
