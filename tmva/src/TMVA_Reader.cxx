/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Reader                                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Reader class to be used in the user application to interpret the trained  *
 *      MVAs in an analysis context                                               *
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
 * (http://tmva.sourceforge.net/license.txt)                                      *
 *                                                                                *
 **********************************************************************************/

#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TClass.h"
#include "TH1D.h"
#include "TKey.h"
#include "TVector.h"
#include <stdlib.h>

#include "TMVA_Reader.h"
#include "TMVA_MethodCuts.h"
#include "TMVA_MethodLikelihood.h"
#include "TMVA_MethodPDERS.h"
#include "TMVA_MethodHMatrix.h"
#include "TMVA_MethodFisher.h"
#include "TMVA_MethodCFMlpANN.h"
#include "TMVA_MethodTMlpANN.h"
#include "TMVA_MethodBDT.h"

ClassImp(TMVA_Reader)

//_______________________________________________________________________
TMVA_Reader::TMVA_Reader( vector<TString>& inputVars, Bool_t verbose )
  : fInputVars( &inputVars ),
    fVerbose  ( verbose )
{
  Init();
}

//_______________________________________________________________________
TMVA_Reader::TMVA_Reader( vector<string>& inputVars, Bool_t verbose )
  : fVerbose  ( verbose )
{
  fInputVars = new vector<TString>;
  for (vector<string>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
    fInputVars->push_back( ivar->c_str() );
  
  Init();
}

//_______________________________________________________________________
TMVA_Reader::TMVA_Reader( const string varNames, Bool_t verbose )
  : fInputVars( 0 ),
    fVerbose  ( verbose )
{
  this->DecodeVarNames(varNames);
  Init();
}

//_______________________________________________________________________
TMVA_Reader::TMVA_Reader( const TString varNames, Bool_t verbose )
  : fInputVars( 0 ),
    fVerbose  ( verbose )
{
  this->DecodeVarNames(varNames);
  Init();
}

//_______________________________________________________________________
TMVA_Reader::~TMVA_Reader( void )
{}  

//_______________________________________________________________________
void TMVA_Reader::Init( void )
{}

//_______________________________________________________________________
Bool_t TMVA_Reader::BookMVA( TMVA_Types::MVA mva, TString weightfile )
{
  switch (mva) {

  case (TMVA_Types::Cuts):
    fMethods.push_back( new TMVA_MethodCuts( fInputVars, weightfile ) );    
    break;

  case (TMVA_Types::Likelihood):
    fMethods.push_back( new TMVA_MethodLikelihood( fInputVars, weightfile ) );
    break; 

  case (TMVA_Types::PDERS):
    fMethods.push_back( new TMVA_MethodPDERS( fInputVars, weightfile ) );
    break; 

  case (TMVA_Types::HMatrix):
    fMethods.push_back( new TMVA_MethodHMatrix( fInputVars, weightfile ) );
    break; 

  case (TMVA_Types::Fisher):
    fMethods.push_back( new TMVA_MethodFisher( fInputVars, weightfile ) );
    break; 

  case (TMVA_Types::CFMlpANN):
    fMethods.push_back( new TMVA_MethodCFMlpANN( fInputVars, weightfile ) );
    break; 

  case (TMVA_Types::TMlpANN):
    fMethods.push_back( new TMVA_MethodTMlpANN( fInputVars, weightfile ) );
    break; 

  case (TMVA_Types::BDT):
    fMethods.push_back( new TMVA_MethodBDT( fInputVars, weightfile ) );
    break; 

  default: 
    cerr << "--- " << GetName() << ": MVA: " << mva << " not yet implemented ==> abort"
	 << endl;
    return kFALSE;
  }  

  cout << "--- " << GetName() << ": booked method: " << fMethods.back()->GetMethodName() 
       << endl;

  // read weight file
  fMethods.back()->ReadWeightsFromFile();

  return kTRUE;
}

//_______________________________________________________________________
Double_t TMVA_Reader::EvaluateMVA( vector<Double_t>& inVar, TMVA_Types::MVA mva, Double_t aux )
{
  // need event
  TMVA_Event e( inVar );

  // iterate over methods and call evaluator
  vector<TMVA_MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA_MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for(; itrMethod != itrMethodEnd; itrMethod++) {
    if ((*itrMethod)->GetMethod() == mva) {
      if (mva == TMVA_Types::Cuts) 
	((TMVA_MethodCuts*)(*itrMethod))->SetTestSignalEfficiency( aux );
      return (*itrMethod)->GetMvaValue( &e );    
    }
  }

  // method not found !
  cerr << "--- Fatal error in " << GetName() << ": method: " << mva << " not found"
       << " ==> abort" << endl;
  exit(1);

  return -1.0;
}  

// ---------------------------------------------------------------------------------------
// ----- methods related to the decoding of the input variable names ---------------------
// ---------------------------------------------------------------------------------------

//_______________________________________________________________________
void TMVA_Reader::DecodeVarNames( const string varNames ) 
{
  fInputVars = new vector<TString>;

  size_t ipos = 0, f = 0;
  while (f != varNames.length()) {
    f = varNames.find( ':', ipos );
    if (f > varNames.length()) f = varNames.length();
    string subs = varNames.substr( ipos, f-ipos ); ipos = f+1;    
    fInputVars->push_back( subs.c_str() );
  }  
}

//_______________________________________________________________________
void TMVA_Reader::DecodeVarNames( const TString varNames )
{
  fInputVars = new vector<TString>;

  TString format;  
  Int_t   n = varNames.Length();
  TString format_obj;

  for (int i=0; i< n+1 ; i++) {
    format.Append(varNames(i));
    if ( (varNames(i)==':') || (i==n)) {
      format.Chop();
      format_obj = TString(format.Data()).ReplaceAll("@","");
      fInputVars->push_back( format_obj );
      format.Resize(0); 
    }
  }
} 
