/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::Reader                                                          *
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

//_______________________________________________________________________
//
//  The Reader class serves to use the MVAs in a specific analysis context.
//  Within an event loop, a vector is filled that corresponds to the variables
//  that were used to train the MVA(s) during the training stage. This vector
//  is transfered to the Reader, who takes care of interpreting the weight 
//  file of the MVA of choice, and to return the MVA's output. This is then 
//  used by the user for further analysis.
//
//  ---------------------------------------------------------------------
//  Usage:
//           
//    // ------ before starting the event loop
//
//    // fill vector with variable names according to the definition and order 
//    // used in the training stage
//    vector<string> inputVars;
//    inputVars.push_back( "var1" );
//    inputVars.push_back( "var2" );
//    inputVars.push_back( "var3" );
//    inputVars.push_back( "var4" );
//
//    // create the Reader object
//    Reader *reader = new TMVA_Reader( inputVars );    
//      
//    // book the MVA of your choice (prior training of these methods, ie, 
//    // existence of weight files is required)
//    reader->BookMVA( TMVA_Types::Fisher,   "weights/Fisher.weights" );
//    reader->BookMVA( TMVA_Types::CFMlpANN, "weights/CFMlpANN.weights" );
//    // ... etc
//    
//    // ------- start the event loop
//
//    for (Long64_t ievt=0; ievt<myTree->GetEntries();ievt++) {
//
//      // fill vector with values of variables
//      vector<double> varValues;	 
//      varValues.push_back( var1 ); 
//      varValues.push_back( var2 ); 
//      varValues.push_back( var3 ); 
//      varValues.push_back( var4 ); 
//            
//      // retrieve the corresponding MVA output
//      double mvaFi  = reader->EvaluateMVA( varValues, TMVA_Types::Fisher );
//      double mvaNN  = reader->EvaluateMVA( varValues, TMVA_Types::CFMlpANN );
//
//      // do something with these ...., e.g., fill them into your ntuple
//
//    } // end of event loop
//
//    delete reader;
//  ---------------------------------------------------------------------
//
//  The example application of the Reader: "TMVApplication.cxx" can be found 
//  in the ROOT tutorial directory.
//_______________________________________________________________________

#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TClass.h"
#include "TH1D.h"
#include "TKey.h"
#include "TVector.h"
#include <stdlib.h>

#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#include "TMVA/MethodLikelihood.h"
#include "TMVA/MethodPDERS.h"
#include "TMVA/MethodHMatrix.h"
#include "TMVA/MethodFisher.h"
#include "TMVA/MethodCFMlpANN.h"
#include "TMVA/MethodTMlpANN.h"
#include "TMVA/MethodBDT.h"

ClassImp(TMVA::Reader)

//_______________________________________________________________________
TMVA::Reader::Reader( vector<TString>& inputVars, Bool_t verbose )
  : fInputVars( &inputVars ),
    fVerbose  ( verbose )
{
  // constructor
  // arguments: names of input variables (vector)
  //            verbose flag
  Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( vector<string>& inputVars, Bool_t verbose )
  : fVerbose  ( verbose )
{
  // constructor
  // arguments: names of input variables (vector)
  //            verbose flag
  fInputVars = new vector<TString>;
  for (vector<string>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
    fInputVars->push_back( ivar->c_str() );
  
  Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const string varNames, Bool_t verbose )
  : fInputVars( 0 ),
    fVerbose  ( verbose )
{
  // constructor
  // arguments: names of input variables given in form: "name1:name2:name3"
  //            verbose flag
  this->DecodeVarNames(varNames);
  Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const TString varNames, Bool_t verbose )
  : fInputVars( 0 ),
    fVerbose  ( verbose )
{
  // constructor
  // arguments: names of input variables given in form: "name1:name2:name3"
  //            verbose flag
  this->DecodeVarNames(varNames);
  Init();
}

//_______________________________________________________________________
TMVA::Reader::~Reader( void )
{
  // destructor
}  

//_______________________________________________________________________
void TMVA::Reader::Init( void )
{
  // default initialisation (no member variables)
}

//_______________________________________________________________________
Bool_t TMVA::Reader::BookMVA( TMVA::Types::MVA mva, TString weightfile )
{
  // books MVA method from weightfile
  switch (mva) {

  case (TMVA::Types::Cuts):
    fMethods.push_back( new TMVA::MethodCuts( fInputVars, weightfile ) );    
    break;

  case (TMVA::Types::Likelihood):
    fMethods.push_back( new TMVA::MethodLikelihood( fInputVars, weightfile ) );
    break; 

  case (TMVA::Types::PDERS):
    fMethods.push_back( new TMVA::MethodPDERS( fInputVars, weightfile ) );
    break; 

  case (TMVA::Types::HMatrix):
    fMethods.push_back( new TMVA::MethodHMatrix( fInputVars, weightfile ) );
    break; 

  case (TMVA::Types::Fisher):
    fMethods.push_back( new TMVA::MethodFisher( fInputVars, weightfile ) );
    break; 

  case (TMVA::Types::CFMlpANN):
    fMethods.push_back( new TMVA::MethodCFMlpANN( fInputVars, weightfile ) );
    break; 

  case (TMVA::Types::TMlpANN):
    fMethods.push_back( new TMVA::MethodTMlpANN( fInputVars, weightfile ) );
    break; 

  case (TMVA::Types::BDT):
    fMethods.push_back( new TMVA::MethodBDT( fInputVars, weightfile ) );
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
Double_t TMVA::Reader::EvaluateMVA( vector<Double_t>& inVar, TMVA::Types::MVA mva, Double_t aux )
{
  // evaluates MVA for given set of input variables
  // the aux value is only needed for MethodCuts: it sets the required signal efficiency 

  // need event
  TMVA::Event e( inVar );

  // iterate over methods and call evaluator
  vector<TMVA::MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA::MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for(; itrMethod != itrMethodEnd; itrMethod++) {
    if ((*itrMethod)->GetMethod() == mva) {
      if (mva == TMVA::Types::Cuts) 
        ((TMVA::MethodCuts*)(*itrMethod))->SetTestSignalEfficiency( aux );
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
void TMVA::Reader::DecodeVarNames( const string varNames ) 
{
  // decodes "name1:name2:..." form
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
void TMVA::Reader::DecodeVarNames( const TString varNames )
{
  // decodes "name1:name2:..." form
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
