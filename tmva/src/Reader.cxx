// @(#)root/tmva $Id: Reader.cxx,v 1.26 2006/10/02 09:10:39 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Reader                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://ttmva.sourceforge.net/LICENSE)                                         *
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
//    // ------ before starting the event loop (eg, in the initialisation step)
//
//    //
//    // create TMVA::Reader object
//    //
//    TMVA::Reader *reader = new TMVA::Reader();    
//
//    // create a set of variables and declare them to the reader
//    // - the variable names must corresponds in name and type to 
//    // those given in the weight file(s) that you use
//    Float_t var1, var2, var3, var4;
//    reader->AddVariable( "var1", &var1 );
//    reader->AddVariable( "var2", &var2 );
//    reader->AddVariable( "var3", &var3 );
//    reader->AddVariable( "var4", &var4 );
//      
//    // book the MVA of your choice (prior training of these methods, ie, 
//    // existence of the weight files is required)
//    reader->BookMVA( "Fisher method",  "weights/Fisher.weights.txt"   );
//    reader->BookMVA( "MLP method",     "weights/MLP.weights.txt" );
//    // ... etc
//    
//    // ------- start your event loop
//
//    for (Long64_t ievt=0; ievt<myTree->GetEntries();ievt++) {
//
//      // fill vector with values of variables computed from those in the tree
//      var1 = myvar1;
//      var2 = myvar2;
//      var3 = myvar3;
//      var4 = myvar4;
//            
//      // retrieve the corresponding MVA output
//      double mvaFi = reader->EvaluateMVA( "Fisher method" );
//      double mvaNN = reader->EvaluateMVA( "MLP method"    );
//
//      // do something with these ...., e.g., fill them into your ntuple
//
//    } // end of event loop
//
//    delete reader;
//  ---------------------------------------------------------------------
//
//  An example application of the Reader can be found in TMVA/macros/TMVApplication.C.
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
#include "TMVA/MethodMLP.h"
#include "TMVA/MethodRuleFit.h"

ClassImp(TMVA::Reader)

//_______________________________________________________________________
TMVA::Reader::Reader( Bool_t verbose )
   : fDataSet  ( new DataSet ),
     fVerbose  ( verbose )
{
   // constructor
   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( vector<TString>& inputVars, Bool_t verbose )
   : fDataSet  ( new DataSet ),
     fVerbose  ( verbose )
{
   // constructor
   // arguments: names of input variables (vector)
   //            verbose flag
   for (vector<TString>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
      Data().AddVariable( *ivar );
      
   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( vector<string>& inputVars, Bool_t verbose )
   : fDataSet( new DataSet ),
     fVerbose( verbose )
{
   // constructor
   // arguments: names of input variables (vector)
   //            verbose flag
   for (vector<string>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
      Data().AddVariable( ivar->c_str() );

   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const string varNames, Bool_t verbose )
   : fDataSet( new DataSet ),
     fVerbose( verbose )
{
   // constructor
   // arguments: names of input variables given in form: "name1:name2:name3"
   //            verbose flag
   this->DecodeVarNames(varNames);
   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const TString varNames, Bool_t verbose )
   : fDataSet( new DataSet ),
     fVerbose( verbose )
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
void TMVA::Reader::AddVariable( const TString& expression, float* datalink) 
{
   Data().AddVariable(expression, 'F', (void*)datalink);
}

//_______________________________________________________________________
void TMVA::Reader::AddVariable( const TString& expression, int* datalink) 
{
   Data().AddVariable(expression, 'I', (void*)datalink);
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( TString methodName, TString weightfile )
{
   // read method name from weight file
   ifstream fin( weightfile );
   if(!fin.good()) { // file not found --> Error
      cout << "--- " << GetName() << "::BookMVA: fatal error: "
           << "unable to open input weight file: " << weightfile << endl;
      exit(1);
   }

   char buf[512];

   // read the method name
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("Method")) fin.getline(buf,512);
   TString ls(buf);
   Int_t idx1 = ls.First(':')+2; Int_t idx2 = ls.Index(' ',idx1)-idx1; if (idx2<0) idx2=ls.Length();
   fin.close();  

   TString MethodTypeFromFile = ls(idx1,idx2);

   MethodBase* method = (MethodBase*)this->BookMVA( TMVA::gTypes.GetMethodType(MethodTypeFromFile), 
                                                    weightfile );
   method->SetMethodTitle(methodName);

   return fMethodMap[methodName] = method;
}


//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( TMVA::Types::MVA methodType, TString weightfile )
{
   IMethod* method = 0;
   // books MVA method from weightfile
   switch (methodType) {

   case (TMVA::Types::Cuts):
      method = new TMVA::MethodCuts( Data(), weightfile );    
      break;

   case (TMVA::Types::Likelihood):
      method = new TMVA::MethodLikelihood( Data(), weightfile );
      break; 

   case (TMVA::Types::PDERS):
      method = new TMVA::MethodPDERS( Data(), weightfile );
      break; 

   case (TMVA::Types::HMatrix):
      method = new TMVA::MethodHMatrix( Data(), weightfile );
      break; 

   case (TMVA::Types::Fisher):
      method = new TMVA::MethodFisher( Data(), weightfile );
      break; 

   case (TMVA::Types::CFMlpANN):
      method = new TMVA::MethodCFMlpANN( Data(), weightfile );
      break; 

   case (TMVA::Types::TMlpANN):
      method = new TMVA::MethodTMlpANN( Data(), weightfile );
      break; 

   case (TMVA::Types::BDT):
      method = new TMVA::MethodBDT( Data(), weightfile );
      break; 

   case (TMVA::Types::MLP):
      method = new TMVA::MethodMLP( Data(), weightfile );
     break;

   case (TMVA::Types::RuleFit):
      method = new TMVA::MethodRuleFit( Data(), weightfile );
      break; 

   default: 
      cerr << "--- " << GetName() << ": MVA method: " << methodType << " not yet implemented ==> abort"
           << endl;
      return 0;
   }  

   cout << "--- " << GetName() << ": booked MVA method: " << method->GetMethodName() << endl;

   // read weight file
   method->ReadStateFromFile();

   return method;
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const std::vector<Float_t>& inputVec, TString methodName, Double_t aux )
{
   for (UInt_t ivar=0; ivar<inputVec.size(); ivar++) Data().Event().SetVal( ivar, inputVec[ivar] );
   
   return EvaluateMVA( methodName, aux );
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const std::vector<Double_t>& inputVec, TString methodName, Double_t aux )
{
   for (UInt_t ivar=0; ivar<inputVec.size(); ivar++) Data().Event().SetVal( ivar, (Float_t)inputVec[ivar] );
   
   return EvaluateMVA( methodName, aux );
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( TString methodName, Double_t aux )
{
   IMethod* method = 0;

   // evaluates MVA for given set of input variables
   std::map<const TString, IMethod*>::iterator it = fMethodMap.find( methodName );
   if (it == fMethodMap.end()) {
      cout << "--- " << GetName() << "EvaluateMVA(TString: fatal error: unknown method in map: " << method
           << " ==> abort" << endl;
      cout << " you looked for " << methodName<< " while the available methods are : " <<endl;
      for ( it = fMethodMap.begin(); it!=fMethodMap.end(); it++) cout << "M" << it->first << endl;

      exit(1);
   }

   else method = it->second;

   return this->EvaluateMVA( method, aux );
}  

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( IMethod* method, Double_t aux )
{
   // evaluates the MVA
  
   if (method->GetPreprocessingMethod() != Types::kNone) Data().BackupEvent();

   // NOTE: in likelihood the preprocessing transformations are inserted by hand in GetMvaValue()
   // (to distinguish signal and background transformations), and hence should not be applied here
   if (method->GetMethodType() != Types::Likelihood) 
      Data().ApplyTransformation( method->GetPreprocessingMethod(), kTRUE );

   // the aux value is only needed for MethodCuts: it sets the required signal efficiency 
   if (method->GetMethodType() == TMVA::Types::Cuts) 
      ((TMVA::MethodCuts*)method)->SetTestSignalEfficiency( aux );

   Double_t mvaVal = method->GetMvaValue();

   if (method->GetPreprocessingMethod() != Types::kNone) Data().RestoreEvent();   

   return mvaVal;
}

// ---------------------------------------------------------------------------------------
// ----- methods related to the decoding of the input variable names ---------------------
// ---------------------------------------------------------------------------------------

//_______________________________________________________________________
void TMVA::Reader::DecodeVarNames( const string varNames ) 
{
   // decodes "name1:name2:..." form
   size_t ipos = 0, f = 0;
   while (f != varNames.length()) {
      f = varNames.find( ':', ipos );
      if (f > varNames.length()) f = varNames.length();
      string subs = varNames.substr( ipos, f-ipos ); ipos = f+1;    
      Data().AddVariable( subs.c_str() );
   }  
}

//_______________________________________________________________________
void TMVA::Reader::DecodeVarNames( const TString varNames )
{
   // decodes "name1:name2:..." form

   TString format;  
   Int_t   n = varNames.Length();
   TString format_obj;

   for (int i=0; i< n+1 ; i++) {
      format.Append(varNames(i));
      if ( (varNames(i)==':') || (i==n)) {
         format.Chop();
         format_obj = TString(format.Data()).ReplaceAll("@","");
         Data().AddVariable( format_obj );
         format.Resize(0); 
      }
   }
} 
