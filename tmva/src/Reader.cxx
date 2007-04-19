// @(#)root/tmva $Id: Reader.cxx,v 1.12 2006/11/20 15:35:28 brun Exp $   
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

#include <fstream>

#include "TMVA/Reader.h"
#include "TMVA/Config.h"
#include "TMVA/Methods.h"

#define TMVA_Reader_TestIO__
#undef  TMVA_Reader_TestIO__

ClassImp(TMVA::Reader)

//_______________________________________________________________________
TMVA::Reader::Reader( TString theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSet( new DataSet ),
     fVerbose( verbose ),
     fColor( kFALSE ),
     fLogger ( this )
{
   // constructor
   DeclareOptions();
   ParseOptions();

   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( vector<TString>& inputVars, TString theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSet( new DataSet ),
     fVerbose( verbose ),
     fColor( kFALSE ),
     fLogger ( this )
{
   // constructor
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables (vector)
   //            verbose flag
   for (vector<TString>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
      Data().AddVariable( *ivar );
      
   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( vector<string>& inputVars, TString theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSet( new DataSet ),
     fVerbose( verbose ),
     fColor( kFALSE ),
     fLogger ( this )
{
   // constructor
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables (vector)
   //            verbose flag
   for (vector<string>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
      Data().AddVariable( ivar->c_str() );

   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const string varNames, TString theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSet( new DataSet ),
     fVerbose( verbose ),
     fColor( kFALSE ),
     fLogger ( this )
{
   // constructor
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables given in form: "name1:name2:name3"
   //            verbose flag
   this->DecodeVarNames(varNames);
   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const TString varNames, TString theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSet( new DataSet ),
     fVerbose( verbose ),
     fColor( kFALSE ),
     fLogger ( this )
{
   // constructor
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables given in form: "name1:name2:name3"
   //            verbose flag
   this->DecodeVarNames(varNames);
   Init();
}

//_______________________________________________________________________
void TMVA::Reader::DeclareOptions() 
{
   // declaration of configuration options
   DeclareOptionRef( fVerbose, "V", "verbose flag" );
   DeclareOptionRef( fColor=kTRUE, "Color", "color flag (default on)" );
   
   ParseOptions(kFALSE);
   
   fLogger.SetMinType( Verbose() ? kVERBOSE : kINFO );
   
   Config::Instance().SetUseColor(fColor);

   if (fDataSet!=0) fDataSet->SetVerbose(Verbose());
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
void TMVA::Reader::AddVariable( const TString& expression, float* datalink ) 
{
   // Add a float variable or expression to the reader
   Data().AddVariable(expression, 'F', (void*)datalink);
}

//_______________________________________________________________________
void TMVA::Reader::AddVariable( const TString& expression, int* datalink ) 
{
   // Add a integer variable or expression to the reader
   Data().AddVariable(expression, 'I', (void*)datalink);
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( TString methodName, TString weightfile )
{
   // read method name from weight file
   ifstream fin( weightfile );
   if(!fin.good()) { // file not found --> Error
      fLogger << kFATAL << "<BookMVA> fatal error: "
              << "unable to open input weight file: " << weightfile << Endl;
   }

   char buf[512];

   // read the method name
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("Method")) fin.getline(buf,512);
   TString ls(buf);
   Int_t idx1 = ls.First(':')+2; Int_t idx2 = ls.Index(' ',idx1)-idx1; if (idx2<0) idx2=ls.Length();
   fin.close();  

   TString MethodTypeFromFile = ls(idx1,idx2);

   MethodBase* method = (MethodBase*)this->BookMVA( Types::Instance().GetMethodType(MethodTypeFromFile), 
                                                    weightfile );
   method->SetMethodTitle(methodName);

   return fMethodMap[methodName] = method;
}


//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( TMVA::Types::EMVA methodType, TString weightfile )
{
   // books MVA method from weightfile
   IMethod* method = 0;
   switch (methodType) {

   case (TMVA::Types::kCuts):
      method = new TMVA::MethodCuts( Data(), weightfile );    
      break;

   case (TMVA::Types::kLikelihood):
      method = new TMVA::MethodLikelihood( Data(), weightfile );
      break; 

   case (TMVA::Types::kPDERS):
      method = new TMVA::MethodPDERS( Data(), weightfile );
      break; 

   case (TMVA::Types::kHMatrix):
      method = new TMVA::MethodHMatrix( Data(), weightfile );
      break; 

   case (TMVA::Types::kFisher):
      method = new TMVA::MethodFisher( Data(), weightfile );
      break; 

   case (TMVA::Types::kCFMlpANN):
      method = new TMVA::MethodCFMlpANN( Data(), weightfile );
      break; 

   case (TMVA::Types::kTMlpANN):
      method = new TMVA::MethodTMlpANN( Data(), weightfile );
      break; 

   case (TMVA::Types::kBDT):
      method = new TMVA::MethodBDT( Data(), weightfile );
      break; 

   case (TMVA::Types::kMLP):
      method = new TMVA::MethodMLP( Data(), weightfile );
      break;

   case (TMVA::Types::kRuleFit):
      method = new TMVA::MethodRuleFit( Data(), weightfile );
      break; 

   case (TMVA::Types::kSVM):
      method = new TMVA::MethodSVM( Data(), weightfile );
      break; 

   case (TMVA::Types::kBayesClassifier):
      method = new TMVA::MethodBayesClassifier( Data(), weightfile );
      break; 

   default: 
      fLogger << kFATAL << "Classifier: " << methodType << " not implemented" << Endl;
      return 0;
   }  

   fLogger << kINFO << "Booked classifier: " << ((MethodBase*)method)->GetMethodName() << Endl;

   // read weight file
   ((MethodBase*)method)->ReadStateFromFile();

#ifdef TMVA_Reader_TestIO__
   // testing the read/write procedure
   std::ofstream tfile( weightfile+".control" );
   ((MethodBase*)method)->WriteStateToStream(tfile);
   tfile.close();
#endif

   return method;
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const std::vector<Float_t>& inputVec, TString methodName, Double_t aux )
{
   // Evaluate a vector<float> of input data for a given method
   // The parameter aux is obligatory for the cuts method where it represents the efficiency cutoff

   for (UInt_t ivar=0; ivar<inputVec.size(); ivar++) Data().GetEvent().SetVal( ivar, inputVec[ivar] );
 
   return EvaluateMVA( methodName, aux );
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const std::vector<Double_t>& inputVec, TString methodName, Double_t aux )
{
   // Evaluate a vector<double> of input data for a given method

   // The parameter aux is obligatory for the cuts method where it represents the efficiency cutoff

   for (UInt_t ivar=0; ivar<inputVec.size(); ivar++) Data().GetEvent().SetVal( ivar, (Float_t)inputVec[ivar] );

   return EvaluateMVA( methodName, aux );
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( TString methodName, Double_t aux )
{
   // evaluates MVA for given set of input variables
   IMethod* method = 0;

   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodName );
   if (it == fMethodMap.end()) {
      fLogger << kINFO << "<EvaluateMVA> unknown classifier in map; "
              << "you looked for \"" << methodName << "\" within available methods: " << Endl;
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) fLogger << " --> " << it->first << Endl;
      fLogger << "Check calling string" << kFATAL << Endl;
   }

   else method = it->second;

   return this->EvaluateMVA( (MethodBase*)method, aux );
}  

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( MethodBase* method, Double_t aux )
{
   // evaluates the MVA

   // the event is currently stored in the DataSet [Data().GetEvent()]
   method->GetVarTransform().GetEventRaw().CopyVarValues(Data().GetEvent());

   // NOTE: in likelihood the variable transformations are inserted by hand in GetMvaValue()
   // (to distinguish signal and background transformations), and hence should not be applied here
   if (method->GetMethodType() != Types::kLikelihood) 
      method->GetVarTransform().ApplyTransformation(Types::kSignal);

   // the aux value is only needed for MethodCuts: it sets the required signal efficiency 
   if (method->GetMethodType() == TMVA::Types::kCuts) 
      ((TMVA::MethodCuts*)method)->SetTestSignalEfficiency( aux );

   return method->GetMvaValue();
}
//_______________________________________________________________________
Double_t TMVA::Reader::GetProba( TString methodName,  Double_t ap_sig, Double_t mvaVal )
{
  // evaluates MVA for given set of input variables
   IMethod* method = 0;
   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodName );
   if (it == fMethodMap.end()) {
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) fLogger << "M" << it->first << Endl;
      fLogger << kFATAL << "<EvaluateMVA> unknown classifier in map: " << method << "; "
              << "you looked for " << methodName<< " while the available methods are : " << Endl;
   }
   else method = it->second;

   MethodBase* kl = (MethodBase*)method;
   if (mvaVal == -9999999) mvaVal = kl->GetMvaValue();

   return kl->GetProba( mvaVal, ap_sig );
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
