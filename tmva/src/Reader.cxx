// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Eckhard von Toerne, Jan Therhaag

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
 * Authors (alphabetical order):                                                  *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <peter.speckmayer@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer <Joerg.Stelzer@cern.ch>    - CERN, Switzerland              *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>          - U of Bonn, Germany        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
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

#include "TMVA/Reader.h"

#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TClass.h"
#include "TH1D.h"
#include "TKey.h"
#include "TVector.h"
#include "TXMLEngine.h"

#include <cstdlib>

#include <string>
#include <vector>
#include <fstream>

#include <iostream>
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#include "TMVA/Config.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/IMethod.h"
#include "TMVA/MethodCuts.h"
#include "TMVA/MethodCategory.h"
#include "TMVA/DataSetManager.h"

ClassImp(TMVA::Reader)

//_______________________________________________________________________
TMVA::Reader::Reader( const TString& theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSetManager( NULL ), // DSMTEST
     fDataSetInfo(),
     fVerbose( verbose ),
     fSilent ( kFALSE ),
     fColor  ( kFALSE ),
     fCalculateError(kFALSE),
     fMvaEventError( 0 ),
     fMvaEventErrorUpper( 0 ),
     fLogger ( 0 )
{
   // constructor
   fDataSetManager = new DataSetManager( fDataInputHandler ); 
   fDataSetManager->AddDataSetInfo(fDataSetInfo); 
   fLogger = new MsgLogger(this);
   SetConfigName( GetName() );
   DeclareOptions();
   ParseOptions();

   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( std::vector<TString>& inputVars, const TString& theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSetManager( NULL ), // DSMTEST
     fDataSetInfo(),
     fVerbose( verbose ),
     fSilent ( kFALSE ),
     fColor  ( kFALSE ),
     fCalculateError(kFALSE),
     fMvaEventError( 0 ),
     fMvaEventErrorUpper( 0 ),   //zjh
     fLogger ( 0 )
{
   // constructor

   fDataSetManager = new DataSetManager( fDataInputHandler ); 
   fDataSetManager->AddDataSetInfo(fDataSetInfo); 
   fLogger = new MsgLogger(this);
   SetConfigName( GetName() );
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables (vector)
   //            verbose flag
   for (std::vector<TString>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
      DataInfo().AddVariable( *ivar );

   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( std::vector<std::string>& inputVars, const TString& theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSetManager( NULL ), // DSMTEST
     fDataSetInfo(),
     fVerbose( verbose ),
     fSilent ( kFALSE ),
     fColor  ( kFALSE ),
     fCalculateError(kFALSE),
     fMvaEventError( 0 ),
     fMvaEventErrorUpper( 0 ),
     fLogger ( 0 )
{
   // constructor
   fDataSetManager = new DataSetManager( fDataInputHandler ); 
   fDataSetManager->AddDataSetInfo(fDataSetInfo); 
   fLogger = new MsgLogger(this);
   SetConfigName( GetName() );
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables (vector)
   //            verbose flag
   for (std::vector<std::string>::iterator ivar = inputVars.begin(); ivar != inputVars.end(); ivar++) 
      DataInfo().AddVariable( ivar->c_str() );

   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const std::string& varNames, const TString& theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSetManager( NULL ), // DSMTEST
     fDataSetInfo(),
     fVerbose( verbose ),
     fSilent ( kFALSE ),
     fColor  ( kFALSE ),
     fCalculateError(kFALSE),
     fMvaEventError( 0 ),
     fMvaEventErrorUpper( 0 ),
     fLogger ( 0 )
{
   // constructor
   fDataSetManager = new DataSetManager( fDataInputHandler ); 
   fDataSetManager->AddDataSetInfo(fDataSetInfo); 
   fLogger = new MsgLogger(this);
   SetConfigName( GetName() );
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables given in form: "name1:name2:name3"
   //            verbose flag
   DecodeVarNames(varNames);
   Init();
}

//_______________________________________________________________________
TMVA::Reader::Reader( const TString& varNames, const TString& theOption, Bool_t verbose )
   : Configurable( theOption ),
     fDataSetManager( NULL ), // DSMTEST
     fDataSetInfo(),
     fVerbose( verbose ),
     fSilent ( kFALSE ),
     fColor  ( kFALSE ),
     fCalculateError(kFALSE),
     fMvaEventError( 0 ),
     fMvaEventErrorUpper( 0 ),
     fLogger ( 0 )
{
   // constructor
   fDataSetManager = new DataSetManager( fDataInputHandler ); 
   fDataSetManager->AddDataSetInfo(fDataSetInfo); 
   fLogger = new MsgLogger(this);
   SetConfigName( GetName() );
   DeclareOptions();
   ParseOptions();

   // arguments: names of input variables given in form: "name1:name2:name3"
   //            verbose flag
   DecodeVarNames(varNames);
   Init();
}

//_______________________________________________________________________
void TMVA::Reader::DeclareOptions()
{
   // declaration of configuration options
   if (gTools().CheckForSilentOption( GetOptions() )) Log().InhibitOutput(); // make sure is silent if wanted to

   DeclareOptionRef( fVerbose,        "V",      "Verbose flag" );
   DeclareOptionRef( fColor,          "Color",  "Color flag (default True)" );
   DeclareOptionRef( fSilent,         "Silent", "Boolean silent flag (default False)" );
   DeclareOptionRef( fCalculateError, "Error",  "Calculates errors (default False)" );
}

//_______________________________________________________________________
TMVA::Reader::~Reader( void )
{
   // destructor

   delete fDataSetManager; // DSMTEST

   delete fLogger;
}

//_______________________________________________________________________
void TMVA::Reader::Init( void )
{
   // default initialisation (no member variables)
   // default initialisation (no member variables)
   if (Verbose()) fLogger->SetMinType( kVERBOSE );

   gConfig().SetUseColor( fColor );
   gConfig().SetSilent  ( fSilent );
}

//_______________________________________________________________________
void TMVA::Reader::AddVariable( const TString& expression, Float_t* datalink )
{
   // Add a float variable or expression to the reader
   DataInfo().AddVariable( expression, "", "", 0, 0, 'F', kFALSE ,(void*)datalink ); // <= should this be F or rather T?
}

//_______________________________________________________________________
void TMVA::Reader::AddVariable( const TString& expression, Int_t* datalink )
{
   Log() << kFATAL << "Reader::AddVariable( const TString& expression, Int_t* datalink ), this function is deprecated, please provide all variables to the reader as floats" << Endl;
   // Add an integer variable or expression to the reader
   Log() << kFATAL << "Reader::AddVariable( const TString& expression, Int_t* datalink ), this function is deprecated, please provide all variables to the reader as floats" << Endl;
   DataInfo().AddVariable(expression, "", "", 0, 0, 'I', kFALSE, (void*)datalink ); // <= should this be F or rather T?
}

//_______________________________________________________________________
void TMVA::Reader::AddSpectator( const TString& expression, Float_t* datalink )
{
   // Add a float spectator or expression to the reader
   DataInfo().AddSpectator( expression, "", "", 0, 0, 'F', kFALSE ,(void*)datalink );
}

//_______________________________________________________________________
void TMVA::Reader::AddSpectator( const TString& expression, Int_t* datalink )
{
   // Add an integer spectator or expression to the reader
   DataInfo().AddSpectator(expression, "", "", 0, 0, 'I', kFALSE, (void*)datalink );
}

//_______________________________________________________________________
TString TMVA::Reader::GetMethodTypeFromFile( const TString& filename ) 
{
   // read the method type from the file

   ifstream fin( filename );
   if (!fin.good()) { // file not found --> Error
      Log() << kFATAL << "<BookMVA> fatal error: "
            << "unable to open input weight file: " << filename << Endl;
   }

   TString fullMethodName("");
   if (filename.EndsWith(".xml")) {
      fin.close();
      void* doc      = gTools().xmlengine().ParseFile(filename);// the default buffer size in TXMLEngine::ParseFile is 100k. Starting with ROOT 5.29 one can set the buffer size, see: http://savannah.cern.ch/bugs/?78864. This might be necessary for large XML files
      void* rootnode = gTools().xmlengine().DocGetRootElement(doc); // node "MethodSetup"
      gTools().ReadAttr(rootnode, "Method", fullMethodName);
      gTools().xmlengine().FreeDoc(doc);
   } 
   else {
      char buf[512];
      fin.getline(buf,512);
      while (!TString(buf).BeginsWith("Method")) fin.getline(buf,512);
      fullMethodName = TString(buf);
      fin.close();
   }
   TString methodType = fullMethodName(0,fullMethodName.Index("::"));
   if (methodType.Contains(" ")) methodType = methodType(methodType.Last(' ')+1,methodType.Length());
   return methodType;
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( const TString& methodTag, const TString& weightfile )
{
   // read method name from weight file

   // assert non-existence
   if (fMethodMap.find( methodTag ) != fMethodMap.end())
      Log() << kFATAL << "<BookMVA> method tag \"" << methodTag << "\" already exists!" << Endl;

   TString methodType(GetMethodTypeFromFile(weightfile));

   Log() << kINFO << "Booking \"" << methodTag << "\" of type \"" << methodType << "\" from " << weightfile << "." << Endl;

   MethodBase* method = dynamic_cast<MethodBase*>(this->BookMVA( Types::Instance().GetMethodType(methodType),
                                                                 weightfile ) );
   if( method && method->GetMethodType() == Types::kCategory ){
      MethodCategory *methCat = (dynamic_cast<MethodCategory*>(method));
      if( !methCat )
         Log() << kFATAL << "Method with type kCategory cannot be casted to MethodCategory. /Reader" << Endl;
      methCat->fDataSetManager = fDataSetManager;
   }

   return fMethodMap[methodTag] = method;
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( TMVA::Types::EMVA methodType, const TString& weightfile )
{
   // books MVA method from weightfile
   IMethod* im = ClassifierFactory::Instance().Create(std::string(Types::Instance().GetMethodName( methodType )),
                                                      DataInfo(), weightfile );

   MethodBase *method = (dynamic_cast<MethodBase*>(im));

   if (method==0) return im;

   if( method->GetMethodType() == Types::kCategory ){
      MethodCategory *methCat = (dynamic_cast<MethodCategory*>(method));
      if( !methCat )
         Log() << kERROR << "Method with type kCategory cannot be casted to MethodCategory. /Reader" << Endl;
      methCat->fDataSetManager = fDataSetManager;
   }

   method->SetupMethod();

   // when reading older weight files, they could include options
   // that are not supported any longer
   method->DeclareCompatibilityOptions();

   // read weight file
   method->ReadStateFromFile();

   // check for unused options
   method->CheckSetup();

   Log() << kINFO << "Booked classifier \"" << method->GetMethodName()
         << "\" of type: \"" << method->GetMethodTypeName() << "\"" << Endl;

   return method;
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::BookMVA( TMVA::Types::EMVA methodType, const char* xmlstr )
{

#if (ROOT_SVN_REVISION >= 32259) && (ROOT_VERSION_CODE >= 334336) // 5.26/00

   // books MVA method from weightfile
   IMethod* im = ClassifierFactory::Instance().Create(std::string(Types::Instance().GetMethodName( methodType )),
                                                      DataInfo(), "" );

   MethodBase *method = (dynamic_cast<MethodBase*>(im));

   if(!method) return 0;

   if( method->GetMethodType() == Types::kCategory ){ 
      MethodCategory *methCat = (dynamic_cast<MethodCategory*>(method)); 
      if( !methCat ) 
         Log() << kFATAL << "Method with type kCategory cannot be casted to MethodCategory. /Reader" << Endl; 
      methCat->fDataSetManager = fDataSetManager; 
   }

   method->SetupMethod();

   // when reading older weight files, they could include options
   // that are not supported any longer
   method->DeclareCompatibilityOptions();

   // read weight file
   method->ReadStateFromXMLString( xmlstr );

   // check for unused options
   method->CheckSetup();

   Log() << kINFO << "Booked classifier \"" << method->GetMethodName()
         << "\" of type: \"" << method->GetMethodTypeName() << "\"" << Endl;

   return method;
#else
   Log() << kFATAL << "Method Reader::BookMVA(TMVA::Types::EMVA methodType = " << methodType 
         << ", const char* xmlstr = " << xmlstr 
         << " ) is not available for ROOT versions prior to 5.26/00." << Endl;
   return 0;
#endif
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const std::vector<Float_t>& inputVec, const TString& methodTag, Double_t aux )
{
   // Evaluate a vector<float> of input data for a given method
   // The parameter aux is obligatory for the cuts method where it represents the efficiency cutoff

   // create a temporary event from the vector.
   IMethod* imeth = FindMVA( methodTag );
   MethodBase* meth = dynamic_cast<TMVA::MethodBase*>(imeth);
   if(meth==0) return 0;

//   Event* tmpEvent=new Event(inputVec, 2); // ToDo resolve magic 2 issue
   Event* tmpEvent=new Event(inputVec, DataInfo().GetNVariables()); // is this the solution?

   if (meth->GetMethodType() == TMVA::Types::kCuts) {
      TMVA::MethodCuts* mc = dynamic_cast<TMVA::MethodCuts*>(meth);
      if(mc)
         mc->SetTestSignalEfficiency( aux );
   }
   Double_t val = meth->GetMvaValue( tmpEvent, (fCalculateError?&fMvaEventError:0));
   delete tmpEvent;
   return val;
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const std::vector<Double_t>& inputVec, const TString& methodTag, Double_t aux )
{
   // Evaluate a vector<double> of input data for a given method
   // The parameter aux is obligatory for the cuts method where it represents the efficiency cutoff

   // performs a copy to float values which are internally used by all methods
   if(fTmpEvalVec.size() != inputVec.size())
      fTmpEvalVec.resize(inputVec.size());

   for (UInt_t idx=0; idx!=inputVec.size(); idx++ )
      fTmpEvalVec[idx]=inputVec[idx];

   return EvaluateMVA( fTmpEvalVec, methodTag, aux );
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( const TString& methodTag, Double_t aux )
{
   // evaluates MVA for given set of input variables
   IMethod* method = 0;

   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodTag );
   if (it == fMethodMap.end()) {
      Log() << kINFO << "<EvaluateMVA> unknown classifier in map; "
              << "you looked for \"" << methodTag << "\" within available methods: " << Endl;
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) Log() << " --> " << it->first << Endl;
      Log() << "Check calling string" << kFATAL << Endl;
   }

   else method = it->second;

   MethodBase * kl = dynamic_cast<TMVA::MethodBase*>(method);

   if(kl==0)
      Log() << kFATAL << methodTag << " is not a method" << Endl;

   return this->EvaluateMVA( kl, aux );
}

//_______________________________________________________________________
Double_t TMVA::Reader::EvaluateMVA( MethodBase* method, Double_t aux )
{
   // evaluates the MVA

   // the aux value is only needed for MethodCuts: it sets the
   // required signal efficiency
   if (method->GetMethodType() == TMVA::Types::kCuts) {
      TMVA::MethodCuts* mc = dynamic_cast<TMVA::MethodCuts*>(method);
      if(mc)
         mc->SetTestSignalEfficiency( aux );
   }

   return method->GetMvaValue( (fCalculateError?&fMvaEventError:0),
                               (fCalculateError?&fMvaEventErrorUpper:0) );
}

//_______________________________________________________________________
const std::vector< Float_t >& TMVA::Reader::EvaluateRegression( const TString& methodTag, Double_t aux )
{
   // evaluates MVA for given set of input variables
   IMethod* method = 0;

   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodTag );
   if (it == fMethodMap.end()) {
      Log() << kINFO << "<EvaluateMVA> unknown method in map; "
              << "you looked for \"" << methodTag << "\" within available methods: " << Endl;
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) Log() << " --> " << it->first << Endl;
      Log() << "Check calling string" << kFATAL << Endl;
   }
   else method = it->second;

   MethodBase * kl = dynamic_cast<TMVA::MethodBase*>(method);

   if(kl==0)
      Log() << kFATAL << methodTag << " is not a method" << Endl;

   return this->EvaluateRegression( kl, aux );
}

//_______________________________________________________________________
const std::vector< Float_t >& TMVA::Reader::EvaluateRegression( MethodBase* method, Double_t /*aux*/ )
{
   // evaluates the regression MVA
   return method->GetRegressionValues();
}


//_______________________________________________________________________
Float_t TMVA::Reader::EvaluateRegression( UInt_t tgtNumber, const TString& methodTag, Double_t aux )
{ 
   // evaluates the regression MVA
   try {
      return EvaluateRegression(methodTag, aux).at(tgtNumber); 
   }
   catch (std::out_of_range e) {
      Log() << kWARNING << "Regression could not be evaluated for target-number " << tgtNumber << Endl;
      return 0;
   }
}



//_______________________________________________________________________
const std::vector< Float_t >& TMVA::Reader::EvaluateMulticlass( const TString& methodTag, Double_t aux )
{
   // evaluates MVA for given set of input variables
   IMethod* method = 0;

   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodTag );
   if (it == fMethodMap.end()) {
      Log() << kINFO << "<EvaluateMVA> unknown method in map; "
              << "you looked for \"" << methodTag << "\" within available methods: " << Endl;
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) Log() << " --> " << it->first << Endl;
      Log() << "Check calling string" << kFATAL << Endl;
   }
   else method = it->second;

   MethodBase * kl = dynamic_cast<TMVA::MethodBase*>(method);

   if(kl==0)
      Log() << kFATAL << methodTag << " is not a method" << Endl;

   return this->EvaluateMulticlass( kl, aux );
}

//_______________________________________________________________________
const std::vector< Float_t >& TMVA::Reader::EvaluateMulticlass( MethodBase* method, Double_t /*aux*/ )
{
   // evaluates the multiclass MVA
   return method->GetMulticlassValues();
}


//_______________________________________________________________________
Float_t TMVA::Reader::EvaluateMulticlass( UInt_t clsNumber, const TString& methodTag, Double_t aux )
{ 
   // evaluates the multiclass MVA
   try {
      return EvaluateMulticlass(methodTag, aux).at(clsNumber); 
   }
   catch (std::out_of_range e) {
      Log() << kWARNING << "Multiclass could not be evaluated for class-number " << clsNumber << Endl;
      return 0;
   }
}


//_______________________________________________________________________
TMVA::IMethod* TMVA::Reader::FindMVA( const TString& methodTag )
{
   // return pointer to method with tag "methodTag"
   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodTag );
   if (it != fMethodMap.end()) return it->second;
   Log() << kERROR << "Method " << methodTag << " not found!" << Endl;
   return 0;
}

//_______________________________________________________________________
TMVA::MethodCuts* TMVA::Reader::FindCutsMVA( const TString& methodTag )
{
   // special function for Cuts to avoid dynamic_casts in ROOT macros,
   // which are not properly handled by CINT
   return dynamic_cast<MethodCuts*>(FindMVA(methodTag));
}

//_______________________________________________________________________
Double_t TMVA::Reader::GetProba( const TString& methodTag,  Double_t ap_sig, Double_t mvaVal )
{
   // evaluates probability of MVA for given set of input variables
   IMethod* method = 0;
   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodTag );
   if (it == fMethodMap.end()) {
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) Log() << "M" << it->first << Endl;
      Log() << kFATAL << "<EvaluateMVA> unknown classifier in map: " << method << "; "
              << "you looked for " << methodTag<< " while the available methods are : " << Endl;
   }
   else method = it->second;

   MethodBase* kl = dynamic_cast<MethodBase*>(method);
   if(kl==0) return -1;

   if (mvaVal == -9999999) mvaVal = kl->GetMvaValue();

   return kl->GetProba( mvaVal, ap_sig );
}

//_______________________________________________________________________
Double_t TMVA::Reader::GetRarity( const TString& methodTag, Double_t mvaVal )
{
   // evaluates the MVA's rarity
   IMethod* method = 0;
   std::map<TString, IMethod*>::iterator it = fMethodMap.find( methodTag );
   if (it == fMethodMap.end()) {
      for (it = fMethodMap.begin(); it!=fMethodMap.end(); it++) Log() << "M" << it->first << Endl;
      Log() << kFATAL << "<EvaluateMVA> unknown classifier in map: \"" << method << "\"; "
              << "you looked for \"" << methodTag<< "\" while the available methods are : " << Endl;
   }
   else method = it->second;

   MethodBase* kl = dynamic_cast<MethodBase*>(method);
   if(kl==0) return -1;

   if (mvaVal == -9999999) mvaVal = kl->GetMvaValue();

   return kl->GetRarity( mvaVal );
}

// ---------------------------------------------------------------------------------------
// ----- methods related to the decoding of the input variable names ---------------------
// ---------------------------------------------------------------------------------------

//_______________________________________________________________________
void TMVA::Reader::DecodeVarNames( const std::string& varNames )
{
   // decodes "name1:name2:..." form
   size_t ipos = 0, f = 0;
   while (f != varNames.length()) {
      f = varNames.find( ':', ipos );
      if (f > varNames.length()) f = varNames.length();
      std::string subs = varNames.substr( ipos, f-ipos ); ipos = f+1;
      DataInfo().AddVariable( subs.c_str() );
   }
}

//_______________________________________________________________________
void TMVA::Reader::DecodeVarNames( const TString& varNames )
{
   // decodes "name1:name2:..." form

   TString format;
   Int_t   n = varNames.Length();
   TString format_obj;

   for (int i=0; i< n+1 ; i++) {
      format.Append(varNames(i));
      if (varNames(i) == ':' || i == n) {
         format.Chop();
         format_obj = format;
         format_obj.ReplaceAll("@","");
         DataInfo().AddVariable( format_obj );
         format.Resize(0);
      }
   }
}
