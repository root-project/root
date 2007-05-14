/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooClassFactory.cxx,v 1.2 2007/05/11 09:11:58 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooClassFactory is a clase like TTree::MakeClass() that generates
// skeleton code for RooAbsPdf and RooAbsReal functions given
// a list of input parameter names
//

#include "RooFit.h"

#include "RooClassFactory.h"
#include "RooClassFactory.h"
#include <fstream>
#include <vector>
#include <string>

ClassImp(RooClassFactory) 
;


RooClassFactory::RooClassFactory()
{
}


RooClassFactory::~RooClassFactory() 
{
}

Bool_t RooClassFactory::makePdf(const char* name, const char* argNames, Bool_t hasAnaInt, Bool_t hasIntGen) 
{
  return makeClass("RooAbsPdf",name,argNames,hasAnaInt,hasIntGen) ;
}

Bool_t RooClassFactory::makeFunction(const char* name, const char* argNames, Bool_t hasAnaInt) 
{
  return makeClass("RooAbsReal",name,argNames,hasAnaInt) ;
}

Bool_t RooClassFactory::makeClass(const char* baseName, const char* className, const char* argNames, 
					Bool_t hasAnaInt, Bool_t hasIntGen)
{
  // Check that arguments were given
  if (!baseName) {
    cout << "RooClassFactory::makeClass: ERROR: a base class name must be given" << endl ;
    return kTRUE ;
  }

  if (!className) {
    cout << "RooClassFactory::makeClass: ERROR: a class name must be given" << endl ;
    return kTRUE ;
  }

  if (!argNames) {
    cout << "RooClassFactory::makeClass: ERROR: A list of input argument names must be given" << endl ;
    return kTRUE ;
  }

  // Parse comma separated list of argument names into list of strings
  char* buf = new char[strlen(argNames)+1] ;
  strcpy(buf,argNames) ;
  char* token = strtok(buf,",") ;
  vector<string> alist ;
  while(token) {
    alist.push_back(token) ;
    token = strtok(0,",") ;
  }
  delete[] buf ;

  TString impFileName(className), hdrFileName(className) ;
  impFileName += ".cxx" ;
  hdrFileName += ".h" ;

  TString ifdefName(className) ;
  ifdefName.ToUpper() ;
  
  ofstream hf(hdrFileName) ;
  hf << "/*****************************************************************************" << endl
     << " * Project: RooFit                                                           *" << endl
     << " *                                                                           *" << endl
     << " * Copyright (c) 2000-2005, Regents of the University of California          *" << endl
     << " *                          and Stanford University. All rights reserved.    *" << endl
     << " *                                                                           *" << endl
     << " * Redistribution and use in source and binary forms,                        *" << endl
     << " * with or without modification, are permitted according to the terms        *" << endl
     << " * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *" << endl
     << " *****************************************************************************/" << endl
     << endl
     << "#ifndef " << ifdefName << endl
     << "#define " << ifdefName << endl
     << "" << endl     
     << "#include \"" << baseName << ".h\"" << endl
     << "#include \"RooRealProxy.h\"" << endl
     << "#include \"RooAbsReal.h\"" << endl
     << " " << endl
     << "class " << className << " : public " << baseName << " {" << endl
     << "public:" << endl
     << "  " << className << "(const char *name, const char *title," << endl ;
  
  // Insert list of input arguments
  unsigned int i ;
  for (i=0 ; i<alist.size() ; i++) { 
    hf << "	      RooAbsReal& _" ;
    hf << alist[i] ;
    if (i==alist.size()-1) {
      hf << ");" << endl ;
    } else {
      hf << "," << endl ;
    }    
  }
  
  hf << "  " << className << "(const " << className << "& other, const char* name=0) ;" << endl
     << "  virtual TObject* clone(const char* newname) const { return new " << className << "(*this,newname); }" << endl
     << "  inline virtual ~" << className << "() { }" << endl
     << endl ;

  if (hasAnaInt) {
    hf << "  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;" << endl
       << "  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;" << endl
       << "" << endl ;
  }

  if (hasIntGen) {
     hf << "  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;" << endl
	<< "  void initGenerator(Int_t code) {} ; // optional pre-generation initialization" << endl
	<< "  void generateEvent(Int_t code);" << endl
	<< endl ;
  }
       
  hf << "protected:" << endl
     << "" << endl ;

  // Insert list of input arguments
  for (i=0 ; i<alist.size() ; i++) { 
    hf << "  RooRealProxy " << alist[i] << " ;" << endl ;
  }
  
  hf << "  " << endl 
     << "  Double_t evaluate() const ;" << endl
     << "" << endl
     << "private:" << endl
     << "" << endl
     << "  ClassDef(" << className << ",0) // Your description goes here..." << endl
     << "};" << endl
     << " " << endl
     << "#endif" << endl ;


  ofstream cf(impFileName) ;

  cf << " /***************************************************************************** " << endl 
     << "  * Project: RooFit                                                           * " << endl 
     << "  *                                                                           * " << endl 
     << "  * Copyright (c) 2000-2005, Regents of the University of California          * " << endl 
     << "  *                          and Stanford University. All rights reserved.    * " << endl 
     << "  *                                                                           * " << endl 
     << "  * Redistribution and use in source and binary forms,                        * " << endl 
     << "  * with or without modification, are permitted according to the terms        * " << endl 
     << "  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             * " << endl 
     << "  *****************************************************************************/ " << endl 
     << endl 

     << " // -- CLASS DESCRIPTION [PDF] -- " << endl 
     << " // Your description goes here... " << endl 
     << endl 

     << " #include <iostream> " << endl 
     << endl 

     << " #include \"" << className << ".h\" " << endl 
     << " #include \"RooAbsReal.h\" " << endl 
     << endl 

     << " ClassImp(" << className << ") " << endl 
     << endl 

     << " " << className << "::" << className << "(const char *name, const char *title, " << endl ;

  // Insert list of proxy constructors
  for (i=0 ; i<alist.size() ; i++) { 
    cf << "                        RooAbsReal& _" << alist[i] ;
    if (i<alist.size()-1) {
      cf << "," ;
    } else {
      cf << ") :" ;
    }
    cf << endl ;
  }

  // Insert base class constructor
  cf << "   " << baseName << "(name,title), " << endl ;
  
  // Insert list of proxy constructors
  for (i=0 ; i<alist.size() ; i++) { 
    cf << "   " << alist[i] << "(\"" << alist[i] << "\",\"" << alist[i] << "\",this,_" << alist[i] << ")" ;
    if (i<alist.size()-1) {
      cf << "," ;
    }
    cf << endl ;
  }
  
  cf << " { " << endl 
     << " } " << endl 
     << endl 
     << endl 

     << " " << className << "::" << className << "(const " << className << "& other, const char* name) :  " << endl 
     << "   " << baseName << "(other,name), " << endl ;

  for (i=0 ; i<alist.size() ; i++) { 
    cf << "   " << alist[i] << "(\"" << alist[i] << "\",this,other." << alist[i] << ")" ;
    if (i<alist.size()-1) {
      cf << "," ;
    }
    cf << endl ;
  }

  cf << " { " << endl 
     << " } " << endl 
     << endl 
     << endl 
     << endl 

     << " Double_t " << className << "::evaluate() const " << endl 
     << " { " << endl 
     << "   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE " << endl 
     << "   return 1.0 ; " << endl
     << " } " << endl 
     << endl 
     << endl 
     << endl ;

  if (hasAnaInt) {
    cf << " Int_t " << className << "::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const  " << endl 
       << " { " << endl 
       << "   // LIST HERE OVER WHICH VARIABLES ANALYTICAL INTEGRATION IS SUPPORTED, " << endl 
       << "   // ASSIGN A NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS " << endl 
       << "   // THE EXAMPLE BELOW ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X" << endl 
       << "   // YOU CAN ALSO IMPLEMENT MORE THAN ONE ANALYTICAL INTEGRAL BY REPEATING THE matchArgs " << endl
       << "   // EXPRESSION MULTIPLE TIMES" << endl 
       << endl 
       << "   // if (matchArgs(allVars,analVars,x)) return 1 ; " << endl 
       << "   return 0 ; " << endl 
       << " } " << endl 
       << endl 
       << endl 
       << endl 

       << " Double_t " << className << "::analyticalIntegral(Int_t code, const char* rangeName) const  " << endl 
       << " { " << endl 
       << "   // RETURN ANALYTICAL INTEGRAL DEFINED BY RETURN CODE ASSIGNED BY getAnalyticalIntegral" << endl
       << "   // THE MEMBER FUNCTION x.min(rangeName) AND x.max(rangeName) WILL RETURN THE INTEGRATION" << endl
       << "   // BOUNDARIES FOR EACH OBSERVABLE x" << endl 
       << endl 
       << "   // assert(code==1) ; " << endl 
       << "   // return (x.max(rangeName)-x.min(rangeName)) ; " << endl 
       << "   return 0 ; " << endl
       << " } " << endl 
       << endl 
       << endl 
       << endl ;
  }
  
  if (hasIntGen) {
    cf << " Int_t " << className << "::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const " << endl 
       << " { " << endl 
       << "   // LIST HERE OVER WHICH VARIABLES INTERNAL GENERATION IS SUPPORTED, " << endl 
       << "   // ASSIGN A NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS " << endl 
       << "   // THE EXAMPLE BELOW ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X" << endl 
       << "   // YOU CAN ALSO IMPLEMENT MORE THAN ONE GENERATOR CONFIGURATION BY REPEATING THE matchArgs " << endl
       << "   // EXPRESSION MULTIPLE TIMES. IF THE FLAG staticInitOK IS TRUE THEN IT IS SAFE TO PRECALCULATE " << endl 
       << "   // INTERMEDIATE QUANTITIES IN initGenerator(), IF IT IS NOT SET THEN YOU SHOULD NOT ADVERTISE" << endl
       << "   // ANY GENERATOR METHOD THAT RELIES ON PRECALCULATIONS IN initGenerator()" << endl 
       << endl 
       << "   // if (matchArgs(directVars,generateVars,x)) return 1 ;   " << endl 
       << "   return 0 ; " << endl 
       << " } " << endl 
       << endl 
       << endl 
       << endl 

       << " void " << className << "::generateEvent(Int_t code) " << endl 
       << " { " << endl 
       << "   // GENERATE SET OF OBSERVABLES DEFINED BY RETURN CODE ASSIGNED BY getGenerator()" << endl
       << "   // RETURN THE GENERATED VALUES BY ASSIGNING THEM TO THE PROXY DATA MEMBERS THAT" << endl
       << "   // REPRESENT THE CHOSEN OBSERVABLES" << endl
       << endl 
       << "   // assert(code==1) ; " << endl 
       << "   // x = 0 ; " << endl 
       << "   return; " << endl 
       << " } " << endl 
       << endl 
       << endl 
       << endl ;
  }

  
  return kFALSE ;
}


