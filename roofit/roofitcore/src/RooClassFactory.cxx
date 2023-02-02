/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

/**
\file RooClassFactory.cxx
\class RooClassFactory
\ingroup Roofitcore

RooClassFactory is a clase like TTree::MakeClass() that generates
skeleton code for RooAbsPdf and RooAbsReal functions given
a list of input parameter names. The factory can also compile
the generated code on the fly, and on request also immediate
instantiate objects.
**/

#include "RooClassFactory.h"

#include "TClass.h"
#include "RooFactoryWSTool.h"
#include "RooErrorHandler.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "TInterpreter.h"
#include "RooWorkspace.h"
#include "RooGlobalFunc.h"
#include "RooAbsPdf.h"
#include "strlcpy.h"
#include <fstream>

using namespace std;

namespace {

class ClassFacIFace : public RooFactoryWSTool::IFace {
public:
  std::string create(RooFactoryWSTool& ft, const char* typeName, const char* instanceName, std::vector<std::string> args) override ;
} ;

static Int_t init();

Int_t dummy = init();

static Int_t init()
{
  RooFactoryWSTool::IFace* iface = new ClassFacIFace ;
  RooFactoryWSTool::registerSpecial("CEXPR",iface) ;
  RooFactoryWSTool::registerSpecial("cexpr",iface) ;
  (void)dummy;
  return 0 ;
}

}


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooClassFactory::RooClassFactory()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooClassFactory::~RooClassFactory()
{
}



////////////////////////////////////////////////////////////////////////////////

bool RooClassFactory::makeAndCompilePdf(const char* name, const char* expression, const RooArgList& vars, const char* intExpression)
{
  string realArgNames,catArgNames ;
  for (RooAbsArg * arg : vars) {
    if (dynamic_cast<RooAbsReal*>(arg)) {
      if (realArgNames.size()>0) realArgNames += "," ;
      realArgNames += arg->GetName() ;
    } else if (dynamic_cast<RooAbsCategory*>(arg)) {
      if (catArgNames.size()>0) catArgNames += "," ;
      catArgNames += arg->GetName() ;
    } else {
      oocoutE(nullptr,InputArguments) << "RooClassFactory::makeAndCompilePdf ERROR input argument " << arg->GetName()
                     << " is neither RooAbsReal nor RooAbsCategory and is ignored" << endl ;
    }
  }

  bool ret = makePdf(name,realArgNames.c_str(),catArgNames.c_str(),expression,intExpression?true:false,false,intExpression) ;
  if (ret) {
    return ret ;
  }

  TInterpreter::EErrorCode ecode;
  gInterpreter->ProcessLineSynch(Form(".L %s.cxx+",name),&ecode) ;
  return (ecode!=TInterpreter::kNoError) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Write, compile and load code for a RooAbsReal implementation with
/// class name 'name', taking all elements of 'vars' as constructor
/// arguments. The initial value expression is taken to be
/// 'expression' which can be any one-line C++ expression in terms of
/// variables that occur in 'vars'. You can add optional expressions
/// for analytical integrals to be advertised by your class in the
/// syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral.

bool RooClassFactory::makeAndCompileFunction(const char* name, const char* expression, const RooArgList& vars, const char* intExpression)
{
  string realArgNames,catArgNames ;
  for (RooAbsArg * arg : vars) {
    if (dynamic_cast<RooAbsReal*>(arg)) {
      if (realArgNames.size()>0) realArgNames += "," ;
      realArgNames += arg->GetName() ;
    } else if (dynamic_cast<RooAbsCategory*>(arg)) {
      if (catArgNames.size()>0) catArgNames += "," ;
      catArgNames += arg->GetName() ;
    } else {
      oocoutE(nullptr,InputArguments) << "RooClassFactory::makeAndCompileFunction ERROR input argument " << arg->GetName()
                   << " is neither RooAbsReal nor RooAbsCategory and is ignored" << endl ;
    }
  }

  bool ret = makeFunction(name,realArgNames.c_str(),catArgNames.c_str(),expression,intExpression?true:false,intExpression) ;
  if (ret) {
    return ret ;
  }

  TInterpreter::EErrorCode ecode;
  gInterpreter->ProcessLineSynch(Form(".L %s.cxx+",name),&ecode) ;
  return (ecode!=TInterpreter::kNoError) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Write, compile and load code and instantiate object for a
/// RooAbsReal implementation with class name 'name', taking all
/// elements of 'vars' as constructor arguments. The initial value
/// expression is taken to be 'expression' which can be any one-line
/// C++ expression in terms of variables that occur in 'vars'.
///
/// The returned object is an instance of the object you just defined
/// connected to the variables listed in 'vars'. The name of the
/// object is 'name', its class name Roo<name>Class.
///
/// This function is an effective compiled replacement of RooFormulaVar
///
/// You can add optional expressions for analytical integrals to be
/// advertised by your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral.

RooAbsReal* RooClassFactory::makeFunctionInstance(const char* name, const char* expression, const RooArgList& vars, const char* intExpression)
{
  // Construct unique class name for this function expression
  string tmpName(name) ;
  tmpName[0] = toupper(tmpName[0]) ;
  string className = Form("Roo%sFunc",tmpName.c_str()) ;

  return makeFunctionInstance(className.c_str(),name,expression,vars,intExpression) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Write, compile and load code and instantiate object for a
/// RooAbsReal implementation with class name 'name', taking all
/// elements of 'vars' as constructor arguments. The initial value
/// expression is taken to be 'expression' which can be any one-line
/// C++ expression in terms of variables that occur in 'vars'.
///
/// The returned object is an instance of the object you just defined
/// connected to the variables listed in 'vars'. The name of the
/// object is 'name', its class name Roo<name>Class.
///
/// This function is an effective compiled replacement of RooFormulaVar
///
/// You can add optional expressions for analytical integrals to be
/// advertised by your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral.

RooAbsReal* RooClassFactory::makeFunctionInstance(const char* className, const char* name, const char* expression, const RooArgList& vars, const char* intExpression)
{
  // Use class factory to compile and link specialized function
  bool error = makeAndCompileFunction(className,expression,vars,intExpression) ;

  // Check that class was created OK
  if (error) {
    RooErrorHandler::softAbort() ;
  }

  // Create interpreter line that instantiates specialized object
  std::string line = std::string("new ") + className + "(\"" + name + "\",\"" + name + "\"";

  // Make list of pointer values (represented in hex ascii) to be passed to cint
  // Note that the order of passing arguments must match the convention in which
  // the class code is generated: first all reals, then all categories

  std::string argList ;
  // First pass the RooAbsReal arguments in the list order
  for(RooAbsArg * var : vars) {
    if (dynamic_cast<RooAbsReal*>(var)) {
      argList += Form(",*((RooAbsReal*)0x%zx)",(size_t)var) ;
    }
  }
  // Next pass the RooAbsCategory arguments in the list order
  for(RooAbsArg * var : vars) {
    if (dynamic_cast<RooAbsCategory*>(var)) {
      argList += Form(",*((RooAbsCategory*)0x%zx)",(size_t)var) ;
    }
  }

  line += argList + ") ;" ;

  // Let interpreter instantiate specialized formula
  return (RooAbsReal*) gInterpreter->ProcessLineSynch(line.c_str()) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Write, compile and load code and instantiate object for a
/// RooAbsPdf implementation with class name 'name', taking all
/// elements of 'vars' as constructor arguments. The initial value
/// expression is taken to be 'expression' which can be any one-line
/// C++ expression in terms of variables that occur in 'vars'.
///
/// The returned object is an instance of the object you just defined
/// connected to the variables listed in 'vars'. The name of the
/// object is 'name', its class name Roo<name>Class.
///
/// This function is an effective compiled replacement of RooGenericPdf
///
/// You can add optional expressions for analytical integrals to be
/// advertised by your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral.

RooAbsPdf* RooClassFactory::makePdfInstance(const char* name, const char* expression,
                   const RooArgList& vars, const char* intExpression)
{
  // Construct unique class name for this function expression
  string tmpName(name) ;
  tmpName[0] = toupper(tmpName[0]) ;
  string className = Form("Roo%sPdf",tmpName.c_str()) ;

  return makePdfInstance(className.c_str(),name,expression,vars,intExpression) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Write, compile and load code and instantiate object for a
/// RooAbsPdf implementation with class name 'name', taking all
/// elements of 'vars' as constructor arguments. The initial value
/// expression is taken to be 'expression' which can be any one-line
/// C++ expression in terms of variables that occur in 'vars'.
///
/// The returned object is an instance of the object you just defined
/// connected to the variables listed in 'vars'. The name of the
/// object is 'name', its class name Roo<name>Class.
///
/// This function is an effective compiled replacement of RooGenericPdf
///
/// You can add optional expressions for analytical integrals to be
/// advertised by your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral.

RooAbsPdf* RooClassFactory::makePdfInstance(const char* className, const char* name, const char* expression,
                   const RooArgList& vars, const char* intExpression)
{
  // Use class factory to compile and link specialized function
  bool error = makeAndCompilePdf(className,expression,vars,intExpression) ;

  // Check that class was created OK
  if (error) {
    RooErrorHandler::softAbort() ;
  }

  // Create interpreter line that instantiates specialized object
  std::string line = std::string("new ") + className + "(\"" + name + "\",\"" + name + "\"";

  // Make list of pointer values (represented in hex ascii) to be passed to cint
  // Note that the order of passing arguments must match the convention in which
  // the class code is generated: first all reals, then all categories

  std::string argList ;
  // First pass the RooAbsReal arguments in the list order
  for (RooAbsArg * var : vars) {
    if (dynamic_cast<RooAbsReal*>(var)) {
      argList += Form(",*((RooAbsReal*)0x%zx)",(size_t)var) ;
    }
  }
  // Next pass the RooAbsCategory arguments in the list order
  for (RooAbsArg * var : vars) {
    if (dynamic_cast<RooAbsCategory*>(var)) {
      argList += Form(",*((RooAbsCategory*)0x%zx)",(size_t)var) ;
    }
  }

  line += argList + ") ;" ;

  // Let interpreter instantiate specialized formula
  return (RooAbsPdf*) gInterpreter->ProcessLineSynch(line.c_str()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Write code for a RooAbsPdf implementation with class name 'name',
/// taking RooAbsReal arguments with names listed in argNames and
/// RooAbsCategory arguments with names listed in catArgNames as
/// constructor arguments (use a comma separated list for multiple
/// arguments). The initial value expression is taken to be
/// 'expression' which can be any one-line C++ expression in terms of
/// variables that occur in 'vars'. Skeleton code for handling of
/// analytical integrals is added if hasAnaInt is true. You can add
/// optional expressions for analytical integrals to be advertised by
/// your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral. Skeleton code for internal event generation is added
/// if hasIntGen is true
///

bool RooClassFactory::makePdf(const char* name, const char* argNames, const char* catArgNames, const char* expression,
            bool hasAnaInt, bool hasIntGen, const char* intExpression)
{
  return makeClass("RooAbsPdf",name,argNames,catArgNames,expression,hasAnaInt,hasIntGen,intExpression) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Write code for a RooAbsReal implementation with class name 'name',
/// taking RooAbsReal arguments with names listed in argNames and
/// RooAbsCategory arguments with names listed in catArgNames as
/// constructor arguments (use a comma separated list for multiple
/// arguments). The initial value expression is taken to be
/// 'expression' which can be any one-line C++ expression in terms of
/// variables that occur in 'vars'. Skeleton code for handling of
/// analytical integrals is added if hasAnaInt is true. You can add
/// optional expressions for analytical integrals to be advertised by
/// your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral.

bool RooClassFactory::makeFunction(const char* name, const char* argNames, const char* catArgNames, const char* expression, bool hasAnaInt, const char* intExpression)
{
  return makeClass("RooAbsReal",name,argNames,catArgNames,expression,hasAnaInt,false,intExpression) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Write code for a 'baseName' implementation with class name 'className',
/// taking RooAbsReal arguments with names listed in argNames and
/// RooAbsCategory arguments with names listed in catArgNames as
/// constructor arguments (use a comma separated list for multiple
/// arguments). The initial value expression is taken to be
/// 'expression' which can be any one-line C++ expression in terms of
/// variables that occur in 'vars'. Skeleton code for handling of
/// analytical integrals is added if hasAnaInt is true. You can add
/// optional expressions for analytical integrals to be advertised by
/// your class in the syntax
/// "<intObsName>:<CPPAnaIntExpression>;<intObsName,intObsName>:<CPPAnaIntExpression>"
/// where "<intObsName>" a name of the observable integrated over and
/// "<CPPAnaIntExpression>" is the C++ expression that calculates that
/// integral. Skeleton code for internal event generation is added
/// if hasIntGen is true
///

bool RooClassFactory::makeClass(const char* baseName, const char* className, const char* realArgNames, const char* catArgNames,
              const char* expression,  bool hasAnaInt, bool hasIntGen, const char* intExpression)
{
  // Check that arguments were given
  if (!baseName) {
    oocoutE(nullptr,InputArguments) << "RooClassFactory::makeClass: ERROR: a base class name must be given" << endl ;
    return true ;
  }

  if (!className) {
    oocoutE(nullptr,InputArguments) << "RooClassFactory::makeClass: ERROR: a class name must be given" << endl ;
    return true ;
  }

  if ((!realArgNames || !*realArgNames) && (!catArgNames || !*catArgNames)) {
    oocoutE(nullptr,InputArguments) << "RooClassFactory::makeClass: ERROR: A list of input argument names must be given" << endl ;
    return true ;
  }

  if (intExpression && !hasAnaInt) {
    oocoutE(nullptr,InputArguments) << "RooClassFactory::makeClass: ERROR no analytical integration code requestion, but expression for analytical integral provided" << endl ;
    return true ;
  }

  // Parse comma separated list of argument names into list of strings
  vector<string> alist ;
  vector<bool> isCat ;

  if (realArgNames && *realArgNames) {
    const size_t bufSize = strlen(realArgNames)+1;
    std::vector<char> buf(bufSize);
    strlcpy(buf.data(),realArgNames,bufSize) ;
    char* token = strtok(buf.data(),",") ;
    while(token) {
      alist.push_back(token) ;
      isCat.push_back(false) ;
      token = strtok(0,",") ;
    }
  }
  if (catArgNames && *catArgNames) {
    const size_t bufSize = strlen(catArgNames)+1;
    std::vector<char> buf(bufSize);
    strlcpy(buf.data(),catArgNames,bufSize) ;
    char* token = strtok(buf.data(),",") ;
    while(token) {
      alist.push_back(token) ;
      isCat.push_back(true) ;
      token = strtok(0,",") ;
    }
  }

  TString impFileName(className), hdrFileName(className) ;
  impFileName += ".cxx" ;
  hdrFileName += ".h" ;

  TString ifdefName(className) ;
  ifdefName.ToUpper() ;

  ofstream hf(hdrFileName) ;
  hf << "/*****************************************************************************" << endl
     << " * Project: RooFit                                                           *" << endl
     << " *                                                                           *" << endl
     << "  * This code was autogenerated by RooClassFactory                            * " << endl
     << " *****************************************************************************/" << endl
     << endl
     << "#ifndef " << ifdefName << endl
     << "#define " << ifdefName << endl
     << "" << endl
     << "#include \"" << baseName << ".h\"" << endl
     << "#include \"RooRealProxy.h\"" << endl
     << "#include \"RooCategoryProxy.h\"" << endl
     << "#include \"RooAbsReal.h\"" << endl
     << "#include \"RooAbsCategory.h\"" << endl
     << " " << endl
     << "class " << className << " : public " << baseName << " {" << endl
     << "public:" << endl
     << "  " << className << "() {} ; " << endl
     << "  " << className << "(const char *name, const char *title," << endl ;

  // Insert list of input arguments
  unsigned int i ;
  for (i=0 ; i<alist.size() ; i++) {
    if (!isCat[i]) {
      hf << "        RooAbsReal& _" ;
    } else {
      hf << "        RooAbsCategory& _" ;
    }
    hf << alist[i] ;
    if (i==alist.size()-1) {
      hf << ");" << endl ;
    } else {
      hf << "," << endl ;
    }
  }

  hf << "  " << className << "(const " << className << "& other, const char* name=0) ;" << endl
     << "  virtual TObject* clone(const char* newname) const override { return new " << className << "(*this,newname); }" << endl
     << "  inline virtual ~" << className << "() { }" << endl
     << endl ;

  if (hasAnaInt) {
    hf << "  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;" << endl
       << "  double analyticalIntegral(Int_t code, const char* rangeName=0) const ;" << endl
       << "" << endl ;
  }

  if (hasIntGen) {
     hf << "  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const;" << endl
   << "  void initGenerator(Int_t code) {} ; // optional pre-generation initialization" << endl
   << "  void generateEvent(Int_t code);" << endl
   << endl ;
  }

  hf << "protected:" << endl
     << "" << endl ;

  // Insert list of input arguments
  for (i=0 ; i<alist.size() ; i++) {
    if (!isCat[i]) {
      hf << "  RooRealProxy " << alist[i] << " ;" << endl ;
    } else {
      hf << "  RooCategoryProxy " << alist[i] << " ;" << endl ;
    }
  }

  hf << "  " << endl
     << "  double evaluate() const override;" << endl
     << "" << endl
     << "private:" << endl
     << "" << endl
     << "  ClassDefOverride(" << className << ",1) // Your description goes here..." << endl
     << "};" << endl
     << " " << endl
     << "#endif" << endl ;


  ofstream cf(impFileName) ;

  cf << "/***************************************************************************** " << endl
     << " * Project: RooFit                                                           * " << endl
     << " *                                                                           * " << endl
     << " * This code was autogenerated by RooClassFactory                            * " << endl
     << " *****************************************************************************/ " << endl
     << endl
     << "// Your description goes here... " << endl
     << endl

     << "#include \"Riostream.h\" " << endl
     << endl

     << "#include \"" << className << ".h\" " << endl
     << "#include \"RooAbsReal.h\" " << endl
     << "#include \"RooAbsCategory.h\" " << endl
     << "#include <math.h> " << endl
     << "#include \"TMath.h\" " << endl
     << endl

     << "ClassImp(" << className << "); " << endl
     << endl

     << " " << className << "::" << className << "(const char *name, const char *title, " << endl ;

  // Insert list of proxy constructors
  for (i=0 ; i<alist.size() ; i++) {
    if (!isCat[i]) {
      cf << "                        RooAbsReal& _" << alist[i] ;
    } else {
      cf << "                        RooAbsCategory& _" << alist[i] ;
    }
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

     << " double " << className << "::evaluate() const " << endl
     << " { " << endl
     << "   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE " << endl
     << "   return " << expression << " ; " << endl
     << " } " << endl
     << endl
     << endl
     << endl ;

  if (hasAnaInt) {

    vector<string> intObs ;
    vector<string> intExpr ;
    // Parse analytical integration expression if provided
    // Expected form is observable:expression,observable,observable:expression;[...]
    if (intExpression && *intExpression) {
      const size_t bufSize = strlen(intExpression)+1;
      std::vector<char> buf(bufSize);
      strlcpy(buf.data(),intExpression,bufSize) ;
      char* ptr = strtok(buf.data(),":") ;
      while(ptr) {
   intObs.push_back(ptr) ;
   intExpr.push_back(strtok(0,";")) ;
   ptr = strtok(0,":") ;
      }
    }

    cf << " Int_t " << className << "::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const  " << endl
       << " { " << endl
       << "   // LIST HERE OVER WHICH VARIABLES ANALYTICAL INTEGRATION IS SUPPORTED, " << endl
       << "   // ASSIGN A NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS " << endl
       << "   // THE EXAMPLE BELOW ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X" << endl
       << "   // YOU CAN ALSO IMPLEMENT MORE THAN ONE ANALYTICAL INTEGRAL BY REPEATING THE matchArgs " << endl
       << "   // EXPRESSION MULTIPLE TIMES" << endl
       << endl  ;

    if (intObs.size()>0) {
      for (UInt_t ii=0 ; ii<intObs.size() ; ii++) {
   cf << "   if (matchArgs(allVars,analVars," << intObs[ii] << ")) return " << ii+1 << " ; " << endl ;
      }
    } else {
      cf << "   // if (matchArgs(allVars,analVars,x)) return 1 ; " << endl ;
    }

    cf << "   return 0 ; " << endl
       << " } " << endl
       << endl
       << endl
       << endl

       << " double " << className << "::analyticalIntegral(Int_t code, const char* rangeName) const  " << endl
       << " { " << endl
       << "   // RETURN ANALYTICAL INTEGRAL DEFINED BY RETURN CODE ASSIGNED BY getAnalyticalIntegral" << endl
       << "   // THE MEMBER FUNCTION x.min(rangeName) AND x.max(rangeName) WILL RETURN THE INTEGRATION" << endl
       << "   // BOUNDARIES FOR EACH OBSERVABLE x" << endl
       << endl ;

    if (intObs.size()>0) {
      for (UInt_t ii=0 ; ii<intObs.size() ; ii++) {
   cf << "   if (code==" << ii+1 << ") { return (" << intExpr[ii] << ") ; } " << endl ;
      }
    } else {
      cf << "   // assert(code==1) ; " << endl
    << "   // return (x.max(rangeName)-x.min(rangeName)) ; " << endl ;
    }

    cf << "   return 0 ; " << endl
       << " } " << endl
       << endl
       << endl
       << endl ;
  }

  if (hasIntGen) {
    cf << " Int_t " << className << "::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const " << endl
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


  return false ;
}

namespace {

////////////////////////////////////////////////////////////////////////////////

std::string ClassFacIFace::create(RooFactoryWSTool& ft, const char* typeName, const char* instanceName, std::vector<std::string> args)
{
  static int classCounter = 0 ;

  string tn(typeName) ;
  if (tn=="CEXPR" || tn=="cexpr") {

    if (args.size()<2) {
      throw string(Form("RooClassFactory::ClassFacIFace::create() ERROR: CEXPR requires at least 2 arguments (expr,var,...), but only %u args found",
         (UInt_t)args.size())) ;
    }

    RooAbsArg* ret ;
    // Strip quotation marks from expression string
    char expr[1024] ;
    strncpy(expr,args[0].c_str()+1,args[0].size()-2) ;
    expr[args[0].size()-2]=0 ;


    RooArgList varList ;

    try {
      if (args.size()==2) {
   // Interpret 2nd arg as list
   varList.add(ft.asLIST(args[1].c_str())) ;
      } else {
   for (unsigned int i=1 ; i<args.size() ; i++) {
     varList.add(ft.asARG(args[i].c_str())) ;
   }
      }
    } catch (const string &err) {
      throw string(Form("RooClassFactory::ClassFacIFace::create() ERROR: %s",err.c_str())) ;
    }

    string className ;
    while(true) {
      className = Form("RooCFAuto%03d%s%s",classCounter,(tn=="CEXPR")?"Pdf":"Func",ft.autoClassNamePostFix()) ;
      TClass* tc =  TClass::GetClass(className.c_str(),true,true) ;
      classCounter++ ;
      if (!tc) {
   break ;
      }
    }

    if (tn=="CEXPR") {
      ret = RooClassFactory::makePdfInstance(className.c_str(),instanceName,expr,varList) ;
    } else {
      ret = RooClassFactory::makeFunctionInstance(className.c_str(),instanceName,expr,varList) ;
    }
    if (!ret) {
      throw string(Form("RooClassFactory::ClassFacIFace::create() ERROR creating %s %s with RooClassFactory",((tn=="CEXPR")?"pdf":"function"),instanceName)) ;
    }

    // Import object
    ft.ws().import(*ret,RooFit::Silence()) ;

    // Import class code as well
    ft.ws().importClassCode(ret->IsA()) ;


  } else {

    throw string(Form("RooClassFactory::ClassFacIFace::create() ERROR: Unknown meta-type %s requested",typeName)) ;

  }
  return string(instanceName) ;
}

} // namespace
