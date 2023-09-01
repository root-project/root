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

#include <ROOT/StringUtils.hxx>

#include <strlcpy.h>
#include <fstream>

using namespace std;

namespace {

class ClassFacIFace : public RooFactoryWSTool::IFace {
public:
  std::string create(RooFactoryWSTool& ft, const char* typeName, const char* instanceName, std::vector<std::string> args) override ;
} ;

static int init();

int dummy = init();

static int init()
{
  RooFactoryWSTool::IFace* iface = new ClassFacIFace ;
  RooFactoryWSTool::registerSpecial("CEXPR",iface) ;
  RooFactoryWSTool::registerSpecial("cexpr",iface) ;
  (void)dummy;
  return 0 ;
}

}



////////////////////////////////////////////////////////////////////////////////

bool RooClassFactory::makeAndCompilePdf(const char* name, const char* expression, const RooArgList& vars, const char* intExpression)
{
  string realArgNames,catArgNames ;
  for (RooAbsArg * arg : vars) {
    if (dynamic_cast<RooAbsReal*>(arg)) {
      if (!realArgNames.empty()) realArgNames += "," ;
      realArgNames += arg->GetName() ;
    } else if (dynamic_cast<RooAbsCategory*>(arg)) {
      if (!catArgNames.empty()) catArgNames += "," ;
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
      if (!realArgNames.empty()) realArgNames += "," ;
      realArgNames += arg->GetName() ;
    } else if (dynamic_cast<RooAbsCategory*>(arg)) {
      if (!catArgNames.empty()) catArgNames += "," ;
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
      argList += Form(",*((RooAbsReal*)0x%zx)",(std::size_t)var) ;
    }
  }
  // Next pass the RooAbsCategory arguments in the list order
  for(RooAbsArg * var : vars) {
    if (dynamic_cast<RooAbsCategory*>(var)) {
      argList += Form(",*((RooAbsCategory*)0x%zx)",(std::size_t)var) ;
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
      argList += Form(",*((RooAbsReal*)0x%zx)",(std::size_t)var) ;
    }
  }
  // Next pass the RooAbsCategory arguments in the list order
  for (RooAbsArg * var : vars) {
    if (dynamic_cast<RooAbsCategory*>(var)) {
      argList += Form(",*((RooAbsCategory*)0x%zx)",(std::size_t)var) ;
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


namespace {

std::string listVars(std::vector<std::string> const &alist, std::vector<bool> const &isCat = {})
{
   std::stringstream ss;
   for (std::size_t i = 0; i < alist.size(); ++i) {
      if (!isCat.empty()) {
         ss << (isCat[i] ? "int" : "double") << " ";
      }
      ss << alist[i];
      if (i < alist.size() - 1) {
         ss << ", ";
      }
   }
   return ss.str();
}

std::string declareVarSpans(std::vector<std::string> const &alist)
{
   std::stringstream ss;
   for (std::size_t i = 0; i < alist.size(); ++i) {
      ss << "   "
         << "std::span<const double> " << alist[i] << "Span = dataMap.at(" << alist[i] << ");\n";
   }
   return ss.str();
}

std::string getFromVarSpans(std::vector<std::string> const &alist)
{
   std::stringstream ss;
   for (std::size_t i = 0; i < alist.size(); ++i) {
      std::string name = alist[i] + "Span";
      ss << name << ".size() > 1 ? " << name << "[i] : " << name << "[0]";
      if (i < alist.size() - 1) {
         ss << ",\n                               ";
      }
   }
   return ss.str();
}

/// Replace all occurences of `what` with `with` inside of `inout`.
void replaceAll(std::string &inout, std::string_view what, std::string_view with)
{
   for (std::string::size_type pos{}; inout.npos != (pos = inout.find(what.data(), pos, what.length()));
        pos += with.length()) {
      inout.replace(pos, what.length(), with.data(), with.length());
   }
}

} // namespace

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

bool RooClassFactory::makeClass(std::string const& baseName, std::string const& className, const char* realArgNames, const char* catArgNames,
              const char* expression,  bool hasAnaInt, bool hasIntGen, const char* intExpression)
{
  // Check that arguments were given

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
    for(auto const& token : ROOT::Split(realArgNames, ",")) {
      alist.push_back(token) ;
      isCat.push_back(false) ;
    }
  }
  if (catArgNames && *catArgNames) {
    for(auto const& token : ROOT::Split(catArgNames, ",")) {
      alist.push_back(token) ;
      isCat.push_back(true) ;
    }
  }

  std::stringstream hf;
  hf << R"(/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * This code was autogenerated by RooClassFactory                            *
 *****************************************************************************/

#ifndef CLASS_NAME_h
#define CLASS_NAME_h

#include <BASE_NAME.h>
#include <RooRealProxy.h>
#include <RooCategoryProxy.h>
#include <RooAbsReal.h>
#include <RooAbsCategory.h>

class CLASS_NAME : public BASE_NAME {
public:
   CLASS_NAME() {}
   CLASS_NAME(const char *name, const char *title,)";

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

  hf << R"(  CLASS_NAME(CLASS_NAME const &other, const char* name=nullptr);
  TObject* clone(const char *newname) const override { return new CLASS_NAME(*this, newname); }
)";

  if (hasAnaInt) {
    hf << R"(
   int getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override;
   double analyticalIntegral(int code, const char* rangeName=0) const override;
)";
  }

  if (hasIntGen) {
     hf << R"(
   int getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
   void initGenerator(int code) override {} // optional pre-generation initialization
   void generateEvent(int code) override;
)";
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

  hf << R"(
  double evaluate() const override;
  void computeBatch(double* output, std::size_t size, RooFit::Detail::DataMap const&) const override;

private:

  ClassDefOverride(CLASS_NAME, 1) // Your description goes here...
};

#endif // CLASS_NAME_h)";

  std::stringstream cf;

  cf << R"(/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * This code was autogenerated by RooClassFactory                            *
 *****************************************************************************/

// Your description goes here...

#include "CLASS_NAME.h"

#include <RooAbsReal.h>
#include <RooAbsCategory.h>

#include <Riostream.h>
#include <TMath.h>

#include <cmath>

ClassImp(CLASS_NAME);

CLASS_NAME::CLASS_NAME(const char *name, const char *title,
)";

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
      cf << ")" ;
    }
    cf << endl ;
  }

  // Insert base class constructor
  cf << "   : BASE_NAME(name,title)," << endl ;

  // Insert list of proxy constructors
  for (i=0 ; i<alist.size() ; i++) {
    cf << "   " << alist[i] << "(\"" << alist[i] << "\",\"" << alist[i] << "\",this,_" << alist[i] << ")" ;
    if (i<alist.size()-1) {
      cf << "," ;
    }
    cf << endl ;
  }

  cf << "{" << endl
     << "}" << endl
     << endl

     << "CLASS_NAME::CLASS_NAME(CLASS_NAME const &other, const char *name)" << endl
     << "   : BASE_NAME(other,name)," << endl ;

  for (i=0 ; i<alist.size() ; i++) {
    cf << "   " << alist[i] << "(\"" << alist[i] << "\",this,other." << alist[i] << ")" ;
    if (i<alist.size()-1) {
      cf << "," ;
    }
    cf << endl ;
  }

  cf << "{\n"
     << "}\n"
     << endl
     << "namespace {\n"
     << endl
     << "inline double evaluateImpl(" << listVars(alist, isCat) << ") " << endl
     << "{\n"
     << "   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE " << endl
     << "   return " << expression << "; " << endl
     << "}\n"
     << endl
     << "} // namespace\n"
     << "\n"
     << "double CLASS_NAME::evaluate() const " << endl
     << "{\n"
     << "   return evaluateImpl(" << listVars(alist) << "); " << endl
     << "}\n"
     << "\n"
     << "void CLASS_NAME::computeBatch(double *output, std::size_t size, RooFit::Detail::DataMap const &dataMap) const " << endl
     << "{ \n"
     << declareVarSpans(alist)
     << "\n"
     << "   for (std::size_t i = 0; i < size; ++i) {\n"
     << "      output[i] = evaluateImpl(" << getFromVarSpans(alist) << ");\n"
     << "   }\n"
     << "} \n";

  if (hasAnaInt) {

    vector<string> intObs ;
    vector<string> intExpr ;
    // Parse analytical integration expression if provided
    // Expected form is observable:expression,observable,observable:expression;[...]
    if (intExpression && *intExpression) {
      const std::size_t bufSize = strlen(intExpression)+1;
      std::vector<char> buf(bufSize);
      strlcpy(buf.data(),intExpression,bufSize) ;
      char* ptr = strtok(buf.data(),":") ;
      while(ptr) {
   intObs.push_back(ptr) ;
   intExpr.push_back(strtok(nullptr,";")) ;
   ptr = strtok(nullptr,":") ;
      }
    }

    cf << R"(
int CLASS_NAME::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
   // LIST HERE OVER WHICH VARIABLES ANALYTICAL INTEGRATION IS SUPPORTED,
   // ASSIGN A NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS. THE EXAMPLE
   // BELOW ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X YOU CAN ALSO
   // IMPLEMENT MORE THAN ONE ANALYTICAL INTEGRAL BY REPEATING THE matchArgs
   // EXPRESSION MULTIPLE TIMES.
)";

    if (!intObs.empty()) {
      for (std::size_t ii=0 ; ii<intObs.size() ; ii++) {
   cf << "   if (matchArgs(allVars,analVars," << intObs[ii] << ")) return " << ii+1 << " ; " << endl ;
      }
    } else {
      cf << "   // if (matchArgs(allVars,analVars,x)) return 1 ; " << endl ;
    }

    cf << "   return 0 ; " << endl
       << "} " << endl
       << endl
       << endl

       << R"(double CLASS_NAME::analyticalIntegral(int code, const char* rangeName) const
{
  // RETURN ANALYTICAL INTEGRAL DEFINED BY RETURN CODE ASSIGNED BY
  // getAnalyticalIntegral(). THE MEMBER FUNCTION x.min(rangeName) AND
  // x.max(rangeName) WILL RETURN THE INTEGRATION BOUNDARIES FOR EACH
  // OBSERVABLE x.
)";

    if (!intObs.empty()) {
      for (std::size_t ii=0 ; ii<intObs.size() ; ii++) {
   cf << "   if (code==" << ii+1 << ") { return (" << intExpr[ii] << ") ; } " << endl ;
      }
    } else {
      cf << "   // assert(code==1) ; " << endl
    << "   // return (x.max(rangeName)-x.min(rangeName)) ; " << endl ;
    }

    cf << "   return 0 ; " << endl
       << "} " << endl;
  }

  if (hasIntGen) {
    cf << R"(
int CLASS_NAME::getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
   // LIST HERE OVER WHICH VARIABLES INTERNAL GENERATION IS SUPPORTED, ASSIGN A
   // NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS. THE EXAMPLE BELOW
   // ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X. YOU CAN ALSO IMPLEMENT
   // MORE THAN ONE GENERATOR CONFIGURATION BY REPEATING THE matchArgs
   // EXPRESSION MULTIPLE TIMES. IF THE FLAG staticInitOK IS TRUE, THEN IT IS
   // SAFE TO PRECALCULATE INTERMEDIATE QUANTITIES IN initGenerator(), IF IT IS
   // NOT SET THEN YOU SHOULD NOT ADVERTISE ANY GENERATOR METHOD THAT RELIES ON
   // PRECALCULATIONS IN initGenerator().

   // if (matchArgs(directVars,generateVars,x)) return 1;
   return 0;
}

void CLASS_NAME::generateEvent(int code)
{
   // GENERATE SET OF OBSERVABLES DEFINED BY RETURN CODE ASSIGNED BY
   // getGenerator(). RETURN THE GENERATED VALUES BY ASSIGNING THEM TO THE
   // PROXY DATA MEMBERS THAT REPRESENT THE CHOSEN OBSERVABLES.

   // assert(code==1);
   // x = 0;
   return;
}
)";

  }

  std::ofstream ohf(className + ".h");
  std::ofstream ocf(className + ".cxx");
  std::string headerCode = hf.str();
  std::string sourceCode = cf.str();
  replaceAll(headerCode, "CLASS_NAME", className);
  replaceAll(sourceCode, "CLASS_NAME", className);
  replaceAll(headerCode, "BASE_NAME", baseName);
  replaceAll(sourceCode, "BASE_NAME", baseName);
  ohf << headerCode;
  ocf << sourceCode;

  return false ;
}

namespace {

////////////////////////////////////////////////////////////////////////////////

std::string ClassFacIFace::create(RooFactoryWSTool& ft, const char* typeName, const char* instanceName, std::vector<std::string> args)
{
  static int classCounter = 0 ;

  string tn(typeName) ;

    if (args.size()<2) {
      throw std::runtime_error(Form("RooClassFactory::ClassFacIFace::create() ERROR: CEXPR requires at least 2 arguments (expr,var,...), but only %u args found",
         (UInt_t)args.size())) ;
    }

    RooAbsArg* ret ;
    // Strip quotation marks from expression string
    char expr[1024] ;
    strncpy(expr,args[0].c_str()+1,args[0].size()-2) ;
    expr[args[0].size()-2]=0 ;


    RooArgList varList ;

      if (args.size()==2) {
   // Interpret 2nd arg as list
   varList.add(ft.asLIST(args[1].c_str())) ;
      } else {
   for (unsigned int i=1 ; i<args.size() ; i++) {
     varList.add(ft.asARG(args[i].c_str())) ;
   }
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
      throw std::runtime_error(Form("RooClassFactory::ClassFacIFace::create() ERROR creating %s %s with RooClassFactory",((tn=="CEXPR")?"pdf":"function"),instanceName)) ;
    }

    // Import object
    ft.ws().import(*ret,RooFit::Silence()) ;

    // Import class code as well
    ft.ws().importClassCode(ret->IsA()) ;

  return string(instanceName) ;
}

} // namespace
