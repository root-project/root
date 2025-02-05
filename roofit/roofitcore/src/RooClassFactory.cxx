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

Similar to TTree::MakeClass(), generates
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
#include "RooFitImplHelpers.h"

#include <ROOT/StringUtils.hxx>

#include <strlcpy.h>
#include <cctype>
#include <fstream>
#include <mutex>

using std::endl, std::vector, std::string;

namespace {

class ClassFacIFace : public RooFactoryWSTool::IFace {
public:
   std::string
   create(RooFactoryWSTool &ft, const char *typeName, const char *instanceName, std::vector<std::string> args) override;
};

static int init();

int dummy = init();

int init()
{
   RooFactoryWSTool::IFace *iface = new ClassFacIFace;
   RooFactoryWSTool::registerSpecial("CEXPR", iface);
   RooFactoryWSTool::registerSpecial("cexpr", iface);
   (void)dummy;
   return 0;
}

bool makeAndCompileClass(std::string const &baseClassName, std::string const &name, std::string const &expression,
                         const RooArgList &vars, std::string const &intExpression)
{
   // A structure to store the inputs to this function, to check if has been
   // called already with the same arguments.
   class ClassInfo {
   public:
      ClassInfo(std::string const &baseClassName, std::string const &name, std::string const &expression,
                const RooArgList &vars, std::string const &intExpression)
         : _baseClassName{baseClassName}, _name{name}, _expression{expression}, _intExpression{intExpression}
      {
         _argNames.reserve(vars.size());
         _argsAreCategories.reserve(vars.size());
         for (RooAbsArg *arg : vars) {
            _argNames.emplace_back(arg->GetName());
            _argsAreCategories.emplace_back(arg->isCategory());
         }
      }
      bool operator==(const ClassInfo &other) const
      {
         return other._baseClassName == _baseClassName && other._name == _name && other._expression == _expression &&
                other._argNames == _argNames && other._argsAreCategories == _argsAreCategories &&
                other._intExpression == _intExpression;
      }

      std::string _baseClassName;
      std::string _name;
      std::string _expression;
      std::vector<std::string> _argNames;
      std::vector<bool> _argsAreCategories;
      std::string _intExpression;
   };

   static std::vector<ClassInfo> infosVec;
   static std::mutex infosVecMutex; // protects infosVec

   ClassInfo info{baseClassName, name, expression, vars, intExpression};

   // Check if this class was already compiled
   auto found = std::find_if(infosVec.begin(), infosVec.end(), [&](auto const &elem) { return elem._name == name; });
   if (found != infosVec.end()) {
      if (*found == info) {
         return false;
      }
      std::stringstream ss;
      ss << "RooClassFactory ERROR The type, expressions, or variables for the class \"" << name
         << "\" are not identical to what you passed last time this class was compiled! This is not allowed.";
      oocoutE(nullptr, InputArguments) << ss.str() << std::endl;
      throw std::runtime_error(ss.str());
   }

   // Making a new compiled class is not thread safe
   const std::lock_guard<std::mutex> lock(infosVecMutex);

   infosVec.emplace_back(info);

   std::string realArgNames;
   std::string catArgNames;
   for (RooAbsArg *arg : vars) {
      if (dynamic_cast<RooAbsReal *>(arg)) {
         if (!realArgNames.empty())
            realArgNames += ",";
         realArgNames += arg->GetName();
      } else if (arg->isCategory()) {
         if (!catArgNames.empty())
            catArgNames += ",";
         catArgNames += arg->GetName();
      } else {
         oocoutE(nullptr, InputArguments) << "RooClassFactory ERROR input argument " << arg->GetName()
                                          << " is neither RooAbsReal nor RooAbsCategory and is ignored" << std::endl;
      }
   }

   bool ret = RooClassFactory::makeClass(baseClassName, name, realArgNames, catArgNames, expression,
                                         !intExpression.empty(), false, intExpression);
   if (ret) {
      return ret;
   }

   TInterpreter::EErrorCode ecode;
   gInterpreter->ProcessLineSynch((".L " + name + ".cxx+").c_str(), &ecode);
   return (ecode != TInterpreter::kNoError);
}

RooAbsReal *makeClassInstance(std::string const &baseClassName, std::string const &className, std::string const &name,
                              std::string const &expression, const RooArgList &vars, std::string const &intExpression)
{
   // Use class factory to compile and link specialized function
   bool error = makeAndCompileClass(baseClassName, className, expression, vars, intExpression);

   // Check that class was created OK
   if (error) {
      RooErrorHandler::softAbort();
   }

   // Create interpreter line that instantiates specialized object
   std::string line = std::string("new ") + className + "(\"" + name + "\",\"" + name + "\"";

   // Make list of pointer values (represented in hex ascii) to be passed to cint
   // Note that the order of passing arguments must match the convention in which
   // the class code is generated: first all reals, then all categories

   std::string argList;
   // First pass the RooAbsReal arguments in the list order
   for (RooAbsArg *var : vars) {
      if (dynamic_cast<RooAbsReal *>(var)) {
         argList += Form(",*reinterpret_cast<RooAbsReal*>(0x%zx)", reinterpret_cast<std::size_t>(var));
      }
   }
   // Next pass the RooAbsCategory arguments in the list order
   for (RooAbsArg *var : vars) {
      if (var->isCategory()) {
         argList += Form(",*reinterpret_cast<RooAbsCategory*>(0x%zx)", reinterpret_cast<std::size_t>(var));
      }
   }

   line += argList + ") ;";

   // Let interpreter instantiate specialized formula
   return reinterpret_cast<RooAbsReal *>(gInterpreter->ProcessLineSynch(line.c_str()));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

bool RooClassFactory::makeAndCompilePdf(std::string const &name, std::string const &expression, const RooArgList &vars,
                                        std::string const &intExpression)
{
   return makeAndCompileClass("RooAbsPdf", name, expression, vars, intExpression);
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

bool RooClassFactory::makeAndCompileFunction(std::string const &name, std::string const &expression,
                                             const RooArgList &vars, std::string const &intExpression)
{
   return makeAndCompileClass("RooAbsReal", name, expression, vars, intExpression);
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

RooAbsReal *RooClassFactory::makeFunctionInstance(std::string const &name, std::string const &expression,
                                                  const RooArgList &vars, std::string const &intExpression)
{
   // Construct unique class name for this function expression
   std::string tmpName(name);
   tmpName[0] = toupper(tmpName[0]);
   string className = "Roo" + tmpName + "Func";

   return makeFunctionInstance(className, name, expression, vars, intExpression);
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

RooAbsReal *RooClassFactory::makeFunctionInstance(std::string const &className, std::string const &name,
                                                  std::string const &expression, const RooArgList &vars,
                                                  std::string const &intExpression)
{
   return static_cast<RooAbsReal *>(makeClassInstance("RooAbsRal", className, name, expression, vars, intExpression));
}

////////////////////////////////////////////////////////////////////////////////
/// Write, compile and load code and instantiate object for a RooAbsPdf
/// implementation. The difference to makeFunctionInstance() is the base
/// class of the written class (RooAbsPdf instead of RooAbsReal).
///
/// \see RooClassFactory::makeFunctionInstance(const char*, const char*, RooArgList const&, const char*)

RooAbsPdf *RooClassFactory::makePdfInstance(std::string const &name, std::string const &expression,
                                            const RooArgList &vars, std::string const &intExpression)
{
   // Construct unique class name for this function expression
   std::string tmpName(name);
   tmpName[0] = toupper(tmpName[0]);
   string className = "Roo" + tmpName + "Pdf";

   return makePdfInstance(className, name, expression, vars, intExpression);
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

RooAbsPdf *RooClassFactory::makePdfInstance(std::string const &className, std::string const &name,
                                            std::string const &expression, const RooArgList &vars,
                                            std::string const &intExpression)
{
   return static_cast<RooAbsPdf *>(makeClassInstance("RooAbsPdf", className, name, expression, vars, intExpression));
}

////////////////////////////////////////////////////////////////////////////////
/// Write code for a RooAbsPdf implementation with class name 'name'.
/// The difference to makePdf() is the base
/// class of the written class (RooAbsPdf instead of RooAbsReal).
///
/// \see RooClassFactory::makePdf(const char*, const char*, std::string const &, const char*, RooArgList const&, bool,
/// bool, const char*)

bool RooClassFactory::makePdf(std::string const &name, std::string const &argNames, std::string const &catArgNames,
                              std::string const &expression, bool hasAnaInt, bool hasIntGen,
                              std::string const &intExpression)
{
   return makeClass("RooAbsPdf", name, argNames, catArgNames, expression, hasAnaInt, hasIntGen, intExpression);
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

bool RooClassFactory::makeFunction(std::string const &name, std::string const &argNames, std::string const &catArgNames,
                                   std::string const &expression, bool hasAnaInt, std::string const &intExpression)
{
   return makeClass("RooAbsReal", name, argNames, catArgNames, expression, hasAnaInt, false, intExpression);
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
         << "std::span<const double> " << alist[i] << "Span = ctx.at(" << alist[i] << ");\n";
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

inline bool isSpecial(char c)
{
   return c != '_' && !std::isalnum(c);
}

bool isComplex(std::string const &expression)
{
   // Let's figure out if the expression contains the imaginary unit

   for (std::size_t i = 0; i < expression.size(); ++i) {
      bool leftOkay = (i == 0) || isSpecial(expression[i - 1]);
      bool rightOkay = (i == expression.size() - 1) || isSpecial(expression[i + 1]);
      if (expression[i] == 'I' && leftOkay && rightOkay)
         return true;
   }
   return false;
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

bool RooClassFactory::makeClass(std::string const &baseName, std::string const &className,
                                std::string const &realArgNames, std::string const &catArgNames,
                                std::string const &expression, bool hasAnaInt, bool hasIntGen,
                                std::string const &intExpression)
{
   // Check that arguments were given

   if (realArgNames.empty() && catArgNames.empty()) {
      oocoutE(nullptr, InputArguments)
         << "RooClassFactory::makeClass: ERROR: A list of input argument names must be given" << std::endl;
      return true;
   }

   if (!intExpression.empty() && !hasAnaInt) {
      oocoutE(nullptr, InputArguments) << "RooClassFactory::makeClass: ERROR no analytical integration code "
                                          "requestion, but expression for analytical integral provided"
                                       << std::endl;
      return true;
   }

   // Parse comma separated list of argument names into list of strings
   vector<string> alist;
   vector<bool> isCat;

   for (auto const &token : ROOT::Split(realArgNames, ",", /*skipEmpyt=*/true)) {
      alist.push_back(token);
      isCat.push_back(false);
   }
   for (auto const &token : ROOT::Split(catArgNames, ",", /*skipEmpyt=*/true)) {
      alist.push_back(token);
      isCat.push_back(true);
   }

   // clang-format off
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

#include <complex>

class CLASS_NAME : public BASE_NAME {
public:
   CLASS_NAME() {}
   CLASS_NAME(const char *name, const char *title,)";

  // Insert list of input arguments
  for (std::size_t i=0 ; i<alist.size() ; i++) {
    if (!isCat[i]) {
      hf << "        RooAbsReal& _" ;
    } else {
      hf << "        RooAbsCategory& _" ;
    }
    hf << alist[i] ;
    if (i==alist.size()-1) {
      hf << ");" << std::endl ;
    } else {
      hf << "," << std::endl ;
    }
  }

  hf << R"(  CLASS_NAME(CLASS_NAME const &other, const char *name=nullptr);
  TObject* clone(const char *newname) const override { return new CLASS_NAME(*this, newname); }
)";

  if (hasAnaInt) {
    hf << R"(
   int getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char *rangeName=nullptr) const override;
   double analyticalIntegral(int code, const char *rangeName=nullptr) const override;
)";
  }

  if (hasIntGen) {
     hf << R"(
   int getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
   void initGenerator(int code) override {} // optional pre-generation initialization
   void generateEvent(int code) override;
)";
  }

  hf << "" << std::endl ;

  // Insert list of input arguments
  for (std::size_t i=0 ; i<alist.size() ; i++) {
    if (!isCat[i]) {
      hf << "  RooRealProxy " << alist[i] << " ;" << std::endl ;
    } else {
      hf << "  RooCategoryProxy " << alist[i] << " ;" << std::endl ;
    }
  }

  hf << R"(
  double evaluate() const override;
  void doEval(RooFit::EvalContext &) const override;

private:

  ClassDefOverride(CLASS_NAME, 1) // Your description goes here...
};

namespace RooFit {
namespace Experimental {

void codegenImpl(CLASS_NAME &arg, CodegenContext &ctx);

} // namespace Experimental
} // namespace RooFit

)";


  hf << "inline double CLASS_NAME_evaluate(" << listVars(alist, isCat) << ")";
  hf << R"(
{)";

  // When Clad is supporting std::complex, we might drop this check and always write the definition of I.
  if (isComplex(expression)) {
    hf << R"(
   // Support also using the imaginary unit
   using namespace std::complex_literals;
   // To be able to also comile C code, we define a variable that behaves like the "I" macro from C.
   constexpr auto I = 1i;
)";
  }

  hf << R"(
   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE

)"
     << "   return " << expression << ";" << std::endl
     << "}\n"
     << std::endl;

  hf << "\n#endif // CLASS_NAME_h";

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


CLASS_NAME::CLASS_NAME(const char *name, const char *title,
)";

  // Insert list of proxy constructors
  for (std::size_t i=0 ; i<alist.size() ; i++) {
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
    cf << std::endl ;
  }

  // Insert base class constructor
  cf << "   : BASE_NAME(name,title)," << std::endl ;

  // Insert list of proxy constructors
  for (std::size_t i=0 ; i<alist.size() ; i++) {
    cf << "   " << alist[i] << "(\"" << alist[i] << "\",\"" << alist[i] << "\",this,_" << alist[i] << ")" ;
    if (i<alist.size()-1) {
      cf << "," ;
    }
    cf << std::endl ;
  }

  cf << "{" << std::endl
     << "}" << std::endl
     << std::endl

     << "CLASS_NAME::CLASS_NAME(CLASS_NAME const &other, const char *name)" << std::endl
     << "   : BASE_NAME(other,name)," << std::endl ;

  for (std::size_t i=0 ; i<alist.size() ; i++) {
    cf << "   " << alist[i] << "(\"" << alist[i] << "\",this,other." << alist[i] << ")" ;
    if (i<alist.size()-1) {
      cf << "," ;
    }
    cf << std::endl ;
  }

  cf << "{\n"
     << "}\n"
     << std::endl
     << "\n"
     << "double CLASS_NAME::evaluate() const " << std::endl
     << "{\n"
     << "   return CLASS_NAME_evaluate(" << listVars(alist) << ");" << std::endl
     << "}\n"
     << "\n"
     << "void CLASS_NAME::doEval(RooFit::EvalContext &ctx) const" << std::endl
     << "{\n"
     << declareVarSpans(alist)
     << "\n"
     << "   std::size_t n = ctx.output().size();\n"
     << "   for (std::size_t i = 0; i < n; ++i) {\n"
     << "      ctx.output()[i] = CLASS_NAME_evaluate(" << getFromVarSpans(alist) << ");\n"
     << "   }\n"
     << "}\n";

   {
   std::stringstream varsGetters;
   for (std::size_t i = 0; i < alist.size(); ++i) {
      varsGetters << "arg." << alist[i];
      if (i < alist.size() - 1) {
         varsGetters << ", ";
      }
   }

   cf << "void RooFit::Experimental::codegenImpl(CLASS_NAME &arg, RooFit::Experimental::CodegenContext &ctx)\n"
      << "{\n"
      << "   ctx.addResult(&arg, ctx.buildCall(\"CLASS_NAME_evaluate\", " << varsGetters.str() << "));\n"
      <<"}\n";
  }

  if (hasAnaInt) {

    vector<string> intObs ;
    vector<string> intExpr ;
    // Parse analytical integration expression if provided
    // Expected form is observable:expression,observable,observable:expression;[...]
    if (!intExpression.empty()) {
      const std::size_t bufSize = intExpression.size()+1;
      std::vector<char> buf(bufSize);
      strlcpy(buf.data(),intExpression.c_str(),bufSize) ;
      char* ptr = strtok(buf.data(),":") ;
      while(ptr) {
   intObs.push_back(ptr) ;
   intExpr.push_back(strtok(nullptr,";")) ;
   ptr = strtok(nullptr,":") ;
      }
    }

    cf << R"(
int CLASS_NAME::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char */*rangeName*/) const
{
   // Support also using the imaginary unit
   using namespace std::complex_literals;
   // To be able to also comile C code, we define a variable that behaves like the "I" macro from C.
   constexpr auto I = 1i;

   // LIST HERE OVER WHICH VARIABLES ANALYTICAL INTEGRATION IS SUPPORTED,
   // ASSIGN A NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS. THE EXAMPLE
   // BELOW ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X YOU CAN ALSO
   // IMPLEMENT MORE THAN ONE ANALYTICAL INTEGRAL BY REPEATING THE matchArgs
   // EXPRESSION MULTIPLE TIMES.
)";

    if (!intObs.empty()) {
      for (std::size_t ii=0 ; ii<intObs.size() ; ii++) {
   cf << "   if (matchArgs(allVars,analVars," << intObs[ii] << ")) return " << ii+1 << " ; " << std::endl ;
      }
    } else {
      cf << "   // if (matchArgs(allVars,analVars,x)) return 1 ; " << std::endl ;
    }

    cf << "   return 0 ; " << std::endl
       << "} " << std::endl
       << std::endl
       << std::endl

       << R"(double CLASS_NAME::analyticalIntegral(int code, const char *rangeName) const
{
  // RETURN ANALYTICAL INTEGRAL DEFINED BY RETURN CODE ASSIGNED BY
  // getAnalyticalIntegral(). THE MEMBER FUNCTION x.min(rangeName) AND
  // x.max(rangeName) WILL RETURN THE INTEGRATION BOUNDARIES FOR EACH
  // OBSERVABLE x.
)";

    if (!intObs.empty()) {
      for (std::size_t ii=0 ; ii<intObs.size() ; ii++) {
   cf << "   if (code==" << ii+1 << ") { return (" << intExpr[ii] << ") ; } " << std::endl ;
      }
    } else {
      cf << "   // assert(code==1) ; " << std::endl
    << "   // return (x.max(rangeName)-x.min(rangeName)) ; " << std::endl ;
    }

    cf << "   return 0 ; " << std::endl
       << "} " << std::endl;
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
   // clang-format on

   std::ofstream ohf(className + ".h");
   std::ofstream ocf(className + ".cxx");
   std::string headerCode = hf.str();
   std::string sourceCode = cf.str();
   RooFit::Detail::replaceAll(headerCode, "CLASS_NAME", className);
   RooFit::Detail::replaceAll(sourceCode, "CLASS_NAME", className);
   RooFit::Detail::replaceAll(headerCode, "BASE_NAME", baseName);
   RooFit::Detail::replaceAll(sourceCode, "BASE_NAME", baseName);
   ohf << headerCode;
   ocf << sourceCode;

   return false;
}

namespace {

////////////////////////////////////////////////////////////////////////////////

std::string ClassFacIFace::create(RooFactoryWSTool &ft, const char *typeName, const char *instanceName,
                                  std::vector<std::string> args)
{
   static int classCounter = 0;

   string tn(typeName);

   if (args.size() < 2) {
      throw std::runtime_error(Form("RooClassFactory::ClassFacIFace::create() ERROR: CEXPR requires at least 2 "
                                    "arguments (expr,var,...), but only %u args found",
                                    (UInt_t)args.size()));
   }

   RooAbsArg *ret;
   // Strip quotation marks from expression string
   char expr[1024];
   strncpy(expr, args[0].c_str() + 1, args[0].size() - 2);
   expr[args[0].size() - 2] = 0;

   RooArgList varList;

   if (args.size() == 2) {
      // Interpret 2nd arg as list
      varList.add(ft.asLIST(args[1].c_str()));
   } else {
      for (unsigned int i = 1; i < args.size(); i++) {
         varList.add(ft.asARG(args[i].c_str()));
      }
   }

   string className;
   while (true) {
      className = Form("RooCFAuto%03d%s%s", classCounter, (tn == "CEXPR") ? "Pdf" : "Func", ft.autoClassNamePostFix());
      TClass *tc = TClass::GetClass(className.c_str(), true, true);
      classCounter++;
      if (!tc) {
         break;
      }
   }

   if (tn == "CEXPR") {
      ret = RooClassFactory::makePdfInstance(className, instanceName, expr, varList);
   } else {
      ret = RooClassFactory::makeFunctionInstance(className, instanceName, expr, varList);
   }
   if (!ret) {
      throw std::runtime_error(
         Form("RooClassFactory::ClassFacIFace::create() ERROR creating %s %s with RooClassFactory",
              ((tn == "CEXPR") ? "pdf" : "function"), instanceName));
   }

   // Import object
   ft.ws().import(*ret, RooFit::Silence());

   // Import class code as well
   ft.ws().importClassCode(ret->IsA());

   return string(instanceName);
}

} // namespace
