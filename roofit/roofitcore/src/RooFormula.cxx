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
\file RooFormula.cxx
\class RooFormula
\ingroup Roofitcore

RooFormula internally uses ROOT's TFormula to compute user-defined expressions
of RooAbsArgs.

The string expression can be any valid TFormula expression referring to the
listed servers either by name or by their ordinal list position. These three are
forms equivalent:
```
  RooFormula("formula", "x*y",       RooArgList(x,y))  or
  RooFormula("formula", "@0*@1",     RooArgList(x,y))
  RooFormula("formula", "x[0]*x[1]", RooArgList(x,y))
```
Note that `x[i]` is an expression reserved for TFormula. If a variable with
the name `x` is given, the RooFormula interprets `x[i]` as a list position,
but `x` without brackets as a variable name.

### Category expressions
State information of RooAbsCategories can be accessed using the '::' operator,
*e.g.*, `tagCat::Kaon` will resolve to the numerical value of
the `Kaon` state of the RooAbsCategory object named `tagCat`.

A formula to switch between lepton categories could look like this:
```
  RooFormula("formulaWithCat",
    "x * (leptonMulti == leptonMulti::one) + y * (leptonMulti == leptonMulti::two)",
    RooArgList(x, y, leptonMulti));
```

### Debugging a formula that won't compile
When the formula is preprocessed, RooFit prints some information in the message stream.
These can be retrieved by activating the RooFit::MsgLevel `RooFit::DEBUG`
and the RooFit::MsgTopic `RooFit::InputArguments`.
Check the tutorial rf506_msgservice.C for details.
**/

#include "RooFormula.h"

#include "RooFit.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "ROOT/RMakeUnique.hxx"

#include <sstream>
#include <regex>

using namespace std;

ClassImp(RooFormula);


////////////////////////////////////////////////////////////////////////////////
/// Default constructor
/// coverity[UNINIT_CTOR]

RooFormula::RooFormula() : TNamed()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Construct a new formula.
/// \param[in] name Name of the formula.
/// \param[in] formula Formula to be evaluated. Parameters/observables are identified by name
/// or ordinal position in `varList`.
/// \param[in] varList List of variables to be passed to the formula.
RooFormula::RooFormula(const char* name, const char* formula, const RooArgList& varList) :
  TNamed(name, formula), _tFormula{nullptr}
{
  _origList.add(varList);
  _isCategory = findCategoryServers(_origList);

  std::string processedFormula = processFormula(formula);

  cxcoutD(InputArguments) << "RooFormula '" << GetName() << "' will be compiled as "
      << "\n\t" << processedFormula
      << "\n  and used as"
      << "\n\t" << reconstructFormula(processedFormula)
      << "\n  with the parameters " << _origList << endl;


  if (!processedFormula.empty())
    _tFormula = std::make_unique<TFormula>(name, processedFormula.c_str());

  if (!_tFormula || !_tFormula->IsValid()) {
    coutF(InputArguments) << "RooFormula '" << GetName() << "' did not compile."
        << "\nInput:\n\t" << formula
        << "\nProcessed:\n\t" << processedFormula << endl;
    _tFormula.reset(nullptr);
  }

  RooArgList useList = usedVariables();
  if (_origList.size() != useList.size()) {
    coutI(InputArguments) << "The formula " << GetName() << " claims to use the variables " << _origList
        << " but only " << useList << " seem to be in use."
        << "\n  inputs:         " << formula
        << "\n  interpretation: " << reconstructFormula(processedFormula) << std::endl;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor
RooFormula::RooFormula(const RooFormula& other, const char* name) : 
  TNamed(name ? name : other.GetName(), other.GetTitle()), RooPrintable(other)
{
  _origList.add(other._origList);
  _isCategory = findCategoryServers(_origList);
  
  TFormula* newTF = nullptr;
  if (other._tFormula) {
    newTF = new TFormula(*other._tFormula);
    newTF->SetName(GetName());
  }

  _tFormula.reset(newTF);
}

#if !defined(__GNUC__) || defined(__clang__) || (__GNUC__ > 4) || ( __GNUC__ == 4 && __GNUC_MINOR__ > 8)
#define ROOFORMULA_HAVE_STD_REGEX
////////////////////////////////////////////////////////////////////////////////
/// Process given formula by replacing all ordinal and name references by
/// `x[i]`, where `i` matches the position of the argument in `_origList`.
/// Further, references to category states such as `leptonMulti:one` are replaced
/// by the category index.
std::string RooFormula::processFormula(std::string formula) const {

  cxcoutD(InputArguments) << "Preprocessing formula step 1: find category tags (catName::catState) in "
      << formula << endl;

  // Step 1: Find all category tags and the corresponding index numbers
  std::regex categoryReg("(\\w+)::(\\w+)");
  std::map<std::string, int> categoryStates;
  for (sregex_iterator matchIt = sregex_iterator(formula.begin(), formula.end(), categoryReg);
       matchIt != sregex_iterator(); ++matchIt) {
    assert(matchIt->size() == 3);
    const std::string fullMatch = (*matchIt)[0];
    const std::string catName = (*matchIt)[1];
    const std::string catState = (*matchIt)[2];

    const auto catVariable = dynamic_cast<const RooAbsCategory*>(_origList.find(catName.c_str()));
    if (!catVariable) {
      cxcoutD(InputArguments) << "Formula " << GetName() << " uses '::' to reference a category state as '" << fullMatch
          << "' but a category '" << catName << "' cannot be found in the input variables." << endl;
      continue;
    }

    const RooCatType* catType = catVariable->lookupType(catState.c_str(), false);
    if (!catType) {
      coutE(InputArguments) << "Formula " << GetName() << " uses '::' to reference a category state as '" << fullMatch
          << "' but the category '" << catName << "' does not seem to have the state '" << catState << "'." << endl;
      throw std::invalid_argument(formula);
    }
    const int catNum = catType->getVal();

    categoryStates[fullMatch] = catNum;
    cxcoutD(InputArguments) << "\n\t" << fullMatch << "\tname=" << catName << "\tstate=" << catState << "=" << catNum;
  }
  cxcoutD(InputArguments) << "-- End of category tags --"<< endl;

  // Step 2: Replace all category tags
  for (const auto& catState : categoryStates) {
    std::stringstream replacement;
    replacement << catState.second;
    formula = std::regex_replace(formula, std::regex(catState.first), replacement.str());
  }

  cxcoutD(InputArguments) << "Preprocessing formula step 2: replace category tags\n\t" << formula << endl;

  // Step 3: Convert `@i`-style references to `x[i]`
  std::regex ordinalRegex("@([0-9]+)");
  formula = std::regex_replace(formula, ordinalRegex, "x[$1]");

  cxcoutD(InputArguments) << "Preprocessing formula step 3: replace '@'-references\n\t" << formula << endl;

  // Step 4: Replace all named references with "x[i]"-style
  for (unsigned int i = 0; i < _origList.size(); ++i) {
    const auto& var = _origList[i];
    std::string regex = "\\b";
    regex += var.GetName();
    regex += "\\b(?!\\[)"; //Negative lookahead. If the variable is called `x`, this might otherwise replace `x[0]`.
    std::regex findParameterRegex(regex);

    std::stringstream replacement;
    replacement << "x[" << i << "]";
    formula = std::regex_replace(formula, findParameterRegex, replacement.str());

    cxcoutD(InputArguments) << "Preprocessing formula step 4: replace named references: "
        << var.GetName() << " --> " << replacement.str()
        << "\n\t" << formula << endl;
  }

  cxcoutD(InputArguments) << "Final formula:\n\t" << formula << endl;

  return formula;
}


////////////////////////////////////////////////////////////////////////////////
/// Analyse internal formula to find out which variables are actually in use.
RooArgList RooFormula::usedVariables() const {
  RooArgList useList;
  if (_tFormula == nullptr)
    return useList;

  const std::string formula(_tFormula->GetTitle());

  std::set<unsigned int> matchedOrdinals;
  std::regex newOrdinalRegex("\\bx\\[([0-9]+)\\]");
  for (sregex_iterator matchIt = sregex_iterator(formula.begin(), formula.end(), newOrdinalRegex);
      matchIt != sregex_iterator(); ++matchIt) {
    assert(matchIt->size() == 2);
    std::stringstream matchString((*matchIt)[1]);
    unsigned int i;
    matchString >> i;

    matchedOrdinals.insert(i);
  }

  for (unsigned int i : matchedOrdinals) {
    useList.add(_origList[i]);
  }

  return useList;
}


////////////////////////////////////////////////////////////////////////////////
/// From the internal representation, construct a formula by replacing all index place holders
/// with the names of the variables that are being used to evaluate it.
std::string RooFormula::reconstructFormula(std::string internalRepr) const {
  for (unsigned int i = 0; i < _origList.size(); ++i) {
    const auto& var = _origList[i];
    std::stringstream regexStr;
    regexStr << "x\\[" << i << "\\]|@" << i;
    std::regex regex(regexStr.str());

    std::string replacement = std::string("[") + var.GetName() + "]";
    internalRepr = std::regex_replace(internalRepr, regex, replacement);
  }

  return internalRepr;
}
#endif //GCC < 4.9 Check

////////////////////////////////////////////////////////////////////////////////
/// Find all input arguments which are categories, and save this information in
/// with the names of the variables that are being used to evaluate it.
std::vector<bool> RooFormula::findCategoryServers(const RooAbsCollection& collection) const {
  std::vector<bool> output;

  for (unsigned int i = 0; i < collection.size(); ++i) {
    output.push_back(dynamic_cast<const RooAbsCategory*>(collection[i]));
  }

  return output;
}


////////////////////////////////////////////////////////////////////////////////
/// Recompile formula with new expression. In case of error, the old formula is
/// retained.
Bool_t RooFormula::reCompile(const char* newFormula)
{
  std::string processed = processFormula(newFormula);
  auto newTF = std::make_unique<TFormula>(GetName(), processed.c_str());

  if (!newTF->IsValid()) {
    coutE(InputArguments) << __func__ << ": new equation doesn't compile, formula unchanged" << endl;
    return true;
  }

  _tFormula = std::move(newTF);
  SetTitle(newFormula);
  return false;
}

void RooFormula::dump() const
{
  printMultiline(std::cout, 0);
}


////////////////////////////////////////////////////////////////////////////////
/// Change used variables to those with the same name in given list.
/// \param[in] newDeps New dependents to replace the old ones.
/// \param[in] mustReplaceAll Will yield an error if one dependent does not have a replacement.
/// \param[in] nameChange Passed down to RooAbsArg::findNewServer(const RooAbsCollection&, Bool_t) const.
Bool_t RooFormula::changeDependents(const RooAbsCollection& newDeps, Bool_t mustReplaceAll, Bool_t nameChange)
{
  //Change current servers to new servers with the same name given in list
  bool errorStat = false;

  for (const auto arg : _origList) {
    RooAbsReal* replace = (RooAbsReal*) arg->findNewServer(newDeps,nameChange) ;
    if (replace) {
      _origList.replace(*arg, *replace);

      if (arg->getStringAttribute("origName")) {
        replace->setStringAttribute("origName",arg->getStringAttribute("origName")) ;
      } else {
        replace->setStringAttribute("origName",arg->GetName()) ;
      }

    } else if (mustReplaceAll) {
      coutE(LinkStateMgmt) << __func__ << ": cannot find replacement for " << arg->GetName() << endl;
      errorStat = true;
    }
  }

  _isCategory = findCategoryServers(_origList);

  return errorStat;
}



////////////////////////////////////////////////////////////////////////////////
/// Evaluate the internal TFormula.
///
/// First, all variables serving this instance are evaluated given the normalisation set,
/// and then the formula is evaluated.
/// \param[in] nset Normalisation set passed to evaluation of arguments serving values.
/// \return The result of the evaluation.
Double_t RooFormula::eval(const RooArgSet* nset) const
{
  if (!_tFormula) {
    coutF(Eval) << __func__ << " (" << GetName() << "): Formula didn't compile: " << GetTitle() << endl;
    std::string what = "Formula ";
    what += GetTitle();
    what += " didn't compile.";
    throw std::invalid_argument(what);
  }

  std::vector<double> pars;
  pars.reserve(_origList.size());
  for (unsigned int i = 0; i < _origList.size(); ++i) {
    if (_isCategory[i]) {
      const auto& cat = static_cast<RooAbsCategory&>(_origList[i]);
      pars.push_back(cat.getIndex());
    } else {
      const auto& real = static_cast<RooAbsReal&>(_origList[i]);
      pars.push_back(real.getVal(nset));
    }
  }

  return _tFormula->EvalPar(pars.data());
}


////////////////////////////////////////////////////////////////////////////////
/// Printing interface

void RooFormula::printMultiline(ostream& os, Int_t /*contents*/, Bool_t /*verbose*/, TString indent) const 
{
  os << indent << "--- RooFormula ---" << endl;
  os << indent << " Formula:        '" << GetTitle() << "'" << endl;
  os << indent << " Interpretation: '" << reconstructFormula(GetTitle()) << "'" << endl;
  indent.Append("  ");
  os << indent << "Servers: " << _origList << "\n";
  os << indent << "In use : " << actualDependents() << endl;
}


////////////////////////////////////////////////////////////////////////////////
/// Print value of formula

void RooFormula::printValue(ostream& os) const 
{
  os << const_cast<RooFormula*>(this)->eval(0) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print name of formula

void RooFormula::printName(ostream& os) const 
{
  os << GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print title of formula

void RooFormula::printTitle(ostream& os) const 
{
  os << GetTitle() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print class name of formula

void RooFormula::printClassName(ostream& os) const 
{
  os << IsA()->GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print arguments of formula, i.e. dependents that are actually used

void RooFormula::printArgs(ostream& os) const 
{
  os << "[ actualVars=";
  for (const auto arg : usedVariables()) {
     os << " " << arg->GetName();
  }
  os << " ]";
}





#ifndef ROOFORMULA_HAVE_STD_REGEX
/*
 * g++ 4.8 doesn't support the std::regex. It has headers, but no implementations of the standard, leading to linker
 * errors. As long as centos 7 needs to be supported, this forces us to have a legacy implementation.
 */

#include "TPRegexp.h"

////////////////////////////////////////////////////////////////////////////////
/// Process given formula by replacing all ordinal and name references by
/// `x[i]`, where `i` matches the position of the argument in `_origList`.
/// Further, references to category states such as `leptonMulti:one` are replaced
/// by the category index.
std::string RooFormula::processFormula(std::string formula) const {
  TString formulaTString = formula.c_str();

  cxcoutD(InputArguments) << "Preprocessing formula step 1: find category tags (catName::catState) in "
      << formulaTString.Data() << endl;

  // Step 1: Find all category tags and the corresponding index numbers
  TPRegexp categoryReg("(\\w+)::(\\w+)");
  std::map<std::string, int> categoryStates;
  int offset = 0;
  do {
    std::unique_ptr<TObjArray> matches(categoryReg.MatchS(formulaTString, "", offset, 3));
    if (matches->GetEntries() == 0)
      break;

    std::string fullMatch = static_cast<TObjString*>(matches->At(0))->GetString().Data();
    std::string catName = static_cast<TObjString*>(matches->At(1))->GetString().Data();
    std::string catState = static_cast<TObjString*>(matches->At(2))->GetString().Data();
    offset = formulaTString.Index(categoryReg, offset) + fullMatch.size();

    const auto catVariable = dynamic_cast<const RooAbsCategory*>(_origList.find(catName.c_str()));
    if (!catVariable) {
      cxcoutD(InputArguments) << "Formula " << GetName() << " uses '::' to reference a category state as '" << fullMatch
          << "' but a category '" << catName << "' cannot be found in the input variables." << endl;
      continue;
    }

    const RooCatType* catType = catVariable->lookupType(catState.c_str(), false);
    if (!catType) {
      coutE(InputArguments) << "Formula " << GetName() << " uses '::' to reference a category state as '" << fullMatch
          << "' but the category '" << catName << "' does not seem to have the state '" << catState << "'." << endl;
      throw std::invalid_argument(formula);
    }
    const int catNum = catType->getVal();

    categoryStates[fullMatch] = catNum;
    cxcoutD(InputArguments) << "\n\t" << fullMatch << "\tname=" << catName << "\tstate=" << catState << "=" << catNum;
  } while (offset != -1);
  cxcoutD(InputArguments) << "-- End of category tags --"<< endl;

  // Step 2: Replace all category tags
  for (const auto& catState : categoryStates) {
    std::stringstream replacement;
    replacement << catState.second;
    formulaTString.ReplaceAll(catState.first.c_str(), replacement.str().c_str());
  }

  cxcoutD(InputArguments) << "Preprocessing formula step 2: replace category tags\n\t" << formulaTString.Data() << endl;

  // Step 3: Convert `@i`-style references to `x[i]`
  TPRegexp ordinalRegex("@([0-9]+)");
  int nsub = 0;
  do {
    nsub = ordinalRegex.Substitute(formulaTString, "x[$1]");
  } while (nsub > 0);

  cxcoutD(InputArguments) << "Preprocessing formula step 3: replace '@'-references\n\t" << formulaTString.Data() << endl;

  // Step 4: Replace all named references with "x[i]"-style
  for (unsigned int i = 0; i < _origList.size(); ++i) {
    const auto& var = _origList[i];
    TString regex = "\\b";
    regex += var.GetName();
    regex += "\\b([^[]|$)"; //Negative lookahead. If the variable is called `x`, this might otherwise replace `x[0]`.
    TPRegexp findParameterRegex(regex);

    std::stringstream replacement;
    replacement << "x[" << i << "]$1";
    int nsub2 = 0;
    do {
      nsub2 = findParameterRegex.Substitute(formulaTString, replacement.str().c_str());
    } while (nsub2 > 0);

    cxcoutD(InputArguments) << "Preprocessing formula step 4: replace named references: "
        << var.GetName() << " --> " << replacement.str()
        << "\n\t" << formulaTString.Data() << endl;
  }

  cxcoutD(InputArguments) << "Final formula:\n\t" << formulaTString << endl;

  return formulaTString.Data();
}


////////////////////////////////////////////////////////////////////////////////
/// Analyse internal formula to find out which variables are actually in use.
RooArgList RooFormula::usedVariables() const {
  RooArgList useList;
  if (_tFormula == nullptr)
    return useList;

  const TString formulaTString = _tFormula->GetTitle();

  std::set<unsigned int> matchedOrdinals;
  TPRegexp newOrdinalRegex("\\bx\\[([0-9]+)\\]");
  int offset = 0;
  do {
    std::unique_ptr<TObjArray> matches(newOrdinalRegex.MatchS(formulaTString, "", offset, 2));
    if (matches->GetEntries() == 0)
      break;

    std::string fullMatch = static_cast<TObjString*>(matches->At(0))->GetString().Data();
    std::string ordinal   = static_cast<TObjString*>(matches->At(1))->GetString().Data();
    offset = formulaTString.Index(newOrdinalRegex, offset) + fullMatch.size();

    std::stringstream matchString(ordinal.c_str());
    unsigned int i;
    matchString >> i;

    matchedOrdinals.insert(i);
  } while (offset != -1);

  for (unsigned int i : matchedOrdinals) {
    useList.add(_origList[i]);
  }

  return useList;
}


////////////////////////////////////////////////////////////////////////////////
/// From the internal representation, construct a formula by replacing all index place holders
/// with the names of the variables that are being used to evaluate it.
std::string RooFormula::reconstructFormula(std::string internalRepr) const {
  TString internalReprT = internalRepr.c_str();

  for (unsigned int i = 0; i < _origList.size(); ++i) {
    const auto& var = _origList[i];
    std::stringstream regexStr;
    regexStr << "x\\[" << i << "\\]|@" << i;
    TPRegexp regex(regexStr.str().c_str());

    std::string replacement = std::string("[") + var.GetName() + "]";
    regex.Substitute(internalReprT, replacement.c_str());
  }

  return internalReprT.Data();
}
#endif //GCC < 4.9 Check
