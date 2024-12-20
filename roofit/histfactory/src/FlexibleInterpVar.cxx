// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
// Author: Giovanni Petrucciani (UCSD) (log-interpolation)
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::HistFactory::FlexibleInterpVar
 *  \ingroup HistFactory
 */

#include <RooMsgService.h>
#include <RooTrace.h>

#include <RooFit/Detail/MathFuncs.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>

#include <Riostream.h>
#include <TMath.h>

ClassImp(RooStats::HistFactory::FlexibleInterpVar);

using namespace RooStats;
using namespace HistFactory;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

FlexibleInterpVar::FlexibleInterpVar()
{
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title,
             const RooArgList& paramList,
             double argNominal, std::vector<double> const& lowVec, std::vector<double> const& highVec) :
  FlexibleInterpVar{name, title, paramList, argNominal, lowVec, highVec, std::vector<int>(lowVec.size(), 0)}
{
}


////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title,
                 const RooArgList& paramList,
                 double argNominal, std::vector<double> const& lowVec, std::vector<double> const& highVec,
                 std::vector<int> const& codes) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(argNominal), _low(lowVec), _high(highVec)
{
  for (auto param : paramList) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName()
             << " is not of type RooAbsReal" << std::endl ;
      // use R__ASSERT which remains also in release mode
      R__ASSERT(0) ;
    }
    _paramList.add(*param) ;
  }

  _interpCode.resize(_paramList.size());
  for (std::size_t i = 0; i < codes.size(); ++i) {
    setInterpCodeForParam(i, codes[i]);
  }

  if (_low.size() != _paramList.size() || _low.size() != _high.size() || _low.size() != _interpCode.size()) {
     coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") invalid input std::vectors " << std::endl;
     R__ASSERT(_low.size() == _paramList.size());
     R__ASSERT(_low.size() == _high.size());
     R__ASSERT(_low.size() == _interpCode.size());
  }

  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor of flat polynomial function

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title) :
  RooAbsReal(name, title),
  _paramList("paramList","List of coefficients",this)
{
  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const FlexibleInterpVar& other, const char* name) :
  RooAbsReal(other, name),
  _paramList("paramList",this,other._paramList),
  _nominal(other._nominal), _low(other._low), _high(other._high), _interpCode(other._interpCode), _interpBoundary(other._interpBoundary)

{
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

FlexibleInterpVar::~FlexibleInterpVar()
{
  TRACE_DESTROY;
}

void FlexibleInterpVar::setInterpCode(RooAbsReal &param, int code)
{
   int index = _paramList.index(&param);
   if (index < 0) {
      coutE(InputArguments) << "FlexibleInterpVar::setInterpCode ERROR:  " << param.GetName() << " is not in list"
                            << std::endl;
      return;
   }
   setInterpCodeForParam(index, code);
}

void FlexibleInterpVar::setAllInterpCodes(int code)
{
   for (std::size_t i = 0; i < _interpCode.size(); ++i) {
      setInterpCodeForParam(i, code);
   }
}

void FlexibleInterpVar::setInterpCodeForParam(int iParam, int code)
{
   RooAbsArg const &param = _paramList[iParam];
   if (code < 0 || code > 5) {
      coutE(InputArguments) << "FlexibleInterpVar::setInterpCode ERROR: " << param.GetName()
                            << " with unknown interpolation code " << code << ", keeping current code "
                            << _interpCode[iParam] << std::endl;
      return;
   }
   if (code == 3) {
      // In the past, code 3 was equivalent to code 2, which confused users.
      // Now, we just say that code 3 doesn't exist and default to code 2 in
      // that case for backwards compatible behavior.
      coutE(InputArguments) << "FlexibleInterpVar::setInterpCode ERROR: " << param.GetName()
                            << " with unknown interpolation code " << code << ", defaulting to code 2" << std::endl;
      code = 2;
   }
   _interpCode.at(iParam) = code;
   setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setNominal(double newNominal){
  coutW(InputArguments) << "FlexibleInterpVar::setNominal : nominal is now " << newNominal << std::endl ;
  _nominal = newNominal;

  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setLow(RooAbsReal& param, double newLow){
  int index = _paramList.index(&param);
  if(index<0){
      coutE(InputArguments) << "FlexibleInterpVar::setLow ERROR:  " << param.GetName()
             << " is not in list" << std::endl ;
  } else {
      coutW(InputArguments) << "FlexibleInterpVar::setLow :  " << param.GetName()
             << " is now " << newLow << std::endl ;
    _low.at(index) = newLow;
  }

  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setHigh(RooAbsReal& param, double newHigh){
  int index = _paramList.index(&param);
  if(index<0){
      coutE(InputArguments) << "FlexibleInterpVar::setHigh ERROR:  " << param.GetName()
             << " is not in list" << std::endl ;
  } else {
      coutW(InputArguments) << "FlexibleInterpVar::setHigh :  " << param.GetName()
             << " is now " << newHigh << std::endl ;
    _high.at(index) = newHigh;
  }

  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::printAllInterpCodes(){
  for(unsigned int i=0; i<_interpCode.size(); ++i){
    coutI(InputArguments) <<"interp code for " << _paramList.at(i)->GetName() << " = " << _interpCode.at(i) << std::endl;
    // GHL: Adding suggestion by Swagato:
    if( _low.at(i) <= 0.001 ) coutE(InputArguments) << GetName() << ", " << _paramList.at(i)->GetName() << ": low value = " << _low.at(i) << std::endl;
    if( _high.at(i) <= 0.001 ) coutE(InputArguments) << GetName() << ", " << _paramList.at(i)->GetName() << ": high value = " << _high.at(i) << std::endl;
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of polynomial

double FlexibleInterpVar::evaluate() const
{
   double total(_nominal);
   for (std::size_t i = 0; i < _paramList.size(); ++i) {
      int code = _interpCode[i];
      // To get consistent codes with the PiecewiseInterpolation
      if (code == 4) {
         code = 5;
      }
      double paramVal = static_cast<const RooAbsReal *>(&_paramList[i])->getVal();
      total += RooFit::Detail::MathFuncs::flexibleInterpSingle(code, _low[i], _high[i], _interpBoundary, _nominal,
                                                               paramVal, total);
   }

   if (total <= 0) {
      total = TMath::Limits<double>::Min();
   }

   return total;
}

void FlexibleInterpVar::doEval(RooFit::EvalContext &ctx) const
{
   double total(_nominal);

   for (std::size_t i = 0; i < _paramList.size(); ++i) {
      int code = _interpCode[i];
      // To get consistent codes with the PiecewiseInterpolation
      if (code == 4) {
         code = 5;
      }
      total += RooFit::Detail::MathFuncs::flexibleInterpSingle(code, _low[i], _high[i], _interpBoundary, _nominal,
                                                             ctx.at(&_paramList[i])[0], total);
   }

   if (total <= 0) {
      total = TMath::Limits<double>::Min();
   }

   ctx.output()[0] = total;
}

void FlexibleInterpVar::printMultiline(std::ostream& os, Int_t contents,
                   bool verbose, TString indent) const
{
  RooAbsReal::printMultiline(os,contents,verbose,indent);
  os << indent << "--- FlexibleInterpVar ---" << std::endl;
  printFlexibleInterpVars(os);
}

void FlexibleInterpVar::printFlexibleInterpVars(std::ostream& os) const
{
  for (int i=0;i<(int)_low.size();i++) {
    auto& param = static_cast<RooAbsReal&>(_paramList[i]);
    os << std::setw(36) << param.GetName()<<": "<<std::setw(7) << _low[i]<<"  "<<std::setw(7) << _high[i]
       <<std::endl;
  }
}
