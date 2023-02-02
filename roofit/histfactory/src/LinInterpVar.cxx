// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::HistFactory::LinInterpVar
 * \ingroup HistFactory
 * RooAbsReal that does piecewise-linear interpolations.
 */

#include <iostream>
#include <cmath>

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"

#include "RooStats/HistFactory/LinInterpVar.h"

using namespace std;

ClassImp(RooStats::HistFactory::LinInterpVar);

using namespace RooStats;
using namespace HistFactory;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

LinInterpVar::LinInterpVar()
{
  _nominal = 0 ;
}


////////////////////////////////////////////////////////////////////////////////

LinInterpVar::LinInterpVar(const char* name, const char* title,
             const RooArgList& paramList,
             double nominal, vector<double> low, vector<double> high) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(nominal), _low(low), _high(high)
{

  for (auto param : paramList) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "LinInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName()
             << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _paramList.add(*param) ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor of flat polynomial function

LinInterpVar::LinInterpVar(const char* name, const char* title) :
  RooAbsReal(name, title),
  _paramList("paramList","List of coefficients",this),
  _nominal(0)
{
  
}

////////////////////////////////////////////////////////////////////////////////

LinInterpVar::LinInterpVar(const LinInterpVar& other, const char* name) :
  RooAbsReal(other, name),
  _paramList("paramList",this,other._paramList),
  _nominal(other._nominal), _low(other._low), _high(other._high)

{
  // Copy constructor

}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of polynomial

double LinInterpVar::evaluate() const
{
  double sum(_nominal) ;
  
  int i=0;
  for(auto const* param: static_range_cast<RooAbsReal *>(_paramList)) {
    if(param->getVal()>0)
      sum +=  param->getVal()*(_high.at(i) - _nominal );
    else
      sum += param->getVal()*(_nominal - _low.at(i));

    ++i;
  }

  if(sum<=0) {
    sum=1E-9;
  }

  return sum;
}



