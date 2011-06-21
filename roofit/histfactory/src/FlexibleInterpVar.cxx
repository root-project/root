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

//_________________________________________________
/*
BEGIN_HTML
<p>
</p>
END_HTML
*/
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>
#include "TMath.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "TMath.h"

#include "RooStats/HistFactory/FlexibleInterpVar.h"

ClassImp(RooStats::HistFactory::FlexibleInterpVar)

using namespace RooStats;
using namespace HistFactory;

//_____________________________________________________________________________
FlexibleInterpVar::FlexibleInterpVar()
{
  // Default constructor
  _paramIter = _paramList.createIterator() ;
}


//_____________________________________________________________________________
FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title, 
		       const RooArgList& paramList, 
		       double nominal, vector<double> low, vector<double> high) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(nominal), _low(low), _high(high)
{

  _paramIter = _paramList.createIterator() ;


  TIterator* paramIter = paramList.createIterator() ;
  RooAbsArg* param ;
  while((param = (RooAbsArg*)paramIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _paramList.add(*param) ;
    _interpCode.push_back(0); // default code
  }
  delete paramIter ;

}

//_____________________________________________________________________________
FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title, 
				     const RooArgList& paramList, 
				     double nominal, vector<double> low, vector<double> high,
				     vector<int> code) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(nominal), _low(low), _high(high), _interpCode(code)
{

  _paramIter = _paramList.createIterator() ;


  TIterator* paramIter = paramList.createIterator() ;
  RooAbsArg* param ;
  while((param = (RooAbsArg*)paramIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _paramList.add(*param) ;
  }
  delete paramIter ;

}

//_____________________________________________________________________________
FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title) :
  RooAbsReal(name, title),
  _paramList("paramList","List of coefficients",this)
{
  // Constructor of flat polynomial function

  _paramIter = _paramList.createIterator() ;
}

//_____________________________________________________________________________
FlexibleInterpVar::FlexibleInterpVar(const FlexibleInterpVar& other, const char* name) :
  RooAbsReal(other, name), 
  _paramList("paramList",this,other._paramList),
  _nominal(other._nominal), _low(other._low), _high(other._high), _interpCode(other._interpCode)
  
{
  // Copy constructor
  _paramIter = _paramList.createIterator() ;
  
}


//_____________________________________________________________________________
FlexibleInterpVar::~FlexibleInterpVar() 
{
  // Destructor
  delete _paramIter ;
}


//_____________________________________________________________________________
void FlexibleInterpVar::setInterpCode(RooAbsReal& param, int code){

  int index = _paramList.index(&param);
  if(index<0){
      coutE(InputArguments) << "FlexibleInterpVar::setInterpCode ERROR:  " << param.GetName() 
			    << " is not in list" << endl ;
  } else {
      coutW(InputArguments) << "FlexibleInterpVar::setInterpCode :  " << param.GetName() 
			    << " is now " << code << endl ;
    _interpCode.at(index) = code;
  }
}

//_____________________________________________________________________________
void FlexibleInterpVar::setAllInterpCodes(int code){

  for(int i=0; i<_interpCode.size(); ++i){
    _interpCode.at(i) = code;
  }
}

//_____________________________________________________________________________
void FlexibleInterpVar::printAllInterpCodes(){

  for(int i=0; i<_interpCode.size(); ++i){
    coutI(InputArguments) <<"interp code for " << _paramList.at(i)->GetName() << " = " << _interpCode.at(i) <<endl;
  }

}

//_____________________________________________________________________________
Double_t FlexibleInterpVar::evaluate() const 
{
  // Calculate and return value of polynomial

  Double_t total(_nominal) ;
  _paramIter->Reset() ;

  RooAbsReal* param ;
  //const RooArgSet* nset = _paramList.nset() ;
  int i=0;

  while((param=(RooAbsReal*)_paramIter->Next())) {
    //    param->Print("v");

    if(_interpCode.at(i)==0){
      // piece-wise linear
      if(param->getVal()>0)
	total +=  param->getVal()*(_high.at(i) - _nominal );
      else
	total += param->getVal()*(_nominal - _low.at(i));
    } else if(_interpCode.at(i)==1){
      // pice-wise log
      if(param->getVal()>=0)
	total *= pow(_high.at(i)/_nominal, +param->getVal());
      else
	total *= pow(_low.at(i)/_nominal,  -param->getVal());
    } else if(_interpCode.at(i)==2){
      // parabolic with linear
      double a = 0.5*(_high.at(i)+_low.at(i))-_nominal;
      double b = 0.5*(_high.at(i)-_low.at(i));
      double c = 0;
      if(param->getVal()>1 ){
	total += (2*a+b)*(param->getVal()-1)+_high.at(i)-_nominal;
      } else if(param->getVal()<-1 ) {
	total += -1*(2*a-b)*(param->getVal()+1)+_low.at(i)-_nominal;
      } else {
	total +=  a*pow(param->getVal(),2) + b*param->getVal()+c;
      }
    } else if(_interpCode.at(i)==3){
      //parabolic version of log-normal
      double a = 0.5*(_high.at(i)+_low.at(i))-_nominal;
      double b = 0.5*(_high.at(i)-_low.at(i));
      double c = 0;
      if(param->getVal()>1 ){
	total += (2*a+b)*(param->getVal()-1)+_high.at(i)-_nominal;
      } else if(param->getVal()<-1 ) {
	total += -1*(2*a-b)*(param->getVal()+1)+_low.at(i)-_nominal;
      } else {
	total +=  a*pow(param->getVal(),2) + b*param->getVal()+c;
      }
	
    } else {
      coutE(InputArguments) << "FlexibleInterpVar::evaluate ERROR:  " << param->GetName() 
			    << " with unknown interpolation code" << endl ;
    }
    ++i;
  }

  if(total<=0) {
    total=1E-9;
  }    

  return total;
}



