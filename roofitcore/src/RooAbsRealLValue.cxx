/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRealLValue.cc,v 1.35 2005/02/14 20:44:21 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooAbsRealLValue is the common abstract base class for objects that represent a
// real value that may appear on the left hand side of an equation ('lvalue')
// Each implementation must provide a setVal() member to allow direct modification 
// of the value. RooAbsRealLValue may be derived, but its functional relation
// to other RooAbsArg must be invertible
//
// This class has methods that export a fit range, but doesn't hold its values
// because these limits may be derived from limits of client object.
// The fit limits serve as integration range when interpreted
// as a dependent and a boundaries when interpreted as a parameter.

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooRandom.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooAbsBinning.hh"
#include "RooFitCore/RooBinning.hh"
#include "RooFitCore/RooUniformBinning.hh"
#include "RooFitCore/RooCmdConfig.hh"
using std::cout;
using std::endl;
using std::istream;
using std::ostream;

ClassImp(RooAbsRealLValue)

RooAbsRealLValue::RooAbsRealLValue(const char *name, const char *title, const char *unit) :
  RooAbsReal(name, title, 0, 0, unit)
{
  // Constructor
}  


RooAbsRealLValue::RooAbsRealLValue(const RooAbsRealLValue& other, const char* name) :
  RooAbsReal(other,name)
{
  // Copy constructor
}


RooAbsRealLValue::~RooAbsRealLValue() 
{
  // Destructor
}



Bool_t RooAbsRealLValue::inRange(Double_t value, Double_t* clippedValPtr) const
{
  // Return kTRUE if the input value is within our fit range. Otherwise, return
  // kFALSE and write a clipped value into clippedValPtr if it is non-zero.

  // Double_t range = getMax() - getMin() ; // ok for +/-INIFINITY
  Double_t clippedValue(value);
  Bool_t inRange(kTRUE) ;

  // test this value against our upper fit limit
  if(hasMax() && value > (getMax()+1e-6)) {
    if (clippedValPtr) {
      cout << "RooAbsRealLValue::inFitRange(" << GetName() << "): value " << value
	   << " rounded down to max limit " << getMax() << endl ;
    }
    clippedValue = getMax();
    inRange = kFALSE ;
  }
  // test this value against our lower fit limit
  if(hasMin() && value < getMin()-1e-6) {
    if (clippedValPtr) {
      cout << "RooAbsRealLValue::inFitRange(" << GetName() << "): value " << value
	   << " rounded up to min limit " << getMin() << endl;
    }
    clippedValue = getMin();
    inRange = kFALSE ;
  } 

  if (clippedValPtr) *clippedValPtr=clippedValue ;
  return inRange ;
}



Bool_t RooAbsRealLValue::isValidReal(Double_t value, Bool_t verbose) const 
{
  // Check if given value is valid
  if (!inRange(value)) {
    if (verbose)
      cout << "RooRealVar::isValid(" << GetName() << "): value " << value
           << " out of range (" << getMin() << " - " << getMax() << ")" << endl ;
    return kFALSE ;
  }
  return kTRUE ;
}                                                                                                                         


Bool_t RooAbsRealLValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  return kTRUE ;
}

void RooAbsRealLValue::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
}


RooAbsArg& RooAbsRealLValue::operator=(Double_t newValue) 
{
  // Assignment operator from a Double_t

  Double_t clipValue ;
  // Clip 
  inRange(newValue,&clipValue) ;
  setVal(clipValue) ;

  return *this ;
}


RooAbsArg& RooAbsRealLValue::operator=(const RooAbsReal& arg) 
{
  return operator=(arg.getVal()) ;
}

RooPlot* RooAbsRealLValue::frame(RooCmdArg arg1, RooCmdArg arg2, RooCmdArg arg3, RooCmdArg arg4,
				 RooCmdArg arg5, RooCmdArg arg6, RooCmdArg arg7, RooCmdArg arg8) const {

  // Define configuration for this method
  RooCmdConfig pc(Form("RooPlot::RooPlot(%s)",GetName())) ;
  pc.defineDouble("min","Range",0,getMin()) ;
  pc.defineDouble("max","Range",1,getMax()) ;
  pc.defineInt("nbins","Bins",0,getBins()) ;
  pc.defineString("rangeName","RangeWithName",0,"") ;
  pc.defineString("name","Name",0,"") ;
  pc.defineString("title","Title",0,"") ;
  pc.defineMutex("Range","RangeWithName") ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  Double_t xmin,xmax ;
  if (pc.hasProcessed("Range")) {
    xmin = pc.getDouble("min") ;
    xmax = pc.getDouble("max") ;
  } else {
    const char* rangeName=pc.getString("rangeName",0,kTRUE) ;
    xmin = getMin(rangeName) ;
    xmax = getMax(rangeName) ;
  }

  Int_t nbins = pc.getInt("nbins") ;
  const char* name = pc.getString("name",0,kTRUE) ;
  const char* title = pc.getString("title",0,kTRUE) ;

  return new RooPlot(name,title,*this,xmin,xmax,nbins) ;
}

RooPlot *RooAbsRealLValue::frame(Double_t xlo, Double_t xhi, Int_t nbins) const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.

  return new RooPlot(*this,xlo,xhi,nbins);
}


RooPlot *RooAbsRealLValue::frame(Double_t xlo, Double_t xhi) const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.

  return new RooPlot(*this,xlo,xhi,getBins());
}



RooPlot *RooAbsRealLValue::frame(Int_t nbins) const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.
  //
  // The current fit range may not be open ended or empty.

  // Plot range of variable may not be infinite or empty
  if (getMin()==getMax()) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: empty fit range, must specify plot range" << endl ;
    return 0 ;
  }
  if (RooNumber::isInfinite(getMin())||RooNumber::isInfinite(getMax())) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: open ended fit range, must specify plot range" << endl ;
    return 0 ;
  }

  return new RooPlot(*this,getMin(),getMax(),nbins);
}



RooPlot *RooAbsRealLValue::frame() const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.
  //
  // The current fit range may not be open ended or empty.

  // Plot range of variable may not be infinite or empty
  if (getMin()==getMax()) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: empty fit range, must specify plot range" << endl ;
    return 0 ;
  }
  if (RooNumber::isInfinite(getMin())||RooNumber::isInfinite(getMax())) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: open ended fit range, must specify plot range" << endl ;
    return 0 ;
  }

  return new RooPlot(*this,getMin(),getMax(),getBins());
}



void RooAbsRealLValue::copyCache(const RooAbsArg* source) 
{
  // Copy cache of another RooAbsArg to our cache

  RooAbsReal::copyCache(source) ;
  setVal(_value) ; // force back-propagation
}



void RooAbsRealLValue::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsReal::printToStream() we add:
  //
  //   Verbose : fit range and error

  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooAbsRealLValue ---" << endl;
    TString unit(_unit);
    if(!unit.IsNull()) unit.Prepend(' ');
    os << indent << "  Fit range is [ ";
    if(hasMin()) {
      os << getMin() << unit << " , ";
    }
    else {
      os << "-INF , ";
    }
    if(hasMax()) {
      os << getMax() << unit << " ]" << endl;
    }
    else {
      os << "+INF ]" << endl;
    }
  }
}

void RooAbsRealLValue::randomize() {
  // Set a new value sampled from a uniform distribution over the fit range.
  // Prints a warning and does nothing if the fit range is not finite.

  if(hasMin() && hasMax()) {
    Double_t range= getMax()-getMin();
    setVal(getMin() + RooRandom::uniform()*range);
  }
  else {
    cout << fName << "::" << ClassName() << ":randomize: fails with unbounded fit range" << endl;
  }
}


void RooAbsRealLValue::setBin(Int_t ibin) 
{
  // Check range of plot bin index
  if (ibin<0 || ibin>=numBins()) {
    cout << "RooAbsRealLValue::setBin(" << GetName() << ") ERROR: bin index " << ibin
	 << " is out of range (0," << getBins()-1 << ")" << endl ;
    return ;
  }
 
  // Set value to center of requested bin
  setVal(getBinning().binCenter(ibin)) ;
}



Bool_t RooAbsRealLValue::fitRangeOKForPlotting() const 
{
  // Check if fit range is usable as plot range, i.e. it is neither
  // open ended, nor empty
  return (hasMin() && hasMax() && (getMin()!=getMax())) ;
}


Bool_t RooAbsRealLValue::inRange(const char* name) const 
{
  // Check if current value is inside range with given name
  return (getVal() >= getMin(name) && getVal() <= getMax(name)) ;
}



TH1F *RooAbsRealLValue::createHistogram(const char *name, const char *yAxisLabel) const {
  // Create an empty 1D-histogram with appropriate scale and labels for this variable.
  // This method uses the default plot range which can be changed using the
  // setPlotMin(),setPlotMax() methods, and the default binning which can be
  // changed with setPlotBins(). The caller takes ownership of the returned
  // object and is responsible for deleting it.

  // Check if the fit range is usable as plot range
  if (!fitRangeOKForPlotting()) {
    cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	 << ") ERROR: fit range empty or open ended, must explicitly specify range" << endl ;
    return 0 ;
  }

  RooArgList list(*this) ;
  Double_t xlo = getMin() ;
  Double_t xhi = getMax() ;
  Int_t nbins = getBins() ;
  return (TH1F*)createHistogram(name, list, yAxisLabel, &xlo, &xhi, &nbins);
}

TH1F *RooAbsRealLValue::createHistogram(const char *name, const char *yAxisLabel, Double_t xlo, Double_t xhi, Int_t nBins) const {
  // Create an empty 1D-histogram with appropriate scale and labels for this variable.
  // This method uses the default plot range which can be changed using the
  // setPlotMin(),setPlotMax() methods, and the default binning which can be
  // changed with setPlotBins(). The caller takes ownership of the returned
  // object and is responsible for deleting it.

  RooArgList list(*this) ;
  return (TH1F*)createHistogram(name, list, yAxisLabel, &xlo, &xhi, &nBins);
}

TH1F *RooAbsRealLValue::createHistogram(const char *name, const char *yAxisLabel, const RooAbsBinning& bins) const {
  // Create an empty 1D-histogram with appropriate scale and labels for this variable.

  RooArgList list(*this) ;
  const RooAbsBinning* pbins = &bins ;
  return (TH1F*)createHistogram(name, list, yAxisLabel, &pbins);
}

TH2F *RooAbsRealLValue::createHistogram(const char *name, const RooAbsRealLValue &yvar, const char *zAxisLabel, 
					Double_t* xlo, Double_t* xhi, Int_t* nBins) const {
  // Create an empty 2D-histogram with appropriate scale and labels for this variable (x)
  // and the specified y variable. This method uses the default plot ranges for x and y which
  // can be changed using the setPlotMin(),setPlotMax() methods, and the default binning which
  // can be changed with setPlotBins(). The caller takes ownership of the returned object
  // and is responsible for deleting it.

  if ((!xlo && xhi) || (xlo && !xhi)) {
    cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	 << ") ERROR must specify either no range, or both limits" << endl ;
    return 0 ;
  }

  Double_t xlo_fit[2] ;
  Double_t xhi_fit[2] ;
  Int_t nbins_fit[2] ;

  Double_t *xlo2 = xlo;
  Double_t *xhi2 = xhi;
  Int_t *nBins2 = nBins;

  if (!xlo2) {

    if (!fitRangeOKForPlotting()) {
      cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	   << ") ERROR: fit range empty or open ended, must explicitly specify range" << endl ;      
      return 0 ;
    }
    if (!yvar.fitRangeOKForPlotting()) {
      cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	   << ") ERROR: fit range of " << yvar.GetName() << " empty or open ended, must explicitly specify range" << endl ;      
      return 0 ;
    }

    xlo_fit[0] = getMin() ;
    xhi_fit[0] = getMax() ;    

    xlo_fit[1] = yvar.getMin() ;
    xhi_fit[1] = yvar.getMax() ;

    xlo2 = xlo_fit ;
    xhi2 = xhi_fit ;
  }
  
  if (!nBins2) {
    nbins_fit[0] = getBins() ;
    nbins_fit[1] = yvar.getBins() ;
    nBins2 = nbins_fit ;
  }


  RooArgList list(*this,yvar) ;
  return (TH2F*)createHistogram(name, list, zAxisLabel, xlo2, xhi2, nBins2);
}

TH2F *RooAbsRealLValue::createHistogram(const char *name, const RooAbsRealLValue &yvar, const char *zAxisLabel, const RooAbsBinning** bins) const {
  // Create an empty 2D-histogram with appropriate scale and labels for this variable (x)
  // and the specified y variable. 

  RooArgList list(*this,yvar) ;
  return (TH2F*)createHistogram(name, list, zAxisLabel, bins);
}

TH3F *RooAbsRealLValue::createHistogram(const char *name, const RooAbsRealLValue &yvar, const RooAbsRealLValue &zvar,
					const char *tAxisLabel, Double_t* xlo, Double_t* xhi, Int_t* nBins) const {
  // Create an empty 3D-histogram with appropriate scale and labels for this variable (x)
  // and the specified y,z variables. This method uses the default plot ranges for x,y,z which
  // can be changed using the setPlotMin(),setPlotMax() methods, and the default binning which
  // can be changed with setPlotBins(). The caller takes ownership of the returned object
  // and is responsible for deleting it.

  if ((!xlo && xhi) || (xlo && !xhi)) {
    cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	 << ") ERROR must specify either no range, or both limits" << endl ;
    return 0 ;
  }

  Double_t xlo_fit[3] ;
  Double_t xhi_fit[3] ;
  Int_t nbins_fit[3] ;

  Double_t *xlo2 = xlo;
  Double_t *xhi2 = xhi;
  Int_t* nBins2 = nBins;
  if (!xlo2) {

    if (!fitRangeOKForPlotting()) {
      cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	   << ") ERROR: fit range empty or open ended, must explicitly specify range" << endl ;      
      return 0 ;
    }
    if (!yvar.fitRangeOKForPlotting()) {
      cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	   << ") ERROR: fit range of " << yvar.GetName() << " empty or open ended, must explicitly specify range" << endl ;      
      return 0 ;
    }
    if (!zvar.fitRangeOKForPlotting()) {
      cout << "RooAbsRealLValue::createHistogram(" << GetName() 
	   << ") ERROR: fit range of " << zvar.GetName() << " empty or open ended, must explicitly specify range" << endl ;      
      return 0 ;
    }

    xlo_fit[0] = getMin() ;
    xhi_fit[0] = getMax() ;    

    xlo_fit[1] = yvar.getMin() ;
    xhi_fit[1] = yvar.getMax() ;

    xlo_fit[2] = zvar.getMin() ;
    xhi_fit[2] = zvar.getMax() ;

    xlo2 = xlo_fit ;
    xhi2 = xhi_fit ;
  }
  
  if (!nBins2) {
    nbins_fit[0] = getBins() ;
    nbins_fit[1] = yvar.getBins() ;
    nbins_fit[2] = zvar.getBins() ;
    nBins2 = nbins_fit ;
  }

  RooArgList list(*this,yvar,zvar) ;
  return (TH3F*)createHistogram(name, list, tAxisLabel, xlo2, xhi2, nBins2);
}


TH3F *RooAbsRealLValue::createHistogram(const char *name, const RooAbsRealLValue &yvar, const RooAbsRealLValue &zvar, 
					const char* tAxisLabel, const RooAbsBinning** bins) const {
  // Create an empty 3D-histogram with appropriate scale and labels for this variable (x)
  // and the specified y,z variables. 

  RooArgList list(*this,yvar,zvar) ;
  return (TH3F*)createHistogram(name, list, tAxisLabel, bins);
}


TH1 *RooAbsRealLValue::createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, Double_t* xlo, Double_t* xhi, Int_t* nBins)
{
  const RooAbsBinning* bin[3] ;
  Int_t ndim = vars.getSize() ;
  bin[0] = new RooUniformBinning(xlo[0],xhi[0],nBins[0]) ;
  bin[1] = (ndim>1) ? new RooUniformBinning(xlo[1],xhi[1],nBins[1]) : 0 ;
  bin[2] = (ndim>2) ? new RooUniformBinning(xlo[2],xhi[2],nBins[2]) : 0 ;

  TH1* ret = createHistogram(name,vars,tAxisLabel,bin) ;

  if (bin[0]) delete bin[0] ;
  if (bin[1]) delete bin[1] ;
  if (bin[2]) delete bin[2] ;
  return ret ;  
}


TH1 *RooAbsRealLValue::createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, const RooAbsBinning** bins) 
{
  // Create a 1,2, or 3D-histogram with appropriate scale and labels.
  // Binning and ranges are taken from the variables themselves and can be changed by
  // calling their setPlotMin/Max() and setPlotBins() methods. A histogram can be filled
  // using RooAbsReal::fillHistogram() or RooTreeData::fillHistogram().
  // The caller takes ownership of the returned object and is responsible for deleting it.

  // Check that we have 1-3 vars
  Int_t dim= vars.getSize();
  if(dim < 1 || dim > 3) {
    cout << "RooAbsReal::createHistogram: dimension not supported: " << dim << endl;
    return 0;
  }

  // Check that all variables are AbsReals and prepare a name of the form <name>_<var1>_...
  TString histName(name);
  histName.Append("_");
  const RooAbsRealLValue *xyz[3];

  Int_t index;
  for(index= 0; index < dim; index++) {
    const RooAbsArg *arg= vars.at(index);
    xyz[index]= dynamic_cast<const RooAbsRealLValue*>(arg);
    if(!xyz[index]) {
      cout << "RooAbsRealLValue::createHistogram: variable is not real lvalue: " << arg->GetName() << endl;
      return 0;
    }
    histName.Append("_");
    histName.Append(arg->GetName());
  }
  TString histTitle(histName);
  histTitle.Prepend("Histogram of ");

  // Create the histogram
  TH1 *histogram = 0;
  switch(dim) {
  case 1:
    histogram= new TH1F(histName.Data(), histTitle.Data(),
			bins[0]->numBins(),bins[0]->array());
    break;
  case 2:
    histogram= new TH2F(histName.Data(), histTitle.Data(),
			bins[0]->numBins(),bins[0]->array(),
			bins[1]->numBins(),bins[1]->array());
    break;
  case 3:
    histogram= new TH3F(histName.Data(), histTitle.Data(),
			bins[0]->numBins(),bins[0]->array(),
			bins[1]->numBins(),bins[1]->array(),
			bins[2]->numBins(),bins[2]->array());
    break;
  default:
    assert(0);
    break;
  }
  if(!histogram) {
    cout << "RooAbsReal::createHistogram: unable to create a new histogram" << endl;
    return 0;
  }

  // Set the histogram coordinate axis labels from the titles of each variable, adding units if necessary.
  for(index= 0; index < dim; index++) {
    TString axisTitle(xyz[index]->getTitle(kTRUE));
    switch(index) {
    case 0:
      histogram->SetXTitle(axisTitle.Data());
      break;
    case 1:
      histogram->SetYTitle(axisTitle.Data());
      break;
    case 2:
      histogram->SetZTitle(axisTitle.Data());
      break;
    default:
      assert(0);
      break;
    }
  }

  // Set the t-axis title if given one
  if((0 != tAxisLabel) && (0 != strlen(tAxisLabel))) {
    TString axisTitle(tAxisLabel);
    axisTitle.Append(" / ( ");
    for(Int_t index= 0; index < dim; index++) {
      Double_t delta= bins[index]->averageBinWidth() ; // xyz[index]->getBins();
      if(index > 0) axisTitle.Append(" x ");
      axisTitle.Append(Form("%g",delta));
      if(strlen(xyz[index]->getUnit())) {
	axisTitle.Append(" ");
	axisTitle.Append(xyz[index]->getUnit());
      }
    }
    axisTitle.Append(" )");
    switch(dim) {
    case 1:
      histogram->SetYTitle(axisTitle.Data());
      break;
    case 2:
      histogram->SetZTitle(axisTitle.Data());
      break;
    case 3:
      // not supported in TH1
      break;
    default:
      assert(0);
      break;
    }
  }

  return histogram;
}

