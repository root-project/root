/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsRealLValue.cc,v 1.21 2001/12/10 22:51:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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
#include "RooFitCore/RooRealFixedBinIter.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooArgList.hh"

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



Bool_t RooAbsRealLValue::inFitRange(Double_t value, Double_t* clippedValPtr) const
{
  // Return kTRUE if the input value is within our fit range. Otherwise, return
  // kFALSE and write a clipped value into clippedValPtr if it is non-zero.

  Double_t range = getFitMax() - getFitMin() ; // ok for +/-INIFINITY
  Double_t clippedValue(value);
  Bool_t inRange(kTRUE) ;

  // test this value against our upper fit limit
  if(hasFitMax() && value > (getFitMax()+1e-6)) {
    if (clippedValPtr) {
      cout << "RooAbsRealLValue::inFitRange(" << GetName() << "): value " << value
	   << " rounded down to max limit " << getFitMax() << endl ;
    }
    clippedValue = getFitMax();
    inRange = kFALSE ;
  }
  // test this value against our lower fit limit
  if(hasFitMin() && value < getFitMin()-1e-6) {
    if (clippedValPtr) {
      cout << "RooAbsRealLValue::inFitRange(" << GetName() << "): value " << value
	   << " rounded up to min limit " << getFitMin() << endl;
    }
    clippedValue = getFitMin();
    inRange = kFALSE ;
  } 

  if (clippedValPtr) *clippedValPtr=clippedValue ;
  return inRange ;
}



Bool_t RooAbsRealLValue::isValidReal(Double_t value, Bool_t verbose) const 
{
  // Check if given value is valid
  if (!inFitRange(value)) {
    if (verbose)
      cout << "RooRealVar::isValid(" << GetName() << "): value " << value
           << " out of range (" << getFitMin() << " - " << getFitMax() << ")" << endl ;
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
  inFitRange(newValue,&clipValue) ;
  setVal(clipValue) ;

  return *this ;
}


RooAbsArg& RooAbsRealLValue::operator=(const RooAbsReal& arg) 
{
  return operator=(arg.getVal()) ;
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

  return new RooPlot(*this,xlo,xhi,getFitBins());
}



RooPlot *RooAbsRealLValue::frame(Int_t nbins) const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.
  //
  // The current fit range may not be open ended or empty.

  // Plot range of variable may not be infinite or empty
  if (getFitMin()==getFitMax()) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: empty fit range, must specify plot range" << endl ;
    return 0 ;
  }
  if (RooNumber::isInfinite(getFitMin())||RooNumber::isInfinite(getFitMax())) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: open ended fit range, must specify plot range" << endl ;
    return 0 ;
  }

  return new RooPlot(*this,getFitMin(),getFitMax(),nbins);
}



RooPlot *RooAbsRealLValue::frame() const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.
  //
  // The current fit range may not be open ended or empty.

  // Plot range of variable may not be infinite or empty
  if (getFitMin()==getFitMax()) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: empty fit range, must specify plot range" << endl ;
    return 0 ;
  }
  if (RooNumber::isInfinite(getFitMin())||RooNumber::isInfinite(getFitMax())) {
    cout << "RooAbsRealLValue::frame(" << GetName() << ") ERROR: open ended fit range, must specify plot range" << endl ;
    return 0 ;
  }

  return new RooPlot(*this,getFitMin(),getFitMax(),getFitBins());
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
    if(hasFitMin()) {
      os << getFitMin() << unit << " , ";
    }
    else {
      os << "-INF , ";
    }
    if(hasFitMax()) {
      os << getFitMax() << unit << " ]" << endl;
    }
    else {
      os << "+INF ]" << endl;
    }
  }
}

void RooAbsRealLValue::randomize() {
  // Set a new value sampled from a uniform distribution over the fit range.
  // Prints a warning and does nothing if the fit range is not finite.

  if(hasFitMin() && hasFitMax()) {
    Double_t range= getFitMax()-getFitMin();
    setVal(getFitMin() + RooRandom::uniform()*range);
  }
  else {
    cout << fName << "::" << ClassName() << ":randomize: fails with unbounded fit range" << endl;
  }
}


void RooAbsRealLValue::setFitBin(Int_t ibin) 
{
  // Check range of plot bin index
  if (ibin<0 || ibin>=numFitBins()) {
    cout << "RooAbsRealLValue::setFitBin(" << GetName() << ") ERROR: bin index " << ibin
	 << " is out of range (0," << getFitBins()-1 << ")" << endl ;
    return ;
  }
 
  // Set value to center of requested bin
  setVal(fitBinCenter(ibin)) ;
}


Int_t RooAbsRealLValue::getFitBin() const 
{
  // Return the fit bin index for the current value
  if (getVal() >= getFitMax()) return numFitBins()-1 ;
  if (getVal() < getFitMin()) return 0 ;

  return Int_t((getVal() - getFitMin())/ fitBinWidth()) ;
}



RooAbsBinIter* RooAbsRealLValue::createFitBinIterator() const 
{
  // Return an iterator over the fit bins of this object
  return new RooRealFixedBinIter(*this) ;
}




Double_t RooAbsRealLValue::fitBinCenter(Int_t i) const 
{
  // Return the central value of the 'i'-th fit bin
  if (i<0 || i>=numFitBins()) {
    cout << "RooAbsRealLValue::fitBinCenter(" << GetName() << ") ERROR: bin index " << i 
	 << " is out of range (0," << getFitBins()-1 << ")" << endl ;
    return 0 ;
  }

  return getFitMin() + (i + 0.5)*fitBinWidth() ;
}


Double_t RooAbsRealLValue::fitBinLow(Int_t i) const 
{
  // Return the low edge of the 'i'-th fit bin
  if (i<0 || i>=numFitBins()) {
    cout << "RooAbsRealLValue::fitBinLow(" << GetName() << ") ERROR: bin index " << i 
	 << " is out of range (0," << getFitBins()-1 << ")" << endl ;
    return 0 ;
  }

  return getFitMin() + i*fitBinWidth() ;
}


Double_t RooAbsRealLValue::fitBinHigh(Int_t i) const 
{
  // Return the high edge of the 'i'-th fit bin
  if (i<0 || i>=numFitBins()) {
    cout << "RooAbsRealLValue::fitBinHigh(" << GetName() << ") ERROR: bin index " << i 
	 << " is out of range (0," << getFitBins()-1 << ")" << endl ;
    return 0 ;
  }

  return getFitMin() + (i + 1)*fitBinWidth() ;
}


Double_t RooAbsRealLValue::fitBinWidth() const 
{
  // Return the low edge of the fit bins
  return (getFitMax()-getFitMin())/getFitBins() ;
}



Bool_t RooAbsRealLValue::fitRangeOKForPlotting() const 
{
  // Check if fit range is usable as plot range, i.e. it is neither
  // open ended, nor empty
  return (hasFitMin() && hasFitMax() && (getFitMin()!=getFitMax())) ;
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
  Double_t xlo = getFitMin() ;
  Double_t xhi = getFitMax() ;
  Int_t nbins = getFitBins() ;
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

  Double_t *xlo2(xlo), *xhi2(xhi);
  Int_t *nBins2(nBins) ;

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

    xlo_fit[0] = getFitMin() ;
    xhi_fit[0] = getFitMax() ;    

    xlo_fit[1] = yvar.getFitMin() ;
    xhi_fit[1] = yvar.getFitMax() ;

    xlo2 = xlo_fit ;
    xhi2 = xhi_fit ;
  }
  
  if (!nBins2) {
    nbins_fit[0] = getFitBins() ;
    nbins_fit[1] = yvar.getFitBins() ;
    nBins2 = nbins_fit ;
  }


  RooArgList list(*this,yvar) ;
  return (TH2F*)createHistogram(name, list, zAxisLabel, xlo2, xhi2, nBins2);
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

  Double_t *xlo2(xlo), *xhi2(xhi) ;
  Int_t* nBins2(nBins) ;
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

    xlo_fit[0] = getFitMin() ;
    xhi_fit[0] = getFitMax() ;    

    xlo_fit[1] = yvar.getFitMin() ;
    xhi_fit[1] = yvar.getFitMax() ;

    xlo_fit[2] = zvar.getFitMin() ;
    xhi_fit[2] = zvar.getFitMax() ;

    xlo2 = xlo_fit ;
    xhi2 = xhi_fit ;
  }
  
  if (!nBins2) {
    nbins_fit[0] = getFitBins() ;
    nbins_fit[1] = yvar.getFitBins() ;
    nbins_fit[2] = zvar.getFitBins() ;
    nBins2 = nbins_fit ;
  }

  RooArgList list(*this,yvar,zvar) ;
  return (TH3F*)createHistogram(name, list, tAxisLabel, xlo2, xhi2, nBins2);
}

TH1 *RooAbsRealLValue::createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, Double_t* xlo, Double_t* xhi, Int_t* nBins)
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
  for(Int_t index= 0; index < dim; index++) {
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
  TH1 *histogram(0);
  switch(dim) {
  case 1:
    histogram= new TH1F(histName.Data(), histTitle.Data(),
			nBins[0], xlo[0], xhi[0]);
    break;
  case 2:
    histogram= new TH2F(histName.Data(), histTitle.Data(),
			nBins[0], xlo[0], xhi[0],
			nBins[1], xlo[1], xhi[1]) ;
    break;
  case 3:
    histogram= new TH3F(histName.Data(), histTitle.Data(),
			nBins[0], xlo[0], xhi[0],
			nBins[1], xlo[1], xhi[1],
			nBins[2], xlo[2], xhi[2]) ;			
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
  for(Int_t index= 0; index < dim; index++) {
    TString axisTitle(xyz[index]->GetTitle());
    if(strlen(xyz[index]->getUnit())) {
      axisTitle.Append(" (");
      axisTitle.Append(xyz[index]->getUnit());
      axisTitle.Append(")");
    }
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
      Double_t delta= (xyz[index]->getFitMax() - xyz[index]->getFitMin())/xyz[index]->getFitBins();
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

