/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: Roo2DKeysPdf.rdl,v 1.5 2002/01/30 09:08:51 bevan Exp $
 * Authors:
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 *
 * History:
 *   25-Jul-2001 AB Created 2D KEYS version based on the 1D template of RooKeysPdf
 *                  by Gerhard Raven.
 *   Wed Aug 15  AB changed grid to 100*100 instead of 50*50
 *   25-Aug-2001 AB Ported to RooFitCore/RooFitModels
 *   08-Dec-2001 AB added a bandwidth scale factor to allow fine tuning of the PDF
 *   29-Jan-2002 AB Altered evaluation to relu on RFC's internal cacheing.  This
 *                  gives the same numerical result withing a fraction of the errors 
 *                  as using the LUT; but gives an exact result for f(x,y) at any 
 *                  point in the fit domain and is comparable speed to fit [~10% 
 *                  slower than with a LUT].
 *
 * Copyright (C) 2001, Liverpool University
 *****************************************************************************/
#ifndef ROO_2DKEYS
#define ROO_2DKEYS

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooDataSet.hh"
#include <iostream.h>

////////////////////////////////////////////////////////////////////////////////////
// Use of the PDF:
//  If you generate the PDF using the same domain as you fit to, for data populated
//  near the edges, there will be no compensating 'feed through from the other side 
//  of the boundary'.  There are the following options available to you to avoid or
//  reduce edge effects in your fits:
//	1) enlarge your fit domain to avoid having significant data at the boundaries
//         of your fit.
//	2) try using the 'm' option.  This mirrors all gaussuans within the domain at 
//         the domain boundary.  This will lead to an overestimate of the PDF near the
//         boundary instead of the previously observed deficit.
//      3) define the fit domain [the RooRealVars in the ctor] to be smaller than the 
//         domain which the data set ctor used.  This PDF is construced using the events
//         in the data set, so if the boundaries are really giving you a problem and 
//         mirroring is not satisfactory solution to the problem, you have to take a hit 
//         in using a reduced data set to construct the PDF from.  Recall that you will 
//         be fitting to data in a different domain [and hence dataset] so there are no 
//         inconsistencies as a result [unless you are really limited by MC stats and 
//         use the same MC to generate the PDF and test the PDF's systematics].
///////////////////////////////////////////////////////////////////////////////////////
class Roo2DKeysPdf : public RooAbsPdf 
{
public:
  Roo2DKeysPdf(const char *name, const char *title,
             RooAbsReal& xx, RooAbsReal &yy, RooDataSet& data, TString options = "a", Double_t widthScaleFactor = 1.0);
  Roo2DKeysPdf(const Roo2DKeysPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new Roo2DKeysPdf(*this,newname); }

  virtual ~Roo2DKeysPdf();

//load in a new dataset and re-calculate the PDF
//return 0 if sucessful
  Int_t    loadDataSet(RooDataSet& data, TString options);

// The Roo2DKeys options available are:
//      a = select an adaptove bandwidth [default]
//      n = select a normal bandwidth
//      m = mirror kernal contributions at edges [fold gaussians back into the x,y plane]
//      d = print debug statements [useful for development only; default is off]
//      v  = print verbose debug statements [useful for development only; default is off]
//      vv = print ludicrously verbose debug statements [useful for development only; default is off]
  void     setOptions(TString options);
  void     getOptions(void) const;

// Set the value of a scale factor to modify the bandwidth by. The default value for this is unity.
// Modification of 'normal' bandwidths is useful when the data are not 'normally distributed', 
// otherwise one expects little departure from that behavior.  Note that both the normal and adaptive
// bandwidth selections are modified by this factor.  Useful for systematic studies.
//           ***********
//           *IMPORTANT* The kernel is proportional to 1/widthScaleFactor.
//           ***********
  inline void     setWidthScaleFactor(Double_t widthScaleFactor);

// choose the kernel bandwith to use.  The default is 0                                               
//    0 = use adaptive kernel estimator (uses local population to vary with of kernels)               
//    1 = use trivial kernel estimator (uses all data and sigma to estimate uniform kernel bandwidth)
  Int_t    calculateBandWidth(Int_t kernel = -999);

  Int_t    getBandWidthType() const;
  Double_t getMean(const char * axis) const;
  Double_t getSigma(const char * axis) const;

// print content and basic information about the data
  void     PrintInfo(ostream &) const;

// save PDF to a file as a TH2F *, TTree * or both
// this is so that you only need to compute the PDF once and 
// are free to use the much faster Roo2DHistPdf class in order 
// to perform fits/do toy studies etc.  
  void     writeToFile(char * outputFile, const char * name) const;
  void     writeHistToFile(char * outputFile, const char * histName) const;
  void     writeNTupleToFile(char * outputFile, const char * name) const;

  RooRealProxy x;
  RooRealProxy y;

  Double_t evaluate() const;

protected:

private:
  // these are used in calculating bandwidths for x and y
  Double_t evaluateFull(Double_t thisX, Double_t thisY) const;
  Double_t g(Double_t var1, Double_t * _var1, Double_t sigma1, Double_t var2, 
	     Double_t * _var2, Double_t sigma2) const; 

  //mirror corrections for the boundaries
  Double_t highBoundaryCorrection(Double_t thisVar, Double_t thisH, Double_t high, Double_t tVar) const;
  Double_t lowBoundaryCorrection(Double_t thisVar, Double_t thisH, Double_t low, Double_t tVar) const;

  Double_t * _x;
  Double_t * _hx;
  Double_t * _y;
  Double_t * _hy;
  Double_t   _norm;
  Double_t   _xMean;    // the (x,y) mean and sigma are properties of the data, not of the PDF
  Double_t   _xSigma;
  Double_t   _yMean;
  Double_t   _ySigma;
  Double_t   _n;         //coefficient of the kernel estimation sum
  Double_t   _n16;       //pow(_nEvents, -1/6)
  Double_t   _sqrt2pi;
  Double_t   _2pi;       // = M_PI*2
  Double_t   _lox,_hix;
  Double_t   _loy,_hiy;
  Double_t   _xoffset;
  Double_t   _yoffset;
  Double_t   _widthScaleFactor; //allow manipulation of the bandwidth by a scale factor

  Int_t      _nEvents;
  Int_t      _BandWidthType;
  Int_t      _MirrorAtBoundary;
  Int_t      _debug;
  Int_t      _verbosedebug;
  Int_t      _vverbosedebug;

  ClassDef(Roo2DKeysPdf,0) // Non-Parametric Multi Variate KEYS PDF
};

inline void  Roo2DKeysPdf::setWidthScaleFactor(Double_t widthScaleFactor) { _widthScaleFactor = widthScaleFactor; }

#endif



