/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu         *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          Liverpool University,                            *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class Roo2DKeysPdf
    \ingroup Roofit

Two-dimensional kernel estimation PDF.

<b>This function has been superseded by the more general RooNDKeysPdf.</b>
*/

#include "Roo2DKeysPdf.h"
#include "RooRealVar.h"
#include "TTree.h"
#include "TH2.h"
#include "TFile.h"
#include "TMath.h"

using std::ostream;

ClassImp(Roo2DKeysPdf);


////////////////////////////////////////////////////////////////////////////////
/// Constructor.
/// \param[in] name
/// \param[in] title
/// \param[in] xx
/// \param[in] yy
/// \param[in] data
/// \param[in] options
/// \param[in] widthScaleFactor

Roo2DKeysPdf::Roo2DKeysPdf(const char *name, const char *title,
                       RooAbsReal& xx, RooAbsReal & yy, RooDataSet& data,  TString options, double widthScaleFactor):
  RooAbsPdf(name,title),
  x("x", "x dimension",this, xx),
  y("y", "y dimension",this, yy)
{
  setWidthScaleFactor(widthScaleFactor);
  loadDataSet(data, options);
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.
/// \param[in] other
/// \param[in] name

Roo2DKeysPdf::Roo2DKeysPdf(const Roo2DKeysPdf & other, const char* name) :
  RooAbsPdf(other,name),
  x("x", this, other.x),
  y("y", this, other.y)
{
  if(_verbosedebug) { std::cout << "Roo2DKeysPdf::Roo2DKeysPdf copy ctor" << std::endl; }

  _xMean   = other._xMean;
  _xSigma  = other._xSigma;
  _yMean   = other._yMean;
  _ySigma  = other._ySigma;
  _n       = other._n;

  _BandWidthType    = other._BandWidthType;
  _MirrorAtBoundary = other._MirrorAtBoundary;
  _widthScaleFactor = other._widthScaleFactor;

  _2pi     = other._2pi;
  _sqrt2pi = other._sqrt2pi;
  _nEvents = other._nEvents;
  _n16     = other._n16;
  _debug   = other._debug;
  _verbosedebug   = other._verbosedebug;
  _vverbosedebug  = other._vverbosedebug;

  _lox       = other._lox;
  _hix       = other._hix;
  _loy       = other._loy;
  _hiy       = other._hiy;
  _xoffset   = other._xoffset;
  _yoffset   = other._yoffset;

  _x  = new double[_nEvents];
  _y  = new double[_nEvents];
  _hx = new double[_nEvents];
  _hy = new double[_nEvents];

  //copy the data and bandwidths
  for(Int_t iEvt = 0; iEvt< _nEvents; iEvt++)
  {
    _x[iEvt]  = other._x[iEvt];
    _y[iEvt]  = other._y[iEvt];
    _hx[iEvt] = other._hx[iEvt];
    _hy[iEvt] = other._hy[iEvt];
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor.

Roo2DKeysPdf::~Roo2DKeysPdf() {
  if(_verbosedebug) { std::cout << "Roo2DKeysPdf::Roo2KeysPdf dtor" << std::endl; }
    delete[] _x;
    delete[] _hx;
    delete[] _y;
    delete[] _hy;
}


////////////////////////////////////////////////////////////////////////////////
/// Loads a new data set into the class instance.
/// Returns 1 in case of error, 0 otherwise.
/// \param[in] data
/// \param[in] options

Int_t Roo2DKeysPdf::loadDataSet(RooDataSet& data, TString options)
{
  if(_verbosedebug) { std::cout << "Roo2DKeysPdf::loadDataSet" << std::endl; }

  setOptions(options);

  if(_verbosedebug) { std::cout << "Roo2DKeysPdf::loadDataSet(RooDataSet& data, TString options)" << std::endl; }

  _2pi       = 2.0*TMath::Pi() ;   //use pi from math.h
  _sqrt2pi   = sqrt(_2pi);
  _nEvents   = (Int_t)data.numEntries();
  if(_nEvents == 0)
  {
    std::cout << "ERROR:  Roo2DKeysPdf::loadDataSet The input data set is empty.  Unable to begin generating the PDF" << std::endl;
    return 1;
  }
  _n16       =  std::pow(_nEvents, -1./6.); // = (4/[n(dim(R) + 2)])^1/(dim(R)+4); dim(R) = 2

  _lox       = x.min();
  _hix       = x.max();
  _loy       = y.min();
  _hiy       = y.max();

  _x  = new double[_nEvents];
  _y  = new double[_nEvents];
  _hx = new double[_nEvents];
  _hy = new double[_nEvents];

  double x0 = 0.0;
  double x1 = 0.0;
  double x_2 = 0.0;
  double y0 = 0.0;
  double y1 = 0.0;
  double y_2 = 0.0;

  //check that the data contain the variable we are interested in
  Int_t bad = 0;
  const RooAbsReal & xx = x.arg();
  const RooAbsReal & yy = y.arg();
  if(! static_cast<RooRealVar*>((const_cast<RooArgSet *>(data.get(0)))->find( xx.GetName() )) )
  {
    std::cout << "Roo2DKeysPdf::Roo2DKeysPdf invalid RooAbsReal name: "<<xx.GetName()<<" not in the data set" << std::endl;
    bad = 1;
  }
  if(! static_cast<RooRealVar*>((const_cast<RooArgSet *>(data.get(0)))->find( yy.GetName() )) )
  {
    std::cout << "Roo2DKeysPdf::Roo2DKeysPdf invalid RooAbsReal name: "<<yy.GetName()<<" not in the data set" << std::endl;
    bad = 1;
  }
  if(bad)
  {
    std::cout << "Roo2DKeysPdf::Roo2DKeysPdf Unable to initialize object; incompatible RooDataSet doesn't contain"<< std::endl;
    std::cout << "                           all of the RooAbsReal arguments"<< std::endl;
    return 1;
  }

  //copy the data into local arrays
  const RooArgSet * values = data.get();
  auto X = static_cast<RooRealVar const*>(values->find(xx.GetName()));
  auto Y = static_cast<RooRealVar const*>(values->find(yy.GetName()));

  for (Int_t j=0;j<_nEvents;++j)
  {
    data.get(j) ;

    _x[j] = X->getVal() ;
    _y[j] = Y->getVal() ;

    x0+=1; x1+=_x[j]; x_2+=_x[j]*_x[j];
    y0+=1; y1+=_y[j]; y_2+=_y[j]*_y[j];
  }

  //==========================================//
  //calculate the mean and sigma for the data //
  //==========================================//
  if(_nEvents == 0)
  {
    std::cout << "Roo2DKeysPdf::Roo2DKeysPdf Empty data set was used; can't generate a PDF"<< std::endl;
  }

  _xMean  = x1/x0;
  _xSigma = sqrt(x_2/_nEvents-_xMean*_xMean);

  _yMean  = y1/y0;
  _ySigma = sqrt(y_2/_nEvents-_yMean*_yMean);

  _n = double(1)/(_2pi*_nEvents*_xSigma*_ySigma);

  //calculate the PDF
  return calculateBandWidth(_BandWidthType);
}


////////////////////////////////////////////////////////////////////////////////

void Roo2DKeysPdf::setOptions(TString options)
{
  if(_verbosedebug) { std::cout << "Roo2DKeysPdf::setOptions" << std::endl; }

  options.ToLower();
  if( options.Contains("a") )   _BandWidthType    = 0;
  else                          _BandWidthType    = 1;
  if( options.Contains("n") )   _BandWidthType    = 1;
  else                          _BandWidthType    = 0;
  if( options.Contains("m") )   _MirrorAtBoundary = 1;
  else                          _MirrorAtBoundary = 0;
  if( options.Contains("d") )   _debug            = 1;
  else                          _debug            = 0;
  if( options.Contains("v") )   { _debug         = 1; _verbosedebug = 1; }
  else                            _verbosedebug  = 0;
  if( options.Contains("vv") )  { _vverbosedebug = 1; }
  else                            _vverbosedebug = 0;

  if( _debug )
  {
    std::cout << "Roo2DKeysPdf::setOptions(TString options)    options = "<< options << std::endl;
    std::cout << "\t_BandWidthType    = " << _BandWidthType    << std::endl;
    std::cout << "\t_MirrorAtBoundary = " << _MirrorAtBoundary << std::endl;
    std::cout << "\t_debug            = " << _debug            << std::endl;
    std::cout << "\t_verbosedebug     = " << _verbosedebug     << std::endl;
    std::cout << "\t_vverbosedebug    = " << _vverbosedebug    << std::endl;
  }
}


////////////////////////////////////////////////////////////////////////////////

void Roo2DKeysPdf::getOptions(void) const
{
  std::cout << "Roo2DKeysPdf::getOptions(void)" << std::endl;
  std::cout << "\t_BandWidthType                           = " << _BandWidthType    << std::endl;
  std::cout << "\t_MirrorAtBoundary                        = " << _MirrorAtBoundary << std::endl;
  std::cout << "\t_debug                                   = " << _debug            << std::endl;
  std::cout << "\t_verbosedebug                            = " << _verbosedebug     << std::endl;
  std::cout << "\t_vverbosedebug                           = " << _vverbosedebug    << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculates the kernel bandwidth for x & y and the probability look up table _p[i][j]
/// \param[in] kernel

Int_t Roo2DKeysPdf::calculateBandWidth(Int_t kernel)
{
  if(_verbosedebug) { std::cout << "Roo2DKeysPdf::calculateBandWidth(Int_t kernel)" << std::endl; }
  if(kernel != -999)
  {
    _BandWidthType = kernel;
  }

  double h = 0.0;

  double sigSum       = _xSigma*_xSigma + _ySigma*_ySigma;
  double sqrtSum      = sqrt( sigSum );
  double sigProd      = _ySigma*_xSigma;
  if(sigProd != 0.0)  h = _n16*sqrt( sigSum/sigProd );
  if(sqrtSum == 0)
  {
    std::cout << "Roo2DKeysPdf::calculateBandWidth The sqr(variance sum) == 0.0. " << " Your dataset represents a delta function."<< std::endl;
    return 1;
  }

  double hXSigma = h * _xSigma;
  double hYSigma = h * _ySigma;
  double xhmin   = hXSigma * sqrt(2.)/10;  //smallest anticipated bandwidth
  double yhmin   = hYSigma * sqrt(2.)/10;

  //////////////////////////////////////
  //calculate bandwidths from the data//
  //////////////////////////////////////
  if(_BandWidthType == 1)  //calculate a trivial bandwidth
  {
    std::cout << "Roo2DKeysPdf::calculateBandWidth Using a normal bandwidth (same for a given dimension) based on"<< std::endl;
    std::cout << "                                 h_j = n^{-1/6}*sigma_j for the j^th dimension and n events * "<<_widthScaleFactor<< std::endl;
    double hxGaussian = _n16 * _xSigma * _widthScaleFactor;
    double hyGaussian = _n16 * _ySigma * _widthScaleFactor;
    for(Int_t j=0;j<_nEvents;++j)
    {
      _hx[j] = hxGaussian;
      _hy[j] = hyGaussian;
      if(_hx[j]<xhmin) _hx[j] = xhmin;
      if(_hy[j]<yhmin) _hy[j] = yhmin;
     }
  }
  else //use an adaptive bandwidth to reduce the dependence on global data distribution
  {
    std::cout << "Roo2DKeysPdf::calculateBandWidth Using an adaptive bandwidth (in general different for all events) [default]"<< std::endl;
    std::cout << "                                 scaled by a factor of "<<_widthScaleFactor<< std::endl;
    double xnorm   = h * std::pow(_xSigma/sqrtSum, 1.5) * _widthScaleFactor;
    double ynorm   = h * std::pow(_ySigma/sqrtSum, 1.5) * _widthScaleFactor;
    for(Int_t j=0;j<_nEvents;++j)
    {
      double f_ti =  std::pow( g(_x[j], _x, hXSigma, _y[j], _y, hYSigma), -0.25 ) ;
      _hx[j] = xnorm * f_ti;
      _hy[j] = ynorm * f_ti;
      if(_hx[j]<xhmin) _hx[j] = xhmin;
      if(_hy[j]<yhmin) _hy[j] = yhmin;
    }
  }

  return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluates the kernel estimation for x,y, interpolating between the points if necessary
///
/// Uses the caching intrinsic in RFC to bypass the grid and remove
/// the grid and extrapolation approximation in the kernel estimation method
/// implementation.

double Roo2DKeysPdf::evaluate() const
{
  if(_vverbosedebug) { std::cout << "Roo2DKeysPdf::evaluate()" << std::endl; }
  return evaluateFull(x,y);
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluates the sum of the product of the 2D kernels
/// for use in calculating the fixed kernel estimate, f,
/// given the bandwidths _hx[j] and _hy[j].
///
/// _n is calculated once in the constructor.
/// \param[in] thisX
/// \param[in] thisY

double Roo2DKeysPdf::evaluateFull(double thisX, double thisY) const
{
  if( _vverbosedebug ) { std::cout << "Roo2DKeysPdf::evaluateFull()" << std::endl; }

  double f=0.0;

  double rx2;
  double ry2;
  double zx;
  double zy;
  if( _MirrorAtBoundary )
  {
    for (Int_t j = 0; j < _nEvents; ++j)
    {
      rx2 = 0.0; ry2 = 0.0; zx = 0.0; zy = 0.0;
      if(_hx[j] != 0.0) rx2 = (thisX - _x[j])/_hx[j];
      if(_hy[j] != 0.0) ry2 = (thisY - _y[j])/_hy[j];

      if(_hx[j] != 0.0) zx = exp(-0.5*rx2*rx2)/_hx[j];
      if(_hy[j] != 0.0) zy = exp(-0.5*ry2*ry2)/_hy[j];

      zx += highBoundaryCorrection(thisX, _hx[j], x.max(), _x[j])
    +   lowBoundaryCorrection(thisX, _hx[j], x.min(), _x[j]);
      zy += highBoundaryCorrection(thisY, _hy[j], y.max(), _y[j])
    +   lowBoundaryCorrection(thisY, _hy[j], y.min(), _y[j]);
      f += zy * zx;
      //      f += _n * zy * zx; // ooops this is a normalisation factor :(
    }
  }
  else
  {
    for (Int_t j = 0; j < _nEvents; ++j)
    {
      rx2 = 0.0; ry2 = 0.0; zx = 0.0; zy = 0.0;
      if(_hx[j] != 0.0) rx2 = (thisX - _x[j])/_hx[j];
      if(_hy[j] != 0.0) ry2 = (thisY - _y[j])/_hy[j];

      if(_hx[j] != 0.0) zx = exp(-0.5*rx2*rx2)/_hx[j];
      if(_hy[j] != 0.0) zy = exp(-0.5*ry2*ry2)/_hy[j];
      f += zy * zx;
      //      f += _n * zy * zx; // ooops this is a normalisation factor  :(
    }
  }
  return f;
}


////////////////////////////////////////////////////////////////////////////////
/// Apply the mirror at boundary correction to a dimension given the space position to evaluate
/// at (thisVar), the bandwidth at this position (thisH), the boundary (high/low) and the
/// value of the data kernel that this correction is being applied to tVar (i.e. the _x[ix] etc.).
/// \param[in] thisVar
/// \param[in] thisH
/// \param[in] high
/// \param[in] tVar

double Roo2DKeysPdf::highBoundaryCorrection(double thisVar, double thisH, double high, double tVar) const
{
  if(_vverbosedebug) { std::cout << "Roo2DKeysPdf::highBoundaryCorrection" << std::endl; }

  if(thisH == 0.0) return 0.0;
  double correction = (thisVar + tVar - 2.0* high )/thisH;
  return exp(-0.5*correction*correction)/thisH;
}


////////////////////////////////////////////////////////////////////////////////

double Roo2DKeysPdf::lowBoundaryCorrection(double thisVar, double thisH, double low, double tVar) const
{
  if(_vverbosedebug) { std::cout << "Roo2DKeysPdf::lowBoundaryCorrection" << std::endl; }

  if(thisH == 0.0) return 0.0;
  double correction = (thisVar + tVar - 2.0* low )/thisH;
  return exp(-0.5*correction*correction)/thisH;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculates f(t_i) for the bandwidths.
/// \f$ g = 1/(N_{evt} * \sigma_j * \sqrt{2\pi})*\sum_{all evts}{\prod d K[ \exp({-(xd - ti)/\sigma_{j}d^2}) ]}\f$
/// \param[in] varMean1
/// \param[in] _var1
/// \param[in] sigma1
/// \param[in] varMean2
/// \param[in] _var2
/// \param[in] sigma2

double Roo2DKeysPdf::g(double varMean1, double * _var1, double sigma1, double varMean2, double * _var2, double sigma2) const
{
  if((_nEvents == 0.0) || (sigma1 == 0.0) || (sigma2 == 0)) return 0.0;

  double c1 = -1.0/(2.0*sigma1*sigma1);
  double c2 = -1.0/(2.0*sigma2*sigma2);
  double d  = 4.0*c1*c2  /(_sqrt2pi*_nEvents);
  double z  = 0.0;

  for (Int_t i = 0; i < _nEvents; ++i)
  {
    double r1 =  _var1[i] - varMean1;
    double r2 =  _var2[i] - varMean2;
    z          += exp( c1 * r1*r1 ) * exp( c2 * r2*r2 );
  }
  z = z*d;
  return z;
}


////////////////////////////////////////////////////////////////////////////////

Int_t Roo2DKeysPdf::getBandWidthType() const
{
  if(_BandWidthType == 1)  std::cout << "The Bandwidth Type selected is Trivial" << std::endl;
  else                     std::cout << "The Bandwidth Type selected is Adaptive" << std::endl;

  return _BandWidthType;
}


////////////////////////////////////////////////////////////////////////////////

double Roo2DKeysPdf::getMean(const char * axis) const
{
  if(!strcmp(axis,x.GetName()) || !strcmp(axis,"x") || !strcmp(axis,"X"))      return _xMean;
  else if(!strcmp(axis,y.GetName()) || !strcmp(axis,"y") || !strcmp(axis,"Y")) return _yMean;
  else
  {
    std::cout << "Roo2DKeysPdf::getMean unknown axis "<<axis<< std::endl;
  }
  return 0.0;
}


////////////////////////////////////////////////////////////////////////////////

double Roo2DKeysPdf::getSigma(const char * axis) const
{
  if(!strcmp(axis,x.GetName()) || !strcmp(axis,"x") || !strcmp(axis,"X"))      return _xSigma;
  else if(!strcmp(axis,y.GetName()) || !strcmp(axis,"y") || !strcmp(axis,"Y")) return _ySigma;
  else
  {
    std::cout << "Roo2DKeysPdf::getSigma unknown axis "<<axis<< std::endl;
  }
  return 0.0;
}



////////////////////////////////////////////////////////////////////////////////

void Roo2DKeysPdf::writeToFile(char * outputFile, const char * name) const
{
  TString histName = name;
  histName        += "_hist";
  TString nName    = name;
  nName           += "_Ntuple";
  writeHistToFile( outputFile,    histName);
  writeNTupleToFile( outputFile,  nName);
}


////////////////////////////////////////////////////////////////////////////////
/// Plots the PDF as a histogram and saves it to a file, so that it can be loaded in
/// as a Roo2DHist PDF in the future to save on calculation time.
/// \param[in] outputFile Name of the file where to store the PDF
/// \param[in] histName PDF histogram name

void Roo2DKeysPdf::writeHistToFile(char * outputFile, const char * histName) const
{
  std::cout << "Roo2DKeysPdf::writeHistToFile This member function is temporarily disabled" << std::endl;
  //make sure that any existing file is not over written
  std::unique_ptr<TFile> file{TFile::Open(outputFile, "UPDATE")};
  if (!file)
  {
    std::cout << "Roo2DKeysPdf::writeHistToFile unable to open file "<< outputFile << std::endl;
    return;
  }


  const RooAbsReal & xx = x.arg();
  const RooAbsReal & yy = y.arg();
  RooArgSet values( RooArgList( xx, yy ));
  RooRealVar * xArg = (static_cast<RooRealVar*>(values.find(xx.GetName())) ) ;
  RooRealVar * yArg = (static_cast<RooRealVar*>(values.find(yy.GetName())) ) ;

  TH2F * hist = (TH2F*)xArg->createHistogram("hist", *yArg);
  hist = static_cast<TH2F*>(this->fillHistogram(hist, RooArgList(*xArg, *yArg) ));
  hist->SetName(histName);

  file->Write();
  file->Close();
}


////////////////////////////////////////////////////////////////////////////////
/// Saves the data and calculated bandwidths to a file,
/// as a record of what produced the PDF and to give a reduced
/// data set in order to facilitate re-calculation in the future.
/// \param[in] outputFile Name of the file where to store the data
/// \param[in] name Name of the tree which will contain the data

void Roo2DKeysPdf::writeNTupleToFile(char * outputFile, const char * name) const
{
  TFile * file = nullptr;

  //make sure that any existing file is not over written
  file = new TFile(outputFile, "UPDATE");
  if (!file)
  {
    std::cout << "Roo2DKeysPdf::writeNTupleToFile unable to open file "<< outputFile << std::endl;
    return;
  }
  RooAbsReal & xArg = (RooAbsReal&)x.arg();
  RooAbsReal & yArg = (RooAbsReal&)y.arg();

  double theX;
  double theY;
  double hx;
  TString label = name;
  label += " the source data for 2D Keys PDF";
  TTree * _theTree =  new TTree(name, label);
  if(!_theTree) { std::cout << "Unable to get a TTree for output" << std::endl; return; }
  _theTree->SetAutoSave(1000000000);  // autosave when 1 Gbyte written

  //name the TBranches the same as the RooAbsReal's
  const char * xname = xArg.GetName();
  const char * yname = yArg.GetName();
  if (!strcmp(xname,"")) xname = "x";
  if (!strcmp(yname,"")) yname = "y";

  _theTree->Branch(xname, &theX, " x/D");
  _theTree->Branch(yname, &theY, " y/D");
  _theTree->Branch("hx",  &hx,   " hx/D");
  _theTree->Branch("hy",  &hx,  " hy/D");

  for(Int_t iEvt = 0; iEvt < _nEvents; iEvt++)
  {
    theX = _x[iEvt];
    theY = _y[iEvt];
    hx   = _hx[iEvt];
    hx   = _hy[iEvt];
    _theTree->Fill();
  }
  file->Write();
  file->Close();
}


////////////////////////////////////////////////////////////////////////////////
/// Prints out _p[_nPoints][_nPoints] indicating the domain limits.
/// \param[out] out Output stream where to print

void Roo2DKeysPdf::PrintInfo(ostream & out) const
{
  out << "Roo2DKeysPDF instance domain information:"<< std::endl;
  out << "\tX_min          = " << _lox << std::endl;
  out << "\tX_max          = " << _hix << std::endl;
  out << "\tY_min          = " << _loy << std::endl;
  out << "\tY_max          = " << _hiy << std::endl;

  out << "Data information:" << std::endl;
  out << "\t<x>             = " << _xMean << std::endl;
  out << "\tsigma(x)       = " << _xSigma << std::endl;
  out << "\t<y>             = " << _yMean << std::endl;
  out << "\tsigma(y)       = " << _ySigma << std::endl;

  out << "END of info for Roo2DKeys pdf instance"<< std::endl;
}
