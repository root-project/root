/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: Roo2DKeysPdf.cc,v 1.5 2001/09/24 23:08:54 verkerke Exp $
 * Authors:
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 *
 * History:
 *   25-Jul-2001 AB Created 2D KEYS version based on the 1D template of RooKeysPdf
 *                  by Gerhard Raven.
 *   25-Aug-2001 AB Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 2001, Liverpool University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

#include "BaBar/BaBar.hh"
#include "RooFitModels/Roo2DKeysPdf.hh"
#include "RooFitCore/RooRealVar.hh"
#include "TH2.h"
#include "TFile.h"
#include "TBranch.h"

#include <math.h>

ClassImp(Roo2DKeysPdf)

Roo2DKeysPdf::Roo2DKeysPdf(const char *name, const char *title,
                       RooAbsReal& xx, RooAbsReal & yy, RooDataSet& data,  TString options):
  RooAbsPdf(name,title),
  x("x", "x dimension",this, xx),
  y("y", "y dimension",this, yy)
{
  loadDataSet(data, options);
}

Roo2DKeysPdf::Roo2DKeysPdf(const Roo2DKeysPdf & other, const char* name) :
  RooAbsPdf(other,name),
  x("x", this, other.x),
  y("y", this, other.y)
{
  _xMean   = other._xMean;
  _xSigma  = other._xSigma;
  _yMean   = other._yMean;
  _ySigma  = other._ySigma;
  _n       = other._n;

  _BandWidthType    = other._BandWidthType;
  _MirrorAtBoundary = other._MirrorAtBoundary;

  _2pi     = other._2pi;
  _sqrt2pi = other._sqrt2pi;
  _nEvents = other._nEvents;
  _n16     = other._n16;

  _lox       = other._lox;
  _hix       = other._hix;
  _loy       = other._loy;
  _hiy       = other._hiy;
  _xbinWidth = other._xbinWidth;
  _ybinWidth = other._ybinWidth;

  _x  = new Double_t[_nEvents];
  _y  = new Double_t[_nEvents];
  _hx = new Double_t[_nEvents];
  _hy = new Double_t[_nEvents];

  //copy the data and bandwidths
  for(Int_t iEvt = 0; iEvt< _nEvents; iEvt++)
  {
    _x[iEvt]  = other._x[iEvt];
    _y[iEvt]  = other._y[iEvt];
    _hx[iEvt] = other._hx[iEvt];
    _hy[iEvt] = other._hy[iEvt];
  }

  //copy the PDF LUT
  for (Int_t _ix=0;_ix<_nPoints;++_ix) 
  {
    for (Int_t _iy=0;_iy<_nPoints;++_iy) 
    {
      _p[_ix][_iy] = other._p[_ix][_iy];
    }
  }
}

Roo2DKeysPdf::~Roo2DKeysPdf() {
    delete[] _x;
    delete[] _hx;
    delete[] _y;
    delete[] _hy;
}

Int_t Roo2DKeysPdf::loadDataSet(RooDataSet& data, TString options)
{
  _2pi       = 2.0*M_PI;   //use pi from math.h
  _sqrt2pi   = sqrt(_2pi);
  _nEvents   = (Int_t)data.numEntries();
  _n16       =  pow (_nEvents, -0.166666666); // = (4/[n(dim(R) + 2)])^1/(dim(R)+4); dim(R) = 2

  _lox       = x.min();
  _hix       = x.max();
  _loy       = y.min();
  _hiy       = y.max();
  _xbinWidth = (_hix-_lox)/(_nPoints-1);
  _ybinWidth = (_hiy-_loy)/(_nPoints-1);

  _x  = new Double_t[_nEvents];
  _y  = new Double_t[_nEvents];
  _hx = new Double_t[_nEvents];
  _hy = new Double_t[_nEvents];

  Double_t x0 = 0.0;
  Double_t x1 = 0.0;
  Double_t x2 = 0.0;
  Double_t y0 = 0.0;
  Double_t y1 = 0.0;
  Double_t y2 = 0.0;

  //check that the data contain the variable we are interested in  
  Int_t bad = 0;
  const RooAbsReal & xx = x.arg();
  const RooAbsReal & yy = y.arg();
  if(! (RooRealVar*)( (RooArgSet*)data.get(0) )->find( xx.GetName() ) )
  {
    cout << "Roo2DKeysPdf::Roo2DKeysPdf invalid RooAbsReal name: "<<xx.GetName()<<" not in the data set" <<endl;
    bad = 1;
  }
  if(! (RooRealVar*)( (RooArgSet*)data.get(0) )->find( yy.GetName() ) )
  {
    cout << "Roo2DKeysPdf::Roo2DKeysPdf invalid RooAbsReal name: "<<yy.GetName()<<" not in the data set" << endl;
    bad = 1;
  }
  if(bad)
  {
    cout << "Roo2DKeysPdf::Roo2DKeysPdf Unable to initilize object; incompatible RooDataSet doesn't contain"<<endl;
    cout << "                           all of the RooAbsReal arguments"<<endl;
    return 1;
  }

  //copy the data into local arrays
  const RooArgSet * values = data.get();
  const RooRealVar* X = ((RooRealVar*)(values->find(xx.GetName())) ) ;
  const RooRealVar* Y = ((RooRealVar*)(values->find(yy.GetName())) ) ;

  for (Int_t j=0;j<_nEvents;++j) 
  {
    data.get(j) ;

    _x[j] = X->getVal() ;
    _y[j] = Y->getVal() ;

    x0+=1; x1+=_x[j]; x2+=_x[j]*_x[j];
    y0+=1; y1+=_y[j]; y2+=_y[j]*_y[j];
  }

  //==========================================//
  //calculate the mean and sigma for the data //
  //==========================================//
  if(_nEvents == 0) 
  {
    cout << "Roo2DKeysPdf::Roo2DKeysPdf Empty data set was used; can't generate a PDF"<<endl;
  }

  _xMean  = x1/x0;
  _xSigma = sqrt(x2/_nEvents-_xMean*_xMean);
  
  _yMean  = y1/y0;
  _ySigma = sqrt(y2/_nEvents-_yMean*_yMean);

  _n=Double_t(1)/(_2pi*_nEvents*_xSigma*_ySigma);

  setOptions(options);

  //calculate the PDF
  return calculateBandWidth(_BandWidthType);
}

void Roo2DKeysPdf::setOptions(TString options)
{
  options.ToLower();
  if(options.Contains("a"))      _BandWidthType    = 0;
  else                           _BandWidthType    = 1;
  if(options.Contains("m"))      _MirrorAtBoundary = 1;
  else                           _MirrorAtBoundary = 0;
}

//=====================================================//
// calculate the kernal bandwith for x & y             //
// & Calculate the probability look up table _p[i][j]  //
//=====================================================//
Int_t Roo2DKeysPdf::calculateBandWidth(Int_t kernel)
{
  if(kernel != -999)
  {
    _BandWidthType = kernel;
  }

  Double_t h = 0.0;

  Double_t sigSum       = _xSigma*_xSigma + _ySigma*_ySigma;
  Double_t sqrtSum      = sqrt( sigSum );
  Double_t sigProd      = _ySigma*_xSigma;
  if(sigProd != 0.0)  h = _n16*sqrt( sigSum/sigProd );
  if(sqrtSum == 0)
  { 
    cout << "Roo2DKeysPdf::calculateBandWidth The sqr(variance sum) == 0.0. " << " Your dataset represents a delta function."<<endl;
    return 1;
  }

  Double_t hXSigma = h * _xSigma; 
  Double_t hYSigma = h * _ySigma; 
  Double_t xhmin   = hXSigma * sqrt(2)/10;  //smallest anticipated bandwidth
  Double_t yhmin   = hYSigma * sqrt(2)/10;
  //////////////////////////////////////
  //calculate bandwidths from the data//
  //////////////////////////////////////
  if(_BandWidthType == 1)  //calculate a trivial bandwith
  {
    cout << "Roo2DKeysPdf::calculateBandWidth Using a normal bandwith (same for a given dimension)"<<endl;
    cout << "based on h_j = n^{-1/6}*sigma_j for the j^th dimension and n events"<<endl;
    Double_t hxGaussian = _n16*_xSigma;
    Double_t hyGaussian = _n16*_ySigma;
    for(Int_t j=0;j<_nEvents;++j) 
    {
      _hx[j] = hxGaussian;
      _hy[j] = hyGaussian;
      if(_hx[j]<xhmin) _hx[j] = xhmin;
      if(_hy[j]<yhmin) _hy[j] = yhmin;
     }
  }
  else //use an adaptive bandwith to reduce the dependance on global data distribution
  {
    cout << "Roo2DKeysPdf::calculateBandWidth Using an adaptive bandwith (in general different for all events) [default]"<<endl;
    Double_t xnorm   = h * pow(_xSigma/sqrtSum, 1.5);
    Double_t ynorm   = h * pow(_ySigma/sqrtSum, 1.5);
    for(Int_t j=0;j<_nEvents;++j) 
    {
      Double_t f_ti =  pow ( g(_x[j], _x, hXSigma, _y[j], _y, hYSigma), -0.25 ) ;
      _hx[j] = xnorm * f_ti;
      _hy[j] = ynorm * f_ti;
      if(_hx[j]<xhmin) _hx[j] = xhmin;
      if(_hy[j]<yhmin) _hy[j] = yhmin;
    }
  }

  ////////////////////////
  //build the PDF table //
  ////////////////////////
  Double_t thisX, thisY;
  for (Int_t _ix=0;_ix<_nPoints;++_ix) 
  {
    for (Int_t _iy=0;_iy<_nPoints;++_iy) 
    {
      thisX        = _lox + _ix * _xbinWidth;
      thisY        = _loy + _iy * _ybinWidth;
      _p[_ix][_iy] = evaluateFull(thisX, thisY);
    }
  }
  return 0;
}

//=======================================================================================//
// evaluate the kernal estimation for x,y, interpolating between the points if necessary //
//=======================================================================================//
Double_t Roo2DKeysPdf::evaluate() const
{
  Int_t ix = (Int_t)((x-_lox)/_xbinWidth);
  Int_t iy = (Int_t)((y-_loy)/_ybinWidth);

  //check x,y is in the valid domain
  if (ix<0) 
  {
    cerr << "got point below lower bound:" << x << " < " << _lox << " -- performing linear extrapolation..." << endl;
    ix=0;
  }
  if (ix>_nPoints-1) 
  {
    cerr << "got point above upper bound:" << x << " > " << _hix << " -- performing linear extrapolation..." << endl;
    ix=_nPoints-1;
  }
  if (iy<0) 
  {
    cerr << "got point below lower bound:"  << y << " < " << _loy << " -- performing linear extrapolation..." << endl;
    iy=0;
  }
  if (iy>_nPoints-1) 
  {
    cerr << "got point above upper bound:"  << y << " > " << _hiy << " -- performing linear extrapolation..." << endl;
    iy=_nPoints-1;
  }
  Double_t dfdx = (_p[ix+1][iy] - _p[ix][iy])/_xbinWidth;
  Double_t dfdy = (_p[ix][iy+1] - _p[ix][iy])/_ybinWidth;

  Double_t dx = (x-( _lox + (Double_t)ix * _xbinWidth));
  Double_t dy = (y-( _loy + (Double_t)iy * _ybinWidth));

  return ( _p[ix][iy] + dx*dfdx + dy*dfdy );
}

/////////////////////////////////////////////////////////
// Evaluate the sum of the product of the 2D kernels   //
// for use in calculating the fixed kernel estimate, f //
// given the bandwiths _hx[j] and _hy[j]               //
/////////////////////////////////////////////////////////
// _n is calculated once in the constructor
Double_t Roo2DKeysPdf::evaluateFull(Double_t thisX, Double_t thisY)
{
  Double_t f=0;

  Double_t rx2, ry2, zx, zy;
  if( _MirrorAtBoundary )
  {
    for (Int_t j = 0; j < _nEvents; ++j) 
    {
      rx2 = 0.0; ry2 = 0.0; zx = 0.0; zy = 0.0;
      if(_hx[j] != 0.0) rx2 = (thisX - _x[j])/_hx[j];
      if(_hy[j] != 0.0) ry2 = (thisY - _y[j])/_hy[j];

      if(_hx[j] != 0.0) zx = exp(-0.5*rx2*rx2)/_hx[j];
      if(_hy[j] != 0.0) zy = exp(-0.5*ry2*ry2)/_hy[j];
      zx += xBoundaryCorrection(thisX, j);
      zy += yBoundaryCorrection(thisY, j);
      f += _n * zy * zx;
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
      f += _n * zy * zx;
    }
  }
  return f;
}

Double_t Roo2DKeysPdf::xBoundaryCorrection(Double_t thisX, Int_t ix)
{
  if(_hx[ix] == 0.0) return 0.0;
  Double_t correction = 0.0;
  Double_t nSigmaLow  = (thisX - _lox)/_hx[ix];
  if((nSigmaLow < ROO2DKEYSPDF_NSIGMAMIROOR) && (nSigmaLow>0.0))
  {
    correction = (thisX + _x[ix] - 2.0* x.min() )/_hx[ix];
    correction = exp(-0.5*correction*correction)/_hx[ix];
    return correction;
  }
  Double_t nSigmaHigh  = (_hix - thisX)/_hx[ix];
  if((nSigmaHigh < ROO2DKEYSPDF_NSIGMAMIROOR) && (nSigmaHigh>0.0))
  {
    correction = (thisX - (2.0*x.max() - _x[ix]) )/_hx[ix];
    correction = exp(-0.5*correction*correction)/_hx[ix];
    return correction;
  }
  return correction;
}

Double_t Roo2DKeysPdf::yBoundaryCorrection(Double_t thisY, Int_t iy)
{
  if(_hy[iy] == 0.0) return 0.0;
  Double_t correction = 0.0;
  Double_t nSigmaLow   = (thisY - _loy)/_hy[iy];
  if((nSigmaLow < ROO2DKEYSPDF_NSIGMAMIROOR) && (nSigmaLow > 0.0))
  {
    correction = (thisY +  _y[iy] -  2.0*y.min() )/_hy[iy];
    correction = exp(-0.5*correction*correction)/_hy[iy];
    return correction;
  }
  Double_t nSigmaHigh  = (_hiy - thisY)/_hy[iy];
  if((nSigmaHigh < ROO2DKEYSPDF_NSIGMAMIROOR) && (nSigmaHigh > 0.0))
  {
    correction = (thisY - (2.0*y.max() - _y[iy]) )/_hy[iy];
    correction = exp(-0.5*correction*correction)/_hy[iy];
    return correction;
  }
  return correction;
}

//==========================================================================================//
// calculate f(t_i) for the bandwidths                                                      //
//                                                                                          //
// g = 1/(Nevt * sigma_j * sqrt2pi)*sum_{all evts}{prod d K[ exp{-(xd - ti)/sigma_jd^2} ]}  //
//                                                                                          //
//==========================================================================================//
Double_t Roo2DKeysPdf::g(Double_t varMean1, Double_t * _var1, Double_t sigma1, Double_t varMean2, Double_t * _var2, Double_t sigma2) 
{
  if((_nEvents == 0.0) || (sigma1 == 0.0) || (sigma2 == 0)) return 0.0;

  Double_t c1 = -1.0/(2.0*sigma1*sigma1);
  Double_t c2 = -1.0/(2.0*sigma2*sigma2);
  Double_t d  = 4.0*c1*c2  /(_sqrt2pi*_nEvents);
  Double_t z  = 0.0;

  for (Int_t i = 0; i < _nEvents; ++i) 
  {
    Double_t r1 =  _var1[i] - varMean1; 
    Double_t r2 =  _var2[i] - varMean2; 
    z          += exp( c1 * r1*r1 ) * exp( c2 * r2*r2 );
  }
  z = z*d;
  return z;
}

Int_t Roo2DKeysPdf::getBandWidthType()
{
  if(_BandWidthType == 1)  cout << "The Bandwidth Type selected is Trivial" << endl;
  else                     cout << "The Bandwidth Type selected is Adaptive" << endl;

  return _BandWidthType;
}

Double_t Roo2DKeysPdf::getMean(const char * axis)
{
  if((axis == x.GetName()) || (axis == "x") || (axis == "X"))      return _xMean;
  else if((axis == y.GetName()) || (axis == "y") || (axis == "Y")) return _yMean;
  else 
  {
    cout << "Roo2DKeysPdf::getMean unknown axis "<<axis<<endl;
  }
  return 0.0;
}

Double_t Roo2DKeysPdf::getSigma(const char * axis)
{
  if((axis == x.GetName()) || (axis == "x") || (axis == "X"))      return _xSigma;
  else if((axis == y.GetName()) || (axis == "y") || (axis == "Y")) return _ySigma;
  else 
  {
    cout << "Roo2DKeysPdf::getSigma unknown axis "<<axis<<endl;
  }
  return 0.0;
}


void Roo2DKeysPdf::writeToFile(char * outputFile, const char * name)
{
  TString histName = name;
  histName        += "_hist";
  TString nName    = name;
  nName           += "_Ntuple";
  writeHistToFile( outputFile,    histName);
  writeNTupleToFile( outputFile,  nName);
}

// plot the PDf as a histogram and save to file
// so that it can be loaded in as a Roo2DHist Pdf in the future to 
// save on calculation time
void Roo2DKeysPdf::writeHistToFile(char * outputFile, const char * histName)
{
  TFile * file = 0;

  //make sure that any existing file is not over written
  file = new TFile(outputFile, "UPDATE"); 
  if (!file)
  {
    cout << "Roo2DKeysPdf::writeHistToFile unable to open file "<< outputFile <<endl;
    return;
  }

  RooAbsReal & xArg = (RooAbsReal&)x.arg();
  RooAbsReal & yArg = (RooAbsReal&)y.arg();

  // make the histogram with a normalization of 1
  // WVE temp disabled TH2F * hist = this->plot(xArg, yArg, 1.0, _nPoints, _nPoints);
  // hist->SetName(histName);
  file->Write();
  file->Close();
}

// save the data and calculated bandwidths to file
// as a record of what produced the PDF and to give a reduced
// data set in order to facilitate re-calculation in the future
void Roo2DKeysPdf::writeNTupleToFile(char * outputFile, const char * name)
{
  TFile * file = 0;

  //make sure that any existing file is not over written
  file = new TFile(outputFile, "UPDATE"); 
  if (!file)
  {
    cout << "Roo2DKeysPdf::writeNTupleToFile unable to open file "<< outputFile <<endl;
    return;
  }
  RooAbsReal & xArg = (RooAbsReal&)x.arg();
  RooAbsReal & yArg = (RooAbsReal&)y.arg();

  Double_t theX, theY, hx, hy;
  TString label = name;
  label += " the source data for 2D Keys PDF";
  TTree * _theTree =  new TTree(name, label);
  if(!_theTree) { cout << "Unable to get a TTree for output" << endl; return; }
  _theTree->SetAutoSave(1000000000);  // autosave when 1 Gbyte written

  //name the TBranches the same as the RooAbsReal's
  const char * xname = xArg.GetName();
  const char * yname = yArg.GetName();
  if(xname == "") xname = "x";
  if(yname == "") yname = "y";

  TBranch * b_x  = _theTree->Branch(xname, &theX, " x/D");
  TBranch * b_y  = _theTree->Branch(yname, &theY, " y/D");
  TBranch * b_hx = _theTree->Branch("hx",  &hx,   " hx/D");
  TBranch * b_hy = _theTree->Branch("hy",   &hx,  " hy/D");

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

