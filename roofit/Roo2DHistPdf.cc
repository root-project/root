/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: Roo2DHistPdf.cc,v 1.3 2001/09/08 02:29:49 bevan Exp $
 * Authors:
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 *
 * History:
 *   08-Aug-2001 AB Created Roo2DHistPdf
 *   25-Aug-2001 AB Ported to RooFitCore/RooFitModels
 *
 * Copyright (C) 2001, Liverpool University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

#include "BaBar/BaBar.hh"
#include "RooFitModels/Roo2DHistPdf.hh"

#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsReal.hh"

#include <iostream.h>
#include <math.h>

ClassImp(Roo2DHistPdf)

Roo2DHistPdf::Roo2DHistPdf(const char * name, const char *title,
                           RooAbsReal& xx, RooAbsReal &yy, const char * rootFile, const char * histName, TString opt):
  RooAbsPdf(name,title),
  x("x", "x dimension",this, xx),
  y("y", "y dimension",this, yy)
{
  _file = new TFile(rootFile);
  if(!_file)
  {
    cout << "Roo2DHistPdf::Roo2DHistPdf Unable to open file "<< rootFile <<endl;
    return;
  }

  _hist = (TH2F*)_file->Get(histName);
  if(!_hist)
  {
    cout << "Roo2DHistPdf::Roo2DHistPdf Unable to get histogram "<< rootFile << " from file " << rootFile<<endl;
    return;
  }

  loadNewHist(_hist, opt);
}

Roo2DHistPdf::Roo2DHistPdf(const char *name, const char *title,
                       RooAbsReal& xx, RooAbsReal & yy, RooDataSet& data, TString opt):
  RooAbsPdf(name,title),
  x("x", "x dimension",this, xx),
  y("y", "y dimension",this, yy)
{
  TH2F * hist = (TH2F*)data.createHistogram(xx,yy);
  loadNewHist(_hist, opt);
}

Roo2DHistPdf::Roo2DHistPdf(const char *name, const char *title,
                       RooAbsReal& xx, RooAbsReal & yy, TH2F * hist, TString opt):
  RooAbsPdf(name,title),
  x("x", "x dimension",this, xx),
  y("y", "y dimension",this, yy)
{
  loadNewHist(_hist, opt);
}

Roo2DHistPdf::Roo2DHistPdf(const char *name, const char *title,
                       RooAbsReal& xx, RooAbsReal & yy, TH2D * hist, TString opt):
  RooAbsPdf(name,title),
  x("x", "x dimension",this, xx),
  y("y", "y dimension",this, yy)
{
  _iWantToExtrapolate = 0;
  loadNewHist( (TH2F*)_hist, opt);
}

// dont take on the file and histogram source from the original
// PDF in case the clone goes out of scope at a different time
// to the original
Roo2DHistPdf::Roo2DHistPdf(const Roo2DHistPdf & other, const char* name) :
  RooAbsPdf(other,name),
  x("x", this, other.x),
  y("y", this, other.y)
{
  _nPointsx = other._nPointsx;
  _nPointsy = other._nPointsy;

  _iWantToSmooth      = other._iWantToSmooth;
  _iWantToExtrapolate = other._iWantToExtrapolate;

  //set boundary values
  _lox = x.min();
  _hix = x.max();
  _xbinWidth = (_hix-_lox)/(_nPointsx);

  _loy = y.min();
  _hiy = y.max();
  _ybinWidth = (_hiy-_loy)/(_nPointsy);

  //read in the table of values
  for(Int_t i = 0; i < _nPointsx; i++)
  {
    for(Int_t j = 0; j < _nPointsy; j++)
    {
      _p[i][j] = other._p[i][j];
    }
  }
}

Roo2DHistPdf::~Roo2DHistPdf() 
{
  if(_file) _file->Close();
}

Int_t Roo2DHistPdf::loadNewHist(TH2F * aNewHist, TString options)
{
  SetOptions(options);
  return GetProbability(aNewHist);
}

// 'e' extrapolation between bins to try and smooth out the logo shape
// 's' smooth the histogram: to be initiated
void Roo2DHistPdf::SetOptions(TString opt)
{
  opt.ToLower();
  if( opt.Contains("e") ) { _iWantToExtrapolate = 1; }
  else _iWantToExtrapolate = 0;
  if( opt.Contains("s") )
  {
    _iWantToSmooth = 1;
  }
  else _iWantToSmooth = 0;
}

//====================//
//calculate the LUT   //
//====================//
Int_t Roo2DHistPdf::GetProbability(TH2F * theHist)
{
  if(theHist == 0)
  {
    cout <<"Roo2DHistPdf::GetProbability Trying to use an null histogram"<<endl;
    return 1;
  }
  Int_t nx = theHist->GetNbinsX();
  Int_t ny = theHist->GetNbinsY();

  if(nx> _nPoints) 
  {
    cout << "WARNING: Truncating histogram as nx>_nPoints (Roo2DHistPdf):"<<nx<<" > "<<_nPoints<<endl;  
    nx = _nPoints;
  }
  if(ny> _nPoints) 
  {
    cout << "WARNING: Truncating histogram as ny>_nPoints (Roo2DHistPdf):"<<ny<<" > "<<_nPoints<<endl;  
    ny = _nPoints;
  }
  _nPointsx = nx;
  _nPointsy = ny;

  //set boundary values
  _lox = x.min();
  _hix = x.max();
  _xbinWidth = (_hix-_lox)/(_nPointsx);

  _loy = y.min();
  _hiy = y.max();
  _ybinWidth = (_hiy-_loy)/(_nPointsy);

  if( (_xbinWidth == 0.0) || (_ybinWidth == 0.0))
  {
    cout << "Roo2DHistPdf::GetProbability histogram bin width = 0" <<endl;
    return 1;
  }
  //read in the table of values
  for(Int_t i = 1; i <= nx; i++)
  {
    for(Int_t j = 1; j <= ny; j++)
    {
      _p[i-1][j-1] = theHist->GetBinContent(i, j);
      if(_p[i-1][j-1] < 0.0)
      {
        cout << "Roo2DHistPdf::GetProbability histogram bin content: "<< _p[i-1][j-1] <<" < 0; setting to probability to 0.0"<<endl;
        _p[i-1][j-1] = 0.0;
      }
    }
  }

  return 0;
}

//=======================================================================================//
// evaluate the kernal estimation for x,y, interpolating between the points if necessary //
//=======================================================================================//
Double_t Roo2DHistPdf::evaluate() const
{
  Int_t    ix =  (Int_t)( (x - _lox) / _xbinWidth);
  Int_t    iy =  (Int_t)( (y - _loy) / _ybinWidth);
  Double_t signX = 1.0;
  Double_t signY = 1.0;

  //check x,y is in the valid domain
  if (ix<0) 
  {
    cerr << "got point below lower bound:" << x << " < " << _lox << endl;
    ix=0;
  }
  if (ix > _nPointsx-1) 
  {
    cerr << "got point above upper bound:" << x << " > " << _hix << " ix = "<<ix<<endl;
    ix=_nPointsx-1;
    signX = -1;
  }
  if (iy<0) 
  {
    cerr << "got point below lower bound:"  << y << " < " << _loy << endl;
    iy=0;
  }
  if (iy > _nPointsy-1)
  {
    cerr << "got point above upper bound:"  << y << " > " << _hiy << " iy = "<<iy<< endl;
    iy=_nPointsy-1;
    signX = -1;
  }

  //give the choice of extrapolation between bins and using the bin content
  if(_iWantToExtrapolate)
  {
    Double_t dfdx = (_p[ix+1][iy] - _p[ix][iy])/_xbinWidth;
    Double_t dfdy = (_p[ix][iy+1] - _p[ix][iy])/_ybinWidth;

    Double_t dx = (x-( _lox + (Double_t)ix * _xbinWidth));
    Double_t dy = (y-( _loy + (Double_t)iy * _ybinWidth));

    return( _p[ix][iy] + dx*dfdx + dy * dfdy );
  }
  else return (_p[ix][iy]);
}

