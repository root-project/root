 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Copyright (c) 2000-2005, Regents of the University of California          *
  *                          and Stanford University. All rights reserved.    *
  *                                                                           *
  * Redistribution and use in source and binary forms,                        *
  * with or without modification, are permitted according to the terms        *
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
  *****************************************************************************/

/** \class RooIntegralMorph
    \ingroup Roofit

Class RooIntegralMorph is an implementation of the histogram interpolation
technique described by Alex Read in 'NIM A 425 (1999) 357-369 'Linear interpolation of histograms'
for continuous functions rather than histograms. The interpolation method, in short,
works as follows.

  - Given a p.d.f f1(x) with c.d.f F1(x) and p.d.f f2(x) with c.d.f F2(x)

  - One finds takes a value 'y' of both c.d.fs and determines the corresponding x
    values x(1,2) at which F(1,2)(x)==y.

  - The value of the interpolated p.d.f fbar(x) is then calculated as
    fbar(alpha*x1+(1-alpha)*x2) = f1(x1)*f2(x2) / ( alpha*f2(x2) + (1-alpha)*f1(x1) ) ;

From a technical point of view class RooIntegralMorph is a p.d.f that takes
two input p.d.fs f1(x,p) an f2(x,q) and an interpolation parameter to
make a p.d.f fbar(x,p,q,alpha). The shapes f1 and f2 are always taken
to be end the end-points of the parameter alpha, regardless of what
the those numeric values are.

Since the value of fbar(x) cannot be easily calculated for a given value
of x, class RooIntegralMorph is an implementation of RooAbsCachedPdf and
calculates the shape of the interpolated p.d.f. fbar(x) for all values
of x for a given value of alpha,p,q and caches these values in a histogram
(as implemented by RooAbsCachedPdf). The binning granularity of the cache
can be controlled by the binning named "cache" on the RooRealVar representing
the observable x. The fbar sampling algorithm is based on a recursive division
mechanism with a built-in precision cutoff: First an initial sampling in
64 equally spaced bins is made. Then the value of fbar is calculated in
the center of each gap. If the calculated value deviates too much from
the value obtained by linear interpolation from the edge bins, gap
is recursively divided. This strategy makes it possible to define a very
fine cache sampling (e.g. 1000 or 10000) bins without incurring a
corresponding CPU penalty.

Note on numeric stability of the algorithm. Since the algorithm relies
on a numeric inversion of cumulative distributions functions, some precision
may be lost at the 'edges' of the same (i.e. at regions in x where the
c.d.f. value is close to zero or one). The general sampling strategy is
to start with 64 equally spaces samples in the range y=(0.01-0.99).
Then the y ranges are pushed outward by reducing y (or the distance of y to 1.0)
by a factor of sqrt(10) iteratively up to the point where the corresponding
x value no longer changes significantly. For p.d.f.s with very flat tails
such as Gaussians some part of the tail may be lost due to limitations
in numeric precision in the CDF inversion step.

An effect related to the above limitation in numeric precision should
be anticipated when floating the alpha parameter in a fit. If a p.d.f
with such flat tails is fitted, it is likely that the dataset contains
events in the flat tail region. If the alpha parameter is varied, the
likelihood contribution from such events may exhibit discontinuities
in alpha, causing discontinuities in the summed likelihood as well
that will cause convergence problems in MINUIT. To mitigate this effect
one can use the setCacheAlpha() method to instruct RooIntegralMorph
to construct a two-dimensional cache for its output values in both
x and alpha. If linear interpolation is requested on the resulting
output histogram, the resulting interpolation of the p.d.f in the
alpha dimension will smooth out the discontinuities in the tail regions
result in a continuous likelihood distribution that can be fitted.
An added advantage of the cacheAlpha option is that if parameters
p,q of f1,f2 are fixed, the cached values in RooIntegralMorph are
valid for the entire fit session and do not need to be recalculated
for each change in alpha, which may result an considerable increase
in calculation speed.

**/

#include "Riostream.h"

#include "RooIntegralMorph.h"
#include "RooAbsCategory.h"
#include "RooBrentRootFinder.h"
#include "RooAbsFunc.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "TH1.h"

using namespace std;

ClassImp(RooIntegralMorph);

////////////////////////////////////////////////////////////////////////////////
/// Constructor with observables x, pdf shapes pdf1 and pdf2 which represent
/// the shapes at the end points of the interpolation parameter alpha
/// If doCacheAlpha is true, a two-dimensional cache is constructed in
/// both alpha and x

RooIntegralMorph::RooIntegralMorph(const char *name, const char *title,
                RooAbsReal& _pdf1,
                RooAbsReal& _pdf2,
                RooAbsReal& _x,
                RooAbsReal& _alpha,
                bool doCacheAlpha) :
  RooAbsCachedPdf(name,title,2),
  pdf1("pdf1","pdf1",this,_pdf1),
  pdf2("pdf2","pdf2",this,_pdf2),
  x("x","x",this,_x),
  alpha("alpha","alpha",this,_alpha),
  _cacheAlpha(doCacheAlpha),
  _cache(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooIntegralMorph::RooIntegralMorph(const RooIntegralMorph& other, const char* name) :
  RooAbsCachedPdf(other,name),
  pdf1("pdf1",this,other.pdf1),
  pdf2("pdf2",this,other.pdf2),
  x("x",this,other.x),
  alpha("alpha",this,other.alpha),
  _cacheAlpha(other._cacheAlpha),
  _cache(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Observable to be cached for given choice of normalization.
/// Returns the 'x' observable unless doCacheAlpha is set in which
/// case a set with both x and alpha

RooArgSet* RooIntegralMorph::actualObservables(const RooArgSet& /*nset*/) const
{
  RooArgSet* obs = new RooArgSet ;
  if (_cacheAlpha) {
    obs->add(alpha.arg()) ;
  }
  obs->add(x.arg()) ;
  return obs ;
}

////////////////////////////////////////////////////////////////////////////////
/// Parameters of the cache. Returns parameters of both pdf1 and pdf2
/// and parameter cache, in case doCacheAlpha is not set.

RooArgSet* RooIntegralMorph::actualParameters(const RooArgSet& /*nset*/) const
{
  RooArgSet* par1 = pdf1.arg().getParameters(RooArgSet()) ;
  RooArgSet* par2 = pdf2.arg().getParameters(RooArgSet()) ;
  par1->add(*par2,true) ;
  par1->remove(x.arg(),true,true) ;
  if (!_cacheAlpha) {
    par1->add(alpha.arg()) ;
  }
  delete par2 ;
  return par1 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return base name component for cache components in this case
/// a string encoding the names of both end point p.d.f.s

const char* RooIntegralMorph::inputBaseName() const
{
  static TString name ;

  name = pdf1.arg().GetName() ;
  name.Append("_MORPH_") ;
  name.Append(pdf2.arg().GetName()) ;
  return name.Data() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the cache with the interpolated shape.

void RooIntegralMorph::fillCacheObject(PdfCacheElem& cache) const
{
  MorphCacheElem& mcache = static_cast<MorphCacheElem&>(cache) ;

  // If cacheAlpha is true employ slice iterator here to fill all slices

  if (!_cacheAlpha) {

    TIterator* dIter = cache.hist()->sliceIterator((RooAbsArg&)x.arg(),RooArgSet()) ;
    mcache.calculate(dIter) ;
    delete dIter ;

  } else {
    TIterator* slIter = cache.hist()->sliceIterator((RooAbsArg&)alpha.arg(),RooArgSet()) ;

    double alphaSave = alpha ;
    RooArgSet alphaSet(alpha.arg()) ;
    coutP(Eval) << "RooIntegralMorph::fillCacheObject(" << GetName() << ") filling multi-dimensional cache" ;
    while(slIter->Next()) {
      alphaSet.assign(*cache.hist()->get()) ;
      TIterator* dIter = cache.hist()->sliceIterator((RooAbsArg&)x.arg(),RooArgSet(alpha.arg())) ;
      mcache.calculate(dIter) ;
      ccoutP(Eval) << "." << flush;
      delete dIter ;
    }
    ccoutP(Eval) << endl ;

    delete slIter ;
    const_cast<RooIntegralMorph*>(this)->alpha = alphaSave ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Create and return a derived MorphCacheElem.

RooAbsCachedPdf::PdfCacheElem* RooIntegralMorph::createCache(const RooArgSet* nset) const
{
  return new MorphCacheElem(const_cast<RooIntegralMorph&>(*this),nset) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return all RooAbsArg components contained in this cache

RooArgList RooIntegralMorph::MorphCacheElem::containedArgs(Action action)
{
  RooArgList ret ;
  ret.add(PdfCacheElem::containedArgs(action)) ;
  ret.add(*_self) ;
  ret.add(*_pdf1) ;
  ret.add(*_pdf2) ;
  ret.add(*_x  ) ;
  ret.add(*_alpha) ;
  ret.add(*_c1) ;
  ret.add(*_c2) ;

  return ret ;
}

////////////////////////////////////////////////////////////////////////////////
/// Construct of cache element, copy relevant input from RooIntegralMorph,
/// create the cdfs from the input p.d.fs and instantiate the root finders
/// on the cdfs to perform the inversion.

RooIntegralMorph::MorphCacheElem::MorphCacheElem(RooIntegralMorph& self, const RooArgSet* nsetIn) : PdfCacheElem(self,nsetIn)
{
  // Mark in base class that normalization of cached pdf is invariant under pdf parameters
  _x = (RooRealVar*)self.x.absArg() ;
  _nset = new RooArgSet(*_x) ;

  _alpha = (RooAbsReal*)self.alpha.absArg() ;
  _pdf1 = (RooAbsPdf*)(self.pdf1.absArg()) ;
  _pdf2 = (RooAbsPdf*)(self.pdf2.absArg()) ;
  _c1 = _pdf1->createCdf(*_x);
  _c2 = _pdf2->createCdf(*_x) ;
  _cb1 = _c1->bindVars(*_x,_nset) ;
  _cb2 = _c2->bindVars(*_x,_nset) ;
  _self = &self ;

  _rf1 = new RooBrentRootFinder(*_cb1) ;
  _rf2 = new RooBrentRootFinder(*_cb2) ;
  _ccounter = 0 ;

  _rf1->setTol(1e-12) ;
  _rf2->setTol(1e-12) ;
  _ycutoff = 1e-7 ;

  // _yatX = 0 ;
  // _calcX = 0 ;

  // Must do this here too: fillCache() may not be called if cache contents is retrieved from EOcache
  pdf()->setUnitNorm(true) ;

  _yatXmax = 0 ;
  _yatXmin = 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooIntegralMorph::MorphCacheElem::~MorphCacheElem()
{
  delete _rf1 ;
  delete _rf2 ;
  // delete[] _yatX ;
  // delete[] _calcX ;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the x value of the output p.d.f at the given cdf value y.
/// The ok boolean is filled with the success status of the operation.

double RooIntegralMorph::MorphCacheElem::calcX(double y, bool& ok)
{
  if (y<0 || y>1) {
    oocoutW(_self,Eval) << "RooIntegralMorph::MorphCacheElem::calcX() WARNING: requested root finding for unphysical CDF value " << y << endl ;
  }
  double x1,x2 ;

  double xmax = _x->getMax("cache") ;
  double xmin = _x->getMin("cache") ;

  ok=true ;
  ok &= _rf1->findRoot(x1,xmin,xmax,y) ;
  ok &= _rf2->findRoot(x2,xmin,xmax,y) ;
  if (!ok) return 0 ;
  _ccounter++ ;

  return _alpha->getVal()*x1 + (1-_alpha->getVal())*x2 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the bin number enclosing the given x value

Int_t RooIntegralMorph::MorphCacheElem::binX(double X)
{
  double xmax = _x->getMax("cache") ;
  double xmin = _x->getMin("cache") ;
  return (Int_t)(_x->numBins("cache")*(X-xmin)/(xmax-xmin)) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate shape of p.d.f for x,alpha values
/// defined by dIter iterator over cache histogram

void RooIntegralMorph::MorphCacheElem::calculate(TIterator* dIter)
{
  double xsave = _self->x ;

  // if (!_yatX) {
  //   _yatX = new double[_x->numBins("cache")+1] ;
  //   _calcX = new double[_x->numBins("cache")+1] ;
  // }

 _yatX.resize(_x->numBins("cache")+1);
 _calcX.resize(_x->numBins("cache")+1);

  RooArgSet nsetTmp(*_x) ;
  _ccounter = 0 ;

  // Get number of bins from PdfCacheElem histogram
  Int_t nbins = _x->numBins("cache") ;

  // Initialize yatX array to 'un-calculated values (-1)'
  for (int i=0 ; i<nbins ; i++) {
    _yatX[i] = -1 ;
    _calcX[i] = 0 ;
  }

  // Find low and high point
  findRange() ;

  // Perform initial scan of 100 points
  for (int i=0 ; i<10 ; i++) {

    // Take a point in y
    double offset = _yatX[_yatXmin] ;
    double delta = (_yatX[_yatXmax] - _yatX[_yatXmin])/10 ;
    double y = offset + i*delta ;

    // Calculate corresponding X
    bool ok ;
    double X = calcX(y,ok) ;
    if (ok) {
      Int_t iX = binX(X) ;
      _yatX[iX] = y ;
      _calcX[iX] =  X ;
    }
  }

  // Now take an iteration filling the 'gaps'
  Int_t igapLow = _yatXmin+1 ;
  while(true) {
    // Find next gap
    Int_t igapHigh = igapLow+1 ;
    while(igapHigh<(_yatXmax) && _yatX[igapHigh]<0) igapHigh++ ;

    // Fill the gap (iteratively and/or using interpolation)
    fillGap(igapLow-1,igapHigh) ;

    // Terminate after processing of last gap
    if (igapHigh>=_yatXmax-1) break ;
    igapLow = igapHigh+1 ;
  }

  // Make one more iteration to recalculate Y value at bin centers
  double xmax = _x->getMax("cache") ;
  double xmin = _x->getMin("cache") ;
  double binw = (xmax-xmin)/_x->numBins("cache") ;
  for (int i=_yatXmin+1 ; i<_yatXmax-1 ; i++) {

    // Calculate additional offset to apply if bin ixlo does not have X value calculated at bin center
    double xBinC = xmin + (i+0.5)*binw ;
    double xOffset = xBinC-_calcX[i] ;
    if (fabs(xOffset/binw)>1e-3) {
      double slope = (_yatX[i+1]-_yatX[i-1])/(_calcX[i+1]-_calcX[i-1]) ;
      double newY = _yatX[i] + slope*xOffset ;
      //cout << "bin " << i << " needs to be re-centered " << xOffset/binw << " slope = " << slope << " origY = " << _yatX[i] << " newY = " << newY << endl ;
      _yatX[i] = newY ;
    }
  }

  // Zero output histogram below lowest calculable X value
  for (int i=0; i<_yatXmin ; i++) {
    dIter->Next() ;
    //_hist->get(i) ;
    hist()->set(0) ;
  }

  double x1 = _x->getMin("cache");
  double x2 = _x->getMin("cache");

  double xMax = _x->getMax("cache");

  // Transfer calculated values to histogram
  for (int i=_yatXmin ; i<_yatXmax ; i++) {

    double y = _yatX[i] ;

    // Little optimization here exploiting the fact that th cumulative
    // distribution functions increase monotonically, so we already know that
    // the next x-value must be higher than the last one as y is increasing. So
    // we can use the previous x values as lower bounds.
    _rf1->findRoot(x1,x1,xMax,y) ;
    _rf2->findRoot(x2,x2,xMax,y) ;

    _x->setVal(x1) ; double f1x1 = _pdf1->getVal(&nsetTmp) ;
    _x->setVal(x2) ; double f2x2 = _pdf2->getVal(&nsetTmp) ;
    double fbarX = f1x1*f2x2 / ( _alpha->getVal()*f2x2 + (1-_alpha->getVal())*f1x1 ) ;

    dIter->Next() ;
    //_hist->get(i) ;
    hist()->set(fbarX) ;
  }
  // Zero output histogram above highest calculable X value
  for (int i=_yatXmax+1 ; i<nbins ; i++) {
    dIter->Next() ;
    //_hist->get(i) ;
    hist()->set(0) ;
  }

  pdf()->setUnitNorm(true) ;
  _self->x = xsave ;

  oocxcoutD(_self,Eval) << "RooIntegralMorph::MorphCacheElem::calculate(" << _self->GetName() << ") calculation required " << _ccounter << " samplings of cdfs" << endl ;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill all empty histogram bins between bins ixlo and ixhi. The value of 'splitPoint'
/// defines the split point for the recursive division strategy to fill the gaps
/// If the midpoint value of y is very close to the midpoint in x, use interpolation
/// to fill the gaps, otherwise the intervals again.

void RooIntegralMorph::MorphCacheElem::fillGap(Int_t ixlo, Int_t ixhi, double splitPoint)
{
  // CONVENTION: _yatX[ixlo] is filled, _yatX[ixhi] is filled, elements in between are empty
  //   cout << "fillGap: gap from _yatX[" << ixlo << "]=" << _yatX[ixlo] << " to _yatX[" << ixhi << "]=" << _yatX[ixhi] << ", size = " << ixhi-ixlo << endl ;

  if (_yatX[ixlo]<0) {
    oocoutE(_self,Eval) << "RooIntegralMorph::MorphCacheElme::fillGap(" << _self->GetName() << "): ERROR in fillgap " << ixlo << " = " << ixhi
         << " splitPoint= " << splitPoint << " _yatX[ixlo] = " << _yatX[ixlo] << endl ;
  }
  if (_yatX[ixhi]<0) {
    oocoutE(_self,Eval) << "RooIntegralMorph::MorphCacheElme::fillGap(" << _self->GetName() << "): ERROR in fillgap " << ixlo << " = " << ixhi
         << " splitPoint " << splitPoint << " _yatX[ixhi] = " << _yatX[ixhi] << endl ;
  }

  // Determine where half-way Y value lands
  double ymid = _yatX[ixlo]*splitPoint + _yatX[ixhi]*(1-splitPoint) ;
  bool ok ;
  double Xmid = calcX(ymid,ok) ;
  if (!ok) {
    oocoutW(_self,Eval) << "RooIntegralMorph::MorphCacheElem::fillGap(" << _self->GetName() << ") unable to calculate midpoint in gap ["
         << ixlo << "," << ixhi << "], resorting to interpolation" << endl ;
    interpolateGap(ixlo,ixhi) ;
  }

  Int_t iX = binX(Xmid) ;
  double cq = (Xmid-_calcX[ixlo])/(_calcX[ixhi]-_calcX[ixlo])-0.5 ;

  // Store midway point
  _yatX[iX] = ymid ;
  _calcX[iX] = Xmid ;


  // Policy: If centration quality is better than 1% OR better than 1/10 of a bin, fill interval with linear interpolation
  if (fabs(cq)<0.01 || fabs(cq*(ixhi-ixlo))<0.1 || ymid<_ycutoff ) {

    // Fill remaining gaps on either side with linear interpolation
    if (iX-ixlo>1) {
      interpolateGap(ixlo,iX) ;
    }
    if (ixhi-iX>1) {
      interpolateGap(iX,ixhi) ;
    }

  } else {

    if (iX==ixlo) {

      if (splitPoint<0.95) {
   // Midway value lands on lowest bin, retry split with higher split point
   double newSplit = splitPoint + 0.5*(1-splitPoint) ;
   fillGap(ixlo,ixhi,newSplit) ;
      } else {
   // Give up and resort to interpolation
   interpolateGap(ixlo,ixhi) ;
      }

    } else if (iX==ixhi) {

      // Midway value lands on highest bin, retry split with lower split point
      if (splitPoint>0.05) {
   double newSplit = splitPoint/2 ;
   fillGap(ixlo,ixhi,newSplit) ;
      } else {
   // Give up and resort to interpolation
   interpolateGap(ixlo,ixhi) ;
      }

    } else {

      // Midway point reasonable, iterate on interval on both sides
      if (iX-ixlo>1) {
   fillGap(ixlo,iX) ;
      }
      if (ixhi-iX>1) {
   fillGap(iX,ixhi) ;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill empty histogram bins between ixlo and ixhi with values obtained
/// from linear interpolation of ixlo,ixhi elements.

void RooIntegralMorph::MorphCacheElem::interpolateGap(Int_t ixlo, Int_t ixhi)
{
  //cout << "filling gap with linear interpolation ixlo=" << ixlo << " ixhi=" << ixhi << endl ;

  double xmax = _x->getMax("cache") ;
  double xmin = _x->getMin("cache") ;
  double binw = (xmax-xmin)/_x->numBins("cache") ;

  // Calculate deltaY in terms of actual X difference calculate, not based on nominal bin width
  double deltaY = (_yatX[ixhi]-_yatX[ixlo])/((_calcX[ixhi]-_calcX[ixlo])/binw) ;

  // Calculate additional offset to apply if bin ixlo does not have X value calculated at bin center
  double xBinC = xmin + (ixlo+0.5)*binw ;
  double xOffset = xBinC-_calcX[ixlo] ;

  for (int j=ixlo+1 ; j<ixhi ; j++) {
    _yatX[j] = _yatX[ixlo]+(xOffset+(j-ixlo))*deltaY ;
    _calcX[j] = xmin + (j+0.5)*binw ;
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Determine which range of y values can be mapped to x values
/// from the numeric inversion of the input c.d.fs.
/// Start with a y range of [0.1-0.9] and push boundaries
/// outward with a factor of 1/sqrt(10). Stop iteration if
/// inverted x values no longer change

void RooIntegralMorph::MorphCacheElem::findRange()
{
  double xmin = _x->getMin("cache") ;
  double xmax = _x->getMax("cache") ;
  Int_t nbins = _x->numBins("cache") ;

  double x1,x2 ;
  bool ok = true ;
  double ymin=0.1,yminSave(-1) ;
  double Xsave(-1),Xlast=xmax ;

  // Find lowest Y value that can be measured
  // Start at 0.1 and iteratively lower limit by sqrt(10)
  while(true) {
    ok &= _rf1->findRoot(x1,xmin,xmax,ymin) ;
    ok &= _rf2->findRoot(x2,xmin,xmax,ymin) ;
    oocxcoutD(_self,Eval) << "RooIntegralMorph::MorphCacheElem::findRange(" << _self->GetName() << ") findMin: x1 = " << x1 << " x2 = " << x2 << " ok = " << (ok?"T":"F") << endl ;

    // Terminate in case of non-convergence
    if (!ok) break ;

    // Terminate if value of X no longer moves by >0.1 bin size
    double X = _alpha->getVal()*x1 + (1-_alpha->getVal())*x2 ;
    if (fabs(X-Xlast)/(xmax-xmin)<0.0001) {
      break ;
    }
    Xlast=X ;

    // Store new Y value
    _yatXmin = (Int_t)(nbins*(X-xmin)/(xmax-xmin)) ;
    _yatX[_yatXmin] = ymin ;
    _calcX[_yatXmin] = X ;
    yminSave = ymin ;
    Xsave=X ;

    // Reduce ymin by half an order of magnitude
    ymin /=sqrt(10.) ;

    // Emergency break
    if (ymin<_ycutoff) break ;
  }
  _yatX[_yatXmin] = yminSave ;
  _calcX[_yatXmin] = Xsave ;

  // Find highest Y value that can be measured
  // Start at 1 - 0.1 and iteratively lower delta by sqrt(10)
  ok = true ;
  double deltaymax=0.1, deltaymaxSave(-1) ;
  Xlast=xmin ;
  while(true) {
    ok &= _rf1->findRoot(x1,xmin,xmax,1-deltaymax) ;
    ok &= _rf2->findRoot(x2,xmin,xmax,1-deltaymax) ;

    oocxcoutD(_self,Eval) << "RooIntegralMorph::MorphCacheElem::findRange(" << _self->GetName() << ") findMax: x1 = " << x1 << " x2 = " << x2 << " ok = " << (ok?"T":"F") << endl ;

    // Terminate in case of non-convergence
    if (!ok) break ;

    // Terminate if value of X no longer moves by >0.1 bin size
    double X = _alpha->getVal()*x1 + (1-_alpha->getVal())*x2 ;
    if (fabs(X-Xlast)/(xmax-xmin)<0.0001) {
      break ;
    }
    Xlast=X ;

    // Store new Y value
    _yatXmax = (Int_t)(nbins*(X-xmin)/(xmax-xmin)) ;
    _yatX[_yatXmax] = 1-deltaymax ;
    _calcX[_yatXmax] = X ;
    deltaymaxSave = deltaymax ;
    Xsave=X ;

    // Reduce ymin by half an order of magnitude
    deltaymax /=sqrt(10.) ;

    // Emergency break
    if (deltaymax<_ycutoff) break ;
  }

  _yatX[_yatXmax] = 1-deltaymaxSave ;
  _calcX[_yatXmax] = Xsave ;


  // Initialize values out of range to 'out-of-range' (-2)
  for (int i=0 ; i<_yatXmin ; i++)  _yatX[i] = -2 ;
  for (int i=_yatXmax+1 ; i<nbins; i++) _yatX[i] = -2 ;
  oocxcoutD(_self,Eval) << "RooIntegralMorph::findRange(" << _self->GetName() << "): ymin = " << _yatX[_yatXmin] << " ymax = " << _yatX[_yatXmax] << endl;
  oocxcoutD(_self,Eval) << "RooIntegralMorph::findRange(" << _self->GetName() << "): xmin = " << _calcX[_yatXmin] << " xmax = " << _calcX[_yatXmax] << endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Dummy

double RooIntegralMorph::evaluate() const
{
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Indicate to the RooAbsCachedPdf base class that for the filling of the
/// cache the traversal of the x should be in the innermost loop, to minimize
/// recalculation of the one-dimensional internal cache for a fixed value of alpha

void RooIntegralMorph::preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const
{
  // Put x last to minimize cache faulting
  orderedObs.removeAll() ;

  orderedObs.add(obs) ;
  RooAbsArg* obsX = obs.find(x.arg().GetName()) ;
  if (obsX) {
    orderedObs.remove(*obsX) ;
    orderedObs.add(*obsX) ;
  }
}
