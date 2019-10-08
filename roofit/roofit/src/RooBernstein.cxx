/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   Kyle Cranmer                                                            *
 *                                                                           *
 *****************************************************************************/

/** \class RooBernstein
    \ingroup Roofit

Bernstein basis polynomials are positive-definite in the range [0,1].
In this implementation, we extend [0,1] to be the range of the parameter.
There are n+1 Bernstein basis polynomials of degree n:
\f[
 B_{i,n}(x) = \begin{pmatrix}n \\\ i \end{pmatrix} x^i \cdot (1-x)^{n-i}
\f]
Thus, by providing n coefficients that are positive-definite, there
is a natural way to have well-behaved polynomial PDFs. For any n, the n+1 polynomials
'form a partition of unity', i.e., they sum to one for all values of x.
They can be used as a basis to span the space of polynomials with degree n or less:
\f[
 PDF(x, c_0, ..., c_n) = \mathcal{N} \cdot \sum_{i=0}^{n} c_i \cdot B_{i,n}(x).
\f]
By giving n+1 coefficients in the constructor, this class constructs the n+1
polynomials of degree n, and sums them to form an element of the space of polynomials
of degree n. \f$ \mathcal{N} \f$ is a normalisation constant that takes care of the
cases where the \f$ c_i \f$ are not all equal to one.

See also
http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf
**/

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>
#include "TMath.h"
#include "RooBernstein.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"

#include "TError.h"

using namespace std;

ClassImp(RooBernstein);

////////////////////////////////////////////////////////////////////////////////

RooBernstein::RooBernstein()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooBernstein::RooBernstein(const char* name, const char* title,
                           RooAbsReal& x, const RooArgList& coefList):
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefficients","List of coefficients",this)
{
  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while((coef = (RooAbsArg*)coefIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooBernstein::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName()
      << " is not of type RooAbsReal" << endl ;
      R__ASSERT(0) ;
    }
    _coefList.add(*coef) ;
  }
  delete coefIter ;
}

////////////////////////////////////////////////////////////////////////////////

RooBernstein::RooBernstein(const RooBernstein& other, const char* name) :
  RooAbsPdf(other, name),
  _x("x", this, other._x),
  _coefList("coefList",this,other._coefList)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBernstein::evaluate() const
{
  Double_t xmin = _x.min();
  Double_t x = (_x - xmin) / (_x.max() - xmin); // rescale to [0,1]
  Int_t degree = _coefList.getSize() - 1; // n+1 polys of degree n
  RooFIter iter = _coefList.fwdIterator();

  if(degree == 0) {

    return ((RooAbsReal *)iter.next())->getVal();

  } else if(degree == 1) {

    Double_t a0 = ((RooAbsReal *)iter.next())->getVal(); // c0
    Double_t a1 = ((RooAbsReal *)iter.next())->getVal() - a0; // c1 - c0
    return a1 * x + a0;

  } else if(degree == 2) {

    Double_t a0 = ((RooAbsReal *)iter.next())->getVal(); // c0
    Double_t a1 = 2 * (((RooAbsReal *)iter.next())->getVal() - a0); // 2 * (c1 - c0)
    Double_t a2 = ((RooAbsReal *)iter.next())->getVal() - a1 - a0; // c0 - 2 * c1 + c2
    return (a2 * x + a1) * x + a0;

  } else if(degree > 2) {

    Double_t t = x;
    Double_t s = 1 - x;

    Double_t result = ((RooAbsReal *)iter.next())->getVal() * s;
    for(Int_t i = 1; i < degree; i++) {
      result = (result + t * TMath::Binomial(degree, i) * ((RooAbsReal *)iter.next())->getVal()) * s;
      t *= x;
    }
    result += t * ((RooAbsReal *)iter.next())->getVal();

    return result;
  }

  // in case list of arguments passed is empty
  return TMath::SignalingNaN();
}

////////////////////////////////////////////////////////////////////////////////
/// No analytical calculation available (yet) of integrals over subranges

Int_t RooBernstein::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  if (rangeName && strlen(rangeName)) {
    return 0 ;
  }

  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBernstein::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1) ;
  Double_t xmin = _x.min(rangeName); Double_t xmax = _x.max(rangeName);
  Int_t degree= _coefList.getSize()-1; // n+1 polys of degree n
  Double_t norm(0) ;

  RooFIter iter = _coefList.fwdIterator() ;
  Double_t temp=0;
  for (int i=0; i<=degree; ++i){
    // for each of the i Bernstein basis polynomials
    // represent it in the 'power basis' (the naive polynomial basis)
    // where the integral is straight forward.
    temp = 0;
    for (int j=i; j<=degree; ++j){ // power basis≈ß
      temp += pow(-1.,j-i) * TMath::Binomial(degree, j) * TMath::Binomial(j,i) / (j+1);
    }
    temp *= ((RooAbsReal*)iter.next())->getVal(); // include coeff
    norm += temp; // add this basis's contribution to total
  }

  norm *= xmax-xmin;
  return norm;
}
