/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooCBShape.cc,v 1.2 2001/08/02 21:40:04 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   05-Jan-2000 DK Created initial version from RooCBShapeProb
 *   19-Jun-2001 JB Ported to RooFitModels.
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooCBShape.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooCBShape)

static const char rcsid[] =
"$Id: RooCBShape.cc,v 1.2 2001/08/02 21:40:04 verkerke Exp $";

RooCBShape::RooCBShape(const char *name, const char *title,
		       RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _sigma,
		       RooAbsReal& _alpha, RooAbsReal& _n) :
  RooAbsPdf(name, title),
  m("m", "Dependent", this, _m),
  m0("m0", "M0", this, _m0),
  sigma("sigma", "Sigma", this, _sigma),
  alpha("alpha", "Alpha", this, _alpha),
  n("n", "Order", this, _n)
{
}

RooCBShape::RooCBShape(const RooCBShape& other, const char* name) :
  RooAbsPdf(other, name), m("m", this, other.m), m0("m0", this, other.m0),
  sigma("sigma", this, other.sigma), alpha("alpha", this, other.alpha),
  n("n", this, other.n)
{
}

Double_t RooCBShape::evaluate() const {

  Double_t t = (m-m0)/sigma;
  if (alpha < 0) t = -t;

  Double_t absAlpha = fabs(alpha);

  if (t > -absAlpha) {
    return exp(-0.5*t*t);
  }
  else {
    Double_t a =  pow(n/absAlpha,n)*exp(-0.5*absAlpha*absAlpha);
    Double_t b= n/absAlpha - absAlpha; 

    return a/pow(b - t, n);
  }
}

