/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooNovosibirsk.cc,v 1.1 2002/02/04 23:42:45 best Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   14-Feb-2000 DK Created initial version from RooLifetime
 *
 * Copyright (C) 1999 Stanford University
 *   20 Jan 2001 Hirohisa A. Tanaka: Novosibirsk PDF for RooFitTools
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// RooNovosibirsk implements the Novosibirsk function 
//

//#include "BaBar/BaBar.hh"

#include <math.h>

#include "RooFitModels/RooNovosibirsk.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooNovosibirsk)

RooNovosibirsk::RooNovosibirsk(const char *name, const char *title,
			     RooRealVar& _x,     RooRealVar& _peak,
			     RooRealVar& _width, RooRealVar& _tail) :
  // The two addresses refer to our first dependent variable and
  // parameter, respectively, as declared in the rdl file
  RooAbsPdf(name, title),
  x("x","x",this,_x),
  peak("peak","peak",this,_peak),
  width("width","width",this,_width),
  tail("tail","tail",this,_tail)
{
}

RooNovosibirsk::RooNovosibirsk(const RooNovosibirsk& other, const char *name):
  RooAbsPdf(other,name),
  x("x",this,other.x),
  peak("peak",this,other.peak),
  width("width",this,other.width),
  tail("tail",this,other.tail)
{
}

Double_t RooNovosibirsk::evaluate() const {
  // Put the formula for your PDF's value here. Use the pre-computed
  // value of _norm to normalize the result.

  double qa=0,qb=0,qc=0,qx=0,qy=0;
  double result=0;

  if(fabs(tail) < 1.e-7) 
    qc = 0.5*pow(((x-peak)/width),2);
  else {
    qa = tail*sqrt(log(4.));
    qb = sinh(qa)/qa;
    qx = (x-peak)/width*qb;
    qy = 1.+tail*qx;
  
    //---- Cutting curve from right side

    if( qy > 1.E-7) 
      qc = 0.5*(pow((log(qy)/tail),2) + tail*tail);
    else
      qc = 15.0;
  }

  //---- Normalize the result

  return exp(-qc);

}
