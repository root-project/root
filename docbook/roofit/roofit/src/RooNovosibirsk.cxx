/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   DB, Dieter Best,     UC Irvine,         best@slac.stanford.edu          *
 *   HT, Hirohisa Tanaka  SLAC               tanaka@slac.stanford.edu        *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooNovosibirsk implements the Novosibirsk function 
// END_HTML
//


#include "RooFit.h"

#include <math.h>
#include "TMath.h"

#include "RooNovosibirsk.h"
#include "RooRealVar.h"

ClassImp(RooNovosibirsk)



//_____________________________________________________________________________
RooNovosibirsk::RooNovosibirsk(const char *name, const char *title,
			     RooAbsReal& _x,     RooAbsReal& _peak,
			     RooAbsReal& _width, RooAbsReal& _tail) :
  // The two addresses refer to our first dependent variable and
  // parameter, respectively, as declared in the rdl file
  RooAbsPdf(name, title),
  x("x","x",this,_x),
  width("width","width",this,_width),
  peak("peak","peak",this,_peak),
  tail("tail","tail",this,_tail)
{
}


//_____________________________________________________________________________
RooNovosibirsk::RooNovosibirsk(const RooNovosibirsk& other, const char *name):
  RooAbsPdf(other,name),
  x("x",this,other.x),
  width("width",this,other.width),
  peak("peak",this,other.peak),
  tail("tail",this,other.tail)
{
}


//_____________________________________________________________________________
Double_t RooNovosibirsk::evaluate() const {
  // Put the formula for your PDF's value here. Use the pre-computed
  // value of _norm to normalize the result.

  double qa=0,qb=0,qc=0,qx=0,qy=0;

  if(TMath::Abs(tail) < 1.e-7) 
    qc = 0.5*TMath::Power(((x-peak)/width),2);
  else {
    qa = tail*sqrt(log(4.));
    qb = sinh(qa)/qa;
    qx = (x-peak)/width*qb;
    qy = 1.+tail*qx;
  
    //---- Cutting curve from right side

    if( qy > 1.E-7) 
      qc = 0.5*(TMath::Power((log(qy)/tail),2) + tail*tail);
    else
      qc = 15.0;
  }

  //---- Normalize the result

  return exp(-qc);

}
