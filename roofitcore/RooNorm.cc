/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   10-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2000 University of California
 *****************************************************************************/

#include "RooFitCore/RooNorm.hh"

ClassImp(RooNorm)
;


RooNorm::RooNorm(const char *name, const char *title, const RooAbsReal& norm) :
  RooAbsPdf(name,title),
  _n("n","Normalization",this,(RooAbsReal&)norm)
{
}



RooNorm::RooNorm(const RooNorm& other, const char* name) :
  RooAbsPdf(other,name),
  _n("n",this,other._n)
{
}


