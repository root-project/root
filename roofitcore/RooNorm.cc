/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooNorm.cc,v 1.1 2001/10/06 06:19:53 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   10-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2000 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//  RooNorm is a dummy PDF with a flat likelihood distribution that introduces a 
//  parameteric extended likelihood term into a PDF. Typically RooNorm is used
//  to add the extended likelihood term to another PDF by multiplying RooNorm
//  with that PDF using RooProdPdf.

#include "RooFitCore/RooNorm.hh"

ClassImp(RooNorm)
;


RooNorm::RooNorm(const char *name, const char *title, const RooAbsReal& norm) :
  RooAbsPdf(name,title),
  _n("n","Normalization",this,(RooAbsReal&)norm)
{
  // Constructor with parameter for expected number of events
}



RooNorm::RooNorm(const RooNorm& other, const char* name) :
  RooAbsPdf(other,name),
  _n("n",this,other._n)
{
  // Constructor
}


