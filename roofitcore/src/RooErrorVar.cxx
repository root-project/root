/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   09-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooErrorVar is an auxilary class that represents the error
// of a RooRealVar as a seperate object. The main reason of
// existence of this class is to facilitate the reuse of existing
// techniques to perform calculations that involve a RooRealVars
// error, such as calculating the pull value.
//

#include "RooFitCore/RooErrorVar.hh"

ClassImp(RooErrorVar)
;


RooErrorVar::RooErrorVar(const char *name, const char *title, const RooRealVar& input) :
  RooAbsReal(name,title),
  _realVar("realVar","RooRealVar with error",this,(RooAbsReal&)input)
{
  // Constuctor
}


RooErrorVar::RooErrorVar(const RooErrorVar& other, const char* name) :
  RooAbsReal(other,name),
  _realVar("realVar",this,other._realVar)
{
  // Copy constructor
}

