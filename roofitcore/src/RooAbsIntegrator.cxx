/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIntegrator.cc,v 1.7 2001/08/02 23:54:23 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   05-Aug-2001 DK Adapted to use RooAbsFunc interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAbsIntegrator is the abstract interface for integrating real-valued
// functions that implement the RooAbsFunc interface.

#include "RooFitCore/RooAbsIntegrator.rdl"

ClassImp(RooAbsIntegrator)
;

static const char rcsid[] =
"$Id: RooAbsFunc.cc,v 1.1 2001/08/03 21:44:56 david Exp $";

RooAbsIntegrator::RooAbsIntegrator(const RooAbsFunc& function) :
  _function(&function), _valid(function.isValid())
{
}
