/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIntegrator.cc,v 1.8 2001/08/08 23:11:23 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   05-Aug-2001 DK Adapted to use RooAbsFunc interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooAbsIntegrator is the abstract interface for integrating real-valued
// functions that implement the RooAbsFunc interface.

#include "RooFitCore/RooAbsIntegrator.rdl"

ClassImp(RooAbsIntegrator)
;

static const char rcsid[] =
"$Id: RooAbsIntegrator.cc,v 1.8 2001/08/08 23:11:23 david Exp $";

RooAbsIntegrator::RooAbsIntegrator(const RooAbsFunc& function) :
  _function(&function), _valid(function.isValid())
{
}
