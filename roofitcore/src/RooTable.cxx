/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTable.cc,v 1.2 2001/05/03 02:15:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooTable is the abstract interface for table objects.
// Table objects are the category equivalent of RooPlot objects
// (which are used for real-valued objects)

#include "RooFitCore/RooTable.hh"

ClassImp(RooTable)


RooTable::RooTable(const char *name, const char *title) : TNamed(name,title)
{
}


RooTable::RooTable(const RooTable& other) : TNamed(other)
{
}


RooTable::~RooTable()
{
}


void RooTable::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  os << indent << "RooTable" << endl ;
}
