/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

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


void RooTable::printToStream(ostream& os, PrintOption opt=Standard) 
{
  os << "RooTable" << endl ;
}
