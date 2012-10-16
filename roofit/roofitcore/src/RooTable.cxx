/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
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
// RooTable is the abstract interface for table objects.
// Table objects are the category equivalent of RooPlot objects
// (which are used for real-valued objects)
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooTable.h"



using namespace std;

ClassImp(RooTable)



//_____________________________________________________________________________
RooTable::RooTable(const char *name, const char *title) : TNamed(name,title)
{
  // Constructor
}



//_____________________________________________________________________________
RooTable::RooTable(const RooTable& other) : TNamed(other), RooPrintable(other)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooTable::~RooTable()
{
  // Destructor
}

