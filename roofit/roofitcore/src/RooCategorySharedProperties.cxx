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
// RooCategorySharedProperties is the container for all properties
// that are shared between instance of RooCategory objects that
// are clones of each other. At present the only property that is
// shared in this way is the list of alternate named range definitions
// END_HTML
//

#include "RooFit.h"
#include "RooCategorySharedProperties.h"

ClassImp(RooCategorySharedProperties)
;


//_____________________________________________________________________________
RooCategorySharedProperties::RooCategorySharedProperties()
{
  // Constructor
} 



//_____________________________________________________________________________
RooCategorySharedProperties::~RooCategorySharedProperties() 
{
  // Destructor
  _altRanges.Delete() ;
} 


