// @(#)root/base:$Id$
// Author: Fons Rademakers   07/05/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TInspectorImp
\ingroup Base

ABC describing GUI independent object inspector (abstraction mainly needed
for Win32. On X11 systems it currently uses a standard TCanvas).
*/

#include "TInspectorImp.h"

ClassImp(TInspectorImp);
