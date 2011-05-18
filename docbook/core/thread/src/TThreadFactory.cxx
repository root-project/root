// @(#)root/thread:$Id$
// Author: Fons Rademakers   01/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TThreadFactory                                                       //
//                                                                      //
// This ABC is a factory for thread components. Depending on which      //
// factory is active one gets either Posix or Win32 threads.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TThreadFactory.h"

TThreadFactory *gThreadFactory = 0;

ClassImp(TThreadFactory)

//______________________________________________________________________________
TThreadFactory::TThreadFactory(const char *name, const char *title)
               : TNamed(name, title)
{
   // TThreadFactory ctor only called by derived classes.
}
