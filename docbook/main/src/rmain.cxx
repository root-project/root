// @(#)root/main:$Id$
// Author: Fons Rademakers   02/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// RMain                                                                //
//                                                                      //
// Main program used to create RINT application.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRint.h"

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // Create an interactive ROOT application
   TRint *theApp = new TRint("Rint", &argc, argv);

   // and enter the event loop...
   theApp->Run();

   delete theApp;

   return 0;
}
