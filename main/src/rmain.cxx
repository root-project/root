// @(#)root/main:$Name:  $:$Id: rmain.cxx,v 1.1.1.1 2000/05/16 17:00:49 rdm Exp $
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

#include "TROOT.h"
#include "TRint.h"

//______________________________________________________________________________
int main(int argc, char **argv)
{
   TROOT root("Rint","The ROOT Interactive Interface");
   
   TRint *theApp = new TRint("Rint", &argc, argv, 0, 0);

   // Enter the event loop...
   theApp->Run();

   delete theApp;

   return 0;
}
