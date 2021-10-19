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
#ifdef _MSC_VER
#include "RConfigure.h"
#include "snprintf.h"
#include <process.h>
#endif

////////////////////////////////////////////////////////////////////////////////
/// Create an interactive ROOT application

int main(int argc, char **argv)
{
#ifdef _MSC_VER
   char **argvv;
   char arg0[kMAXPATHLEN];
   int notebook = 0; // index of --notebook args, all other args will be re-directed to nbmain
   int i;
   for (i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--notebook")) { notebook = i; break; }
   }
   if (notebook > 0) {
      // Build command
#ifdef ROOTBINDIR
      if (getenv("ROOTIGNOREPREFIX"))
#endif
         snprintf(arg0, sizeof(arg0), "%s/bin/rootnb.exe", getenv("ROOTSYS"));
#ifdef ROOTBINDIR
      else
         snprintf(arg0, sizeof(arg0), "%s/rootnb.exe", ROOTBINDIR);
#endif
      int numnbargs = 1 + (argc - notebook);
      argvv = new char* [numnbargs+1];
      argvv[0] = arg0;
      for (i = 1; i < numnbargs; i++)
         argvv[i] = argv[notebook + i];
      argvv[numnbargs] = nullptr;
      // Execute ROOT notebook binary
      _execv(arg0, argvv);
      // Exec failed
      fprintf(stderr, "%s: can't start ROOT notebook -- this option is only available when building with CMake, please check that %s exists\n",
              argv[0], arg0);
      delete [] argvv;
      return 1;
   }
#endif
   TRint *theApp = new TRint("Rint", &argc, argv);

   // and enter the event loop...
   theApp->Run();

   delete theApp;

   return 0;
}
