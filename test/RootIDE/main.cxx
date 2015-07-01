// @(#)root/test/RootIDE/:$Id$
// Author: Bertrand Bellenot   20/04/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>

#include <TSystem.h>
#include <TApplication.h>
#include <TRint.h>
#include "TGRootIDE.h"

//----------------------------------------------------------------------

int main(int argc, char *argv[])
{

   TString *fname = 0;
   for (int i = 0; i < argc; i++) {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
         printf("Usage: %s [-h | -?] [filename]\n", argv[0]);
         printf("    -h, -?:     this message\n");
         printf("  filename:     name of the file to open in ROOT IDE\n");
         return 0;
      }
      if ((i > 0) && (!gSystem->AccessPathName(argv[i]))) {
         fname = new TString(argv[i]);
         // don't pass filename arg to TRint (avoid processing file)
         argv[i] = 0;
         argc--;
      }
   }

   TApplication *theApp = new TRint("App", &argc, argv);

   new TGRootIDE(fname ? fname->Data() : 0);

   theApp->Run();
   return 0;
}
