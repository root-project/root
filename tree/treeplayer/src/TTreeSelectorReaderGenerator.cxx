// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 22/06/2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeSelectorReaderGenerator.h"
#include <stdio.h>

namespace ROOT {

   TTreeSelectorReaderGenerator::TTreeSelectorReaderGenerator(TTree* tree,
                                            const char *classname, UInt_t maxUnrolling)
   {
      // Constructor.

      printf("TTreeSelectorReaderGenerator::ctor\n");
   }
}
