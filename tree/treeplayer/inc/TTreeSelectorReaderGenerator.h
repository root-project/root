// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 22/06/2015

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeSelectorReaderGenerator
#define ROOT_TTreeSelectorReaderGenerator

#ifndef ROOT_Tlist
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TTree;

namespace ROOT {

   class TTreeSelectorReaderGenerator
   {
   public:
      TTreeSelectorReaderGenerator(TTree* tree, const char *classname, UInt_t maxUnrolling);
   };

}

using ROOT::TTreeSelectorReaderGenerator;

#endif
