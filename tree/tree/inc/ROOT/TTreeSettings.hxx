// Author: Brian Bockelman UNL 09/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTREE_SETTINGS
#define ROOT_TTREE_SETTINGS

#include "TBasket.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <sstream>
#include <typeinfo>
#include <type_traits> // is_same, enable_if

class TTree;

namespace ROOT {

namespace Experimental {

class TTreeSettings {
public:
   TTreeSettings(TTree &tree) : fTree(tree) {}

   void ClearFeature(TBasket::EIOBits bits);
   bool SetFeature(TBasket::EIOBits bits);
   bool TestFeature(TBasket::EIOBits bits);
   UChar_t GetFeatures();

private:
   TTree &fTree;
};

class TBranchSettings {
public:
  TBranchSettings(TBranch &br) : fBranch(br) {}

  UChar_t GetFeatures();

private:
  TBranch &fBranch;
};

}

}

#endif  // ROOT_TTREE_SETTINGS
