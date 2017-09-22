// Author: Brian Bockelman UNL 09/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TIO_FEATURES
#define ROOT_TIO_FEATURES

#include "TBasket.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <sstream>
#include <typeinfo>
#include <type_traits> // is_same, enable_if

namespace ROOT {

namespace Experimental {

class TIOFeatures {
friend class ::TTree;
friend class ::TBranch;
friend class ::TBasket;

public:
   TIOFeatures() {}

   void Clear(TBasket::EIOBits bits);
   bool Set(TBasket::EIOBits bits);
   bool Test(TBasket::EIOBits bits) const;

private:
   TIOFeatures(UChar_t IOBits) : fIOBits(IOBits) {}
   UChar_t GetFeatures() const;

   UChar_t fIOBits{0};
};

}

}

#endif  // ROOT_TIO_FEATURES
