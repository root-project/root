// Author: Danilo Piparo May 2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RConcurrentHashColl
#define ROOT_RConcurrentHashColl

#include <memory>

namespace ROOT {

class TRWSpinLock;

namespace Internal {

struct RHashSet;

/// This class is a TS set of unsigned set
class RConcurrentHashColl {
private:
   mutable std::unique_ptr<RHashSet> fHashSet;
   mutable std::unique_ptr<ROOT::TRWSpinLock> fRWLock;

public:
   RConcurrentHashColl();
   ~RConcurrentHashColl();
   /// If the hash is there, return false. Otherwise, insert the hash and return true;
   bool Insert(char *buf, int len) const;
};

} // End NS Internal
} // End NS ROOT

#endif
