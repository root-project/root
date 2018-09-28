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
#include "Rtypes.h"

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
   class HashValue {
       friend std::ostream &operator<<(std::ostream &os, const RConcurrentHashColl::HashValue &h);
   private:
      ULong64_t fDigest[4] = {0, 0, 0, 0};

   public:
      HashValue() = default;
      HashValue(const char *data, int len);
      ULong64_t const *Get() const { return fDigest; }
   };

   RConcurrentHashColl();
   ~RConcurrentHashColl();

   /// Return true if the hash is already in already there
   bool Find(const HashValue &hash) const;

   /// If the hash is there, return false. Otherwise, insert the hash and return true;
   bool Insert(char *buf, int len) const;

   /// If the hash is there, return false. Otherwise, insert the hash and return true;
   bool Insert(const HashValue &hash) const;

   /// Return the hash object corresponding to the buffer.
   static HashValue Hash(char *buf, int len);
};

inline bool operator==(const RConcurrentHashColl::HashValue &lhs, const RConcurrentHashColl::HashValue &rhs)
{
   auto l = lhs.Get();
   auto r = rhs.Get();
   return l[0] == r[0] && l[1] == r[1] && l[2] == r[2] && l[3] == r[3];
}

} // End NS Internal
} // End NS ROOT

namespace std {
template <>
struct less<ROOT::Internal::RConcurrentHashColl::HashValue> {
   bool operator()(const ROOT::Internal::RConcurrentHashColl::HashValue &lhs, const ROOT::Internal::RConcurrentHashColl::HashValue &rhs) const
   {
      /// Check piece by piece the 4 64 bits ints which make up the hash.
      auto l = lhs.Get();
      auto r = rhs.Get();
      // clang-format off
      return l[0] < r[0] ? true :
               l[0] > r[0] ? false :
                 l[1] < r[1] ? true :
                   l[1] > r[1] ? false :
                     l[2] < r[2] ? true :
                       l[2] > r[2] ? false :
                         l[3] < r[3] ? true : false;
      // clang-format on
   }
};
} // End NS std

#endif
