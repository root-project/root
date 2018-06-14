#include <ROOT/RConcurrentHashColl.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/TRWSpinLock.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/RSha256.hxx>

#include <set>

namespace ROOT {
namespace Internal {

struct RHashSet {
   std::set<ROOT::Internal::RSha256Hash> fSet;
};

RConcurrentHashColl::RConcurrentHashColl()
   : fHashSet(std::make_unique<RHashSet>()), fRWLock(std::make_unique<ROOT::TRWSpinLock>()){};
RConcurrentHashColl::~RConcurrentHashColl() = default;

/// If the buffer is there, return false. Otherwise, insert the hash and return true
bool RConcurrentHashColl::Insert(char *buffer, int len) const
{
   RSha256Hash hash(buffer, len);

   {
      ROOT::TRWSpinLockReadGuard rg(*fRWLock);
      if (fHashSet->fSet.end() != fHashSet->fSet.find(hash))
         return false;
   }
   {
      ROOT::TRWSpinLockWriteGuard wg(*fRWLock);
      fHashSet->fSet.insert(hash);
      return true;
   }
}

} // End NS Internal
} // End NS ROOT