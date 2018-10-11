#include <ROOT/RConcurrentHashColl.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/TRWSpinLock.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/RSha256.hxx>

#include <set>

namespace ROOT {
namespace Internal {


std::ostream &operator<<(std::ostream &os, const RConcurrentHashColl::HashValue &h)
{
   auto digest = h.Get();
   os << digest[0] << "-" << digest[1] << "-" << digest[2] << "-" << digest[3];
   return os;
}

RConcurrentHashColl::HashValue::HashValue(const char *data, int len)
{
   // The cast here is because in the TBuffer ecosystem, the type used is char*
   Sha256(reinterpret_cast<const unsigned char *>(data), len, fDigest);
}

struct RHashSet {
   std::set<ROOT::Internal::RConcurrentHashColl::HashValue> fSet;
};

RConcurrentHashColl::RConcurrentHashColl()
   : fHashSet(std::make_unique<RHashSet>()), fRWLock(std::make_unique<ROOT::TRWSpinLock>()){};

RConcurrentHashColl::~RConcurrentHashColl() = default;

/// Return true if the hash is already in already there
bool RConcurrentHashColl::Find(const HashValue &hash) const
{
   ROOT::TRWSpinLockReadGuard rg(*fRWLock);
   return (fHashSet->fSet.end() != fHashSet->fSet.find(hash));
}

/// If the buffer is there, return false. Otherwise, insert the hash and return true
RConcurrentHashColl::HashValue RConcurrentHashColl::Hash(char *buffer, int len)
{
   return HashValue(buffer, len);
}

/// If the buffer is there, return false. Otherwise, insert the hash and return true
bool RConcurrentHashColl::Insert(char *buffer, int len) const
{
   HashValue hash(buffer, len);

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

/// If the buffer is there, return false. Otherwise, insert the hash and return true
bool RConcurrentHashColl::Insert(const HashValue &hash) const
{
   ROOT::TRWSpinLockWriteGuard wg(*fRWLock);
   auto ret = fHashSet->fSet.insert(hash);
   return ret.second;
}

} // End NS Internal
} // End NS ROOT