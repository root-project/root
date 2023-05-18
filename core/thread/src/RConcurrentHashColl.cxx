#include <ROOT/RConcurrentHashColl.hxx>
#include <ROOT/TRWSpinLock.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/RSha256.hxx>

#include <memory>
#include <set>
#include <unordered_map>

namespace std
{
template <> struct hash<ROOT::Internal::RConcurrentHashColl::HashValue>
{
   std::size_t operator()(const ROOT::Internal::RConcurrentHashColl::HashValue& key) const noexcept
   {
      return key.Hash();
   }
};
}
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

struct RHashMap {
   std::unordered_map<ROOT::Internal::RConcurrentHashColl::HashValue, RUidColl> fHashMap;
};

RConcurrentHashColl::RConcurrentHashColl()
   : fHashMap(std::make_unique<RHashMap>()), fRWLock(std::make_unique<ROOT::TRWSpinLock>()){};

RConcurrentHashColl::~RConcurrentHashColl() = default;

/// Return true if the hash is already in already there
const RUidColl* RConcurrentHashColl::Find(const HashValue &hash) const
{
   ROOT::TRWSpinLockReadGuard rg(*fRWLock);
   auto iter = fHashMap->fHashMap.find(hash);
   if (iter != fHashMap->fHashMap.end())
      return &(iter->second);
   else
      return nullptr;
}

/// If the buffer is there, return false. Otherwise, insert the hash and return true
RConcurrentHashColl::HashValue RConcurrentHashColl::Hash(char *buffer, int len)
{
   return HashValue(buffer, len);
}

/// If the buffer is there, return false. Otherwise, insert the hash and return true
bool RConcurrentHashColl::Insert(const HashValue &hash, RUidColl &&values) const
{
   ROOT::TRWSpinLockWriteGuard wg(*fRWLock);
   auto ret = fHashMap->fHashMap.insert({hash, std::move(values)});
   return ret.second;
}

} // End NS Internal
} // End NS ROOT