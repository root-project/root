/// \file ROOT/RCluster.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-09-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RCluster
#define ROOT7_RCluster

#include <ROOT/RNTupleUtil.hxx>

#include <cstdint>
#include <unordered_map>

namespace ROOT {
namespace Experimental {
namespace Detail {

struct RSheetKey {
   DescriptorId_t fColumnId;
   NTupleSize_t fPageNo;
   RSheetKey(DescriptorId_t columnId, NTupleSize_t pageNo) : fColumnId(columnId), fPageNo(pageNo) {}
   bool operator == (const RSheetKey &other) const {
      return fColumnId == other.fColumnId && fPageNo == other.fPageNo;
   }
};

class RSheet {
private:
   const void *fAddress = nullptr;
   std::size_t fSize = 0;

public:
   RSheet() = default;
   RSheet(void *address, std::size_t size) : fAddress(address), fSize(size) {}

   const void *GetAddress() const { return fAddress; }
   std::size_t GetSize() const { return fSize; }

   bool IsNull() const { return fAddress == nullptr; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

namespace std
{
   template <>
   struct hash<ROOT::Experimental::Detail::RSheetKey>
   {
      size_t operator()(const ROOT::Experimental::Detail::RSheetKey &key) const
      {
         return ((std::hash<ROOT::Experimental::DescriptorId_t>()(key.fColumnId) ^
                 (hash<ROOT::Experimental::NTupleSize_t>()(key.fPageNo) << 1)) >> 1);
      }
   };
}

namespace ROOT {
namespace Experimental {
namespace Detail {

class RRawFile;

class RCluster {
public:
   using ClusterHandle_t = void *;

protected:
   ClusterHandle_t fHandle;
   DescriptorId_t fClusterId;
   std::unordered_map<RSheetKey, RSheet> fSheets;

public:
   RCluster(ClusterHandle_t handle, DescriptorId_t clusterId) : fHandle(handle), fClusterId(clusterId) {}
   RCluster(const RCluster &other) = delete;
   RCluster &operator =(const RCluster &other) = delete;
   virtual ~RCluster();

   void InsertSheet(const RSheetKey &key, const RSheet &sheet) { fSheets[key] = sheet; }

   ClusterHandle_t GetHandle() const { return fHandle; }
   DescriptorId_t GetId() const { return fClusterId; }
   const RSheet *GetSheet(const RSheetKey &key) const;
};

class RHeapCluster : public RCluster {
public:
   RHeapCluster(ClusterHandle_t handle, DescriptorId_t clusterId) : RCluster(handle, clusterId) {}
   ~RHeapCluster();
};

class RMMapCluster : public RCluster {
private:
   RRawFile &fFile;
   std::size_t fLength;
public:
   RMMapCluster(ClusterHandle_t handle, DescriptorId_t clusterId, std::size_t length, RRawFile &file)
      : RCluster(handle, clusterId), fFile(file), fLength(length) {}
   ~RMMapCluster();
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
