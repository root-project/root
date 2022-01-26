/// \file ROOT/RPageSourceFriends.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2020-08-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageSourceFriends
#define ROOT7_RPageSourceFriends

#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RSpan.hxx>
#include <ROOT/RStringView.hxx>

#include <memory>
#include <vector>
#include <unordered_map>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceFriends
\ingroup NTuple
\brief Virtual storage that combines several other sources horizontally
*/
// clang-format on
class RPageSourceFriends final : public RPageSource {
private:
   struct ROriginId {
      std::size_t fSourceIdx = 0;
      DescriptorId_t fId = kInvalidDescriptorId;
   };

   /// A bi-directional map of descriptor IDs that translates from physical to virtual column, field, and
   /// cluster IDs and vice versa.
   struct RIdBiMap {
      std::unordered_map<DescriptorId_t, ROriginId> fVirtual2Origin;
      std::vector<std::unordered_map<DescriptorId_t, DescriptorId_t>> fOrigin2Virtual;

      void Insert(ROriginId originId, DescriptorId_t virtualId)
      {
         fOrigin2Virtual.resize(originId.fSourceIdx + 1);
         fOrigin2Virtual[originId.fSourceIdx][originId.fId] = virtualId;
         fVirtual2Origin[virtualId] = originId;
      }

      void Clear()
      {
         fVirtual2Origin.clear();
         fOrigin2Virtual.clear();
      }

      DescriptorId_t GetVirtualId(const ROriginId &originId) const
      {
         return fOrigin2Virtual[originId.fSourceIdx].at(originId.fId);
      }

      ROriginId GetOriginId(DescriptorId_t virtualId) const
      {
         return fVirtual2Origin.at(virtualId);
      }
   };

   RNTupleMetrics fMetrics;
   std::vector<std::unique_ptr<RPageSource>> fSources;
   RIdBiMap fIdBiMap;

   RNTupleDescriptorBuilder fBuilder;
   DescriptorId_t fNextId = 1;  ///< 0 is reserved for the friend zero field

   void AddVirtualField(const RNTupleDescriptor &originDesc, std::size_t originIdx, const RFieldDescriptor &originField,
                        DescriptorId_t virtualParent, const std::string &virtualName);

protected:
   RNTupleDescriptor AttachImpl() final;

public:
   RPageSourceFriends(std::string_view ntupleName, std::span<std::unique_ptr<RPageSource>> sources);

   std::unique_ptr<RPageSource> Clone() const final;
   ~RPageSourceFriends() final;

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) final;
   void DropColumn(ColumnHandle_t columnHandle) final;

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   void LoadSealedPage(DescriptorId_t columnId, const RClusterIndex &clusterIndex, RSealedPage &sealedPage) final;

   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};


} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
