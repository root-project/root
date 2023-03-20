/// \file ROOT/RDaos.hxx
/// \ingroup NTuple ROOT7
/// \author Javier Lopez-Gomez <j.lopez@cern.ch>
/// \date 2020-11-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDaos
#define ROOT7_RDaos

#include <ROOT/RStringView.hxx>
#include <ROOT/TypeTraits.hxx>
#include <ROOT/RSpan.hxx>

#include <daos.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <optional>
#include <unordered_map>

#ifndef DAOS_UUID_STR_SIZE
#define DAOS_UUID_STR_SIZE 37
#endif

namespace ROOT {

namespace Experimental {
namespace Detail {

struct RDaosEventQueue {
   daos_handle_t fQueue;
   RDaosEventQueue();
   ~RDaosEventQueue();

   /// \brief Sets event barrier for a given parent event and waits for the completion of all children launched before
   /// the barrier (must have at least one child).
   /// \return 0 on success; a DAOS error code otherwise (< 0).
   int WaitOnParentBarrier(daos_event_t *ev_ptr);
   /// \brief Reserve event in queue, optionally tied to a parent event.
   /// \return 0 on success; a DAOS error code otherwise (< 0).
   int InitializeEvent(daos_event_t *ev_ptr, daos_event_t *parent_ptr = nullptr);
   /// \brief Release event data from queue.
   /// \return 0 on success; a DAOS error code otherwise (< 0).
   int FinalizeEvent(daos_event_t *ev_ptr);
};

class RDaosContainer;

/**
  \class RDaosPool
  \brief A RDaosPool provides access to containers in a specific DAOS pool.
  */
class RDaosPool {
   friend class RDaosContainer;
private:
   daos_handle_t fPoolHandle{};
   uuid_t fPoolUuid{};
   std::string fPoolLabel{};
   std::unique_ptr<RDaosEventQueue> fEventQueue;

public:
   RDaosPool(const RDaosPool&) = delete;
   RDaosPool(std::string_view poolId);
   ~RDaosPool();

   RDaosPool& operator=(const RDaosPool&) = delete;
   std::string GetPoolUuid();
};

/**
  \class RDaosObject
  \brief Provides low-level access to DAOS objects in a container.
  */
class RDaosObject {
private:
   daos_handle_t fObjectHandle;
public:
   using DistributionKey_t = std::uint64_t;
   using AttributeKey_t = std::uint64_t;
   /// \brief Wrap around a `daos_oclass_id_t`. An object class describes the schema of data distribution
   /// and protection.
   struct ObjClassId {
      daos_oclass_id_t fCid;

      ObjClassId(daos_oclass_id_t cid) : fCid(cid) {}
      ObjClassId(const std::string &name) : fCid(daos_oclass_name2id(name.data())) {}

      bool IsUnknown() const { return fCid == OC_UNKNOWN; }
      std::string ToString() const;

      /// This limit is currently not defined in any header and any call to
      /// `daos_oclass_id2name()` within DAOS uses a stack-allocated buffer
      /// whose length varies from 16 to 50, e.g. `https://github.com/daos-stack/daos/blob/master/src/utils/daos_dfs_hdlr.c#L78`.
      /// As discussed with the development team, 64 is a reasonable limit.
      static constexpr std::size_t kOCNameMaxLength = 64;
   };

   /// \brief Contains required information for a single fetch/update operation.
   struct FetchUpdateArgs {
      FetchUpdateArgs() = default;
      FetchUpdateArgs(const FetchUpdateArgs &) = delete;
      FetchUpdateArgs(FetchUpdateArgs &&fua);
      FetchUpdateArgs(DistributionKey_t d, std::span<const AttributeKey_t> as, std::span<d_iov_t> vs,
                      bool is_async = false);
      FetchUpdateArgs &operator=(const FetchUpdateArgs &) = delete;
      daos_event_t *GetEventPointer();

      /// \brief A `daos_key_t` is a type alias of `d_iov_t`. This type stores a pointer and a length.
      /// In order for `fDistributionKey` to point to memory that we own, `fDkey` holds the distribution key.
      /// `fAkeys` and `fIovs` are sequential containers assumed to remain valid throughout the fetch/update operation.
      DistributionKey_t fDkey{};
      std::span<const AttributeKey_t> fAkeys{};
      std::span<d_iov_t> fIovs{};

      /// \brief The distribution key, as used by the `daos_obj_{fetch,update}` functions.
      daos_key_t fDistributionKey{};
      std::vector<daos_iod_t> fIods{};
      std::vector<d_sg_list_t> fSgls{};
      std::optional<daos_event_t> fEvent{};
   };

   RDaosObject() = delete;
   /// Provides low-level access to an object. If `cid` is OC_UNKNOWN, the user is responsible for
   /// calling `daos_obj_generate_oid()` to fill the reserved bits in `oid` before calling this constructor.
   RDaosObject(RDaosContainer &container, daos_obj_id_t oid, ObjClassId cid = OC_UNKNOWN);
   ~RDaosObject();

   int Fetch(FetchUpdateArgs &args);
   int Update(FetchUpdateArgs &args);
};

/**
  \class RDaosContainer
  \brief A RDaosContainer provides read/write access to objects in a given container.
  */
class RDaosContainer {
   friend class RDaosObject;
public:
   using DistributionKey_t = RDaosObject::DistributionKey_t;
   using AttributeKey_t = RDaosObject::AttributeKey_t;
   using ObjClassId_t = RDaosObject::ObjClassId;

   /// \brief A pair of <object ID, distribution key> that can be used to issue a fetch/update request for multiple
   /// attribute keys.
   struct ROidDkeyPair {
      daos_obj_id_t oid{};
      DistributionKey_t dkey{};

      inline bool operator==(const ROidDkeyPair &other) const
      {
         return this->oid.lo == other.oid.lo && this->oid.hi == other.oid.hi && this->dkey == other.dkey;
      }

      struct Hash {
         auto operator()(const ROidDkeyPair &x) const
         {
            /// Implementation borrowed from `boost::hash_combine`. Comparable to initial seeding with `oid.hi` followed
            /// by two subsequent hash calls for `oid.lo` and `dkey`.
            auto seed = std::hash<uint64_t>{}(x.oid.hi);
            seed ^= std::hash<uint64_t>{}(x.oid.lo) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<DistributionKey_t>{}(x.dkey) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
         }
      };
   };

   /// \brief Describes a read/write operation on multiple objects; see the `ReadV`/`WriteV` functions.
   struct RWOperation {
      RWOperation() = default;
      RWOperation(daos_obj_id_t o, DistributionKey_t d, const std::vector<AttributeKey_t> &as,
                  const std::vector<d_iov_t> &vs)
         : fOid(o), fDistributionKey(d), fAttributeKeys(as), fIovs(vs){};
      RWOperation(ROidDkeyPair &k) : fOid(k.oid), fDistributionKey(k.dkey){};
      daos_obj_id_t fOid{};
      DistributionKey_t fDistributionKey{};
      std::vector<AttributeKey_t> fAttributeKeys{};
      std::vector<d_iov_t> fIovs{};

      void insert(AttributeKey_t attr, const d_iov_t &vec)
      {
         fAttributeKeys.emplace_back(attr);
         fIovs.emplace_back(vec);
      }
   };

   using MultiObjectRWOperation_t = std::unordered_map<ROidDkeyPair, RWOperation, ROidDkeyPair::Hash>;

   std::string GetContainerUuid();

private:
   daos_handle_t fContainerHandle{};
   uuid_t fContainerUuid{};
   std::string fContainerLabel{};
   std::shared_ptr<RDaosPool> fPool;
   ObjClassId_t fDefaultObjectClass{OC_SX};

   /**
     \brief Perform a vector read/write operation on different objects.
     \param map A `MultiObjectRWOperation_t` that describes read/write operations to perform.
     \param cid The `daos_oclass_id_t` used to qualify OIDs.
     \param fn Either `&RDaosObject::Fetch` (read) or `&RDaosObject::Update` (write).
     \return 0 if the operation succeeded; a negative DAOS error number otherwise.
     */
   int VectorReadWrite(MultiObjectRWOperation_t &map, ObjClassId_t cid,
                       int (RDaosObject::*fn)(RDaosObject::FetchUpdateArgs &));

public:
   RDaosContainer(std::shared_ptr<RDaosPool> pool, std::string_view containerId, bool create = false);
   ~RDaosContainer();

   ObjClassId_t GetDefaultObjectClass() const { return fDefaultObjectClass; }
   void SetDefaultObjectClass(const ObjClassId_t cid) { fDefaultObjectClass = cid; }

   /**
     \brief Read data from a single object attribute key to the given buffer.
     \param buffer The address of a buffer that has capacity for at least `length` bytes.
     \param length Length of the buffer.
     \param oid A 128-bit DAOS object identifier.
     \param dkey The distribution key used for this operation.
     \param akey The attribute key used for this operation.
     \param cid An object class ID.
     \return 0 if the operation succeeded; a negative DAOS error number otherwise.
     */
   int ReadSingleAkey(void *buffer, std::size_t length, daos_obj_id_t oid,
                      DistributionKey_t dkey, AttributeKey_t akey, ObjClassId_t cid);
   int ReadSingleAkey(void *buffer, std::size_t length, daos_obj_id_t oid,
                      DistributionKey_t dkey, AttributeKey_t akey)
   { return ReadSingleAkey(buffer, length, oid, dkey, akey, fDefaultObjectClass); }

   /**
     \brief Write the given buffer to a single object attribute key.
     \param buffer The address of the source buffer.
     \param length Length of the buffer.
     \param oid A 128-bit DAOS object identifier.
     \param dkey The distribution key used for this operation.
     \param akey The attribute key used for this operation.
     \param cid An object class ID.
     \return 0 if the operation succeeded; a negative DAOS error number otherwise.
     */
   int WriteSingleAkey(const void *buffer, std::size_t length, daos_obj_id_t oid,
                       DistributionKey_t dkey, AttributeKey_t akey, ObjClassId_t cid);
   int WriteSingleAkey(const void *buffer, std::size_t length, daos_obj_id_t oid,
                       DistributionKey_t dkey, AttributeKey_t akey)
   { return WriteSingleAkey(buffer, length, oid, dkey, akey, fDefaultObjectClass); }

   /**
     \brief Perform a vector read operation on multiple objects.
     \param map A `MultiObjectRWOperation_t` that describes read operations to perform.
     \param cid An object class ID.
     \return Number of operations that could not complete.
     */
   int ReadV(MultiObjectRWOperation_t &map, ObjClassId_t cid) { return VectorReadWrite(map, cid, &RDaosObject::Fetch); }
   int ReadV(MultiObjectRWOperation_t &map) { return ReadV(map, fDefaultObjectClass); }

   /**
     \brief Perform a vector write operation on multiple objects.
     \param map A `MultiObjectRWOperation_t` that describes write operations to perform.
     \param cid An object class ID.
     \return Number of operations that could not complete.
     */
   int WriteV(MultiObjectRWOperation_t &map, ObjClassId_t cid)
   {
      return VectorReadWrite(map, cid, &RDaosObject::Update);
   }
   int WriteV(MultiObjectRWOperation_t &map) { return WriteV(map, fDefaultObjectClass); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
