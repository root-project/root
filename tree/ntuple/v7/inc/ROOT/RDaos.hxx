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

#include <daos.h>
// Avoid depending on `gurt/common.h` as the only required declaration is `d_rank_list_free()`.
// Also, this header file is known to provide macros that conflict with std::min()/std::max().
extern "C" void d_rank_list_free(d_rank_list_t *rank_list);

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ROOT {

namespace Experimental {
namespace Detail {
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

public:
   RDaosPool(const RDaosPool&) = delete;
   RDaosPool(std::string_view poolUuid, std::string_view serviceReplicas);
   ~RDaosPool();

   RDaosPool& operator=(const RDaosPool&) = delete;
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
      FetchUpdateArgs(const FetchUpdateArgs&) = delete;
      FetchUpdateArgs(FetchUpdateArgs&& fua);
      FetchUpdateArgs(DistributionKey_t &d, AttributeKey_t &a, std::vector<d_iov_t> &v, daos_event_t *p = nullptr);
      FetchUpdateArgs& operator=(const FetchUpdateArgs&) = delete;

      /// \brief A `daos_key_t` is a type alias of `d_iov_t`. This type stores a pointer and a length.
      /// In order for `fDistributionKey` and `fIods` to point to memory that we own, `fDkey` and
      /// `fAkey` store a copy of the distribution and attribute key, respectively.
      DistributionKey_t fDkey{};
      AttributeKey_t fAkey{};

      /// \brief The distribution key, as used by the `daos_obj_{fetch,update}` functions.
      daos_key_t fDistributionKey{};
      daos_iod_t fIods[1] = {};
      d_sg_list_t fSgls[1] = {};
      std::vector<d_iov_t> fIovs{};
      daos_event_t *fEv = nullptr;
   };

   RDaosObject() = delete;
   /// Provides low-level access to an object. If `cid` is OC_UNKNOWN, the user is responsible for
   /// calling `daos_obj_generate_id()` to fill the reserved bits in `oid` before calling this constructor.
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
  
   /// \brief Describes a read/write operation on multiple objects; see the `ReadV`/`WriteV` functions.
   struct RWOperation {
      RWOperation() = default;
      RWOperation(daos_obj_id_t o, DistributionKey_t d, AttributeKey_t a, std::vector<d_iov_t> &v)
         : fOid(o), fDistributionKey(d), fAttributeKey(a), fIovs(v) {};
      daos_obj_id_t fOid{};
      DistributionKey_t fDistributionKey{};
      AttributeKey_t fAttributeKey{};
      std::vector<d_iov_t> fIovs{};
   };

private:
   struct DaosEventQueue {
      std::size_t fSize;
      std::unique_ptr<daos_event_t[]> fEvs;
      daos_handle_t fQueue;
      DaosEventQueue(std::size_t size);
      ~DaosEventQueue();
      /**
        \brief Wait for all events in this event queue to complete.
        \return Number of events still in the queue. This should be 0 on success.
       */
      int Poll();
   };

   daos_handle_t fContainerHandle{};
   uuid_t fContainerUuid{};
   std::shared_ptr<RDaosPool> fPool;
   ObjClassId_t fDefaultObjectClass{OC_SX};

   /**
     \brief Perform a vector read/write operation on different objects.
     \param vec A `std::vector<RWOperation>` that describes read/write operations to perform.
     \param cid The `daos_oclass_id_t` used to qualify OIDs.
     \param fn Either `std::mem_fn<&RDaosObject::Fetch>` (read) or `std::mem_fn<&RDaosObject::Update>` (write).
     \return Number of requests that did not complete; this should be 0 after a successful call.
     */
   template <typename Fn>
   int VectorReadWrite(std::vector<RWOperation> &vec, ObjClassId_t cid, Fn fn) {
      int ret;
      DaosEventQueue eventQueue(vec.size());
      {
         std::vector<std::tuple<std::unique_ptr<RDaosObject>, RDaosObject::FetchUpdateArgs>> requests{};
         requests.reserve(vec.size());
         for (size_t i = 0; i < vec.size(); ++i) {
           requests.push_back(std::make_tuple(std::unique_ptr<RDaosObject>(new RDaosObject(*this, vec[i].fOid, cid.fCid)),
                                               RDaosObject::FetchUpdateArgs{
                                                 vec[i].fDistributionKey, vec[i].fAttributeKey,
                                                 vec[i].fIovs, &eventQueue.fEvs[i]}));
            fn(std::get<0>(requests.back()).get(), std::get<1>(requests.back()));
         }
         ret = eventQueue.Poll();
      }
      return ret;
   }

public:
   RDaosContainer(std::shared_ptr<RDaosPool> pool, std::string_view containerUuid, bool create = false);
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
     \brief Perform a vector read operation on (possibly) multiple objects.
     \param vec A `std::vector<RWOperation>` that describes read operations to perform.
     \param cid An object class ID.
     \return Number of operations that could not complete.
     */
   int ReadV(std::vector<RWOperation> &vec, ObjClassId_t cid)
   { return VectorReadWrite(vec, cid, std::mem_fn(&RDaosObject::Fetch)); }
   int ReadV(std::vector<RWOperation> &vec) { return ReadV(vec, fDefaultObjectClass); }

   /**
     \brief Perform a vector write operation on (possibly) multiple objects.
     \param vec A `std::vector<RWOperation>` that describes write operations to perform.
     \param cid An object class ID.
     \return Number of operations that could not complete.
     */
   int WriteV(std::vector<RWOperation> &vec, ObjClassId_t cid)
   { return VectorReadWrite(vec, cid, std::mem_fn(&RDaosObject::Update)); }
   int WriteV(std::vector<RWOperation> &vec) { return WriteV(vec, fDefaultObjectClass); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
