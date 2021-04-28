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
   RDaosObject(RDaosContainer &container, daos_obj_id_t oid, daos_oclass_id_t cid = OC_RP_XSF);
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
   using DistributionKey_t = std::uint64_t;
   using AttributeKey_t = std::uint64_t;
  
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
   /// OID that will be used by the next call to `WriteObject(const void *, std::size_t, DistributionKey_t, AttributeKey_t)`.
   daos_obj_id_t fSequentialWrOid{};

   /**
     \brief Perform a vector read/write operation on different objects.
     \param vec A `std::vector<RWOperation>` that describes read/write operations to perform.
     \param fn Either `std::mem_fn<&RDaosObject::Fetch>` (read) or `std::mem_fn<&RDaosObject::Update>` (write).
     \return Number of requests that did not complete; this should be 0 after a successful call.
     */
   template <typename Fn>
   int VectorReadWrite(std::vector<RWOperation> &vec, Fn fn) {
      int ret;
      DaosEventQueue eventQueue(vec.size());
      {
         std::vector<std::tuple<std::unique_ptr<RDaosObject>, RDaosObject::FetchUpdateArgs>> requests{};
         requests.reserve(vec.size());
         for (size_t i = 0; i < vec.size(); ++i) {
            requests.push_back(std::make_tuple(std::unique_ptr<RDaosObject>(new RDaosObject(*this, vec[i].fOid)),
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

   /**
     \brief Read data from an object in this container to the given buffer.
     \param oid A 128-bit DAOS object identifier.
     \param buffer The address of a buffer that has capacity for at least `length` bytes.
     \param length Length of the buffer.
     \param dkey The distribution key used for this operation.
     \param akey The attribute key used for this operation.
     \return 0 if the operation succeeded; a negative DAOS error number otherwise.
     */
   int ReadObject(daos_obj_id_t oid, void *buffer, std::size_t length, DistributionKey_t dkey, AttributeKey_t akey);

   /**
     \brief Write the given buffer to an object in this container.
     \param oid A 128-bit DAOS object identifier.
     \param buffer The address of the source buffer.
     \param length Length of the buffer.
     \param dkey The distribution key used for this operation.
     \param akey The attribute key used for this operation.
     \return 0 if the operation succeeded; a negative DAOS error number otherwise.
     */
   int WriteObject(daos_obj_id_t oid, const void *buffer, std::size_t length, DistributionKey_t dkey, AttributeKey_t akey);

   /**
     \brief Write the given buffer to an object in this container and return a generated OID.
     \param buffer The address of the source buffer.
     \param length Length of the buffer.
     \param dkey The distribution key used for this operation.
     \param akey The attribute key used for this operation.
     \return A `std::tuple<>` that contains the generated OID and a DAOS error number (0 if the operation succeeded).
     */
   std::tuple<daos_obj_id_t, int>
   WriteObject(const void *buffer, std::size_t length, DistributionKey_t dkey, AttributeKey_t akey);

   /**
     \brief Perform a vector read operation on (possibly) multiple objects.
     \param vec A `std::vector<RWOperation>` that describes read operations to perform.
     \return Number of operations that could not complete.
     */
   int ReadV(std::vector<RWOperation> &vec)
   { return VectorReadWrite(vec, std::mem_fn(&RDaosObject::Fetch)); }

   /**
     \brief Perform a vector write operation on (possibly) multiple objects.
     \param vec A `std::vector<RWOperation>` that describes write operations to perform.
     \return Number of operations that could not complete.
     */
   int WriteV(std::vector<RWOperation> &vec)
   { return VectorReadWrite(vec, std::mem_fn(&RDaosObject::Update)); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
