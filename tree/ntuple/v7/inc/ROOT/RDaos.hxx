/// \file ROOT/RDaos.hxx
/// \ingroup NTuple ROOT7
/// \author Javier Lopez-Gomez <j.lopez@cern.ch>
/// \date 2020-11-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
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
#include <numeric>
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
   daos_pool_info_t fPoolInfo{};
   uuid_t fPoolUuid{};

public:
   RDaosPool() = delete;
   RDaosPool(std::string_view poolUuid, std::string_view serviceReplicas);
   ~RDaosPool();
};

/**
  \class RDaosObject
  \brief Provides low-level access to DAOS objects in a container.
  */
template <typename DKeyT, typename AKeyT>
class RDaosObject {
private:
   daos_handle_t fObjectHandle;

   // Provide support for UINT64 dkey/akey.
   template <typename T, typename std::enable_if<!ROOT::TypeTraits::HasDataAndSize<T>::value, int>::type = 0>
   static size_t key_size(T& x) { return sizeof(x); }

   template <typename T, typename std::enable_if<!ROOT::TypeTraits::HasDataAndSize<T>::value, int>::type = 0>
   static typename std::add_pointer<T>::type key_data(T& x) { return &x; }

   // Provide support for std::string/std::vector dkey/akey.
   template <typename T, typename std::enable_if<ROOT::TypeTraits::HasDataAndSize<T>::value, int>::type = 0>
   static size_t key_size(T& x) { return x.size(); }

   template <typename T, typename std::enable_if<ROOT::TypeTraits::HasDataAndSize<T>::value, int>::type = 0>
   static typename T::pointer key_data(T& x) { return const_cast<typename T::pointer>(x.data()); }

public:
   using DistributionKey_t = DKeyT;
   using AttributeKey_t = AKeyT;
  
   /// \brief Contains required information for a single fetch/update operation.
   struct FetchUpdateArgs {
      FetchUpdateArgs() = default;
      FetchUpdateArgs(const FetchUpdateArgs&) = delete;
      FetchUpdateArgs(FetchUpdateArgs&& fua)
        : fDkey(fua.fDkey), fAkey(fua.fAkey), fIods{fua.fIods[0]}, fSgls{fua.fSgls[0]}, fIovs(std::move(fua.fIovs)), fEv(fua.fEv)
      {
         d_iov_set(&fDistributionKey, key_data(fDkey), key_size(fDkey));
         d_iov_set(&fIods[0].iod_name, key_data(fAkey), key_size(fAkey));
      }
      FetchUpdateArgs(DistributionKey_t &d, AttributeKey_t &a, std::vector<d_iov_t> &v, daos_event_t *p = nullptr)
        : fDkey(d), fAkey(a), fIovs(v), fEv(p)
      {
         d_iov_set(&fDistributionKey, key_data(fDkey), key_size(fDkey));

         d_iov_set(&fIods[0].iod_name, key_data(fAkey), key_size(fAkey));
         fIods[0].iod_nr = 1;
         fIods[0].iod_size = std::accumulate(v.begin(), v.end(), 0,
                                             [](daos_size_t _a, d_iov_t _b) { return _a + _b.iov_len; });
         fIods[0].iod_recxs = nullptr;
         fIods[0].iod_type = DAOS_IOD_SINGLE;

         fSgls[0].sg_nr_out = 0;
         fSgls[0].sg_nr = fIovs.size();
         fSgls[0].sg_iovs = fIovs.data();
      }
      DistributionKey_t fDkey{};
      AttributeKey_t fAkey{};
      daos_key_t fDistributionKey{};
      daos_iod_t fIods[1] = {};
      d_sg_list_t fSgls[1] = {};
      std::vector<d_iov_t> fIovs{};
      daos_event_t *fEv = nullptr;
   };

   RDaosObject() = delete;
   RDaosObject(RDaosContainer &container, daos_obj_id_t oid);
   ~RDaosObject() {
      daos_obj_close(fObjectHandle, nullptr);
   }

   int Fetch(FetchUpdateArgs &args) {
      args.fIods[0].iod_size = (daos_size_t)DAOS_REC_ANY;
      return daos_obj_fetch(fObjectHandle, DAOS_TX_NONE, 0, &args.fDistributionKey, 1,
                            args.fIods, args.fSgls, nullptr, args.fEv);
   }
   int Update(FetchUpdateArgs &args) {
      return daos_obj_update(fObjectHandle, DAOS_TX_NONE, DAOS_COND_DKEY_INSERT, &args.fDistributionKey, 1,
                             args.fIods, args.fSgls, args.fEv);
   }
};

/**
  \class RDaosContainer
  \brief A RDaosContainer provides read/write access to objects in a given container.
  */
class RDaosContainer {
   template <typename DKeyT, typename AKeyT>
   friend class RDaosObject;
public:
   /// \brief Describes a read/write operation on multiple objects; see the `ReadV`/`WriteV` functions.
   template <typename DKeyT, typename AKeyT>
   struct RWOperation {
      RWOperation() = default;
      RWOperation(daos_obj_id_t o, DKeyT d, AKeyT a, std::vector<d_iov_t> &v)
         : fOid(o), fDistributionKey(d), fAttributeKey(a), fIovs(v) {};
      daos_obj_id_t fOid{};
      DKeyT fDistributionKey{};
      AKeyT fAttributeKey{};
      std::vector<d_iov_t> fIovs{};
   };

private:
   struct DaosEventQueue {
      std::size_t fSize;
      std::unique_ptr<daos_event_t[]> fEvs;
      daos_handle_t fQueue;
      DaosEventQueue(std::size_t size);
      ~DaosEventQueue();
      int Poll();
   };

   daos_handle_t fContainerHandle{};
   daos_cont_info_t fContainerInfo{};
   uuid_t fContainerUuid{};
   std::shared_ptr<RDaosPool> fPool;
   /// OID that will be used by the next call to `WriteObject(const void *, std::size_t, DKeyT, AKeyT)`.
   daos_obj_id_t fSequentialWrOid{};

   /** \brief Perform a vector read/write operation on different objects.
     \param vec A `std::vector<RWOperation>` that describes read/write operations to perform.
     \param fn Either `std::mem_fn<&RDaosObject::Fetch>` (read) or `std::mem_fn<&RDaosObject::Update>` (write).
     */
   template <typename Fn, typename DKeyT, typename AKeyT>
   int VectorReadWrite(std::vector<RWOperation<DKeyT, AKeyT>> &vec, Fn fn) {
      using _RDaosObject = RDaosObject<DKeyT, AKeyT>;
      int ret;
      DaosEventQueue eventQueue(vec.size());
      {
         std::vector<std::tuple<std::unique_ptr<_RDaosObject>, typename _RDaosObject::FetchUpdateArgs>> requests{};
         requests.reserve(vec.size());
         for (size_t i = 0; i < vec.size(); ++i) {
            requests.push_back(std::make_tuple(std::unique_ptr<_RDaosObject>(new _RDaosObject(*this, vec[i].fOid)),
                                               typename _RDaosObject::FetchUpdateArgs{vec[i].fDistributionKey, vec[i].fAttributeKey, vec[i].fIovs, &eventQueue.fEvs[i]}));
            fn(std::get<0>(requests.back()).get(), std::get<1>(requests.back()));
         }
         ret = eventQueue.Poll();
      }
      return ret;
   }

public:
   RDaosContainer(std::shared_ptr<RDaosPool> pool, std::string_view containerUuid, bool create = false);
   ~RDaosContainer();

   /** \brief Read data from an object in this container to the given buffer. */
   template <typename DKeyT, typename AKeyT>
   int ReadObject(daos_obj_id_t oid, void *buffer, std::size_t length, DKeyT dkey, AKeyT akey)
   {
      std::vector<d_iov_t> iovs(1);
      d_iov_set(&iovs[0], buffer, length);
      typename RDaosObject<DKeyT, AKeyT>::FetchUpdateArgs args(dkey, akey, iovs);
      return RDaosObject<DKeyT, AKeyT>(*this, oid).Fetch(args);
   }

   /** \brief Write the given buffer to an object in this container. */
   template <typename DKeyT, typename AKeyT>
   int WriteObject(daos_obj_id_t oid, const void *buffer, std::size_t length, DKeyT dkey, AKeyT akey)
   {
      std::vector<d_iov_t> iovs(1);
      d_iov_set(&iovs[0], const_cast<void *>(buffer), length);
      typename RDaosObject<DKeyT, AKeyT>::FetchUpdateArgs args(dkey, akey, iovs);
      return RDaosObject<DKeyT, AKeyT>(*this, oid).Update(args);
   }

   /** \brief Write the given buffer to an object in this container and return a generated OID. */
   template <typename DKeyT, typename AKeyT>
   std::tuple<daos_obj_id_t, int> WriteObject(const void *buffer, std::size_t length, DKeyT dkey, AKeyT akey)
   {
      auto ret = std::make_tuple(fSequentialWrOid,
                                 WriteObject(fSequentialWrOid, buffer, length, dkey, akey));
      fSequentialWrOid.lo++;
      return ret;
   }

   /** \brief Perform a vector read operation on (possibly) multiple objects.
     \param vec A `std::vector<RWOperation>` that describes read operations to perform. */
   template <typename DKeyT, typename AKeyT>
   int ReadV(std::vector<RWOperation<DKeyT, AKeyT>> &vec)
   { return VectorReadWrite(vec, std::mem_fn(&RDaosObject<DKeyT, AKeyT>::Fetch)); }

   /** \brief Perform a vector write operation on (possibly) multiple objects.
     \param vec A `std::vector<RWOperation>` that describes write operations to perform. */
   template <typename DKeyT, typename AKeyT>
   int WriteV(std::vector<RWOperation<DKeyT, AKeyT>> &vec)
   { return VectorReadWrite(vec, std::mem_fn(&RDaosObject<DKeyT, AKeyT>::Update)); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

template <typename DKeyT, typename AKeyT>
ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::RDaosObject(RDaosContainer &container, daos_obj_id_t oid) {
   daos_ofeat_t ofeats{};
   if (std::is_same<std::uint64_t, DKeyT>::value)
      ofeats |= DAOS_OF_DKEY_UINT64;
   if (std::is_same<std::uint64_t, AKeyT>::value)
      ofeats |= DAOS_OF_AKEY_UINT64;
   daos_obj_generate_id(&oid, ofeats /*| DAOS_OF_ARRAY_BYTE*/, OC_RP_XSF, 0);
   if (int err = daos_obj_open(container.fContainerHandle, oid, DAOS_OO_RW, &fObjectHandle, nullptr))
      throw std::runtime_error("daos_obj_open: error: " + std::string(d_errstr(err)));
}

#endif
