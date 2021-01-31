/// \file RDaos.cxx
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

#include <ROOT/RDaos.hxx>

#include <numeric>
#include <stdexcept>

ROOT::Experimental::Detail::RDaosPool::RDaosPool(std::string_view poolUuid, std::string_view serviceReplicas) {
   struct SvcRAII {
      d_rank_list_t *rankList;
      SvcRAII(std::string_view ranks) { rankList = daos_rank_list_parse(ranks.data(), "_"); }
      ~SvcRAII() { d_rank_list_free(rankList); }
   } Svc(serviceReplicas);

   uuid_parse(poolUuid.data(), fPoolUuid);
   if (int err = daos_pool_connect(fPoolUuid, nullptr, Svc.rankList, DAOS_PC_RW, &fPoolHandle, &fPoolInfo, nullptr))
      throw std::runtime_error("daos_pool_connect: error: " + std::string(d_errstr(err)));
}

ROOT::Experimental::Detail::RDaosPool::~RDaosPool() {
   daos_pool_disconnect(fPoolHandle, nullptr);
}


////////////////////////////////////////////////////////////////////////////////


template <typename DKeyT, typename AKeyT>
ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::FetchUpdateArgs::FetchUpdateArgs(FetchUpdateArgs&& fua)
  : fDkey(fua.fDkey), fAkey(fua.fAkey),
    fIods{fua.fIods[0]}, fSgls{fua.fSgls[0]}, fIovs(std::move(fua.fIovs)), fEv(fua.fEv)
{
   d_iov_set(&fDistributionKey, key_data(fDkey), key_size(fDkey));
   d_iov_set(&fIods[0].iod_name, key_data(fAkey), key_size(fAkey));
}

template <typename DKeyT, typename AKeyT>
ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::FetchUpdateArgs::FetchUpdateArgs
(DistributionKey_t &d, AttributeKey_t &a, std::vector<d_iov_t> &v, daos_event_t *p)
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

template <typename DKeyT, typename AKeyT>
ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::RDaosObject(RDaosContainer &container, daos_obj_id_t oid,
                                                                   daos_oclass_id_t cid)
{
   daos_ofeat_t ofeats{};
   if (std::is_same<std::uint64_t, DKeyT>::value)
      ofeats |= DAOS_OF_DKEY_UINT64;
   if (std::is_same<std::uint64_t, AKeyT>::value)
      ofeats |= DAOS_OF_AKEY_UINT64;
   daos_obj_generate_id(&oid, ofeats /*| DAOS_OF_ARRAY_BYTE*/, cid, 0);
   if (int err = daos_obj_open(container.fContainerHandle, oid, DAOS_OO_RW, &fObjectHandle, nullptr))
      throw std::runtime_error("daos_obj_open: error: " + std::string(d_errstr(err)));
}

template <typename DKeyT, typename AKeyT>
ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::~RDaosObject()
{
   daos_obj_close(fObjectHandle, nullptr);
}

template <typename DKeyT, typename AKeyT>
int ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::Fetch(FetchUpdateArgs &args)
{
   args.fIods[0].iod_size = (daos_size_t)DAOS_REC_ANY;
   return daos_obj_fetch(fObjectHandle, DAOS_TX_NONE, 0, &args.fDistributionKey, 1,
                         args.fIods, args.fSgls, nullptr, args.fEv);
}

template <typename DKeyT, typename AKeyT>
int ROOT::Experimental::Detail::RDaosObject<DKeyT, AKeyT>::Update(FetchUpdateArgs &args)
{
   return daos_obj_update(fObjectHandle, DAOS_TX_NONE, DAOS_COND_DKEY_INSERT, &args.fDistributionKey, 1,
                          args.fIods, args.fSgls, args.fEv);
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RDaosContainer::DaosEventQueue::DaosEventQueue(std::size_t size)
  : fSize(size), fEvs(std::unique_ptr<daos_event_t[]>(new daos_event_t[size]))
{
   daos_eq_create(&fQueue);
   for (std::size_t i = 0; i < fSize; ++i)
      daos_event_init(&fEvs[i], fQueue, nullptr);
}

ROOT::Experimental::Detail::RDaosContainer::DaosEventQueue::~DaosEventQueue() {
   for (std::size_t i = 0; i < fSize; ++i)
      daos_event_fini(&fEvs[i]);
   daos_eq_destroy(fQueue, 0);
}

int ROOT::Experimental::Detail::RDaosContainer::DaosEventQueue::Poll() {
   auto evp = std::unique_ptr<daos_event_t*[]>(new daos_event_t*[fSize]);
   std::size_t n = fSize;
   while (n) {
      int c;
      if ((c = daos_eq_poll(fQueue, 0, DAOS_EQ_WAIT, n, evp.get())) < 0)
         break;
      n -= c;
   }
   return n;
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RDaosContainer::RDaosContainer(std::shared_ptr<RDaosPool> pool,
                                                           std::string_view containerUuid, bool create)
  : fPool(pool)
{
   uuid_parse(containerUuid.data(), fContainerUuid);
   if (create) {
      if (int err = daos_cont_create(fPool->fPoolHandle, fContainerUuid, nullptr, nullptr))
         throw std::runtime_error("daos_cont_create: error: " + std::string(d_errstr(err)));
   }
   if (int err = daos_cont_open(fPool->fPoolHandle, fContainerUuid, DAOS_COO_RW,
         &fContainerHandle, &fContainerInfo, nullptr))
      throw std::runtime_error("daos_cont_open: error: " + std::string(d_errstr(err)));
}

ROOT::Experimental::Detail::RDaosContainer::~RDaosContainer() {
   daos_cont_close(fContainerHandle, nullptr);
}


////////////////////////////////////////////////////////////////////////////////


namespace {
static struct RDaosRAII {
   RDaosRAII() { daos_init(); }
   ~RDaosRAII() { daos_fini(); }
} RAII{};
} // anonymous namespace

// Explicit instantiations of `RDaosObject` for specific dkey/akey types
template class ROOT::Experimental::Detail::RDaosObject<uint64_t, uint64_t>;
