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
#include <ROOT/RError.hxx>

#include <numeric>
#include <stdexcept>

ROOT::Experimental::Detail::RDaosPool::RDaosPool(std::string_view poolLabel)
{
   {
      static struct RDaosRAII {
         RDaosRAII() { daos_init(); }
         ~RDaosRAII() { daos_fini(); }
      } RAII = {};
   }

   fPoolLabel = poolLabel;

   daos_pool_info_t poolInfo{};
   if (int err = daos_pool_connect(fPoolLabel.data(), nullptr, DAOS_PC_RW, &fPoolHandle, &poolInfo, nullptr)) {
      throw RException(R__FAIL("daos_pool_connect: error: " + std::string(d_errstr(err))));
   }
}

ROOT::Experimental::Detail::RDaosPool::~RDaosPool()
{
   daos_pool_disconnect(fPoolHandle, nullptr);
}

////////////////////////////////////////////////////////////////////////////////


std::string ROOT::Experimental::Detail::RDaosObject::ObjClassId::ToString() const
{
   char name[kOCNameMaxLength + 1] = {};
   daos_oclass_id2name(fCid, name);
   return std::string{name};
}


ROOT::Experimental::Detail::RDaosObject::FetchUpdateArgs::FetchUpdateArgs(FetchUpdateArgs&& fua)
  : fDkey(fua.fDkey), fAkey(fua.fAkey),
    fIods{fua.fIods[0]}, fSgls{fua.fSgls[0]}, fIovs(std::move(fua.fIovs)), fEv(fua.fEv)
{
   d_iov_set(&fDistributionKey, &fDkey, sizeof(fDkey));
   d_iov_set(&fIods[0].iod_name, &fAkey, sizeof(fAkey));
}

ROOT::Experimental::Detail::RDaosObject::FetchUpdateArgs::FetchUpdateArgs
(DistributionKey_t &d, AttributeKey_t &a, std::vector<d_iov_t> &v, daos_event_t *p)
  : fDkey(d), fAkey(a), fIovs(v), fEv(p)
{
   d_iov_set(&fDistributionKey, &fDkey, sizeof(fDkey));

   d_iov_set(&fIods[0].iod_name, &fAkey, sizeof(fAkey));
   fIods[0].iod_nr = 1;
   fIods[0].iod_size = std::accumulate(v.begin(), v.end(), 0,
                                       [](daos_size_t _a, d_iov_t _b) { return _a + _b.iov_len; });
   fIods[0].iod_recxs = nullptr;
   fIods[0].iod_type = DAOS_IOD_SINGLE;

   fSgls[0].sg_nr_out = 0;
   fSgls[0].sg_nr = fIovs.size();
   fSgls[0].sg_iovs = fIovs.data();
}

ROOT::Experimental::Detail::RDaosObject::RDaosObject(RDaosContainer &container, daos_obj_id_t oid,
                                                     ObjClassId cid)
{
   if (!cid.IsUnknown())
      daos_obj_generate_oid(container.fContainerHandle, &oid, DAOS_OT_MULTI_UINT64, cid.fCid,
                            DAOS_OCH_RDD_DEF | DAOS_OCH_SHD_DEF, 0);

   if (int err = daos_obj_open(container.fContainerHandle, oid, DAOS_OO_RW, &fObjectHandle, nullptr))
      throw RException(R__FAIL("daos_obj_open: error: " + std::string(d_errstr(err))));
}

ROOT::Experimental::Detail::RDaosObject::~RDaosObject()
{
   daos_obj_close(fObjectHandle, nullptr);
}

int ROOT::Experimental::Detail::RDaosObject::Fetch(FetchUpdateArgs &args)
{
   args.fIods[0].iod_size = (daos_size_t)DAOS_REC_ANY;
   return daos_obj_fetch(fObjectHandle, DAOS_TX_NONE,
                         DAOS_COND_DKEY_FETCH | DAOS_COND_AKEY_FETCH,
                         &args.fDistributionKey, 1, args.fIods, args.fSgls, nullptr, args.fEv);
}

int ROOT::Experimental::Detail::RDaosObject::Update(FetchUpdateArgs &args)
{
   return daos_obj_update(fObjectHandle, DAOS_TX_NONE, 0, &args.fDistributionKey, 1,
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
                                                           std::string_view containerLabel, bool create)
   : fPool(pool)
{
   daos_cont_info_t containerInfo{};

   fContainerLabel = containerLabel;

   if (create) {
      if (int err =
             daos_cont_create_with_label(fPool->fPoolHandle, fContainerLabel.data(), nullptr, nullptr, nullptr)) {
         if (err != -DER_EXIST)
            throw RException(R__FAIL("daos_cont_create_with_label: error: " + std::string(d_errstr(err))));
      }
   }
   if (int err = daos_cont_open(fPool->fPoolHandle, fContainerLabel.data(), DAOS_COO_RW, &fContainerHandle,
                                &containerInfo, nullptr))
      throw RException(R__FAIL("daos_cont_open: error: " + std::string(d_errstr(err))));
}

ROOT::Experimental::Detail::RDaosContainer::~RDaosContainer() {
   daos_cont_close(fContainerHandle, nullptr);
}

int ROOT::Experimental::Detail::RDaosContainer::ReadSingleAkey(void *buffer, std::size_t length, daos_obj_id_t oid,
                                                               DistributionKey_t dkey, AttributeKey_t akey,
                                                               ObjClassId_t cid)
{
   std::vector<d_iov_t> iovs(1);
   d_iov_set(&iovs[0], buffer, length);
   RDaosObject::FetchUpdateArgs args(dkey, akey, iovs);
   return RDaosObject(*this, oid, cid.fCid).Fetch(args);
}

int ROOT::Experimental::Detail::RDaosContainer::WriteSingleAkey(const void *buffer, std::size_t length, daos_obj_id_t oid,
                                                                DistributionKey_t dkey, AttributeKey_t akey,
                                                                ObjClassId_t cid)
{
   std::vector<d_iov_t> iovs(1);
   d_iov_set(&iovs[0], const_cast<void *>(buffer), length);
   RDaosObject::FetchUpdateArgs args(dkey, akey, iovs);
   return RDaosObject(*this, oid, cid.fCid).Update(args);
}
