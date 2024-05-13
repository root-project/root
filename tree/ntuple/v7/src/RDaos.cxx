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

ROOT::Experimental::Internal::RDaosPool::RDaosPool(std::string_view poolId)
{
   {
      static struct RDaosRAII {
         RDaosRAII() { daos_init(); }
         ~RDaosRAII() { daos_fini(); }
      } RAII = {};
   }

   daos_pool_info_t poolInfo{};

   if (daos_label_is_valid(poolId.data()))
      fPoolLabel = std::string(poolId);

   if (int err = daos_pool_connect(poolId.data(), nullptr, DAOS_PC_RW, &fPoolHandle, &poolInfo, nullptr)) {
      throw RException(R__FAIL("daos_pool_connect: error: " + std::string(d_errstr(err))));
   }
   uuid_copy(fPoolUuid, poolInfo.pi_uuid);

   fEventQueue = std::make_unique<RDaosEventQueue>();
}

ROOT::Experimental::Internal::RDaosPool::~RDaosPool()
{
   daos_pool_disconnect(fPoolHandle, nullptr);
}

std::string ROOT::Experimental::Internal::RDaosPool::GetPoolUuid()
{
   char id[DAOS_UUID_STR_SIZE];
   uuid_unparse(fPoolUuid, id);
   return std::string(id);
}

////////////////////////////////////////////////////////////////////////////////

std::string ROOT::Experimental::Internal::RDaosObject::ObjClassId::ToString() const
{
   char name[kOCNameMaxLength + 1] = {};
   daos_oclass_id2name(fCid, name);
   return std::string{name};
}

ROOT::Experimental::Internal::RDaosObject::FetchUpdateArgs::FetchUpdateArgs(FetchUpdateArgs &&fua) noexcept
   : fDkey(fua.fDkey),
     fRequests(fua.fRequests),
     fIods(std::move(fua.fIods)),
     fSgls(std::move(fua.fSgls)),
     fEvent(fua.fEvent)
{
   d_iov_set(&fDistributionKey, &fDkey, sizeof(fDkey));
}

ROOT::Experimental::Internal::RDaosObject::FetchUpdateArgs::FetchUpdateArgs(DistributionKey_t d,
                                                                            std::span<RAkeyRequest> rs, bool is_async)
   : fDkey(d), fRequests(rs)
{
   if (is_async)
      fEvent.emplace();

   fSgls.reserve(fRequests.size());
   fIods.reserve(fRequests.size());
   d_iov_set(&fDistributionKey, &fDkey, sizeof(fDkey));

   for (auto &r : fRequests) {
      daos_iod_t iod;
      iod.iod_nr = 1;
      iod.iod_size =
         std::accumulate(r.fIovs.begin(), r.fIovs.end(), 0, [](size_t c, d_iov_t &iov) { return c + iov.iov_len; });
      iod.iod_recxs = nullptr;
      iod.iod_type = DAOS_IOD_SINGLE;
      d_iov_set(&iod.iod_name, const_cast<AttributeKey_t *>(&r.fAkey), sizeof(r.fAkey));
      fIods.push_back(iod);

      d_sg_list_t sgl;
      sgl.sg_nr_out = 0;
      sgl.sg_nr = r.fIovs.size();
      sgl.sg_iovs = r.fIovs.data();
      fSgls.push_back(sgl);
   }
}

daos_event_t *ROOT::Experimental::Internal::RDaosObject::FetchUpdateArgs::GetEventPointer()
{
   return fEvent ? &(fEvent.value()) : nullptr;
}

ROOT::Experimental::Internal::RDaosObject::RDaosObject(RDaosContainer &container, daos_obj_id_t oid, ObjClassId cid)
{
   if (!cid.IsUnknown())
      daos_obj_generate_oid(container.fContainerHandle, &oid, DAOS_OT_MULTI_UINT64, cid.fCid,
                            DAOS_OCH_RDD_DEF | DAOS_OCH_SHD_DEF, 0);

   if (int err = daos_obj_open(container.fContainerHandle, oid, DAOS_OO_RW, &fObjectHandle, nullptr))
      throw RException(R__FAIL("daos_obj_open: error: " + std::string(d_errstr(err))));
}

ROOT::Experimental::Internal::RDaosObject::~RDaosObject()
{
   daos_obj_close(fObjectHandle, nullptr);
}

int ROOT::Experimental::Internal::RDaosObject::Fetch(FetchUpdateArgs &args)
{
   return daos_obj_fetch(fObjectHandle, DAOS_TX_NONE, DAOS_COND_DKEY_FETCH | DAOS_COND_AKEY_FETCH,
                         &args.fDistributionKey, args.fIods.size(), args.fIods.data(), args.fSgls.data(), nullptr,
                         args.GetEventPointer());
}

int ROOT::Experimental::Internal::RDaosObject::Update(FetchUpdateArgs &args)
{
   return daos_obj_update(fObjectHandle, DAOS_TX_NONE, 0, &args.fDistributionKey, args.fIods.size(), args.fIods.data(),
                          args.fSgls.data(), args.GetEventPointer());
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RDaosEventQueue::RDaosEventQueue()
{
   if (int ret = daos_eq_create(&fQueue))
      throw RException(R__FAIL("daos_eq_create: error: " + std::string(d_errstr(ret))));
}

ROOT::Experimental::Internal::RDaosEventQueue::~RDaosEventQueue()
{
   daos_eq_destroy(fQueue, 0);
}

int ROOT::Experimental::Internal::RDaosEventQueue::InitializeEvent(daos_event_t *ev_ptr, daos_event_t *parent_ptr) const
{
   return daos_event_init(ev_ptr, fQueue, parent_ptr);
}

int ROOT::Experimental::Internal::RDaosEventQueue::FinalizeEvent(daos_event_t *ev_ptr)
{
   return daos_event_fini(ev_ptr);
}

int ROOT::Experimental::Internal::RDaosEventQueue::WaitOnParentBarrier(daos_event_t *ev_ptr)
{
   int err;
   bool flag;

   if ((err = daos_event_parent_barrier(ev_ptr)) < 0)
      return err;

   if ((err = daos_event_test(ev_ptr, DAOS_EQ_WAIT, &flag)) < 0)
      return err;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RDaosContainer::RDaosContainer(std::shared_ptr<RDaosPool> pool,
                                                             std::string_view containerId, bool create)
   : fPool(pool)
{
   daos_cont_info_t containerInfo{};

   // Creating containers supported only with a valid label (not UUID).
   if (create && daos_label_is_valid(containerId.data())) {
      fContainerLabel = std::string(containerId);
      if (int err =
             daos_cont_create_with_label(fPool->fPoolHandle, fContainerLabel.data(), nullptr, nullptr, nullptr)) {
         // Ignore error for re-creating existing container.
         if (err != -DER_EXIST)
            throw RException(R__FAIL("daos_cont_create_with_label: error: " + std::string(d_errstr(err))));
      }
   }

   // Opening containers is supported by valid label or UUID
   if (int err = daos_cont_open(fPool->fPoolHandle, containerId.data(), DAOS_COO_RW, &fContainerHandle, &containerInfo,
                                nullptr))
      throw RException(R__FAIL("daos_cont_open: error: " + std::string(d_errstr(err))));
   uuid_copy(fContainerUuid, containerInfo.ci_uuid);
}

ROOT::Experimental::Internal::RDaosContainer::~RDaosContainer()
{
   daos_cont_close(fContainerHandle, nullptr);
}

std::string ROOT::Experimental::Internal::RDaosContainer::GetContainerUuid()
{
   char id[DAOS_UUID_STR_SIZE];
   uuid_unparse(fContainerUuid, id);
   return std::string(id);
}

int ROOT::Experimental::Internal::RDaosContainer::ReadSingleAkey(void *buffer, std::size_t length, daos_obj_id_t oid,
                                                                 DistributionKey_t dkey, AttributeKey_t akey,
                                                                 ObjClassId_t cid)
{
   std::vector<d_iov_t> iovs(1);
   d_iov_set(&iovs[0], const_cast<void *>(buffer), length);
   RDaosObject::RAkeyRequest requests[] = {{akey, std::move(iovs)}};
   RDaosObject::FetchUpdateArgs args(dkey, requests);
   return RDaosObject(*this, oid, cid.fCid).Fetch(args);
}

int ROOT::Experimental::Internal::RDaosContainer::WriteSingleAkey(const void *buffer, std::size_t length,
                                                                  daos_obj_id_t oid, DistributionKey_t dkey,
                                                                  AttributeKey_t akey, ObjClassId_t cid)
{

   std::vector<d_iov_t> iovs(1);
   d_iov_set(&iovs[0], const_cast<void *>(buffer), length);
   RDaosObject::RAkeyRequest requests[] = {{akey, std::move(iovs)}};
   RDaosObject::FetchUpdateArgs args(dkey, requests);
   return RDaosObject(*this, oid, cid.fCid).Update(args);
}

int ROOT::Experimental::Internal::RDaosContainer::VectorReadWrite(
   MultiObjectRWOperation_t &map, ObjClassId_t cid, int (RDaosObject::*fn)(RDaosObject::FetchUpdateArgs &))
{
   using request_t = std::tuple<std::unique_ptr<RDaosObject>, RDaosObject::FetchUpdateArgs>;

   int ret;
   std::vector<request_t> requests{};
   requests.reserve(map.size());

   // Initialize parent event used for grouping and waiting for completion of all requests
   daos_event_t parent_event{};
   if ((ret = fPool->fEventQueue->InitializeEvent(&parent_event)) < 0)
      return ret;

   for (auto &[key, batch] : map) {
      requests.emplace_back(
         std::make_unique<RDaosObject>(*this, batch.fOid, cid.fCid),
         RDaosObject::FetchUpdateArgs{batch.fDistributionKey, batch.fDataRequests, /*is_async=*/true});

      if ((ret = fPool->fEventQueue->InitializeEvent(std::get<1>(requests.back()).GetEventPointer(), &parent_event)) <
          0)
         return ret;

      // Launch operation
      if ((ret = (std::get<0>(requests.back()).get()->*fn)(std::get<1>(requests.back()))) < 0)
         return ret;
   }

   // Sets parent barrier and waits for all children launched before it.
   if ((ret = fPool->fEventQueue->WaitOnParentBarrier(&parent_event)) < 0)
      return ret;

   return fPool->fEventQueue->FinalizeEvent(&parent_event);
}
