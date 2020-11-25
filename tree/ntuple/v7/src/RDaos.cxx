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
