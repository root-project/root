/// \file libdaos_mock.cxx
/// \ingroup NTuple ROOT7
/// \author Javier Lopez-Gomez <j.lopez@cern.ch>
/// \date 2021-01-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RStringView.hxx>

#include <daos.h>

#include <array>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

using label_t = std::string;
namespace std {
// Required by `std::unordered_map<daos_obj_id, ...>`. Based on boost::hash_combine().
template <>
struct hash<daos_obj_id_t> {
   std::size_t operator()(const daos_obj_id_t &oid) const
   {
      auto seed = std::hash<uint64_t>{}(oid.lo);
      seed ^= std::hash<uint64_t>{}(oid.hi) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
   }
};
} // namespace std

inline bool operator==(const daos_obj_id_t &lhs, const daos_obj_id_t &rhs)
{
   return (lhs.lo == rhs.lo) && (lhs.hi == rhs.hi);
}

namespace {
// clang-format off
/**
\class RDaosFakeObject
\brief Manages in-memory storage for a fake DAOS object.

Currently, only 1 I/O descriptor/scather-gather list is supported.
*/
// clang-format on
class RDaosFakeObject {
private:
   std::mutex fMutexStorage;
   std::unordered_map<std::string, std::string> fStorage;

   /// \brief Return the internal storage key by concatenating both dkey and akey.
   static std::string GetKey(daos_key_t *dkey, daos_key_t *akey)
   {
      return std::string{reinterpret_cast<char *>(dkey->iov_buf), dkey->iov_buf_len}.append(
         reinterpret_cast<char *>(akey->iov_buf), akey->iov_buf_len);
   }

public:
   RDaosFakeObject() = default;
   ~RDaosFakeObject() = default;

   int Fetch(daos_key_t *dkey, unsigned int nr, daos_iod_t *iods, d_sg_list_t *sgls);
   int Update(daos_key_t *dkey, unsigned int nr, daos_iod_t *iods, d_sg_list_t *sgls);
};

int RDaosFakeObject::Fetch(daos_key_t *dkey, unsigned int nr, daos_iod_t *iods, d_sg_list_t *sgls)
{
   /* For documentation see DAOS' daos_obj_fetch */

   std::lock_guard<std::mutex> lock(fMutexStorage);
   /* Iterate over pairs of (I/O descriptor, scatter-gather list) */
   for (unsigned i = 0; i < nr; i++) {
      /* Retrieve entry data for (dkey, akey). Fails if not found */
      auto data = fStorage.find(GetKey(dkey, /*akey=*/&iods[i].iod_name));
      if (data == fStorage.end())
         return -DER_INVAL;

      /* In principle, we can safely assume that each attribute key is associated to a single value,
       * i.e. one extent per I/O descriptor; and that the corresponding data is copied to exactly one
       * I/O vector. */
      if (iods[i].iod_nr != 1 || iods[i].iod_type != DAOS_IOD_SINGLE)
         return -DER_INVAL;
      if (sgls[i].sg_nr != 1)
         return -DER_INVAL;

      d_iov_t &iov = sgls[i].sg_iovs[0];
      std::copy_n(std::begin(data->second), std::min(iov.iov_buf_len, data->second.size()),
                  reinterpret_cast<char *>(iov.iov_buf));
   }
   return 0;
}

int RDaosFakeObject::Update(daos_key_t *dkey, unsigned int nr, daos_iod_t *iods, d_sg_list_t *sgls)
{
   /* For documentation see DAOS' daos_obj_update */

   std::lock_guard<std::mutex> lock(fMutexStorage);
   /* Process each I/O descriptor and associated SG list */
   for (unsigned i = 0; i < nr; i++) {
      auto &data = fStorage[GetKey(dkey, /*akey=*/&iods[i].iod_name)];
      /* We assume each attribute key is associated to a single value whose corresponding data is
       * updated from exactly one I/O vector. */
      if (iods[i].iod_nr != 1 || iods[i].iod_type != DAOS_IOD_SINGLE)
         return -DER_INVAL;
      if (sgls[i].sg_nr != 1)
         return -DER_INVAL;

      d_iov_t &iov = sgls[i].sg_iovs[0];
      data.assign(reinterpret_cast<char *>(iov.iov_buf), iov.iov_buf_len); // Write to buffer
   }
   return 0;
}

// clang-format off
/**
\class RDaosFakeContainer
\brief Manages objects in a fake DAOS container.
*/
// clang-format on
class RDaosFakeContainer {
private:
   std::mutex fMutexObjects;
   std::unordered_map<daos_obj_id_t, std::unique_ptr<RDaosFakeObject>> fObjects;

public:
   RDaosFakeContainer() = default;
   ~RDaosFakeContainer() = default;

   RDaosFakeObject *GetObject(daos_obj_id_t oid, unsigned int /*mode*/)
   {
      std::lock_guard<std::mutex> lock(fMutexObjects);
      auto &obj = fObjects[oid];
      if (!obj)
         obj = std::make_unique<RDaosFakeObject>();
      return obj.get();
   }
};

// clang-format off
/**
\class RDaosFakePool
\brief Manages user-defined containers in a fake DAOS pool.
*/
// clang-format on
class RDaosFakePool {
private:
   static std::mutex fMutexPools;
   static std::unordered_map<label_t, std::unique_ptr<RDaosFakePool>> fPools;

   std::mutex fMutexContainers;
   std::unordered_map<label_t, std::unique_ptr<RDaosFakeContainer>> fContainers;

public:
   /// \brief Get a pointer to a RDaosFakePool object associated to the given UUID.
   /// Non-existent pools shall be created on-demand.
   static RDaosFakePool *GetPool(const label_t &label)
   {
      std::lock_guard<std::mutex> lock(fMutexPools);
      auto &pool = fPools[label];
      if (!pool)
         pool = std::make_unique<RDaosFakePool>();
      return pool.get();
   }

   RDaosFakePool() = default;
   ~RDaosFakePool() = default;

   void CreateContainer(const label_t &label)
   {
      std::lock_guard<std::mutex> lock(fMutexContainers);
      fContainers.emplace(label, std::make_unique<RDaosFakeContainer>());
   }

   RDaosFakeContainer *GetContainer(const label_t &label)
   {
      std::lock_guard<std::mutex> lock(fMutexContainers);
      auto it = fContainers.find(label);
      return (it != fContainers.end()) ? it->second.get() : nullptr;
   }
};

std::mutex RDaosFakePool::fMutexPools;
std::unordered_map<label_t, std::unique_ptr<RDaosFakePool>> RDaosFakePool::fPools;

// clang-format off
/**
\class RDaosHandle
\brief Translates a `daos_handle_t` to a pointer to object and viceversa.

A `daos_handle_t` is used by some API functions (in particular, those that work
with pools, containers, or objects) to reference an entity. This type (aka
`uint64_t`) is large enough for a pointer in all architectures. However, an
indirection layer is added in order to detect the use of invalidated handles.
*/
// clang-format on
class RDaosHandle {
private:
   /// \brief Wrapper over a `void *` that may help to detect the use of invalid handles.
   struct Cookie {
      Cookie(void *p) : fPointer(p) {}
      ~Cookie() { fPointer = nullptr; }
      void *GetPointer() { return fPointer; }

      void *fPointer;
   };

public:
   template <typename T>
   static inline daos_handle_t ToHandle(const T &p)
   {
      return {reinterpret_cast<decltype(daos_handle_t::cookie)>(new Cookie(p))};
   }

   template <typename T>
   static inline typename std::add_pointer<T>::type ToPointer(const daos_handle_t h)
   {
      return reinterpret_cast<typename std::add_pointer<T>::type>(reinterpret_cast<Cookie *>(h.cookie)->GetPointer());
   }

   static inline void Invalidate(daos_handle_t h) { delete reinterpret_cast<Cookie *>(h.cookie); }
};

} // anonymous namespace

extern "C" {
int daos_init(void)
{
   R__LOG_WARNING(ROOT::Experimental::NTupleLog()) << "This RNTuple build uses libdaos_mock. Use only for testing!";
   return 0;
}

int daos_fini(void)
{
   return 0;
}

const char *d_errstr(int rc)
{
   return rc ? "DER_INVAL" : "Success";
}

int daos_oclass_name2id(const char *name)
{
   if (strcmp(name, "SX") == 0)
      return OC_SX;
   if (strcmp(name, "RP_XSF") == 0)
      return OC_RP_XSF;
   return OC_UNKNOWN;
}

int daos_oclass_id2name(daos_oclass_id_t oc_id, char *name)
{
   switch (oc_id) {
   case OC_SX:
      strcpy(name, "SX"); // NOLINT
      return 0;
   case OC_RP_XSF:
      strcpy(name, "RP_XSF"); // NOLINT
      return 0;
   }
   return -1;
}

////////////////////////////////////////////////////////////////////////////////

int daos_cont_create_with_label(daos_handle_t poh, const char *label, daos_prop_t * /*cont_prop*/, uuid_t * /*uuid*/,
                                daos_event_t * /*ev*/)
{
   auto pool = RDaosHandle::ToPointer<RDaosFakePool>(poh);
   if (!pool)
      return -DER_INVAL;

   if (!daos_label_is_valid(label))
      return -DER_INVAL;

   pool->CreateContainer(label_t(label));
   return 0;
}

int daos_cont_open(daos_handle_t poh, const char *label, unsigned int /*flags*/, daos_handle_t *coh,
                   daos_cont_info_t * /*info*/, daos_event_t * /*ev*/)
{
   auto pool = RDaosHandle::ToPointer<RDaosFakePool>(poh);
   if (!pool)
      return -DER_INVAL;

   if (!daos_label_is_valid(label))
      return -DER_INVAL;

   auto cont = pool->GetContainer(label_t(label));
   if (!cont)
      return -DER_INVAL;
   *coh = RDaosHandle::ToHandle(cont);
   return 0;
}

int daos_cont_close(daos_handle_t coh, daos_event_t * /*ev*/)
{
   RDaosHandle::Invalidate(coh);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

int daos_eq_create(daos_handle_t * /*eqh*/)
{
   return 0;
}

int daos_eq_destroy(daos_handle_t /*eqh*/, int /*flags*/)
{
   return 0;
}

int daos_eq_poll(daos_handle_t /*eqh*/, int /*wait_running*/, int64_t /*timeout*/, unsigned int nevents,
                 daos_event_t ** /*events*/)
{
   return nevents;
}

int daos_event_test(daos_event * /*ev*/, int64_t /*timeout*/, bool *flag)
{
   if (flag != nullptr)
      *flag = true;
   return 0;
}

int daos_event_parent_barrier(daos_event_t * /*ev*/)
{
   return 0;
}

int daos_event_init(daos_event_t * /*ev*/, daos_handle_t /*eqh*/, daos_event_t * /*parent*/)
{
   return 0;
}

int daos_event_fini(daos_event_t * /*ev*/)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

int daos_obj_open(daos_handle_t coh, daos_obj_id_t oid, unsigned int mode, daos_handle_t *oh, daos_event_t * /*ev*/)
{
   auto cont = RDaosHandle::ToPointer<RDaosFakeContainer>(coh);
   if (!cont)
      return -DER_INVAL;
   auto obj = cont->GetObject(oid, mode);
   *oh = RDaosHandle::ToHandle(obj);
   return 0;
}

int daos_obj_close(daos_handle_t oh, daos_event_t * /*ev*/)
{
   RDaosHandle::Invalidate(oh);
   return 0;
}

int daos_obj_generate_oid(daos_handle_t /*coh*/, daos_obj_id_t *oid, enum daos_otype_t type, daos_oclass_id_t cid,
                          daos_oclass_hints_t /*hints*/, uint32_t /*args*/)
{
   uint64_t hdr;

   /* Validate user-specified bits are not reserved by DAOS */
   if (!daos_otype_t_is_valid(type))
      return -DER_INVAL;

   oid->hi &= (1ULL << OID_FMT_INTR_BITS) - 1;
   // | Upper bits contain
   // | OID_FMT_TYPE_BITS (object type)
   // | OID_FMT_META_BITS (object class metadata)
   // | OID_FMT_CLASS_BITS (object class)
   // | 96-bit for upper layer
   hdr = ((uint64_t)type << OID_FMT_TYPE_SHIFT);
   hdr |= ((uint64_t)0 << OID_FMT_META_SHIFT);
   hdr |= ((uint64_t)cid << OID_FMT_CLASS_SHIFT);
   oid->hi |= hdr;

   return 0;
}

int daos_obj_fetch(daos_handle_t oh, daos_handle_t /*th*/, uint64_t /*flags*/, daos_key_t *dkey, unsigned int nr,
                   daos_iod_t *iods, d_sg_list_t *sgls, daos_iom_t * /*ioms*/, daos_event_t * /*ev*/)
{
   auto obj = RDaosHandle::ToPointer<RDaosFakeObject>(oh);
   if (!obj)
      return -DER_INVAL;
   return obj->Fetch(dkey, nr, iods, sgls);
}

int daos_obj_update(daos_handle_t oh, daos_handle_t /*th*/, uint64_t /*flags*/, daos_key_t *dkey, unsigned int nr,
                    daos_iod_t *iods, d_sg_list_t *sgls, daos_event_t * /*ev*/)
{
   auto obj = RDaosHandle::ToPointer<RDaosFakeObject>(oh);
   if (!obj)
      return -DER_INVAL;
   return obj->Update(dkey, nr, iods, sgls);
}

////////////////////////////////////////////////////////////////////////////////

int daos_pool_connect(const char *label, const char * /*grp*/, unsigned int /*flags*/, daos_handle_t *poh,
                      daos_pool_info_t * /*info*/, daos_event_t * /*ev*/)
{

   *poh = RDaosHandle::ToHandle(RDaosFakePool::GetPool(label_t(label)));
   return 0;
}

int daos_pool_disconnect(daos_handle_t poh, daos_event_t * /*ev*/)
{
   RDaosHandle::Invalidate(poh);
   return 0;
}

} // extern "C"
