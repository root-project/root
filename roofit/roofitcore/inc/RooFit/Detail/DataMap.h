/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  12/2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_DataMap_h
#define RooFit_Detail_DataMap_h

#include <RooAbsArg.h>
#include <RooSpan.h>

#include <TNamed.h>
#include <TObject.h>

#include <map>
#include <stdexcept>
#include <sstream>

template <class T>
class RooTemplateProxy;

/// \class RooFit::DataKey
/// To use as a key type for RooFit data maps and containers. A RooFit::DataKey
/// can be constructed with no runtime overhead from a RooAbsArg (or any
/// templated RooFit proxy for convenience). Compared to using the RooAbsArg
/// pointer directly, this has the advantage that one can easily change the way
/// the key is constructed from the object, just by changing the implementation
/// of the DataKey. For example, it is trivial to move from using the RooAbsArg
/// pointer to using the unique name pointer retrieved by RooAbsArg::namePtr().

namespace RooFit {
namespace Detail {

class DataKey {
public:
   inline DataKey(RooAbsArg const *arg) : _ptr{arg->namePtr()} {}
   inline DataKey(TNamed const *arg) : _ptr{arg} {}
   template <class T>
   inline DataKey(RooTemplateProxy<T> const &proxy) : DataKey{&*proxy}
   {
   }

   // Comparison operators that wrap the pointer comparisons.
   friend inline bool operator==(const DataKey &k1, const DataKey &k2) { return k1._ptr == k2._ptr; }
   friend inline bool operator!=(const DataKey &k1, const DataKey &k2) { return k1._ptr != k2._ptr; }
   friend inline bool operator<(const DataKey &k1, const DataKey &k2) { return k1._ptr < k2._ptr; }

   // Implementing pointer-style operators.
   inline TObject const &operator*() const { return *_ptr; }
   inline TObject const *operator->() const { return _ptr; }

private:
   TObject const *_ptr;
};

} // namespace Detail
} // namespace RooFit

namespace std {

template <>
struct hash<RooFit::Detail::DataKey> {
   std::size_t operator()(const RooFit::Detail::DataKey &k) const { return hash<TObject const *>{}(&*k); }
};

} // namespace std

namespace RooFit {
namespace Detail {

class DataMap {
public:
   auto empty() const { return _dataMap.empty(); }
   auto begin() { return _dataMap.begin(); }
   auto end() { return _dataMap.end(); }
   auto begin() const { return _dataMap.begin(); }
   auto end() const { return _dataMap.end(); }
   auto size() const { return _dataMap.size(); }
   auto resize(std::size_t n) { return _dataMap.resize(n); }

   inline auto &at(RooAbsArg const *arg, RooAbsArg const * /*caller*/ = nullptr)
   {
      std::size_t idx = arg->dataToken();
      return _dataMap[idx];
   }

   inline auto &at(RooAbsArg const *arg, RooAbsArg const *caller = nullptr) const
   {
      return const_cast<DataMap *>(this)->at(arg, caller);
   }

   template <class T>
   inline auto &at(RooTemplateProxy<T> const &proxy)
   {
      return at(&proxy.arg(), proxy.owner());
   }

   template <class T>
   inline auto &at(RooTemplateProxy<T> const &proxy) const
   {
      return at(&proxy.arg(), proxy.owner());
   }

private:
   std::vector<RooSpan<const double>> _dataMap;
};

} // namespace Detail
} // namespace RooFit

#endif
