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
   inline DataKey(RooAbsArg const* arg) : _ptr{arg->namePtr()} {}
   inline DataKey(TNamed const* arg) : _ptr{arg} {}
   template <class T>
   inline DataKey(RooTemplateProxy<T> const& proxy) : DataKey{&*proxy} {}

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

using DataMap = std::map<DataKey, RooSpan<const double>>;

} // namespace Detail
} // namespace RooFit

namespace std {

template <>
struct hash<RooFit::Detail::DataKey> {
   std::size_t operator()(const RooFit::Detail::DataKey &k) const { return hash<TObject const *>{}(&*k); }
};

} // namespace std

#endif
