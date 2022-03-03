/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  12/2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOFIT_BATCHCOMPUTE_DATAKEY_H
#define ROOFIT_BATCHCOMPUTE_DATAKEY_H

#include <TObject.h>

/// \class RooBatchCompute::DataKey
/// To use as a key type for RooBatchCompute data maps and containers. A
/// RooBatchCompute::DataKey can be constructed with no runtime overhead from
/// the pointer to any class T that implements a casting operator to DataKey.
/// Compared to using the pointer to the object of type T directly, this has
/// the advantage that one can easily change the way the key is constructed
/// from the object, just by changing the implementation of the casting
/// operator. For example, it is trivial to move from using the RooAbsArg
/// pointer to using the unique name pointer retrieved by RooAbsArg::namePtr().

namespace RooBatchCompute {
class DataKey {
public:
   /// The only public DataKey constructor from any object of type T that
   /// implements a casting operator to DataKey.
   template <class T>
   constexpr inline DataKey(T const *ptr) : DataKey{static_cast<DataKey>(*ptr)}
   {
   }

   /// If a class T implements a casting operator to DataKey, it needs a way to
   /// actually construct the DataKey.
   constexpr inline static DataKey create(TObject const *obj) { return DataKey(Intermediate{obj}); }

   // Comparison operators that wrap the pointer comparisons.
   friend constexpr inline bool operator==(const DataKey &k1, const DataKey &k2) { return k1._ptr == k2._ptr; }
   friend constexpr inline bool operator!=(const DataKey &k1, const DataKey &k2) { return k1._ptr != k2._ptr; }
   friend constexpr inline bool operator<(const DataKey &k1, const DataKey &k2) { return k1._ptr < k2._ptr; }

   // Implementing pointer-style operators.
   constexpr inline TObject const &operator*() const { return *_ptr; }
   constexpr inline TObject const *operator->() const { return _ptr; }

private:
   /// Intermediate struct to construct the DataKey from, such that
   /// constructing the DataKey directly from a TObject-derived class without
   /// implementing a casting operator is not possible.
   struct Intermediate {
      TObject const *_ptr;
   };

   // The construction operator is explicit so we cond accidentally create a
   // DataKey from a TObject without going over an implicit casting operator.
   constexpr inline explicit DataKey(Intermediate &&intermediate) : _ptr{intermediate._ptr} {}

   TObject const *_ptr;
};

} // namespace RooBatchCompute

#endif
