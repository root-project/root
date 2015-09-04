/// \file TDirectory.h
/// \ingroup Base
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDirectory
#define ROOT7_TDirectory

#include "ROOT/TCoopPtr.h"
#include "ROOT/TLogger.h"
#include "ROOT/TKey.h"

#include <unordered_map>
#include <experimental/string_view>

namespace ROOT {

/**
 Key/value store of objects.

 Given a name, a `TDirectory` can store and retrieve an object. It will manage
 ownership through a `TCoopPtr`: if you delete the object, the object will be
 gone from the `TDirectory`. Once the `TDirectory `is destructed, the objects
 it contains are destructed (unless other `TCoopPtr`s reference the same
 object).

 Example:
  TDirectory dirBackgrounds;
  TDirectory dirQCD;
  TDirectory dirHiggs;

 */
class TDirectory {
  // Don't keep a unique_ptr as value in a map.
  /// The values referenced by a TDirectory are type erased `TCoopPtr`s: they
  /// can be of any type; the actual type is determined through the virtual
  /// interface of Internal::TCoopPtrTypeErasedBase.
  using Value_t = std::shared_ptr<Internal::TCoopPtrTypeErasedBase>;

  /// The directory content is a hashed map of `TKey` => `Value_t`.
  /// TODO: really? Or just std::string => Value_t - that should be enough!
  /// Rather add some info (time stamp etc) to the Value_t.
  using ContentMap_t = std::unordered_map<TKey, Value_t>;

  /// The `TDirectory`'s content.
  ContentMap_t fContent;

public:

  /// Create an object of type `T` (passing some arguments to its constructor).
  /// The `TDirectory` will register the object.
  ///
  /// \param name - Key of the object.
  /// \param args... - arguments to be passed to the constructor of `T`
  template <class T, class... ARGS>
  TCoopPtr<T> Create(const std::string& name, ARGS... args) {
    TCoopPtr<T> ptr(new T(args...));
    Add(name, ptr);
    return ptr;
  }

  // TODO: The key should probably be a simple std::string. Do we need the TKey
  // at all? Even for TFile, the key should stay a std::string, and the value
  // should have an additional meta payload (timestamp). Think of object store.
  const TKey* FindKey(const std::string& name) const {
    auto idx = fContent.find(name);
    if (idx == fContent.end())
      return nullptr;
    return &idx->first;
  }

  /// Add an existing object (rather a `TCoopPtr` to it) to the TDirectory. The
  /// TDirectory will not delete the object but it will need to be notified once
  /// the object is deleted.
  template <class T>
  const TKey& Add(const std::string& name, TCoopPtr<T> ptr) {
    ContentMap_t::iterator idx = fContent.find(name);
    if (idx != fContent.end()) {
      R__LOG_HERE(ELogLevel::kWarning, "CORE")
        << "Replacing object with name " << name;
      idx->second = std::make_unique<Internal::TCoopPtrTypeErased<T>>(ptr);
      // The cast is fine: we do not change the key's value (hash)
      const_cast<TKey&>(idx->first).SetChanged();
      return idx->first;
    }
    return fContent.insert({name,
                            std::make_shared<Internal::TCoopPtrTypeErased<T>>(ptr)})
       .first->first;
  }

  /// Dedicated, process-wide TDirectory.
  ///
  /// \note This is *not* thread-safe. You will need to syncronize yourself. In
  /// general it's a bad idea to use a global collection in a multi-threaded
  /// environment; ROOT itself does not make use of it. It is merely offered for
  /// historical, process-wide object registration by name. Instead, pass a
  /// pointer to the object where you need to access it - this is also much
  /// faster than a lookup by name.
  static TDirectory& Heap();
};

}

#endif
