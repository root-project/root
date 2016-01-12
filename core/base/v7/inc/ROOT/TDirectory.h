/// \file TDirectory.h
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDirectory
#define ROOT7_TDirectory

#include "ROOT/TLogger.h"
#include "ROOT/TDirectoryEntry.h"

#include <memory>
#include <unordered_map>
#include <experimental/string_view>

namespace ROOT {
namespace Experimental {

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
  /// The directory content is a hashed map of
  /// name => `Internal::TDirectoryEntryPtr`.
  using ContentMap_t
    = std::unordered_map<std::string,
                         std::unique_ptr<Internal::TDirectoryEntryPtrBase>>;

  /// The `TDirectory`'s content.
  ContentMap_t fContent;

public:

  /// Create an object of type `T` (passing some arguments to its constructor).
  /// The `TDirectory` will have shared ownership of the object.
  ///
  /// \param name  - Key of the object.
  /// \param args  - arguments to be passed to the constructor of `T`
  template <class T, class... ARGS>
  std::weak_ptr<T> Create(const std::string& name, ARGS... args) {
    return Add(name, std::make_shared<T>(std::forward<ARGS...>(args...)));
  }

  /// FIXME: this should return an iterator of some sort.
  const Internal::TDirectoryEntryPtrBase* FindKey(const std::string& name) const {
    auto idx = fContent.find(name);
    if (idx == fContent.end())
      return nullptr;
    return idx->second.get();
  }

  /// Add an existing object (rather a `shared_ptr` to it) to the TDirectory.
  /// The TDirectory will have shared ownership.
  template <class T>
  void Add(const std::string& name, const std::shared_ptr<T>& ptr) {
    ContentMap_t::iterator idx = fContent.find(name);
    if (idx != fContent.end()) {
      R__LOG_HERE(ELogLevel::kWarning, "CORE")
        << "Replacing object with name " << name;
      idx->second.swap(std::make_unique<Internal::TDirectoryEntryPtr<T>>(ptr));
    }
    fContent.insert({name, ptr});
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

} // namespace Experimental
} // namespace ROOT

#endif
