/// \file TDirectoryEntry.h
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

#ifndef ROOT7_TDirectoryEntry
#define ROOT7_TDirectoryEntry

#include <chrono>
#include <memory>
#include <typeinfo>

namespace ROOT {
namespace Experimental {

namespace Internal {

/**
  Abstract base for a `TDirectory` element.
 */
class TDirectoryEntryPtrBase {
public:
  virtual ~TDirectoryEntryPtrBase() {}
  using clock_t = std::chrono::system_clock;
  using time_point_t = std::chrono::time_point<clock_t>;

  /// Get the last change date of the entry.
  const time_point_t& GetDate() const { return fDate; }

  /// Inform the entry that it has been modified, and needs to update its
  /// last-changed time stamp.
  void SetChanged() { fDate = clock_t::now(); }

  /// Abstract interface to retrieve the address of the object represented by
  /// this entry, if any.
  virtual void* GetObjectAddr() const = 0;

  /// Abstract interface to retrieve the type of the object represented by this
  /// entry.
  virtual const std::type_info& GetTypeInfo() const = 0;

private:
  time_point_t fDate = clock_t::now(); ///< Time of last change
};


/**
 Type-aware part of an entry in a TDirectory. It can manage the lifetime of an
 object, or only inform about last-change date and the type of an object.
 */

template <class T>
class TDirectoryEntryPtr: public TDirectoryEntryPtrBase {
public:
  /// Initialize a TDirectoryEntryPtr from an existing object ("write").
  TDirectoryEntryPtr(T&& obj): fObj(obj) {}

  /// Initialize a TDirectoryEntryPtr from an existing object ("write").
  TDirectoryEntryPtr(const std::shared_ptr<T>& ptr): fObj(ptr) {}

  virtual ~TDirectoryEntryPtr() {}

  /// Retrieve the `shared_ptr` of the referenced object.
  std::shared_ptr<T>& GetPointer() const { return fObj; }

  /// Retrieve the address of the object managed by this entry.
  /// Returns `nullptr` if no object is managed by this entry.
  void* GetObjectAddr() const final { return fObj.get(); }

  /// Type of the object represented by this entry.
  const std::type_info& GetTypeInfo() const final { return typeid(T); }

private:
  std::shared_ptr<T> fObj; ///< Object referenced by this entry, if any.
};

} // namespace Internal

} // namespace Experimental
} // namespace ROOT
#endif
