/// \file ROOT/TDirectoryEntry.h
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

#include "TClass.h"

#include <chrono>
#include <memory>
#include <typeinfo>

namespace ROOT {
namespace Experimental {

namespace Internal {

class TDirectoryEntry {
public:
  using clock_t = std::chrono::system_clock;
  using time_point_t = std::chrono::time_point<clock_t>;

public:
  TDirectoryEntry(): TDirectoryEntry(nullptr) {}
  TDirectoryEntry(std::nullptr_t):
    TDirectoryEntry(std::make_shared<std::nullptr_t>(nullptr)) {}
  template<class T>
  explicit TDirectoryEntry(T* ptr):
    TDirectoryEntry(std::make_shared<T>(*ptr)) {}
  template<class T>
  explicit TDirectoryEntry(const std::shared_ptr<T>& ptr):
    fTypeInfo(&typeid(T)),
    fType(TClass::GetClass(typeid(T))),
    fObj(ptr) {}

  /// Get the last change date of the entry.
  const time_point_t& GetDate() const { return fDate; }

  /// Inform the entry that it has been modified, and needs to update its
  /// last-changed time stamp.
  void SetChanged() { fDate = clock_t::now(); }

  /// Type of the object represented by this entry.
  const std::type_info& GetTypeInfo() const { return *fTypeInfo; }

  /// Get the object's type.
  TClass* GetType() const { return fType; }

  /// Retrieve the `shared_ptr` of the referenced object.
  std::shared_ptr<void>& GetPointer() { return fObj; }
  const std::shared_ptr<void>& GetPointer() const { return fObj; }

  template<class U>
  std::shared_ptr<U> CastPointer() const;

  explicit operator bool() const { return *fTypeInfo != typeid(std::nullptr_t); }

private:
  time_point_t fDate = clock_t::now(); ///< Time of last change
  const std::type_info* fTypeInfo;
  TClass* fType;
  std::shared_ptr<void> fObj;
};

template<class U>
std::shared_ptr<U> TDirectoryEntry::CastPointer() const
{
  if (auto ptr = fType->DynamicCast(TClass::GetClass(typeid(U)), fObj.get()))
    return std::shared_ptr<U>(fObj, static_cast<U*>(ptr));
  return std::shared_ptr<U>();
}

inline bool operator==(const TDirectoryEntry& lhs, const TDirectoryEntry& rhs)
{
  return lhs.GetPointer() == rhs.GetPointer()
    && lhs.GetTypeInfo() == rhs.GetTypeInfo();
}

inline bool operator!=(const TDirectoryEntry& lhs, const TDirectoryEntry& rhs)
{ return !(lhs == rhs); }

} // namespace Internal

} // namespace Experimental
} // namespace ROOT
#endif
