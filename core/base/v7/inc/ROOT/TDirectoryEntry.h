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

class TDirectoryEntryPtrBase {
public:
  virtual ~TDirectoryEntryPtrBase() {}
  using clock_t = std::chrono::system_clock;
  using time_point_t = std::chrono::time_point<clock_t>;

  const time_point_t& GetDate() const { return fDate; }
  void SetChanged() { fDate = clock_t::now(); }

  virtual void* GetObjectAddr() const = 0;
  virtual const std::type_info& GetTypeInfo() const = 0;

private:
  time_point_t fDate = clock_t::now();
};

template <class T>
class TDirectoryEntryPtr: public TDirectoryEntryPtrBase {
  std::shared_ptr<T> fObj;
public:
  TDirectoryEntryPtr(T&& obj): fObj(obj) {}
  TDirectoryEntryPtr(const std::shared_ptr<T>& ptr): fObj(ptr) {}

  virtual ~TDirectoryEntryPtr() {}

  std::shared_ptr<T>& GetPointer() const { return fObj; }

  void* GetObjectAddr() const final { return fObj.get(); }
  const std::type_info& GetTypeInfo() const final { return typeid(T); }
};

} // namespace Internal

} // namespace Experimental
} // namespace ROOT
#endif
