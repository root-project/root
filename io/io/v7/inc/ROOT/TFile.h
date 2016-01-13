/// \file TFile.h
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

#ifndef ROOT7_TFile
#define ROOT7_TFile

#include "ROOT/TDirectory.h"

#include <memory>
#include <experimental/string_view>

namespace ROOT {
namespace Experimental {

namespace Internal {
/** \class TFileImplBase
 Base class for storage-specific ROOT file implementations.

 A class deriving from `TFileImplBase` is an object store: it can serialize any
 object for which ROOT I/O is available (generally: an object which has a
 dictionary), and it stores the object's data under a key name.

 */
class TFileImplBase: public TDirectory {
public:
  /// Must not call Write() of all attached objects:
  /// some might not be needed to be written or writing might be aborted due to
  /// an exception; require explicit Write().
  ~TFileImplBase() = default;

  /// Save all objects associated with this directory (including file header) to
  /// the storage medium.
  virtual void Flush() = 0;

  /// Flush() and make the file non-writable: close it.
  virtual void Close() = 0;

  /// Write an object that is not lifetime managed by this TFileImplBase.
  template <class T>
  void Write(const std::string& /*name*/, const T& /*obj*/) {}

  /// Write an object that is lifetime managed by this TFileImplBase.
  void Write(const std::string& /*name*/) {}

};
}


/**
 \class TFilePtr
 \brief Points to an object that stores or reads objects in ROOT's binary
 format.

 */

class TFilePtr {
std::shared_ptr<Internal::TFileImplBase> fImpl;

public:
  ///\name Generator functions
  ///\{

  /// Open a file with `name` for reading.
  static TFilePtr OpenForRead(std::string_view name);

  /// Open an existing file with `name` for reading and writing. If a file with
  /// that name does not exist, an invalid TFilePtr will be returned.
  static TFilePtr OpenForUpdate(std::string_view name);

  /// Open a file with `name` for reading and writing. Fail (return an invalid
  /// `TFilePtr`) if a file with this name already exists.
  static TFilePtr Create(std::string_view name);

  /// Open a file with `name` for reading and writing. If a file with this name
  /// already exists, delete it and create a new one. Else simply create a new file.
  static TFilePtr Recreate(std::string_view name);

  ///\}

  /// Dereference the file pointer, giving access to the TFileImplBase object.
  Internal::TFileImplBase* operator ->() { return fImpl.get(); }

  /// Dereference the file pointer, giving access to the TFileImplBase object.
  /// const overload.
  const Internal::TFileImplBase* operator ->() const { return fImpl.get(); }

  /// Check the validity of the file pointer.
  operator bool() const { return fImpl.get(); }

private:
  /// Constructed by Open etc.
  TFilePtr(std::unique_ptr<Internal::TFileImplBase>&&);
};

} // namespace Experimental
} // namespace ROOT
#endif
