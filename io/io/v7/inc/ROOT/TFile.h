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

namespace Internal {
/** \class TFileImplBase
 Base class for storage-specific ROOT file implementations.

 A class deriving from `TFileImplBase` is an object store: it can serialize any
 object for which ROOT I/O is available (generally: an object which has a
 dictionary), and it stores the object's data under a key name.

 A `TFileImplBase` stores whatever was added to it as a `TDirectory`, when the
 `TFileImplBase` object is destructed. It can store non-lifetime managed objects
 by passing them to `Save()`.

 */
class TFileImplBase: public TDirectory {
public:
  ~TFileImplBase() = default;

  /// Save all objects associated with this directory to the storage medium.
  virtual void Flush() = 0;

  /// Flush() and make the file non-writable: close it.
  virtual void Close() = 0;

  template <class T>
  void Write(const std::string& /*name*/, const T& /*ptr*/) {}

};
}


/**
 \class TFilePtr
 \brief Points to an object that stores or reads objects in ROOT's binary
 format.

 */

class TFilePtr {
TCoopPtr<Internal::TFileImplBase> fImpl;

public:
  ///\name Generator functions
  ///\{
  /// Open a file with `name` for reading.
  static TFilePtr Read(std::string_view name);

  /// Open a file with `name` for reading and writing. Fail (return an invalid
  /// `TFilePtr`) if a file with this name already exists.
  static TFilePtr Create(std::string_view name);

  /// Open a file with `name` for reading and writing. If a file with this name
  /// already exists, delete it and create a new one. Else simply create a new file.
  static TFilePtr Recreate(std::string_view name);

  /// Open an existing file with `name` for reading and writing. If a file with
  /// that name does not exist, an invalid TFilePtr will be returned.
  static TFilePtr Update(std::string_view name);
  ///\}

  /// Dereference the file pointer, giving access to the TFileImplBase object.
  Internal::TFileImplBase* operator ->() { return fImpl.Get(); }

  /// Dereference the file pointer, giving access to the TFileImplBase object.
  /// const overload.
  const Internal::TFileImplBase* operator ->() const { return fImpl.Get(); }

  /// Check the validity of the file pointer.
  operator bool() const { return fImpl; }

private:
  /// Constructed by
  TFilePtr(TCoopPtr<Internal::TFileImplBase>);
};
}
#endif
