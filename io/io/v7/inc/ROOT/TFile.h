/// \file ROOT/TFile.h
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

  /// Read the object for a key. `T` must be the object's type.
  /// This will re-read the object for each call, returning a new copy; whether
  /// the `TDirectory` is managing an object attached to this key or not.
  /// \returns a `unique_ptr` to the object.
  /// \throws TDirectoryUnknownKey if no object is stored under this name.
  /// \throws TDirectoryTypeMismatch if the object stored under this name is of
  ///   a type different from `T`.
  template <class T>
  std::unique_ptr<T> Read(const std::string& name) {
    // FIXME: need separate collections for a TDirectory's key/value and registered objects. Here, we want to emit a read and must look through the key/values without attaching an object to the TDirectory.
    if (const Internal::TDirectoryEntryPtrBase* dep = Find(name)) {
      // FIXME: implement upcast!
      // FIXME: do not register read object in TDirectory
      // FIXME: implement actual read
      if (auto depT = dynamic_cast<const Internal::TDirectoryEntryPtr<T>*>(dep)) {
        //FIXME: for now, copy out of whatever the TDirectory manages.
        return std::make_unique<T>(*depT->GetPointer());
      }
      // FIXME: add expected versus actual type name as c'tor args
      throw TDirectoryTypeMismatch(name);
    }
    throw TDirectoryUnknownKey(name);
    return std::shared_ptr<T>(); // never happens
  }


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

 FIXME: implement async open; likely using std::future, possibly removing the
 Option_t element.

 */

class TFilePtr {
std::shared_ptr<Internal::TFileImplBase> fImpl;

public:
  /// Options for TFilePtr construction.
  struct Options_t {
    /// Default constructor needed for member inits.
    Options_t() {}

    /// Whether the file should be opened asynchronously, if available.
    bool fAsynchronousOpen = false;

    /// Timeout for asynchronous opening.
    int fAsyncTimeout = 0;

    /// Whether the file should be cached before reading. Only available for
    /// "remote" file protocols. If the download fails, the file will be opened
    /// remotely.
    bool fCachedRead = false;

    /// Where to cache the file. If empty, defaults to TFilePtr::GetCacheDir().
    std::string fCacheDir;
  };

  ///\name Generator functions
  ///\{

  /// Open a file with `name` for reading.
  static TFilePtr Open(std::string_view name, const Options_t& opts = Options_t());

  /// Open an existing file with `name` for reading and writing. If a file with
  /// that name does not exist, an invalid TFilePtr will be returned.
  static TFilePtr OpenForUpdate(std::string_view name, const Options_t& opts = Options_t());

  /// Open a file with `name` for reading and writing. Fail (return an invalid
  /// `TFilePtr`) if a file with this name already exists.
  static TFilePtr Create(std::string_view name, const Options_t& opts = Options_t());

  /// Open a file with `name` for reading and writing. If a file with this name
  /// already exists, delete it and create a new one. Else simply create a new file.
  static TFilePtr Recreate(std::string_view name, const Options_t& opts = Options_t());

  ///\}

  /// Set the new directory used for cached reads, returns the old directory.
  static std::string SetCacheDir(std::string_view path);

  /// Get the directory used for cached reads.
  static std::string GetCacheDir();

  /// Constructed by Open etc.
  TFilePtr(std::unique_ptr<Internal::TFileImplBase>&&);

  /// Dereference the file pointer, giving access to the TFileImplBase object.
  Internal::TFileImplBase* operator ->() { return fImpl.get(); }

  /// Dereference the file pointer, giving access to the TFileImplBase object.
  /// const overload.
  const Internal::TFileImplBase* operator ->() const { return fImpl.get(); }

  /// Check the validity of the file pointer.
  operator bool() const { return fImpl.get(); }
};

} // namespace Experimental
} // namespace ROOT
#endif
