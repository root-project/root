/// \file TFile.h
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

#ifndef ROOT7_TFile
#define ROOT7_TFile

#include "ROOT/TDirectory.h"

#include <memory>
#include <experimental/string_view>

namespace ROOT {

/**
 \class TFile
 \brief Stores or reads objects in ROOT's binary format. Storage-agnostic base.

 A `TFile` is an object store: it can serialize any object for which ROOT I/O is
 available (generally: an object which has a dictionary), and its stores the
 object's data under a key name.

 A `TFile` stores whatever was added to it as a `TDirectory`, when the `TFile`
 object is destructed. It can store non-lifetime managed objects by passing them
 to `Save()`.
 */

class TFile: public TDirectory {
public:
  static TCoopPtr<TFile> Read(std::string_view name);
  static TCoopPtr<TFile> Create(std::string_view name);
  static TCoopPtr<TFile> Recreate(std::string_view name);

  template <class T>
  void Write(const std::string& name, const T& ptr) {}

  /// Save all objects associated with this directory to the storage medium.
  void Flush() {}

  /// Flush() and make the file non-writable: close it.
  void Close();

private:
  TFile() = default;
};
}
#endif
