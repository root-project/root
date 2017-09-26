/// \file ROOT/TDrawingOptsBase.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawingOptsBase
#define ROOT7_TDrawingOptsBase

#include <RStringView.h>
#include <map>
#include <string>

namespace ROOT {
namespace Experimental {
namespace Internal {
/// Implementation.
std::map<std::string, std::string> ReadDrawingOptsDefaultConfig(std::string_view section);
} // namespace Internal

/** \class ROOT::Experimental::TDrawingOptsBase
  Base class for drawing options. Implements access to the default and the default's initialization
  from a config file.

  Derived classes must implement `InitializeDefaultFromFile()`.
  */
template <class DERIVED>
class TDrawingOptsBase {
protected:
   const DERIVED &ToDerived() const { return static_cast<DERIVED &>(*this); }
   DERIVED &ToDerived() { return static_cast<DERIVED &>(*this); }

   /// Read the configuration for section from the config file. Used by derived classes to
   /// initialize their default, in `InitializeDefaultFromFile()`.
   static std::map<std::string, std::string> ReadConfig(std::string_view section)
   {
      return Internal::ReadDrawingOptsDefaultConfig(section);
   }

   /// Default implementation: no configuration variables in config file, simply default-initialize
   /// the drawing options.
   static DERIVED InitializeDefaultFromFile() { return DERIVED(); }

public:
   /// Retrieve the default drawing options for `DERIVED`. Can be used to query and adjust the
   /// default options.
   static DERIVED &Default()
   {
      static DERIVED defaultOpts = DERIVED::InitializeDefaultFromFile();
      return defaultOpts;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
