/// \file ROOT/TDisplayItem.h
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TObjectDrawable
#define ROOT7_TObjectDrawable

#include <ROOT/TDrawable.hxx>
#include <ROOT/TDrawingOptsBase.hxx>
#include "RStringView.h"
#include <memory>
#include <string>

class TObject;

namespace ROOT {
namespace Experimental {

class TPadBase;

/// \class ROOT::Experimental::Internal::TObjectDrawingOpts Drawing options for TObject.
class TObjectDrawingOpts: public TDrawingOptsBase {
   std::string fOpts;                   ///< The drawing options

public:
   TObjectDrawingOpts(const std::string &opts = "") : fOpts(opts) {}

   const std::string &GetOptionString() const { return fOpts; }
};

/// \class ROOT::Experimental::Internal::TObjectDrawable
/// Provides v7 drawing facilities for TObject types (TGraph etc).
class TObjectDrawable: public TDrawableBase<TObjectDrawable> {
   const std::shared_ptr<TObject> fObj; ///< The object to be painted
   TObjectDrawingOpts fOpts;
public:
   TObjectDrawable() = default;

   TObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt): fObj(obj), fOpts(opt) {}

   /// Paint the histogram
   void Paint(Internal::TVirtualCanvasPainter &canv) final;

   /// Fill menu items for the object
   void PopulateMenu(TMenuItems &) final;

   /// Get the options - a string!
   const TObjectDrawingOpts &GetOptions() const { return fOpts; }

   TObjectDrawingOpts &GetOptions() { return fOpts; }

   /// Executes menu item
   void Execute(const std::string &) final;
};
} // namespace Experimental
} // namespace ROOT

/// Interface to graphics taking a shared_ptr<TObject>.
/// Must be on global scope, else lookup cannot find it (no ADL for TObject).
inline std::unique_ptr<ROOT::Experimental::TObjectDrawable>
GetDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt = "")
{
   return std::make_unique<ROOT::Experimental::TObjectDrawable>(obj, opt);
}

#endif
