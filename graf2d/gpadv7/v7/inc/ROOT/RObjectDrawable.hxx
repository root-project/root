/// \file ROOT/RObjectDrawable.hxx
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

#ifndef ROOT7_RObjectDrawable
#define ROOT7_RObjectDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include "ROOT/RStringView.hxx"
#include "ROOT/RDisplayItem.hxx"

#include <memory>
#include <string>
#include <vector>

class TObject;

namespace ROOT {
namespace Experimental {

class RPadBase;

/// \class ROOT::Experimental::Internal::RObjectDrawingOpts Drawing options for TObject.
class RObjectDrawingOpts : public RDrawingOptsBase {
   std::string fOpts; ///< The drawing options

public:
   RObjectDrawingOpts(const std::string &opts = "") : fOpts(opts) {}

   const std::string &GetOptionString() const { return fOpts; }
};

/// \class ROOT::Experimental::Internal::RObjectDrawable
/// Provides v7 drawing facilities for TObject types (TGraph etc).
class RObjectDrawable : public RDrawableBase<RObjectDrawable> {
   const std::shared_ptr<TObject> fObj; ///< The object to be painted
   RObjectDrawingOpts fOpts;

public:
   RObjectDrawable() = default;

   RObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt) : fObj(obj), fOpts(opt) {}

   /// Paint the object
   void Paint(Internal::RPadPainter &canv) final;

   /// Fill menu items for the object
   void PopulateMenu(RMenuItems &) final;

   /// Get the options - a string!
   const RObjectDrawingOpts &GetOptions() const { return fOpts; }

   RObjectDrawingOpts &GetOptions() { return fOpts; }

   /// Executes menu item
   void Execute(const std::string &) final;
};

class RObjectDisplayItem : public RDisplayItem {
protected:
   const TObject *fObject; ///< object to draw
   std::string fOption;    ///< v6 draw options
public:
   RObjectDisplayItem(TObject *obj, const std::string &opt) : fObject(obj), fOption(opt) {}
};

} // namespace Experimental
} // namespace ROOT

/// Interface to graphics taking a shared_ptr<TObject>.
/// Must be on global scope, else lookup cannot find it (no ADL for TObject).
inline std::shared_ptr<ROOT::Experimental::RObjectDrawable>
GetDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt = "")
{
   return std::make_shared<ROOT::Experimental::RObjectDrawable>(obj, opt);
}

#endif
