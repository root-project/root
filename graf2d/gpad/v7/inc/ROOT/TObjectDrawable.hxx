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
#include "RStringView.h"
#include <memory>

class TObject;

namespace ROOT {
namespace Experimental {
namespace Internal {
/// \class ROOT::Experimental::Internal::TObjectDrawable
/// Provides v7 drawing facilities for TObject types (TGraph etc).
class TObjectDrawable : public TDrawable {
   const std::shared_ptr<TObject> fObj; ///< The object to be painted
   std::string fOpts;                   ///< The drawing options

public:
   TObjectDrawable(const std::shared_ptr<TObject> &obj, std::string_view opts) : fObj(obj), fOpts(opts) {}

   /// Paint the histogram
   void Paint(TVirtualCanvasPainter &canv) final;

   /// Fill menu items for the object
   void PopulateMenu(TMenuItems &) final;

   /// Executes menu item
   void Execute(const std::string &) final;
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

/// Interface to graphics taking a shared_ptr<TObject>.
/// Must be on global scope, else lookup cannot find it (no ADL for TObject).
inline std::unique_ptr<ROOT::Experimental::Internal::TDrawable> GetDrawable(const std::shared_ptr<TObject> &obj,
                                                                            std::string_view opts = {})
{
   return std::make_unique<ROOT::Experimental::Internal::TObjectDrawable>(obj, opts);
}

#endif
