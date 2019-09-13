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

#include <memory>
#include <string>

class TObject;

namespace ROOT {
namespace Experimental {

class RPadBase;

/// \class ROOT::Experimental::Internal::RObjectDrawable
/// Provides v7 drawing facilities for TObject types (TGraph etc).
class RObjectDrawable : public RDrawable {

   const std::shared_ptr<TObject> fObj; ///< The object to be painted

   std::string fOpts;  ///< drawing options

public:
   RObjectDrawable() = default;

   RObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt) : fObj(obj), fOpts(opt) {}

   /// Fill menu items for the object
   void PopulateMenu(RMenuItems &) final;

   /// Executes menu item
   void Execute(const std::string &) final;
};

} // namespace Experimental
} // namespace ROOT


#endif
