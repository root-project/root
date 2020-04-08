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

class TObject;

namespace ROOT {
namespace Experimental {

class RPadBase;

/** \class RObjectDrawable
\ingroup GpadROOT7
\brief Provides v7 drawing facilities for TObject types (TGraph etc).
\author Sergey Linev
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RObjectDrawable final : public RDrawable {

   Internal::RIOShared<TObject> fObj; ///< The object to be painted

   std::string fOpts;  ///< drawing options

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) final { vect.emplace_back(&fObj); }

   std::unique_ptr<RDisplayItem> Display(const RPadBase &, Version_t) const override;

public:
   RObjectDrawable() : RDrawable("tobject") {}

   RObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt) : RDrawable("tobject"), fObj(obj), fOpts(opt) {}

   /// Fill menu items for the object
   void PopulateMenu(RMenuItems &) final;

   /// Executes menu item
   void Execute(const std::string &) final;
};

} // namespace Experimental
} // namespace ROOT


#endif
