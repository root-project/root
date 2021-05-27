/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPad
#define ROOT7_RPad

#include "ROOT/RPadBase.hxx"

namespace ROOT {
namespace Experimental {

/** \class RPad
\ingroup GpadROOT7
\brief Graphic container for `RDrawable`-s.
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2017-07-06
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RPad: public RPadBase, public RAttrLine {

   /// Pad containing this pad as a sub-pad.
   RPadBase *fParent{nullptr};             ///< The parent pad, if this pad has one.

   RPadPos fPos;                           ///< pad position
   RPadExtent fSize;                       ///< pad size

protected:

   std::unique_ptr<RDisplayItem> Display(const RDisplayContext &) final;

public:
   /// Create a topmost, non-paintable pad.
   RPad() : RPadBase(), RAttrLine((RDrawable *) this, "border") {}

   /// Create a child pad.
   RPad(RPadBase *parent, const RPadPos &pos, const RPadExtent &size): RPad() { fParent = parent; fPos = pos; fSize = size; }

   /// Destructor to have a vtable.
   virtual ~RPad();

   /// Access to the parent pad (const version).
   const RPadBase *GetParent() const { return fParent; }

   /// Access to the parent pad (non-const version).
   RPadBase *GetParent() { return fParent; }

   /// Access to the top-most canvas (const version).
   const RCanvas *GetCanvas() const override { return fParent ? fParent->GetCanvas() : nullptr; }

   /// Access to the top-most canvas (non-const version).
   RCanvas *GetCanvas() override { return fParent ? fParent->GetCanvas() : nullptr; }

   /// Get the position of the pad in parent (!) coordinates.
   const RPadPos &GetPos() const { return fPos; }

   /// Set position
   void SetPos(const RPadPos &p) { fPos = p; }

   /// Get the size of the pad in parent (!) coordinates.
   const RPadExtent &GetSize() const { return fSize; }

   /// Set the size of the pad in parent (!) coordinates.
   void SetSize(const RPadExtent &sz) { fSize = sz; }

};

} // namespace Experimental
} // namespace ROOT

#endif
