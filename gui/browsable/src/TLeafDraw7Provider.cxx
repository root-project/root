/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLeafProvider.hxx"

#include <ROOT/RCanvas.hxx>
#include <ROOT/RObjectDrawable.hxx>

/** Provider for drawing of ROOT7 classes */

class TLeafDraw7Provider : public TLeafProvider {
public:
   TLeafDraw7Provider()
   {
      RegisterDraw7(TLeaf::Class(), [](std::shared_ptr<ROOT::Experimental::RPadBase> &subpad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         auto hist = TLeafProvider::DrawLeaf(obj);

         if (!hist)
            return false;

         if (subpad->NumPrimitives() > 0) {
            subpad->Wipe();
            subpad->GetCanvas()->Modified();
            subpad->GetCanvas()->Update(true);
         }

         std::shared_ptr<TH1> shared;
         shared.reset(hist);

         subpad->Draw<ROOT::Experimental::RObjectDrawable>(shared, opt);

         return true;
      });

   }

} newTLeafDraw7Provider;

