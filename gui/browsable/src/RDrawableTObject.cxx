/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDrawableProvider.hxx>

#include "TObject.h"
#include "TVirtualPad.h"

#include <ROOT/RCanvas.hxx>
#include <ROOT/RObjectDrawable.hxx>


// ============================================================================================

using namespace ROOT::Experimental;

/** Provider for drawing of ROOT6 classes */

class RV6DrawProvider : public RDrawableProvider {
public:

   RV6DrawProvider()
   {
      RegisterV6(nullptr, [](TVirtualPad *pad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // do not handle TLeaf - needs special plugin
         if (obj->GetClass()->InheritsFrom("TLeaf"))
            return false;

         // try take object without ownership
         auto tobj = obj->get_object<TObject>();
         if (!tobj) {
            auto utobj = obj->get_unique<TObject>();
            if (!utobj)
               return false;
            tobj = utobj.release();
            tobj->SetBit(TObject::kMustCleanup); // TCanvas should care about cleanup
         }

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(tobj, opt.c_str());

         return true;
      });
   }

} newRV6DrawProvider;


/** Provider for drawing of ROOT7 classes */

class RV7DrawProvider : public RDrawableProvider {
public:
   RV7DrawProvider()
   {
      RegisterV7(nullptr, [] (std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // do not handle TLeaf - needs special plugin
         if (obj->GetClass()->InheritsFrom("TLeaf"))
            return false;

         // here clear ownership is required
         // If it possible, TObject will be cloned by TObjectHolder
         auto tobj = obj->get_shared<TObject>();
         if (!tobj) return false;

         if (subpad->NumPrimitives() > 0) {
            subpad->Wipe();
            subpad->GetCanvas()->Modified();
            subpad->GetCanvas()->Update(true);
         }

         subpad->Draw<RObjectDrawable>(tobj, opt);
         return true;
      });

   }

} newRV7DrawProvider;
