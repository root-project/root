/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RProvider.hxx>

#include "TObject.h"
#include "TVirtualPad.h"

using namespace ROOT::Experimental::Browsable;

/** Provider for drawing of ROOT6 classes */

class TObjectDraw6Provider : public RProvider {
public:

   TObjectDraw6Provider()
   {
      RegisterDraw6(nullptr, [](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         // try take object without ownership
         auto tobj = obj->get_object<TObject>();
         if (!tobj) {
            auto utobj = obj->get_unique<TObject>();
            if (!utobj)
               return false;
            tobj = utobj.release();
            tobj->SetBit(TObject::kMustCleanup); // TCanvas should care about cleanup
         }

         pad->GetListOfPrimitives()->Add(tobj, opt.c_str());

         return true;
      });
   }

} TObjectDraw6Provider;

