/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLeafProvider.hxx"

#include "TVirtualPad.h"

/** Provider for drawing of ROOT6 classes */

class TLeafDraw6Provider : public TLeafProvider {
public:
   TLeafDraw6Provider()
   {
      RegisterDraw6(TLeaf::Class(), [](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         auto hist = TLeafProvider::DrawLeaf(obj);

         if (!hist)
            return false;

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(hist, opt.c_str());

         return true;
      });

   }

} newTLeafDraw6Provider;

