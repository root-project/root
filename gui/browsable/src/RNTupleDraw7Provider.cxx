/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClass.h"

#include <ROOT/RCanvas.hxx>
#include <ROOT/TObjectDrawable.hxx>

#include "RFieldProvider.hxx"

using namespace ROOT::Experimental;

// ==============================================================================================

/** \class RNTupleDraw7Provider
\ingroup rbrowser
\brief Provider for RNTuple drawing on RCanvas
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleDraw7Provider : public RFieldProvider {

public:

   RNTupleDraw7Provider()
   {
      RegisterDraw7(TClass::GetClass<ROOT::Experimental::RNTuple>(), [this](std::shared_ptr<RPadBase> &subpad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         auto h1 = DrawField(dynamic_cast<RFieldHolder*> (obj.get()));
         if (!h1) return false;

         if (subpad->NumPrimitives() > 0) {
            subpad->Wipe();
            subpad->GetCanvas()->Modified();
            subpad->GetCanvas()->Update(true);
         }

         std::shared_ptr<TH1> shared;
         shared.reset(h1);

         subpad->Draw<ROOT::Experimental::TObjectDrawable>(shared, opt);
         subpad->GetCanvas()->Update(true);
         return true;
      });
   }

} newRNTupleDraw7Provider;

