/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPad.h"
#include "TClass.h"

#include "RFieldProvider.hxx"


// ==============================================================================================

/** \class RNTupleDraw6Provider
\ingroup rbrowser
\brief Provider for RNTuple drawing on TCanvas
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleDraw6Provider : public RFieldProvider {

public:

   RNTupleDraw6Provider()
   {
      RegisterDraw6(TClass::GetClass<ROOT::Experimental::RNTuple>(), [this](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         auto h1 = DrawField(dynamic_cast<RFieldHolder*> (obj.get()));
         if (!h1) return false;

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(h1, opt.c_str());

         return true;
      });


   }

} newRNTupleDraw6Provider;

