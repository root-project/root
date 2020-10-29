/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RObjectDrawable.hxx>

#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RMenuItems.hxx>

#include "TROOT.h"
#include "TStyle.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TObjString.h"

#include <exception>
#include <sstream>
#include <iostream>


using namespace ROOT::Experimental;

RObjectDrawable::RObjectDrawable(EKind kind, const std::string &opt) : RDrawable("tobject"), fKind(kind), fOpts(opt)
{
   switch (fKind) {
      case kColors: {
         // convert list of colors into strings
         auto arr = std::make_shared<TObjArray>();
         arr->SetOwner(kTRUE);

         auto cols = gROOT->GetListOfColors();
         for (int n = 0; n <= cols->GetLast(); ++n) {
            auto col = dynamic_cast<TColor *>(cols->At(n));
            if (!col) continue;
            auto code = TString::Format("%d=%s", n, GetColorCode(col));
            arr->Add(new TObjString(code));
         }

         fObj = arr;

         break;
      }
      case kStyle: {  // create copy of gStyle
         fObj = std::make_shared<TStyle>(*gStyle);
         break;
      }
      case kPalette: { // copy color palette

         auto arr = std::make_shared<TObjArray>();
         arr->SetOwner(kTRUE);

         auto palette = TColor::GetPalette();
         for (int n = 0; n < palette.GetSize(); ++n) {
            auto col = gROOT->GetColor(palette[n]);
            arr->Add(new TObjString(GetColorCode(col)));
         }

         fObj = arr;
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////
/// Convert TColor to RGB string for using with SVG

const char *RObjectDrawable::GetColorCode(TColor *col)
{
   static TString code;

   if (col->GetAlpha() == 1)
      code.Form("rgb(%d,%d,%d)", (int) (255*col->GetRed()), (int) (255*col->GetGreen()), (int) (255*col->GetBlue()));
   else
      code.Form("rgba(%d,%d,%d,%5.3f)", (int) (255*col->GetRed()), (int) (255*col->GetGreen()), (int) (255*col->GetBlue()), col->GetAlpha());

   return code.Data();
}


std::unique_ptr<RDisplayItem> RObjectDrawable::Display(const RDisplayContext &ctxt)
{
   if (GetVersion() > ctxt.GetLastVersion())
      return std::make_unique<RObjectDisplayItem>(fKind, fObj.get(), fOpts);

   return nullptr;
}

void RObjectDrawable::PopulateMenu(RMenuItems &items)
{
   // fill context menu items for the ROOT class
   if (fKind == kObject)
      items.PopulateObjectMenu(fObj.get(), fObj.get()->IsA());
}

void RObjectDrawable::Execute(const std::string &exec)
{
   if (fKind != kObject) return;

   TObject *obj = fObj.get();

   std::stringstream cmd;
   cmd << "((" << obj->ClassName() << "* ) " << std::hex << std::showbase << (size_t)obj << ")->" << exec << ";";
   std::cout << "RObjectDrawable::Execute Obj " << obj->GetName() << "Cmd " << cmd.str() << std::endl;
   gROOT->ProcessLine(cmd.str().c_str());
}
