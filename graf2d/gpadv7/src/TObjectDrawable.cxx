/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TObjectDrawable.hxx>

#include <ROOT/TObjectDisplayItem.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RMenuItems.hxx>

#include "TArrayI.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TStyle.h"

#include <exception>
#include <sstream>
#include <iostream>
#include <cstring>

using namespace ROOT::Experimental;

std::string TObjectDrawable::DetectCssType(const TObject *obj)
{
   const char *clname = obj ? obj->ClassName() : "TObject";
   if (strncmp(clname, "TH1", 3) == 0) return "th1";
   if (strncmp(clname, "TH2", 3) == 0) return "th2";
   if (strncmp(clname, "TH3", 3) == 0) return "th3";
   if (strncmp(clname, "TGraph", 6) == 0) return "tgraph";
   if (strcmp(clname, "TLine") == 0) return "tline";
   if (strcmp(clname, "TBox") == 0) return "tbox";
   return "tobject";
}

////////////////////////////////////////////////////////////////////
/// Convert TColor to RGB string for using with SVG

const char *TObjectDrawable::GetColorCode(TColor *col)
{
   static TString code;

   if (col->GetAlpha() == 1)
      code.Form("rgb(%d,%d,%d)", (int) (255*col->GetRed()), (int) (255*col->GetGreen()), (int) (255*col->GetBlue()));
   else
      code.Form("rgba(%d,%d,%d,%5.3f)", (int) (255*col->GetRed()), (int) (255*col->GetGreen()), (int) (255*col->GetBlue()), col->GetAlpha());

   return code.Data();
}

////////////////////////////////////////////////////////////////////
/// Create instance of requested special object

std::unique_ptr<TObject> TObjectDrawable::CreateSpecials(int kind)
{
   switch (kind) {
      case kColors: {
         // convert list of colors into strings
         auto arr = std::make_unique<TObjArray>();
         arr->SetOwner(kTRUE);

         auto cols = gROOT->GetListOfColors();
         for (int n = 0; n <= cols->GetLast(); ++n) {
            auto col = dynamic_cast<TColor *>(cols->At(n));
            if (!col) continue;
            auto code = TString::Format("%d=%s", n, GetColorCode(col));
            arr->Add(new TObjString(code));
         }

         return arr;
      }
      case kStyle: {  // create copy of gStyle
         return std::make_unique<TStyle>(*gStyle);
      }
      case kPalette: { // copy color palette

         auto arr = std::make_unique<TObjArray>();
         arr->SetOwner(kTRUE);

         auto palette = TColor::GetPalette();
         for (int n = 0; n < palette.GetSize(); ++n) {
            auto col = gROOT->GetColor(palette[n]);
            arr->Add(new TObjString(GetColorCode(col)));
         }

         return arr;
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////
/// Check if object has specified color value and store it in display item
/// Ensure that color matches on client side too

void TObjectDrawable::ExtractTColor(std::unique_ptr<TObjectDisplayItem> &item, const char *class_name, const char *class_member)
{
   TClass *cl = fObj->IsA();
   if (!cl->GetBaseClass(class_name)) return;

   auto offset = cl->GetDataMemberOffset(class_member);
   if (offset <= 0) return;

   Color_t *icol = (Color_t *)((char *) fObj.get() + offset);
   if (*icol < 10) return;

   TColor *col = gROOT->GetColor(*icol);
   if (col) item->AddTColor(*icol, col->AsHexString());
}


////////////////////////////////////////////////////////////////////
/// Create display item which will be delivered to the client

std::unique_ptr<RDisplayItem> TObjectDrawable::Display(const RDisplayContext &ctxt)
{
   if (GetVersion() > ctxt.GetLastVersion()) {
      if ((fKind == kObject) || fObj) {
         auto item = std::make_unique<TObjectDisplayItem>(*this, fKind, fObj.get());
         if ((fKind == kObject) && fObj) {
            ExtractTColor(item, "TAttLine", "fLineColor");
            ExtractTColor(item, "TAttFill", "fFillColor");
            ExtractTColor(item, "TAttMarker", "fMarkerColor");
            ExtractTColor(item, "TAttText", "fTextColor");
            ExtractTColor(item, "TAttPad", "fFrameFillColor");
            ExtractTColor(item, "TAttPad", "fFrameLineColor");
            ExtractTColor(item, "TAttAxis", "fAxisColor");
            ExtractTColor(item, "TAttAxis", "fLabelColor");
            ExtractTColor(item, "TAttAxis", "fTitleColor");
         }

         return item;
      }

      auto specials = CreateSpecials(fKind);
      return std::make_unique<TObjectDisplayItem>(fKind, specials.release());
   }

   return nullptr;
}

////////////////////////////////////////////////////////////////////
/// fill context menu items for the ROOT class

void TObjectDrawable::PopulateMenu(RMenuItems &items)
{
   if (fKind == kObject)
      items.PopulateObjectMenu(fObj.get(), fObj.get()->IsA());
}

////////////////////////////////////////////////////////////////////
/// Execute object method

void TObjectDrawable::Execute(const std::string &exec)
{
   if (fKind != kObject) return;

   TObject *obj = fObj.get();

   std::string sub, ex = exec;
   if (ex.compare(0, 6, "xaxis#") == 0) {
      ex.erase(0,6);
      ex.insert(0, "GetXaxis()->");
   } else if (ex.compare(0, 6, "yaxis#") == 0) {
      ex.erase(0,6);
      ex.insert(0, "GetYaxis()->");
   } else if (ex.compare(0, 6, "zaxis#") == 0) {
      ex.erase(0,6);
      ex.insert(0, "GetZaxis()->");
   }

   std::stringstream cmd;
   cmd << "((" << obj->ClassName() << " *) " << std::hex << std::showbase << (size_t)obj << ")->" << ex << ";";
   std::cout << "TObjectDrawable::Execute Obj " << obj->GetName() << "Cmd " << cmd.str() << std::endl;
   gROOT->ProcessLine(cmd.str().c_str());
}
