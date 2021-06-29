/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TObjectDrawable.hxx>

#include <ROOT/TObjectDisplayItem.hxx>
#include <ROOT/RColor.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RMenuItems.hxx>

#include "TArrayI.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TList.h"
#include "TStyle.h"
#include "TMethodCall.h"

#include <exception>
#include <sstream>
#include <iostream>
#include <cstring>

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////
/// Checks object ownership - used for TH1 directory handling and TF1 globals lists

void TObjectDrawable::CheckOwnership(TObject *obj)
{
   if (obj && obj->InheritsFrom("TH1")) {
      TMethodCall call(obj->IsA(), "SetDirectory", "nullptr");
      call.Execute((void *)(obj));
   } else if (obj && obj->InheritsFrom("TF1")) {
      TMethodCall call(obj->IsA(), "AddToGlobalList", "kFALSE");
      call.Execute((void *)(obj));
   }
}

////////////////////////////////////////////////////////////////////
/// Provide css type

std::string TObjectDrawable::DetectCssType(const TObject *obj)
{
   if (!obj) return "tobject";

   const char *clname = obj->ClassName();
   if (strncmp(clname, "TH3", 3) == 0) return "th3";
   if (strncmp(clname, "TH2", 3) == 0) return "th2";
   if ((strncmp(clname, "TH1", 3) == 0) || obj->InheritsFrom("TH1")) return "th1";
   if (strncmp(clname, "TGraph", 6) == 0) return "tgraph";
   if (strcmp(clname, "TLine") == 0) return "tline";
   if (strcmp(clname, "TBox") == 0) return "tbox";
   return "tobject";
}

////////////////////////////////////////////////////////////////////
/// Default constructor

TObjectDrawable::TObjectDrawable() : RDrawable("tobject")
{
}

////////////////////////////////////////////////////////////////////
/// Constructor, can take ownership over the object is isowner specified

TObjectDrawable::TObjectDrawable(TObject *obj, bool isowner) : RDrawable(DetectCssType(obj))
{
   fKind = kObject;
   if (isowner) {
      CheckOwnership(obj);
      fObj = std::shared_ptr<TObject>(obj);
   } else {
      fExtObj = obj;
   }
}

////////////////////////////////////////////////////////////////////
/// Constructor, can take ownership over the object is isowner specified

TObjectDrawable::TObjectDrawable(TObject *obj, const std::string &opt, bool isowner) : TObjectDrawable(obj, isowner)
{
   drawopt = opt;
}

////////////////////////////////////////////////////////////////////
/// Constructor

TObjectDrawable::TObjectDrawable(const std::shared_ptr<TObject> &obj) : RDrawable(DetectCssType(obj.get()))
{
   fKind = kObject;
   CheckOwnership(obj.get());
   fObj = obj;
}


////////////////////////////////////////////////////////////////////
/// Constructor

TObjectDrawable::TObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt) : TObjectDrawable(obj)
{
   drawopt = opt;
}

////////////////////////////////////////////////////////////////////
/// Creates special kind for for palette or list of colors
/// One can create persistent, which does not updated to actual values

TObjectDrawable::TObjectDrawable(EKind kind, bool persistent) : RDrawable("tobject")
{
   fKind = kind;

   if (persistent)
      fObj = CreateSpecials(kind);
}

////////////////////////////////////////////////////////////////////
/// Destructor

TObjectDrawable::~TObjectDrawable() = default;


////////////////////////////////////////////////////////////////////
/// Reset object

void TObjectDrawable::Reset()
{
   fKind = kNone;
   fObj = nullptr;
   fExtObj = nullptr;
}


////////////////////////////////////////////////////////////////////
/// Return assigned object

const TObject *TObjectDrawable::Get()
{
   // weak pointer - when object deleted outside only indirect way to handle it :(
   if (fExtObj) {
      if (!fExtObj->TestBit(TObject::kNotDeleted)) fExtObj = nullptr;
      return fExtObj;
   }

   return fObj.get();
}


////////////////////////////////////////////////////////////////////
/// Set object

void TObjectDrawable::Set(TObject *obj, bool isowner)
{
   Reset();

   SetCssType(DetectCssType(obj));
   fKind = kObject;

   if (isowner) {
      CheckOwnership(obj);
      fObj = std::shared_ptr<TObject>(obj);
   } else {
      fExtObj = obj;
   }
}

////////////////////////////////////////////////////////////////////
/// Set object

void TObjectDrawable::Set(TObject *obj, const std::string &opt, bool isowner)
{
   Set(obj, isowner);
   drawopt = opt;
}

////////////////////////////////////////////////////////////////////
/// Convert TColor to RGB string for using with SVG

std::string TObjectDrawable::GetColorCode(TColor *col)
{
   RColor rcol((uint8_t) (255*col->GetRed()), (uint8_t) (255*col->GetGreen()), (uint8_t) (255*col->GetBlue()));
   if (col->GetAlpha() != 1)
      rcol.SetAlphaFloat(col->GetAlpha());

   return rcol.AsSVG();
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
            auto code = TString::Format("%d=%s", n, GetColorCode(col).c_str());
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
            arr->Add(new TObjString(GetColorCode(col).c_str()));
         }

         return arr;
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////
/// Check if object has specified color value and store it in display item
/// Ensure that color matches on client side too

void TObjectDrawable::ExtractObjectColors(std::unique_ptr<TObjectDisplayItem> &item, const TObject *obj)
{
   if (!obj) return;

   TClass *cl = obj->IsA();

   auto ExtractColor = [&item, cl, obj](const char *class_name, const char *class_member) {
      if (!cl->GetBaseClass(class_name)) return;

      auto offset = cl->GetDataMemberOffset(class_member);
      if (offset <= 0) return;

      Color_t *icol = (Color_t *)((char *) obj + offset);
      if (*icol < 10) return;

      TColor *col = gROOT->GetColor(*icol);
      if (col) item->UpdateColor(*icol, GetColorCode(col));
   };

   ExtractColor("TAttLine", "fLineColor");
   ExtractColor("TAttFill", "fFillColor");
   ExtractColor("TAttMarker", "fMarkerColor");
   ExtractColor("TAttText", "fTextColor");
   ExtractColor("TAttPad", "fFrameFillColor");
   ExtractColor("TAttPad", "fFrameLineColor");
   ExtractColor("TAttAxis", "fAxisColor");
   ExtractColor("TAttAxis", "fLabelColor");
   ExtractColor("TAttAxis", "fTitleColor");

   if (cl->InheritsFrom("TH1")) {
      auto offx = cl->GetDataMemberOffset("fXaxis");
      if (offx > 0) ExtractObjectColors(item, (TObject *) ((char *) obj + offx));
      auto offy = cl->GetDataMemberOffset("fYaxis");
      if (offy > 0) ExtractObjectColors(item, (TObject *) ((char *) obj + offy));
      auto offz = cl->GetDataMemberOffset("fZaxis");
      if (offz > 0) ExtractObjectColors(item, (TObject *) ((char *) obj + offz));
   }
}


////////////////////////////////////////////////////////////////////
/// Create display item which will be delivered to the client

std::unique_ptr<RDisplayItem> TObjectDrawable::Display(const RDisplayContext &ctxt)
{
   if (GetVersion() > ctxt.GetLastVersion()) {

      auto obj = Get();

      if ((fKind == kObject) || obj) {
         auto item = std::make_unique<TObjectDisplayItem>(*this, fKind, obj);
         if ((fKind == kObject) && obj) {
            ExtractObjectColors(item, obj);

            // special handling of THStack to support any custom colors inside
            if (strcmp(obj->ClassName(), "THStack") == 0) {
               TClass *cl = gROOT->GetClass("THStack");
               // do not call stack->GetHistogram() to avoid it auto-creation
               auto off1 = cl->GetDataMemberOffset("fHistogram");
               if (off1 > 0) ExtractObjectColors(item, *((TObject **) ((char *) obj + off1)));
               // here make identical to fHistogram, one also can use TMethodCall
               auto off2 = cl->GetDataMemberOffset("fHists");
               if (off2 > 0) {
                  TIter iter(*(TList **) (((char *) obj + off2)));
                  TObject *hist = nullptr;
                  while ((hist = iter()) != nullptr)
                     ExtractObjectColors(item, hist);
               }

            }
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
   auto obj = Get();
   if ((fKind == kObject) && obj)
      items.PopulateObjectMenu((void *)obj, obj->IsA());
}

////////////////////////////////////////////////////////////////////
/// Execute object method

void TObjectDrawable::Execute(const std::string &exec)
{
   auto obj = Get();

   if ((fKind != kObject) || !obj) return;

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
