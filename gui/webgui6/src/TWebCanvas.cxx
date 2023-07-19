// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebCanvas.h"

#include "TWebSnapshot.h"
#include "TWebPadPainter.h"
#include "TWebPS.h"
#include "TWebMenuItem.h"

#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TButton.h"
#include "TFrame.h"
#include "TPaveText.h"
#include "TPaveStats.h"
#include "TText.h"
#include "TROOT.h"
#include "TClass.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TArrayI.h"
#include "TList.h"
#include "TF1.h"
#include "TF2.h"
#include "TH1.h"
#include "TH1K.h"
#include "TH2.h"
#include "THStack.h"
#include "TMultiGraph.h"
#include "TEnv.h"
#include "TError.h"
#include "TGraph.h"
#include "TGaxis.h"
#include "TScatter.h"
#include "TCutG.h"
#include "TBufferJSON.h"
#include "TBase64.h"
#include "TAtt3D.h"
#include "TView.h"
#include "TExec.h"
#include "TVirtualX.h"
#include "TMath.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

/** \class TWebCanvas
\ingroup webgui6

Basic TCanvasImp ABI implementation for Web-based Graphics
Provides painting of main ROOT classes in web browsers using [JSROOT](https://root.cern/js/)

Following settings parameters can be useful for TWebCanvas:

     WebGui.FullCanvas:       1     read-only mode (0), full-functional canvas (1) (default - 1)
     WebGui.StyleDelivery:    1     provide gStyle object to JSROOT client (default - 1)
     WebGui.PaletteDelivery:  1     provide color palette to JSROOT client (default - 1)
     WebGui.TF1UseSave:       0     used saved values for function drawing (1) or calculate function on the client side (0) (default - 0)

TWebCanvas is used by default in interactive ROOT session. To use web-based canvas in batch mode for image
generation, one should explicitly specify `--web` option when starting ROOT:

    [shell] root -b --web tutorials/hsimple.root -e 'hpxpy->Draw("colz"); c1->SaveAs("image.png");'

If for any reasons TWebCanvas does not provide required functionality, one always can disable it.
Either by specifying `root --web=off` when starting ROOT or by setting `Canvas.Name: TRootCanvas` in rootrc file.

*/

using namespace std::string_literals;

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TWebCanvas::TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height, Bool_t readonly)
   : TCanvasImp(c, name, x, y, width, height)
{
   fReadOnly = readonly;
   fStyleDelivery = gEnv->GetValue("WebGui.StyleDelivery", 1);
   fPaletteDelivery = gEnv->GetValue("WebGui.PaletteDelivery", 1);
   fPrimitivesMerge = gEnv->GetValue("WebGui.PrimitivesMerge", 100);
   fTF1UseSave = gEnv->GetValue("WebGui.TF1UseSave", (Int_t) 0) > 0;
   fJsonComp = gEnv->GetValue("WebGui.JsonComp", TBufferJSON::kSameSuppression + TBufferJSON::kNoSpaces);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize window for the web canvas
/// At this place canvas is not yet register to the list of canvases - one cannot call RWebWindow::Show()

Int_t TWebCanvas::InitWindow()
{
   return 111222333; // should not be used at all
}

////////////////////////////////////////////////////////////////////////////////
/// Creates web-based pad painter

TVirtualPadPainter *TWebCanvas::CreatePadPainter()
{
   return new TWebPadPainter();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE when object is fully supported on JSROOT side
/// In ROOT7 Paint function will just return appropriate flag that object can be displayed on JSROOT side

Bool_t TWebCanvas::IsJSSupportedClass(TObject *obj, Bool_t many_primitives)
{
   if (!obj)
      return kTRUE;

   static const struct {
      const char *name{nullptr};
      bool with_derived{false};
      bool reduse_by_many{false};
   } supported_classes[] = {{"TH1", true},
                            {"TF1", true},
                            {"TGraph", true},
                            {"TScatter"},
                            {"TFrame"},
                            {"THStack"},
                            {"TMultiGraph"},
                            {"TGraphPolargram", true},
                            {"TPave", true},
                            {"TGaxis"},
                            {"TPave", true},
                            {"TArrow"},
                            {"TBox", false, true},  // can be handled via TWebPainter, disable for large number of primitives (like in greyscale.C)
                            {"TWbox"}, // some extra calls which cannot be handled via TWebPainter
                            {"TLine", false, true}, // can be handler via TWebPainter, disable for large number of primitives (like in greyscale.C)
                            {"TEllipse", true, true},  // can be handled via TWebPainter, disable for large number of primitives (like in greyscale.C)
                            {"TText"},
                            {"TLatex"},
                            {"TAnnotation"},
                            {"TMathText"},
                            {"TMarker"},
                            {"TPolyMarker"},
                            {"TPolyLine", true, true}, // can be handled via TWebPainter, simplify colors handling
                            {"TPolyMarker3D"},
                            {"TPolyLine3D"},
                            {"TGraphTime"},
                            {"TGraph2D"},
                            {"TGraph2DErrors"},
                            {"TGraphTime"},
                            {"TASImage"},
                            {"TRatioPlot"},
                            {"TSpline"},
                            {"TSpline3"},
                            {"TSpline5"},
                            {"TGeoManager"},
                            {"TGeoVolume"},
                            {}};

   // fast check of class name
   for (int i = 0; supported_classes[i].name != nullptr; ++i)
      if ((!many_primitives || !supported_classes[i].reduse_by_many) && (strcmp(supported_classes[i].name, obj->ClassName()) == 0))
         return kTRUE;

   // now check inheritance only for configured classes
   for (int i = 0; supported_classes[i].name != nullptr; ++i)
      if (supported_classes[i].with_derived && (!many_primitives || !supported_classes[i].reduse_by_many))
         if (obj->InheritsFrom(supported_classes[i].name))
            return kTRUE;

   return IsCustomClass(obj->IsA());
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Configures custom script for canvas.
/// If started from "load:" or "assert:" prefix will be loaded with JSROOT.AssertPrerequisites function
/// Script should implement custom user classes, which transferred as is to client
/// In the script draw handler for appropriate classes would be assigned

void TWebCanvas::SetCustomScripts(const std::string &src)
{
   fCustomScripts = src;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Assign custom class

void TWebCanvas::AddCustomClass(const std::string &clname, bool with_derived)
{
   if (with_derived)
      fCustomClasses.emplace_back("+"s + clname);
   else
      fCustomClasses.emplace_back(clname);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Checks if class belongs to custom

bool TWebCanvas::IsCustomClass(const TClass *cl) const
{
   for (auto &name : fCustomClasses) {
      if (name[0] == '+') {
         if (cl->InheritsFrom(name.substr(1).c_str()))
            return true;
      } else if (name.compare(cl->GetName()) == 0) {
         return true;
      }
   }
   return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Creates representation of the object for painting in web browser

void TWebCanvas::CreateObjectSnapshot(TPadWebSnapshot &master, TPad *pad, TObject *obj, const char *opt, TWebPS *masterps)
{
   if (IsJSSupportedClass(obj, masterps != nullptr)) {
      master.NewPrimitive(obj, opt).SetSnapshot(TWebSnapshot::kObject, obj);
      return;
   }

   // painter is not necessary for batch canvas, but keep configuring it for a while
   auto *painter = dynamic_cast<TWebPadPainter *>(Canvas()->GetCanvasPainter());

   TView *view = nullptr;
   auto savepad = gPad;

   pad->cd();

   if (obj->InheritsFrom(TAtt3D::Class()) && !pad->GetView()) {
      pad->GetViewer3D("pad");
      view = TView::CreateView(1, 0, 0); // Cartesian view by default
      pad->SetView(view);

      // Set view to perform first auto-range (scaling) pass
      view->SetAutoRange(kTRUE);
   }

   TVirtualPS *saveps = gVirtualPS;

   TWebPS ps;
   gVirtualPS = masterps ? masterps : &ps;
   if (painter)
      painter->SetPainting(ps.GetPainting());

   // calling Paint function for the object
   obj->Paint(opt);

   if (view) {
      view->SetAutoRange(kFALSE);
      // call 3D paint once again to make real drawing
      obj->Paint(opt);
      pad->SetView(nullptr);
   }

   if (painter)
      painter->SetPainting(nullptr);

   gVirtualPS = saveps;
   if (savepad)
      savepad->cd();

   // if there are master PS, do not create separate entries
   if (!masterps && !ps.IsEmptyPainting())
      master.NewPrimitive(obj, opt).SetSnapshot(TWebSnapshot::kSVG, ps.TakePainting(), kTRUE);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Calculate hash function for all colors and palette

UInt_t TWebCanvas::CalculateColorsHash()
{
   UInt_t hash = 0;

   TObjArray *colors = (TObjArray *)gROOT->GetListOfColors();

   if (colors) {
      for (Int_t n = 0; n <= colors->GetLast(); ++n)
         if (colors->At(n))
            hash += TString::Hash(colors->At(n), TColor::Class()->Size());
   }

   TArrayI pal = TColor::GetPalette();

   hash += TString::Hash(pal.GetArray(), pal.GetSize() * sizeof(Int_t));

   return hash;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Add special canvas objects like colors list at selected palette

void TWebCanvas::AddColorsPalette(TPadWebSnapshot &master)
{
   TObjArray *colors = (TObjArray *)gROOT->GetListOfColors();

   if (!colors)
      return;

   //Int_t cnt = 0;
   //for (Int_t n = 0; n <= colors->GetLast(); ++n)
   //   if (colors->At(n))
   //      cnt++;
   //if (cnt <= 598)
   //   return; // normally there are 598 colors defined

   TArrayI pal = TColor::GetPalette();

   auto *listofcols = new TWebPainting;
   for (Int_t n = 0; n <= colors->GetLast(); ++n)
      if (colors->At(n))
         listofcols->AddColor(n, (TColor *)colors->At(n));

   // store palette in the buffer
   auto *tgt = listofcols->Reserve(pal.GetSize());
   for (Int_t i = 0; i < pal.GetSize(); i++)
      tgt[i] = pal[i];
   listofcols->FixSize();

   master.NewSpecials().SetSnapshot(TWebSnapshot::kColors, listofcols, kTRUE);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Create snapshot for pad and all primitives
/// Callback function is used to create JSON in the middle of data processing -
/// when all misc objects removed from canvas list of primitives or histogram list of functions
/// After that objects are moved back to their places

void TWebCanvas::CreatePadSnapshot(TPadWebSnapshot &paddata, TPad *pad, Long64_t version, PadPaintingReady_t resfunc)
{
   auto &pad_status = fPadsStatus[pad];

   // send primitives if version 0 or actual pad version grater than already send version
   bool process_primitives = (version == 0) || (pad_status.fVersion > version);

   if (paddata.IsSetObjectIds()) {
      paddata.SetActive(pad == gPad);
      paddata.SetObjectIDAsPtr(pad);
   }
   paddata.SetSnapshot(TWebSnapshot::kSubPad, pad); // add ref to the pad
   paddata.SetWithoutPrimitives(!process_primitives);
   paddata.SetHasExecs(pad->GetListOfExecs()); // if pad execs are there provide more events from client

   // check style changes every time when creating canvas snapshot
   if (resfunc && (GetStyleDelivery() > 0)) {

      if (fStyleVersion != fCanvVersion) {
         auto hash = TString::Hash(gStyle, TStyle::Class()->Size());
         if ((hash != fStyleHash) || (fStyleVersion == 0)) {
            fStyleHash = hash;
            fStyleVersion = fCanvVersion;
         }
      }

      if (fStyleVersion > version)
         paddata.NewPrimitive().SetSnapshot(TWebSnapshot::kStyle, gStyle);
   }

   TList *primitives = pad->GetListOfPrimitives();

   if (primitives) fPrimitivesLists.Add(primitives); // add list of primitives

   TWebPS masterps;
   bool usemaster = primitives ? (primitives->GetSize() > fPrimitivesMerge) : false;

   TIter iter(primitives);
   TObject *obj = nullptr;
   TFrame *frame = nullptr;
   TPaveText *title = nullptr;
   bool need_frame = false, has_histo = false, need_palette = false;
   std::string need_title;

   while (process_primitives && ((obj = iter()) != nullptr)) {
      TString opt = iter.GetOption();
      opt.ToUpper();

      if (obj->InheritsFrom(THStack::Class())) {
         // workaround for THStack, create extra components before sending to client
         auto hs = static_cast<THStack *>(obj);
         TVirtualPad::TContext ctxt(pad, kFALSE);
         hs->BuildPrimitives(iter.GetOption());
         has_histo = true;
      } else if (obj->InheritsFrom(TMultiGraph::Class())) {
         // workaround for TMultiGraph
         if (opt.Contains("A")) {
            auto mg = static_cast<TMultiGraph *>(obj);
            TVirtualPad::TContext ctxt(kFALSE);
            mg->GetHistogram(); // force creation of histogram without any drawings
            has_histo = true;
         }
      } else if (obj->InheritsFrom(TFrame::Class())) {
         frame = static_cast<TFrame *>(obj);
      } else if (obj->InheritsFrom(TH1::Class())) {
         need_frame = true;
         has_histo = true;
         if (!obj->TestBit(TH1::kNoTitle) && !opt.Contains("SAME") && (strlen(obj->GetTitle()) > 0))
            need_title = obj->GetTitle();
         if (obj->InheritsFrom(TH2::Class()) && (opt.Contains("COLZ") || opt.Contains("LEGO2Z") || opt.Contains("LEGO4Z") || opt.Contains("SURF2Z")))
            need_palette = true;
      } else if (obj->InheritsFrom(TGraph::Class())) {
         if (opt.Contains("A")) {
            need_frame = true;
            if (!has_histo && (strlen(obj->GetTitle()) > 0))
               need_title = obj->GetTitle();
         }
      } else if (obj->InheritsFrom(TScatter::Class())) {
         need_frame = need_palette = true;
         if (strlen(obj->GetTitle()) > 0)
            need_title = obj->GetTitle();
      } else if (obj->InheritsFrom(TF1::Class())) {
         need_frame = !obj->InheritsFrom(TF2::Class());
         if (!has_histo && (strlen(obj->GetTitle()) > 0))
            need_title = obj->GetTitle();
      } else if (obj->InheritsFrom(TPaveText::Class())) {
         if (strcmp(obj->GetName(), "title") == 0)
            title = static_cast<TPaveText *>(obj);
      }
   }

   if (need_frame && !frame && primitives && CanCreateObject("TFrame")) {
      if (!IsReadOnly() && need_palette && (pad->GetRightMargin() < 0.12) && (pad->GetRightMargin() == gStyle->GetPadRightMargin()))
         pad->SetRightMargin(0.12);

      frame = pad->GetFrame();
      if(frame)
         primitives->AddFirst(frame);
   }

   if (!need_title.empty() && gStyle->GetOptTitle()) {
      if (title) {
         auto line0 = title->GetLine(0);
         if (line0 && !IsReadOnly()) line0->SetTitle(need_title.c_str());
      } else if (primitives && CanCreateObject("TPaveText")) {
         title = new TPaveText(0, 0, 0, 0, "blNDC");
         title->SetFillColor(gStyle->GetTitleFillColor());
         title->SetFillStyle(gStyle->GetTitleStyle());
         title->SetName("title");
         title->SetBorderSize(gStyle->GetTitleBorderSize());
         title->SetTextColor(gStyle->GetTitleTextColor());
         title->SetTextFont(gStyle->GetTitleFont(""));
         if (gStyle->GetTitleFont("") % 10 > 2)
            title->SetTextSize(gStyle->GetTitleFontSize());
         title->AddText(need_title.c_str());
         title->SetBit(kCanDelete);
         primitives->Add(title);
      }
   }

   auto flush_master = [&]() {
      if (!usemaster || masterps.IsEmptyPainting()) return;

      paddata.NewPrimitive(pad).SetSnapshot(TWebSnapshot::kSVG, masterps.TakePainting(), kTRUE);
      masterps.CreatePainting(); // create for next operations
   };

   iter.Reset();

   bool first_obj = true;

   if (process_primitives)
      pad_status._has_specials = false;

   while ((obj = iter()) != nullptr) {
      if (obj->InheritsFrom(TPad::Class())) {
         flush_master();
         CreatePadSnapshot(paddata.NewSubPad(), (TPad *)obj, version, nullptr);
      } else if (!process_primitives) {
         continue;
      } else if (obj->InheritsFrom(TH1K::Class())) {
         flush_master();
         TH1K *hist = (TH1K *)obj;

         Int_t nbins = hist->GetXaxis()->GetNbins();

         TH1D *h1 = new TH1D("__dummy_name__", hist->GetTitle(), nbins, hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
         h1->SetDirectory(nullptr);
         h1->SetName(hist->GetName());
         hist->TAttLine::Copy(*h1);
         hist->TAttFill::Copy(*h1);
         hist->TAttMarker::Copy(*h1);
         for (Int_t n = 1; n <= nbins; ++n)
             h1->SetBinContent(n, hist->GetBinContent(n));

         TIter fiter(hist->GetListOfFunctions());
         while (auto fobj = fiter())
            h1->GetListOfFunctions()->Add(fobj->Clone());

         TString hopt = iter.GetOption();
         if (title && first_obj) hopt.Append(";;use_pad_title");

         paddata.NewPrimitive(obj, hopt.Data()).SetSnapshot(TWebSnapshot::kObject, h1, kTRUE);

      } else if (obj->InheritsFrom(TH1::Class())) {
         flush_master();

         TH1 *hist = (TH1 *)obj;
         TIter fiter(hist->GetListOfFunctions());
         TObject *fobj = nullptr;
         TPaveStats *stats = nullptr;
         TObject *palette = nullptr;

         hist->BufferEmpty();

         while ((fobj = fiter()) != nullptr) {
           if (fobj->InheritsFrom(TPaveStats::Class()))
              stats = dynamic_cast<TPaveStats *> (fobj);
           else if (fobj->InheritsFrom("TPaletteAxis"))
              palette = fobj;
         }

         if (!stats && first_obj && (gStyle->GetOptStat() > 0) && CanCreateObject("TPaveStats")) {
            stats = new TPaveStats(
                           gStyle->GetStatX() - gStyle->GetStatW(),
                           gStyle->GetStatY() - gStyle->GetStatH(),
                           gStyle->GetStatX(),
                           gStyle->GetStatY(), "brNDC");

             stats->SetParent(hist);
             // do not set optfit and optstat, they calling pad->Update,
             // values correctly set already in TPaveStats constructor
             // stats->SetOptFit(gStyle->GetOptFit());
             // stats->SetOptStat(gStyle->GetOptStat());
             stats->SetFillColor(gStyle->GetStatColor());
             stats->SetFillStyle(gStyle->GetStatStyle());
             stats->SetBorderSize(gStyle->GetStatBorderSize());
             stats->SetTextFont(gStyle->GetStatFont());
             if (gStyle->GetStatFont()%10 > 2)
                stats->SetTextSize(gStyle->GetStatFontSize());
             stats->SetFitFormat(gStyle->GetFitFormat());
             stats->SetStatFormat(gStyle->GetStatFormat());
             stats->SetName("stats");

             stats->SetTextColor(gStyle->GetStatTextColor());
             stats->SetTextAlign(12);
             stats->SetBit(kCanDelete);
             stats->SetBit(kMustCleanup);

             hist->GetListOfFunctions()->Add(stats);
         }

         TString hopt = iter.GetOption();
         TString o = hopt;
         o.ToUpper();

         if (!palette && CanCreateObject("TPaletteAxis") && (hist->GetDimension() > 1) &&
              (o.Contains("COLZ") || o.Contains("LEGO2Z") || o.Contains("LEGO4Z") || o.Contains("SURF2Z"))) {
            std::stringstream exec;
            exec << "new TPaletteAxis(0,0,0,0, (TH1*)" << std::hex << std::showbase << (size_t)hist << ");";
            palette = (TObject *)gROOT->ProcessLine(exec.str().c_str());
            if (palette)
               hist->GetListOfFunctions()->AddFirst(palette);
         }

         if (title && first_obj) hopt.Append(";;use_pad_title");

         // if (stats) hopt.Append(";;use_pad_stats");

         if (palette) hopt.Append(";;use_pad_palette");

         paddata.NewPrimitive(obj, hopt.Data()).SetSnapshot(TWebSnapshot::kObject, obj);

         if (hist->GetDimension() == 2) {
            TString opt = iter.GetOption();
            auto p1 = opt.Index("["), p2 = opt.Index("]");
            if ((p1 != kNPOS) && (p2 != kNPOS) && p2 > p1 + 1) {
               TString cutname = opt(p1 + 1, p2 - p1 - 1);
               TObject *cutg = primitives->FindObject(cutname.Data());
               if (!cutg || (cutg->IsA() != TCutG::Class())) {
                  cutg = gROOT->GetListOfSpecials()->FindObject(cutname.Data());
                  if (cutg && cutg->IsA() == TCutG::Class())
                     paddata.NewPrimitive(cutg, "__ignore_drawing__").SetSnapshot(TWebSnapshot::kObject, cutg);
               }
            }
         }

         // do not extract objects from list of functions - stats and func need to be handled together with hist
         //
         // fiter.Reset();
         // while ((fobj = fiter()) != nullptr)
         //    CreateObjectSnapshot(paddata, pad, fobj, fiter.GetOption());

         // fPrimitivesLists.Add(hist->GetListOfFunctions());

         first_obj = false;
      } else if (obj->InheritsFrom(TGraph::Class())) {
         flush_master();

         TGraph *gr = (TGraph *)obj;
         auto funcs = gr->GetListOfFunctions();

         TIter fiter(funcs);
         TObject *fobj = nullptr;
         TPaveStats *stats = nullptr;

         while ((fobj = fiter()) != nullptr) {
           if (fobj->InheritsFrom(TPaveStats::Class()))
               stats = dynamic_cast<TPaveStats *> (fobj);
         }

         // ensure histogram exists on server to draw it properly on clients side
         if (!IsReadOnly() && first_obj)
            gr->GetHistogram();

         TString gropt = iter.GetOption();
         if (title && first_obj) gropt.Append(";;use_pad_title");
         if (stats) gropt.Append(";;use_pad_stats");

         paddata.NewPrimitive(obj, gropt.Data()).SetSnapshot(TWebSnapshot::kObject, obj);

         fiter.Reset();
         while ((fobj = fiter()) != nullptr)
            CreateObjectSnapshot(paddata, pad, fobj, fiter.GetOption());

         if (funcs)
            fPrimitivesLists.Add(funcs);
         first_obj = false;
      } else if (obj->InheritsFrom(TScatter::Class())) {
         flush_master();

         TScatter *scatter = (TScatter *)obj;
         auto funcs = scatter->GetGraph()->GetListOfFunctions();

         TIter fiter(funcs);
         TObject *fobj = nullptr, *palette = nullptr;

         while ((fobj = fiter()) != nullptr) {
            if (fobj->InheritsFrom("TPaletteAxis"))
               palette = fobj;
         }

         // ensure histogram exists on server to draw it properly on clients side
         if (!IsReadOnly() && first_obj)
            scatter->GetHistogram();

         if (!palette && CanCreateObject("TPaletteAxis")) {
            std::stringstream exec;
            exec << "new TPaletteAxis(0,0,0,0,0,0);";
            palette = (TObject *)gROOT->ProcessLine(exec.str().c_str());
            if (palette)
               funcs->AddFirst(palette);
         }

         TString scopt = iter.GetOption();
         if (title && first_obj) scopt.Append(";;use_pad_title");
         if (palette) scopt.Append(";;use_pad_palette");

         paddata.NewPrimitive(obj, scopt.Data()).SetSnapshot(TWebSnapshot::kObject, obj);

         fiter.Reset();
         while ((fobj = fiter()) != nullptr)
            CreateObjectSnapshot(paddata, pad, fobj, fiter.GetOption());

         if (funcs)
            fPrimitivesLists.Add(funcs);

         first_obj = false;
      } else if (obj->InheritsFrom(TF1::Class())) {
         flush_master();
         auto f1 = static_cast<TF1 *> (obj);

         TString f1opt = iter.GetOption();

         if (f1->IsA() == TF1::Class() || f1->IsA() == TF2::Class()) {
            if (paddata.IsBatchMode() || fTF1UseSave)
               f1->Save(0, 0, 0, 0, 0, 0);
            if (fTF1UseSave)
               f1opt.Append(";force_saved");
         }

         if (first_obj) {
            auto hist = f1->GetHistogram();
            paddata.NewPrimitive(hist, "__ignore_drawing__").SetSnapshot(TWebSnapshot::kObject, hist);
            f1opt.Append(";webcanv_hist");
         }

         paddata.NewPrimitive(f1, f1opt.Data()).SetSnapshot(TWebSnapshot::kObject, f1);

         first_obj = false;

      } else if (obj->InheritsFrom(TGaxis::Class())) {
         flush_master();
         auto gaxis = static_cast<TGaxis *> (obj);
         auto func = gaxis->GetFunction();
         if (func)
            paddata.NewPrimitive(func, "__ignore_drawing__").SetSnapshot(TWebSnapshot::kObject, func);

         paddata.NewPrimitive(obj, iter.GetOption()).SetSnapshot(TWebSnapshot::kObject, obj);
      } else if (IsJSSupportedClass(obj, usemaster)) {
         flush_master();
         paddata.NewPrimitive(obj, iter.GetOption()).SetSnapshot(TWebSnapshot::kObject, obj);
      } else {
         CreateObjectSnapshot(paddata, pad, obj, iter.GetOption(), usemaster ? &masterps : nullptr);
         pad_status._has_specials = true;
      }
   }

   flush_master();

   bool provide_colors = false;

   if ((GetPaletteDelivery() > 2) || ((GetPaletteDelivery() == 2) && resfunc)) {
      // provide colors: either for each subpad (> 2) or only for canvas (== 2)
      provide_colors = process_primitives;
   } else if ((GetPaletteDelivery() == 1) && resfunc) {
      // check that colors really changing, using hash

      if (fColorsVersion != fCanvVersion) {
         auto hash = CalculateColorsHash();
         if ((hash != fColorsHash) || (fColorsVersion == 0)) {
            fColorsHash = hash;
            fColorsVersion = fCanvVersion;
         }
      }

      provide_colors = fColorsVersion > version;
   }

   // add colors after painting is performed - new colors may be generated only during painting
   if (provide_colors)
      AddColorsPalette(paddata);

   if (!resfunc)
      return;

   // now move all primitives and functions into separate list to perform I/O

   TList save_lst;
   TIter diter(&fPrimitivesLists);
   TList *dlst = nullptr;
   while ((dlst = (TList *)diter()) != nullptr) {
      TIter fiter(dlst);
      while ((obj = fiter()) != nullptr)
         save_lst.Add(obj, fiter.GetOption());
      save_lst.Add(dlst); // add list itself to have marker
      dlst->Clear("nodelete");
   }

   // execute function to prevent storing of colors with custom TCanvas streamer
   // TODO: Olivier - we need to change logic here!
   TColor::DefinedColors();

   // invoke callback for master painting
   resfunc(&paddata);

   TIter siter(&save_lst);
   diter.Reset();
   while ((dlst = (TList *)diter()) != nullptr) {
      while ((obj = siter()) != nullptr) {
         if (obj == dlst)
            break;
         dlst->Add(obj, siter.GetOption());
      }
   }

   save_lst.Clear("nodelete");

   fPrimitivesLists.Clear("nodelete");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Add control message for specified connection
/// Same control message can be overwritten many time before it really sends to the client
/// If connid == 0, message will be add to all connections
/// If parameter send_immediately specified, tries to submit message immediately
/// Otherwise short timer is activated and message send afterwards

void TWebCanvas::AddCtrlMsg(unsigned connid, const std::string &key, const std::string &value, Bool_t send_immediately)
{
   for (auto &conn : fWebConn)
      if ((conn.fConnId == connid) || (connid == 0))
         conn.fCtrl[key] = value;

   if (send_immediately)
      CheckDataToSend(connid);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Add message to send queue for specified connection
/// If connid == 0, message will be add to all connections
/// Return kFALSE if queue is full or connection is not exists

Bool_t TWebCanvas::AddToSendQueue(unsigned connid, const std::string &msg)
{
   Bool_t res = kFALSE;
   for (auto &conn : fWebConn) {
      if ((conn.fConnId == connid) || (connid == 0)) {
         conn.fSend.emplace(msg);
         res = kTRUE;
      }
   }
   return res;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Check if any data should be send to client
/// If connid != 0, only selected connection will be checked

void TWebCanvas::CheckDataToSend(unsigned connid)
{
   if (!Canvas() || !fWindow)
      return;

   for (auto &conn : fWebConn) {
      if (connid && (conn.fConnId != connid))
         continue;

      // check if direct data sending is possible
      if (!fWindow->CanSend(conn.fConnId, true))
         continue;

      std::string buf;

      if ((conn.fCheckedVersion < fCanvVersion) && (conn.fSendVersion == conn.fDrawVersion)) {

         buf = "SNAP6:"s + std::to_string(fCanvVersion) + ":"s;

         TCanvasWebSnapshot holder(IsReadOnly(), true, false); // readonly, set ids, batchmode

         holder.SetFixedSize(fFixedSize); // set fixed size flag

         // scripts send only when canvas drawn for the first time
         if (!conn.fSendVersion)
            holder.SetScripts(fCustomScripts);

         holder.SetHighlightConnect(Canvas()->HasConnection("Highlighted(TVirtualPad*,TObject*,Int_t,Int_t)"));

         CreatePadSnapshot(holder, Canvas(), conn.fSendVersion, [&buf, &conn, this](TPadWebSnapshot *snap) {
            auto json = TBufferJSON::ToJSON(snap, fJsonComp);
            auto hash = json.Hash();
            if (conn.fLastSendHash && (conn.fLastSendHash == hash) && conn.fSendVersion) {
               // prevent looping when same data send many times
               buf.clear();
            } else {
               buf.append(json.Data());
               conn.fLastSendHash = hash;
            }
         });

         conn.fCheckedVersion = fCanvVersion;

         conn.fSendVersion = fCanvVersion;

         if (buf.empty())
            conn.fDrawVersion = fCanvVersion;

      } else if (!conn.fSend.empty()) {

         std::swap(buf, conn.fSend.front());
         conn.fSend.pop();

      } else if (!conn.fCtrl.empty()) {
         buf = "CTRL:"s + TBufferJSON::ToJSON(&conn.fCtrl, TBufferJSON::kMapAsObject + TBufferJSON::kNoSpaces);
         conn.fCtrl.clear();
      }

      if (!buf.empty())
         fWindow->Send(conn.fConnId, buf);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Close web canvas - not implemented

void TWebCanvas::Close()
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Show canvas in specified place.
/// If parameter args not specified, default ROOT web display will be used

void TWebCanvas::ShowWebWindow(const ROOT::Experimental::RWebDisplayArgs &args)
{
   if (!fWindow) {
      fWindow = ROOT::Experimental::RWebWindow::Create();

      fWindow->SetConnLimit(0); // configure connections limit

      fWindow->SetDefaultPage("file:rootui5sys/canv/canvas6.html");

      fWindow->SetCallBacks(
         // connection
         [this](unsigned connid) {
            fWebConn.emplace_back(connid);
            CheckDataToSend(connid);
         },
         // data
         [this](unsigned connid, const std::string &arg) {
            ProcessData(connid, arg);
            CheckDataToSend(connid);
         },
         // disconnect
         [this](unsigned connid) {
            unsigned indx = 0;
            for (auto &c : fWebConn) {
               if (c.fConnId == connid) {
                  fWebConn.erase(fWebConn.begin() + indx);
                  break;
               }
               indx++;
            }
         });
   }

   auto w = Canvas()->GetWindowWidth(), h = Canvas()->GetWindowHeight();
   if ((w > 0) && (w < 50000) && (h > 0) && (h < 30000))
      fWindow->SetGeometry(w, h);

   if ((args.GetBrowserKind() == ROOT::Experimental::RWebDisplayArgs::kQt5) ||
       (args.GetBrowserKind() == ROOT::Experimental::RWebDisplayArgs::kQt6) ||
       (args.GetBrowserKind() == ROOT::Experimental::RWebDisplayArgs::kCEF))
      SetLongerPolling(kTRUE);

   fWindow->Show(args);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Show canvas in browser window

void TWebCanvas::Show()
{
   if (gROOT->IsWebDisplayBatch())
      return;

   ROOT::Experimental::RWebDisplayArgs args;
   args.SetWidgetKind("TCanvas");
   args.SetSize(Canvas()->GetWindowWidth(), Canvas()->GetWindowHeight());
   args.SetPos(Canvas()->GetWindowTopX(), Canvas()->GetWindowTopY());

   ShowWebWindow(args);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Function used to send command to browser to toggle menu, toolbar, editors, ...

void TWebCanvas::ShowCmd(const std::string &arg, Bool_t show)
{
   if (AddToSendQueue(0, "SHOW:"s + arg + (show ? ":1"s : ":0"s)))
      CheckDataToSend();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Activate object in editor in web browser

void TWebCanvas::ActivateInEditor(TPad *pad, TObject *obj)
{
   if (!pad || !obj) return;

   UInt_t hash = TString::Hash(&obj, sizeof(obj));

   if (AddToSendQueue(0, "EDIT:"s + std::to_string(hash)))
      CheckDataToSend();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if web canvas has graphical editor

Bool_t TWebCanvas::HasEditor() const
{
   return (fClientBits & TCanvas::kShowEditor) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if web canvas has menu bar

Bool_t TWebCanvas::HasMenuBar() const
{
   return (fClientBits & TCanvas::kMenuBar) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if web canvas has status bar

Bool_t TWebCanvas::HasStatusBar() const
{
   return (fClientBits & TCanvas::kShowEventStatus) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if tooltips are activated in web canvas

Bool_t TWebCanvas::HasToolTips() const
{
   return (fClientBits & TCanvas::kShowToolTips) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window position of web canvas

void TWebCanvas::SetWindowPosition(Int_t x, Int_t y)
{
   AddCtrlMsg(0, "x"s, std::to_string(x));
   AddCtrlMsg(0, "y"s, std::to_string(y));
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window size of web canvas

void TWebCanvas::SetWindowSize(UInt_t w, UInt_t h)
{
   AddCtrlMsg(0, "w"s, std::to_string(w));
   AddCtrlMsg(0, "h"s, std::to_string(h));
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window title of web canvas

void TWebCanvas::SetWindowTitle(const char *newTitle)
{
   AddCtrlMsg(0, "title"s, newTitle);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set canvas size of web canvas

void TWebCanvas::SetCanvasSize(UInt_t cw, UInt_t ch)
{
   fFixedSize = kTRUE;
   AddCtrlMsg(0, "cw"s, std::to_string(cw));
   AddCtrlMsg(0, "ch"s, std::to_string(ch));
   AddCtrlMsg(0, "fixed_size"s, "true"s);
   if ((cw > 0) && (ch > 0)) {
      Canvas()->fCw = cw;
      Canvas()->fCh = ch;
   } else {
      // temporary value, will be reported back from client
      Canvas()->fCw = Canvas()->fWindowWidth;
      Canvas()->fCh = Canvas()->fWindowHeight;
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Assign clients bits

void TWebCanvas::AssignStatusBits(UInt_t bits)
{
   fClientBits = bits;
   Canvas()->SetBit(TCanvas::kShowEventStatus, bits & TCanvas::kShowEventStatus);
   Canvas()->SetBit(TCanvas::kShowEditor, bits & TCanvas::kShowEditor);
   Canvas()->SetBit(TCanvas::kShowToolTips, bits & TCanvas::kShowToolTips);
   Canvas()->SetBit(TCanvas::kMenuBar, bits & TCanvas::kMenuBar);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Decode all pad options, which includes ranges plus objects options

Bool_t TWebCanvas::DecodePadOptions(const std::string &msg, bool process_execs)
{
   if (IsReadOnly() || msg.empty())
      return kFALSE;

   auto arr = TBufferJSON::FromJSON<std::vector<TWebPadOptions>>(msg);

   if (!arr)
      return kFALSE;

   Bool_t need_update = kFALSE;

   TPad *pad_with_execs = nullptr;
   TExec *hist_exec = nullptr;

   for (unsigned n = 0; n < arr->size(); ++n) {
      auto &r = arr->at(n);
      TPad *pad = dynamic_cast<TPad *>(FindPrimitive(r.snapid));

      if (!pad)
         continue;

      if (pad == Canvas()) {
         AssignStatusBits(r.bits);
         Canvas()->fCw = r.cw;
         Canvas()->fCh = r.ch;
         if (r.w.size() == 4) {
            fWindowGeometry = r.w;
            Canvas()->fWindowTopX = fWindowGeometry[0];
            Canvas()->fWindowTopY = fWindowGeometry[1];
            Canvas()->fWindowWidth = fWindowGeometry[2];
            Canvas()->fWindowHeight = fWindowGeometry[3];
         }
      }

      if (r.active && (pad != gPad)) gPad = pad;

      if ((pad->GetTickx() != r.tickx) || (pad->GetTicky() != r.ticky))
         pad->SetTicks(r.tickx, r.ticky);
      if ((pad->GetGridx() != (r.gridx > 0)) || (pad->GetGridy() != (r.gridy > 0)))
         pad->SetGrid(r.gridx, r.gridy);
      pad->fLogx = r.logx;
      pad->fLogy = r.logy;
      pad->fLogz = r.logz;

      pad->SetLeftMargin(r.mleft);
      pad->SetRightMargin(r.mright);
      pad->SetTopMargin(r.mtop);
      pad->SetBottomMargin(r.mbottom);

      if (r.ranges) {
         // avoid call of original methods, set members directly
         // pad->Range(r.px1, r.py1, r.px2, r.py2);
         // pad->RangeAxis(r.ux1, r.uy1, r.ux2, r.uy2);

         pad->fX1 = r.px1;
         pad->fX2 = r.px2;
         pad->fY1 = r.py1;
         pad->fY2 = r.py2;

         pad->fUxmin = r.ux1;
         pad->fUxmax = r.ux2;
         pad->fUymin = r.uy1;
         pad->fUymax = r.uy2;
      }

      // pad->SetPad(r.mleft, r.mbottom, 1-r.mright, 1-r.mtop);

      pad->fAbsXlowNDC = r.xlow;
      pad->fAbsYlowNDC = r.ylow;
      pad->fAbsWNDC = r.xup - r.xlow;
      pad->fAbsHNDC = r.yup - r.ylow;

      if (pad == Canvas()) {
         pad->fXlowNDC = r.xlow;
         pad->fYlowNDC = r.ylow;
         pad->fXUpNDC = r.xup;
         pad->fYUpNDC = r.yup;
         pad->fWNDC = r.xup - r.xlow;
         pad->fHNDC = r.yup - r.ylow;
      } else {
         auto mother = pad->GetMother();
         if (mother->GetAbsWNDC() > 0. && mother->GetAbsHNDC() > 0.) {
            pad->fXlowNDC = (r.xlow - mother->GetAbsXlowNDC()) / mother->GetAbsWNDC();
            pad->fYlowNDC = (r.ylow - mother->GetAbsYlowNDC()) / mother->GetAbsHNDC();
            pad->fXUpNDC = (r.xup - mother->GetAbsXlowNDC()) / mother->GetAbsWNDC();
            pad->fYUpNDC = (r.yup - mother->GetAbsYlowNDC()) / mother->GetAbsHNDC();
            pad->fWNDC = (r.xup - r.xlow) / mother->GetAbsWNDC();
            pad->fHNDC = (r.yup - r.ylow) / mother->GetAbsHNDC();
         }
      }

      // copy of code from TPad::ResizePad()

      Double_t pxlow   = r.xlow * r.cw;
      Double_t pylow   = (1-r.ylow) * r.ch;
      Double_t pxrange = (r.xup - r.xlow) * r.cw;
      Double_t pyrange = -1*(r.yup - r.ylow) * r.ch;

      Double_t rounding = 0.00005;
      Double_t xrange  = r.px2 - r.px1;
      Double_t yrange  = r.py2 - r.py1;

      if ((xrange != 0.) && (pxrange != 0)) {
         // Linear X axis
         pad->fXtoAbsPixelk = rounding + pxlow - pxrange*r.px1/xrange;      //origin at left
         pad->fXtoPixelk = rounding +  -pxrange*r.px1/xrange;
         pad->fXtoPixel = pxrange/xrange;
         pad->fAbsPixeltoXk = r.px1 - pxlow*xrange/pxrange;
         pad->fPixeltoXk = r.px1;
         pad->fPixeltoX = xrange/pxrange;
      }

      if ((yrange != 0.) && (pyrange != 0.)) {
         // Linear Y axis
         pad->fYtoAbsPixelk = rounding + pylow - pyrange*r.py1/yrange;      //origin at top
         pad->fYtoPixelk = rounding +  -pyrange - pyrange*r.py1/yrange;
         pad->fYtoPixel = pyrange/yrange;
         pad->fAbsPixeltoYk = r.py1 - pylow*yrange/pyrange;
         pad->fPixeltoYk = r.py1;
         pad->fPixeltoY = yrange/pyrange;
      }

      pad->SetFixedAspectRatio(kFALSE);

      TObjLink *objlnk = nullptr;

      TH1 *hist = static_cast<TH1 *>(FindPrimitive("histogram", 1, pad, &objlnk));

      if (hist) {

         TObject *hist_holder = objlnk ? objlnk->GetObject() : nullptr;
         if (hist_holder == hist)
            hist_holder = nullptr;

         Bool_t no_entries = hist->GetEntries();

         Double_t hmin = 0., hmax = 0.;

         if (r.zx1 == r.zx2)
            hist->GetXaxis()->SetRange(0,0);
         else
            hist->GetXaxis()->SetRangeUser(r.zx1, r.zx2);

         if (hist->GetDimension() == 1) {
            hmin = r.zy1;
            hmax = r.zy2;
            if ((hmin == hmax) && !no_entries) {
               // if there are no zooming on Y and histogram has no entries, hmin/hmax should be set to full range
               hmin = pad->fLogy ? TMath::Power(pad->fLogy < 2 ? 10 : pad->fLogy, r.uy1) : r.uy1;
               hmax = pad->fLogy ? TMath::Power(pad->fLogy < 2 ? 10 : pad->fLogy, r.uy2) : r.uy2;
            }
         } else if (r.zy1 == r.zy2) {
            hist->GetYaxis()->SetRange(0., 0.);
         } else {
            hist->GetYaxis()->SetRangeUser(r.zy1, r.zy2);
         }

         if (hist->GetDimension() == 2) {
            hmin = r.zz1;
            hmax = r.zz2;
            if ((hmin == hmax) && !no_entries) {
               // z scale is not transformed
               hmin = r.uz1;
               hmax = r.uz2;
            }
         } else if (hist->GetDimension() == 3) {
            if (r.zz1 == r.zz2) {
               hist->GetZaxis()->SetRange(0., 0.);
            } else {
              hist->GetZaxis()->SetRangeUser(r.zz1, r.zz2);
            }
         }

         if (hmin == hmax)
            hmin = hmax = -1111;

         if (!hist_holder || (hist_holder->IsA() == TScatter::Class())) {
            hist->SetMinimum(hmin);
            hist->SetMaximum(hmax);
         } else {
            auto SetMember = [hist_holder](const char *name, Double_t value) {
               auto offset = hist_holder->IsA()->GetDataMemberOffset(name);
               if (offset > 0)
                  *((Double_t *)((char*) hist_holder + offset)) = value;
               else
                  ::Error("SetMember", "Cannot find %s data member in %s", name, hist_holder->ClassName());
            };

            // directly set min/max in classes like THStack, TGraph, TMultiGraph
            SetMember("fMinimum", hmin);
            SetMember("fMaximum", hmax);
         }

         TIter next(hist->GetListOfFunctions());
         while (auto fobj = next())
            if (!hist_exec && fobj->InheritsFrom(TExec::Class())) {
               hist_exec = (TExec *) fobj;
               need_update = kTRUE;
            }
      }

      std::map<std::string, int> idmap;

      for (auto &item : r.primitives) {
         auto iter = idmap.find(item.snapid);
         int idcnt = 1;
         if (iter == idmap.end())
            idmap[item.snapid] = 1;
         else
            idcnt = ++iter->second;

         ProcessObjectOptions(item, pad, idcnt);
      }

      // without special objects no need for explicit update of the pad
      if (fPadsStatus[pad]._has_specials) {
         pad->Modified(kTRUE);
         need_update = kTRUE;
      }

      if (process_execs && (gPad == pad))
         pad_with_execs = pad;
   }

   ProcessExecs(pad_with_execs, hist_exec);

   if (fUpdatedSignal) fUpdatedSignal(); // invoke signal

   return need_update;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Process TExec objects in the pad

void TWebCanvas::ProcessExecs(TPad *pad, TExec *extra)
{
   auto execs = pad ? pad->GetListOfExecs() : nullptr;

   if ((!execs || !execs->GetSize()) && !extra)
      return;

   auto saveps = gVirtualPS;
   TWebPS ps;
   gVirtualPS = &ps;

   auto savex = gVirtualX;
   TVirtualX x;
   gVirtualX = &x;

   TIter next(execs);
   while (auto obj = next()) {
      auto exec = dynamic_cast<TExec *>(obj);
      if (exec)
         exec->Exec();
   }

   if (extra)
      extra->Exec();

   gVirtualPS = saveps;
   gVirtualX = savex;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Execute one or several methods for selected object
/// String can be separated by ";;" to let execute several methods at once
void TWebCanvas::ProcessLinesForObject(TObject *obj, const std::string &lines)
{
   std::string buf = lines;

   while (obj && !buf.empty()) {
      std::string sub = buf;
      auto pos = buf.find(";;");
      if (pos == std::string::npos) {
         sub = buf;
         buf.clear();
      } else {
         sub = buf.substr(0,pos);
         buf = buf.substr(pos+2);
      }
      if (sub.empty()) continue;

      std::stringstream exec;
      exec << "((" << obj->ClassName() << " *) " << std::hex << std::showbase << (size_t)obj << ")->" << sub << ";";
      Info("ProcessLinesForObject", "Obj %s Execute %s", obj->GetName(), exec.str().c_str());
      gROOT->ProcessLine(exec.str().c_str());
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Handle data from web browser
/// Returns kFALSE if message was not processed

Bool_t TWebCanvas::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg.empty())
      return kTRUE;

   // try to identify connection for given WS request
   unsigned indx = 0;
   for (auto &c : fWebConn) {
      if (c.fConnId == connid) break;
      indx++;
   }
   if (indx >= fWebConn.size())
      return kTRUE;

   struct FlagGuard {
      Bool_t &flag;
      FlagGuard(Bool_t &_flag) : flag(_flag) { flag = true; }
      ~FlagGuard() { flag = false; }
   };

   FlagGuard guard(fProcessingData);

   const char *cdata = arg.c_str();

   if (arg == "KEEPALIVE") {
      // do nothing

   } else if (arg == "QUIT") {

      // use window manager to correctly terminate http server
      fWindow->TerminateROOT();

   } else if (arg.compare(0, 7, "READY6:") == 0) {

      // this is reply on drawing of ROOT6 snapshot
      // it confirms when drawing of specific canvas version is completed

      cdata += 7;

      const char *separ = strchr(cdata, ':');
      if (!separ) {
         fWebConn[indx].fDrawVersion = std::stoll(cdata);
      } else {
         fWebConn[indx].fDrawVersion = std::stoll(std::string(cdata, separ - cdata));
         if ((indx == 0) && !IsReadOnly())
            if (DecodePadOptions(separ+1, false)) {
               CheckCanvasModified();
               CheckDataToSend();
            }
      }

   } else if (arg == "RELOAD") {

      // trigger reload of canvas data
      fWebConn[indx].reset();

   } else if (arg.compare(0, 5, "SAVE:") == 0) {

      // save image produced by the client side - like png or svg
      const char *img = cdata + 5;

      const char *separ = strchr(img, ':');
      if (separ) {
         TString filename(img, separ - img);
         img = separ + 1;

         std::ofstream ofs(filename.Data());

         if (filename.Index(".svg") != kNPOS) {
            // ofs << "<?xml version=\"1.0\" standalone=\"no\"?>";
            ofs << img;
         } else {
            TString binary = TBase64::Decode(img);
            ofs.write(binary.Data(), binary.Length());
         }
         ofs.close();

         Info("ProcessData", "File %s has been created", filename.Data());
      }

   } else if (arg.compare(0, 8, "PRODUCE:") == 0) {

      // create ROOT, PDF, ... files using native ROOT functionality
      Canvas()->Print(arg.c_str() + 8);

   } else if (arg.compare(0, 9, "OPTIONS6:") == 0) {

      if ((indx == 0) && !IsReadOnly())
         if (DecodePadOptions(arg.substr(9), true)) {

            CheckCanvasModified();
            CheckDataToSend();
         }

   } else if (arg.compare(0, 11, "STATUSBITS:") == 0) {

      if (indx == 0) {
         AssignStatusBits(std::stoul(arg.substr(11)));
         if (fUpdatedSignal) fUpdatedSignal(); // invoke signal
      }
   } else if (arg.compare(0, 10, "HIGHLIGHT:") == 0) {
      if (indx == 0) {
         auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(10));
         if (!arr || (arr->size() != 4)) {
            Error("ProcessData", "Wrong arguments count %d in highlight message", (int)(arr ? arr->size() : -1));
         } else {
            auto pad = dynamic_cast<TVirtualPad *>(FindPrimitive(arr->at(0)));
            auto obj = FindPrimitive(arr->at(1));
            int argx = std::stoi(arr->at(2));
            int argy = std::stoi(arr->at(3));
            if (pad && obj) {
              Canvas()->Highlighted(pad, obj, argx, argy);
              CheckCanvasModified();
            }
         }
      }
   } else if (ROOT::Experimental::RWebWindow::IsFileDialogMessage(arg)) {

      ROOT::Experimental::RWebWindow::EmbedFileDialog(fWindow, connid, arg);

   } else if (IsReadOnly()) {

      // all following messages are not allowed in readonly mode
      return kFALSE;

   } else if (arg.compare(0, 6, "EVENT:") == 0) {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(6));
      if (!arr || (arr->size() != 5)) {
         Error("ProcessData", "Wrong arguments count %d in event message", (int)(arr ? arr->size() : -1));
      } else {
         auto pad = dynamic_cast<TPad *>(FindPrimitive(arr->at(0)));
         std::string kind = arr->at(1);
         int event = -1;
         if (kind == "move"s) event = kMouseMotion;
         int argx = std::stoi(arr->at(2));
         int argy = std::stoi(arr->at(3));
         auto selobj = FindPrimitive(arr->at(4));

         if ((event >= 0) && pad && (pad == gPad)) {
            Canvas()->fEvent = event;
            Canvas()->fEventX = argx;
            Canvas()->fEventY = argy;

            Canvas()->fSelected = selobj;

            ProcessExecs(pad);
         }
      }

   } else if (arg.compare(0, 8, "GETMENU:") == 0) {

      TObject *obj = FindPrimitive(arg.substr(8));
      if (!obj)
         obj = Canvas();

      TWebMenuItems items(arg.c_str() + 8);
      items.PopulateObjectMenu(obj, obj->IsA());
      std::string buf = "MENU:";
      buf.append(TBufferJSON::ToJSON(&items, 103).Data());

      AddToSendQueue(connid, buf);

   } else if (arg.compare(0, 8, "PRIMIT6:") == 0) {

      if (IsFirstConn(connid) && !IsReadOnly()) { // only first connection can modify object

         auto opt = TBufferJSON::FromJSON<TWebObjectOptions>(arg.c_str() + 8);

         if (opt) {
            TPad *modpad = ProcessObjectOptions(*opt, nullptr);

            // indicate that pad was modified
            if (modpad)
               modpad->Modified();
         }
      }

   } else if (arg.compare(0, 11, "PADCLICKED:") == 0) {

      auto click = TBufferJSON::FromJSON<TWebPadClick>(arg.c_str() + 11);

      if (click && IsFirstConn(connid) && !IsReadOnly()) {

         TPad *pad = dynamic_cast<TPad *>(FindPrimitive(click->padid));

         if (pad && pad->InheritsFrom(TButton::Class())) {
            auto btn = (TButton *) pad;
            const char *mthd = btn->GetMethod();
            if (mthd && *mthd) {
               TVirtualPad::TContext ctxt(gROOT->GetSelectedPad(), kTRUE, kTRUE);
               gROOT->ProcessLine(mthd);
            }
            return kTRUE;
         }

         if (pad && (pad != gPad)) {
            gPad = pad;
            Canvas()->SetClickSelectedPad(pad);
            if (fActivePadChangedSignal)
               fActivePadChangedSignal(pad);
         }

         if (!click->objid.empty()) {
            auto selobj = FindPrimitive(click->objid);
            Canvas()->SetClickSelected(selobj);
            Canvas()->fSelected = selobj;
            if (pad && selobj && fObjSelectSignal)
               fObjSelectSignal(pad, selobj);
         }

         if ((click->x >= 0) && (click->y >= 0)) {
            Canvas()->fEvent = click->dbl ? kButton1Double : kButton1Up;
            Canvas()->fEventX = click->x;
            Canvas()->fEventY = click->y;
            if (click->dbl && fPadDblClickedSignal)
               fPadDblClickedSignal(pad, click->x, click->y);
            else if (!click->dbl && fPadClickedSignal)
               fPadClickedSignal(pad, click->x, click->y);
         }

         ProcessExecs(pad);
      }

   } else if (arg.compare(0, 8, "OBJEXEC:") == 0) {

     auto buf = arg.substr(8);
     auto pos = buf.find(":");

     if ((pos > 0) && IsFirstConn(connid) && !IsReadOnly()) { // only first client can execute commands
        auto sid = buf.substr(0, pos);
        buf.erase(0, pos + 1);

        TObjLink *lnk = nullptr;
        TPad *objpad = nullptr;

        TObject *obj = FindPrimitive(sid, 1, nullptr, &lnk, &objpad);

        if (obj && !buf.empty()) {

           ProcessLinesForObject(obj, buf);

           if (objpad)
              objpad->Modified();
           else
              Canvas()->Modified();

           CheckCanvasModified();
        }
     }

   } else if (arg.compare(0, 12, "EXECANDSEND:") == 0) {

      // execute method and send data, used by drawing projections

      std::string buf = arg.substr(12);
      std::string reply;
      TObject *obj = nullptr;

      auto pos = buf.find(":");

      if ((pos > 0) && IsFirstConn(connid) && !IsReadOnly()) {
         // only first client can execute commands
         reply = buf.substr(0, pos);
         buf.erase(0, pos + 1);
         pos = buf.find(":");
         if (pos > 0) {
            auto sid = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            obj = FindPrimitive(sid);
         }
      }

      if (obj && !buf.empty() && !reply.empty()) {
         std::stringstream exec;
         exec << "((" << obj->ClassName() << " *) " << std::hex << std::showbase << (size_t)obj
              << ")->" << buf << ";";
         if (gDebug > 1)
            Info("ProcessData", "Obj %s Exec %s", obj->GetName(), exec.str().c_str());

         auto res = gROOT->ProcessLine(exec.str().c_str());
         TObject *resobj = (TObject *)(res);
         if (resobj) {
            std::string send = reply;
            send.append(":");
            send.append(TBufferJSON::ToJSON(resobj, 23).Data());
            AddToSendQueue(connid, send);
            if (reply[0] == 'D')
               delete resobj; // delete object if first symbol in reply is D
         }
      }

   } else if (arg.compare(0, 6, "CLEAR:") == 0) {
      std::string snapid = arg.substr(6);

      TPad *pad = dynamic_cast<TPad *>(FindPrimitive(snapid));

      if (pad) {
         pad->Clear();
         pad->Modified();
         PerformUpdate();
      } else {
         printf("Not found pad with id %s\n", snapid.c_str());
      }
   } else if (arg.compare(0, 7, "DIVIDE:") == 0) {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
      if (arr && arr->size() == 2) {
         TPad *pad = dynamic_cast<TPad *>(FindPrimitive(arr->at(0)));
         int nn = 0, n1 = 0, n2 = 0;

         std::string divide = arr->at(1);
         auto p = divide.find('x');
         if (p == std::string::npos)
            p = divide.find('X');

         if (p != std::string::npos) {
            n1 = std::stoi(divide.substr(0,p));
            n2 = std::stoi(divide.substr(p+1));
         } else {
            nn = std::stoi(divide);
         }

         if (pad && ((nn > 1) || (n1*n2 > 1))) {
            pad->Clear();
            pad->Modified();
            if (nn > 1)
               pad->DivideSquare(nn);
            else
               pad->Divide(n1, n2);
            pad->cd(1);
            PerformUpdate();
         }
      }

   } else if (arg.compare(0, 8, "DRAWOPT:") == 0) {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(8));
      if (arr && arr->size() == 2) {
         TObjLink *objlnk = nullptr;
         FindPrimitive(arr->at(0), 1, nullptr, &objlnk);
         if (objlnk)
            objlnk->SetOption(arr->at(1).c_str());
      }
   } else if (arg.compare(0, 8, "RESIZED:") == 0) {
      auto arr = TBufferJSON::FromJSON<std::vector<int>>(arg.substr(8));
      if (arr && arr->size() == 7) {
         // set members directly to avoid redrawing of the client again
         Canvas()->fCw = arr->at(4);
         Canvas()->fCh = arr->at(5);
         fFixedSize = arr->at(6) > 0;
         arr->resize(4);
         fWindowGeometry = *arr;
         Canvas()->fWindowTopX = fWindowGeometry[0];
         Canvas()->fWindowTopY = fWindowGeometry[1];
         Canvas()->fWindowWidth = fWindowGeometry[2];
         Canvas()->fWindowHeight = fWindowGeometry[3];
      }
   } else if (arg.compare(0, 7, "POPOBJ:") == 0) {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
      if (arr && arr->size() == 2) {
         TPad *pad = dynamic_cast<TPad *>(FindPrimitive(arr->at(0)));
         TObject *obj = FindPrimitive(arr->at(1), 0, pad);
         if (pad && obj && (obj != pad->GetListOfPrimitives()->Last())) {
            TIter next(pad->GetListOfPrimitives());
            while (auto o = next())
               if (obj == o) {
                  TString opt = next.GetOption();
                  pad->GetListOfPrimitives()->Remove(obj);
                  pad->GetListOfPrimitives()->AddLast(obj, opt.Data());
                  pad->Modified();
                  break;
               }
         }
      }
   } else if (arg == "INTERRUPT"s) {
      gROOT->SetInterrupt();
   } else {
      // unknown message, probably should be processed by other implementation
      return kFALSE;
   }

   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns true if any pad in the canvas were modified
/// Reset modified flags, increment canvas version (if inc_version is true)

void TWebCanvas::CheckPadModified(TPad *pad)
{
   if (fPadsStatus.find(pad) == fPadsStatus.end())
      fPadsStatus[pad] = PadStatus{0, true, true};

   auto &entry = fPadsStatus[pad];
   entry._detected = true;
   if (pad->IsModified()) {
      pad->Modified(kFALSE);
      entry._modified = true;
   }

   TIter iter(pad->GetListOfPrimitives());
   while (auto obj = iter()) {
      if (obj->InheritsFrom(TPad::Class()))
         CheckPadModified(static_cast<TPad *>(obj));
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Check if any pad on the canvas was modified
/// If yes, increment version of correspondent pad

void TWebCanvas::CheckCanvasModified(bool force_modified)
{
   // clear temporary flags
   for (auto &entry : fPadsStatus) {
      entry.second._detected = false;
      entry.second._modified = force_modified;
   }

   // scan sub-pads
   CheckPadModified(Canvas());

   // remove no-longer existing pads
   bool is_any_modified = false;
   for(auto iter = fPadsStatus.begin(); iter != fPadsStatus.end(); ) {
      if (iter->second._modified)
         is_any_modified = true;
      if (!iter->second._detected)
         fPadsStatus.erase(iter++);
      else
         iter++;
   }

   // if any pad modified, increment canvas version and set version of modified pads
   if (is_any_modified) {
      fCanvVersion++;
      for(auto &entry : fPadsStatus)
         if (entry.second._modified)
            entry.second.fVersion = fCanvVersion;
   }
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Returns window geometry including borders and menus

UInt_t TWebCanvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   if (fWindowGeometry.size() == 4) {
      x = fWindowGeometry[0];
      y = fWindowGeometry[1];
      w = fWindowGeometry[2];
      h = fWindowGeometry[3];
   } else {
      x = Canvas()->fWindowTopX;
      y = Canvas()->fWindowTopY;
      w = Canvas()->fWindowWidth;
      h = Canvas()->fWindowHeight;
   }
   return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////
/// if canvas or any subpad was modified,
/// scan all primitives in the TCanvas and subpads and convert them into
/// the structure which will be delivered to JSROOT client

Bool_t TWebCanvas::PerformUpdate()
{
   if (!fWindow)
      return kTRUE;

   CheckCanvasModified();

   CheckDataToSend();

   if (!fProcessingData && !IsAsyncMode())
      WaitWhenCanvasPainted(fCanvVersion);

   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Increment canvas version and force sending data to client - do not wait for reply

void TWebCanvas::ForceUpdate()
{
   if (!fWindow)
      return;

   CheckCanvasModified(true);

   CheckDataToSend();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Wait when specified version of canvas was painted and confirmed by browser

Bool_t TWebCanvas::WaitWhenCanvasPainted(Long64_t ver)
{
   if (!fWindow)
      return kTRUE;

   // simple polling loop until specified version delivered to the clients
   // first 500 loops done without sleep, then with 1ms sleep and last 500 with 100 ms sleep

   long cnt = 0, cnt_limit = GetLongerPolling() ? 5500 : 1500;

   if (gDebug > 2)
      Info("WaitWhenCanvasPainted", "version %ld", (long)ver);

   while (cnt++ < cnt_limit) {

      if (!fWindow->HasConnection(0, false)) {
         if (gDebug > 2)
            Info("WaitWhenCanvasPainted", "no connections - abort");
         return kFALSE; // wait ~1 min if no new connection established
      }

      if ((fWebConn.size() > 0) && (fWebConn.front().fDrawVersion >= ver)) {
         if (gDebug > 2)
            Info("WaitWhenCanvasPainted", "ver %ld got painted", (long)ver);
         return kTRUE;
      }

      gSystem->ProcessEvents();
      if (cnt > 500)
         gSystem->Sleep((cnt < cnt_limit - 500) ? 1 : 100); // increase sleep interval when do very often
   }

   if (gDebug > 2)
      Info("WaitWhenCanvasPainted", "timeout");

   return kFALSE;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Create JSON painting output for given pad
/// Produce JSON can be used for offline drawing with JSROOT

TString TWebCanvas::CreatePadJSON(TPad *pad, Int_t json_compression, Bool_t batchmode)
{
   TString res;
   if (!pad)
      return res;

   TCanvas *c = dynamic_cast<TCanvas *>(pad);
   if (c) {
      res = CreateCanvasJSON(c, json_compression, batchmode);
   } else {
      auto imp = std::make_unique<TWebCanvas>(pad->GetCanvas(), pad->GetName(), 0, 0, pad->GetWw(), pad->GetWh(), kTRUE);

      TPadWebSnapshot holder(true, false, batchmode); // readonly, no ids, batchmode

      imp->CreatePadSnapshot(holder, pad, 0, [&res, json_compression](TPadWebSnapshot *snap) {
         res = TBufferJSON::ToJSON(snap, json_compression);
      });
   }

   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Create JSON painting output for given canvas
/// Produce JSON can be used for offline drawing with JSROOT

TString TWebCanvas::CreateCanvasJSON(TCanvas *c, Int_t json_compression, Bool_t batchmode)
{
   TString res;

   if (!c)
      return res;

   {
      auto imp = std::make_unique<TWebCanvas>(c, c->GetName(), 0, 0, c->GetWw(), c->GetWh(), kTRUE);

      TCanvasWebSnapshot holder(true, false, batchmode); // readonly, no ids, batchmode

      imp->CreatePadSnapshot(holder, c, 0, [&res, json_compression](TPadWebSnapshot *snap) {
         res = TBufferJSON::ToJSON(snap, json_compression);
      });
   }

   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Create JSON painting output for given canvas and store into the file
/// See TBufferJSON::ExportToFile() method for more details about option
/// If option string starts with symbol 'b', JSON for batch mode will be generated

Int_t TWebCanvas::StoreCanvasJSON(TCanvas *c, const char *filename, const char *option)
{
   Int_t res = 0;
   Bool_t batchmode = kFALSE;
   if (option && *option == 'b') {
      batchmode = kTRUE;
      ++option;
   }

   if (!c)
      return res;

   {
      auto imp = std::make_unique<TWebCanvas>(c, c->GetName(), 0, 0, c->GetWw(), c->GetWh(), kTRUE);

      TCanvasWebSnapshot holder(true, false, batchmode); // readonly, no ids, batchmode

      imp->CreatePadSnapshot(holder, c, 0, [&res, filename, option](TPadWebSnapshot *snap) {
         res = TBufferJSON::ExportToFile(filename, snap, option);
      });
   }

   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Create image using batch (headless) capability of Chrome or Firefox browsers
/// Supported png, jpeg, svg, pdf formats

bool TWebCanvas::ProduceImage(TPad *pad, const char *fileName, Int_t width, Int_t height)
{
   if (!pad)
      return false;

   auto json = CreatePadJSON(pad, TBufferJSON::kNoSpaces + TBufferJSON::kSameSuppression, kTRUE);
   if (!json.Length())
      return false;

   if (!width && !height) {
      if ((pad->GetCanvas() == pad) || (pad->IsA() == TCanvas::Class())) {
         width = pad->GetWw();
         height = pad->GetWh();
      } else {
         width = (Int_t) (pad->GetAbsWNDC() * pad->GetCanvas()->GetWw());
         height = (Int_t) (pad->GetAbsHNDC() * pad->GetCanvas()->GetWh());
      }
   }

   return ROOT::Experimental::RWebDisplayHandle::ProduceImage(fileName, json.Data(), width, height);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Create images for several pads using batch (headless) capability of Chrome or Firefox browsers
/// Supported png, jpeg, svg, pdf formats
/// For png/jpeg/svg filename can include % symbol which will be replaced by image index.
/// For pdf format all images will be stored in single PDF file

bool TWebCanvas::ProduceImages(std::vector<TPad *> pads, const char *filename, Int_t width, Int_t height)
{
   if (pads.empty())
      return false;

   std::vector<std::string> jsons;
   std::vector<Int_t> widths, heights;

   for (auto pad: pads) {
      auto json = CreatePadJSON(pad, TBufferJSON::kNoSpaces + TBufferJSON::kSameSuppression, kTRUE);
      if (!json.Length())
         continue;

      Int_t w = width, h = height;

      if (!w && !h) {
         if ((pad->GetCanvas() == pad) || (pad->IsA() == TCanvas::Class())) {
            w = pad->GetWw();
            h = pad->GetWh();
         } else {
            w = (Int_t) (pad->GetAbsWNDC() * pad->GetCanvas()->GetWw());
            h = (Int_t) (pad->GetAbsHNDC() * pad->GetCanvas()->GetWh());
         }
      }

      jsons.emplace_back(json.Data());
      widths.emplace_back(w);
      heights.emplace_back(h);
   }

   return ROOT::Experimental::RWebDisplayHandle::ProduceImages(filename, jsons, widths, heights);
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Process data for single primitive
/// Returns object pad if object was modified

TPad *TWebCanvas::ProcessObjectOptions(TWebObjectOptions &item, TPad *pad, int idcnt)
{
   TObjLink *lnk = nullptr;
   TPad *objpad = nullptr;
   TObject *obj = FindPrimitive(item.snapid, idcnt, pad, &lnk, &objpad);

   if (item.fcust.compare("exec") == 0) {
      auto pos = item.opt.find("(");
      if (obj && (pos != std::string::npos) && obj->IsA()->GetMethodAllAny(item.opt.substr(0,pos).c_str())) {
         std::stringstream exec;
         exec << "((" << obj->ClassName() << " *) " << std::hex << std::showbase
                      << (size_t)obj << ")->" << item.opt << ";";
         Info("ProcessObjectOptions", "Obj %s Execute %s", obj->GetName(), exec.str().c_str());
         gROOT->ProcessLine(exec.str().c_str());
      } else {
         Error("ProcessObjectOptions", "Fail to execute %s for object %p %s", item.opt.c_str(), obj, obj ? obj->ClassName() : "---");
         objpad = nullptr;
      }
      return objpad;
   }

   bool modified = false;

   if (obj && lnk) {
      auto pos = item.opt.find(";;use_"); // special coding of extra options
      if (pos != std::string::npos) item.opt.resize(pos);

      if (gDebug > 1)
         Info("ProcessObjectOptions", "Set draw option %s for object %s %s", item.opt.c_str(),
               obj->ClassName(), obj->GetName());

      lnk->SetOption(item.opt.c_str());

      modified = true;
   }

   if (item.fcust.compare(0,10,"auto_exec:") == 0) {
      ProcessLinesForObject(obj, item.fcust.substr(10));
   } else if (item.fcust.compare("frame") == 0) {
      if (obj && obj->InheritsFrom(TFrame::Class())) {
         TFrame *frame = static_cast<TFrame *>(obj);
         if (item.fopt.size() >= 4) {
            frame->SetX1(item.fopt[0]);
            frame->SetY1(item.fopt[1]);
            frame->SetX2(item.fopt[2]);
            frame->SetY2(item.fopt[3]);
            modified = true;
         }
      }
   } else if (item.fcust.compare(0,4,"pave") == 0) {
      if (obj && obj->InheritsFrom(TPave::Class())) {
         TPave *pave = static_cast<TPave *>(obj);
         if ((item.fopt.size() >= 4) && objpad) {
            TVirtualPad::TContext ctxt(objpad, kFALSE);

            // first time need to overcome init problem
            pave->ConvertNDCtoPad();

            pave->SetX1NDC(item.fopt[0]);
            pave->SetY1NDC(item.fopt[1]);
            pave->SetX2NDC(item.fopt[2]);
            pave->SetY2NDC(item.fopt[3]);
            modified = true;

            pave->ConvertNDCtoPad();
         }
         if ((item.fcust.length() > 4) && pave->InheritsFrom(TPaveStats::Class())) {
            // add text lines for statsbox
            auto stats = static_cast<TPaveStats *>(pave);
            stats->Clear();
            size_t pos_start = 6, pos_end;
            while ((pos_end = item.fcust.find(";;", pos_start)) != std::string::npos) {
               stats->AddText(item.fcust.substr(pos_start, pos_end - pos_start).c_str());
               pos_start = pos_end + 2;
            }
            stats->AddText(item.fcust.substr(pos_start).c_str());
         }
      }
   } else if (item.fcust.compare(0,9,"func_fail") == 0) {
      if (!fTF1UseSave) {
         fTF1UseSave = kTRUE;
         modified = true;
      } else {
         Error("ProcessObjectOptions", "Client fails to calculate function %s cl %s but it should not try!", obj ? obj->GetName() : "---", obj ? obj->ClassName() : "---");
      }
   }

   return modified ? objpad : nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Search of object with given id in list of primitives
/// One could specify pad where search could be start
/// Also if object is in list of primitives, one could ask for entry link for such object,
/// This can allow to change draw option

TObject *TWebCanvas::FindPrimitive(const std::string &sid, int idcnt, TPad *pad, TObjLink **objlnk, TPad **objpad)
{
   if (sid.empty() || (sid == "0"s))
      return nullptr;

   if (!pad)
      pad = Canvas();

   std::string kind;
   auto separ = sid.find("#");
   long unsigned id = 0;
   bool search_hist = false;

   if (sid == "histogram") {
      search_hist = true;
   } else if (separ == std::string::npos) {
      id = std::stoul(sid);
   } else {
      kind = sid.substr(separ + 1);
      id = std::stoul(sid.substr(0, separ));
   }

   if (!search_hist && TString::Hash(&pad, sizeof(pad)) == id)
      return pad;

   auto getHistogram = [](TObject *obj) -> TH1* {
      auto offset = obj->IsA()->GetDataMemberOffset("fHistogram");
      if (offset > 0)
         return *((TH1 **)((char*) obj + offset));
      ::Error("getHistogram", "Cannot access fHistogram data member in %s", obj->ClassName());
      return nullptr;
   };

   for (auto lnk = pad->GetListOfPrimitives()->FirstLink(); lnk != nullptr; lnk = lnk->Next()) {
      TObject *obj = lnk->GetObject();
      if (!obj) continue;

      TString opt = lnk->GetOption();
      opt.ToUpper();

      TH1 *h1 = obj->InheritsFrom(TH1::Class()) ? static_cast<TH1 *>(obj) : nullptr;
      TGraph *gr = obj->InheritsFrom(TGraph::Class()) ? static_cast<TGraph *>(obj) : nullptr;
      TScatter *scatter = obj->InheritsFrom(TScatter::Class()) ? static_cast<TScatter *>(obj) : nullptr;
      TMultiGraph *mg = obj->InheritsFrom(TMultiGraph::Class()) ? static_cast<TMultiGraph *>(obj) : nullptr;
      THStack *hs = obj->InheritsFrom(THStack::Class()) ? static_cast<THStack *>(obj) : nullptr;
      TF1 *f1 = obj->InheritsFrom(TF1::Class()) ? static_cast<TF1 *>(obj) : nullptr;

      if (search_hist) {
         if (objlnk)
            *objlnk = lnk;

         if (h1)
            return h1;
         if (gr)
            return getHistogram(gr);
         if (scatter)
            return getHistogram(scatter);
         if (mg && opt.Contains("A"))
            return getHistogram(mg);
         if (hs)
            return getHistogram(hs);
         if (f1)
            return getHistogram(f1);

         if (objlnk)
            *objlnk = nullptr;

         continue;
      }

      if ((TString::Hash(&obj, sizeof(obj)) == id) && (--idcnt <= 0)) {
         if (objpad)
            *objpad = pad;

         if (kind.compare(0, 4, "hist") == 0) {
            if (gr)
               obj = h1 = getHistogram(gr);
            else if (mg)
               obj = h1 = getHistogram(mg);
            else if (hs)
               obj = h1 = getHistogram(hs);
            else if (scatter)
               obj = h1 = getHistogram(scatter);
            else if (f1)
               obj = h1 = getHistogram(f1);

            kind.erase(0,4);
            if (!kind.empty() && (kind[0]=='#')) kind.erase(0,1);
            objlnk = nullptr;
         }

         if (h1 && (kind == "x"))
            return h1->GetXaxis();
         if (h1 && (kind == "y"))
            return h1->GetYaxis();
         if (h1 && (kind == "z"))
            return h1->GetZaxis();

         if ((h1 || gr || scatter) && !kind.empty() && (kind.compare(0,5,"func_") == 0)) {
            auto funcname = kind.substr(5);
            TCollection *col = h1 ? h1->GetListOfFunctions() : (gr ? gr->GetListOfFunctions() : scatter->GetGraph()->GetListOfFunctions());
            return col ? col->FindObject(funcname.c_str()) : nullptr;
         }

         if ((h1 || gr) && !kind.empty() && (kind.compare(0,5,"indx_") == 0)) {
            auto col = h1 ? h1->GetListOfFunctions() : gr->GetListOfFunctions();
            return col ? col->At(std::stoi(kind.substr(5))) : nullptr;
         }

         if (!kind.empty() && (kind.compare(0,7,"member_") == 0)) {
            auto member = kind.substr(7);
            auto offset = obj->IsA() ? obj->IsA()->GetDataMemberOffset(member.c_str()) : 0;
            if (offset > 0) {
               TObject **mobj = (TObject **)((char*) obj + offset);
               return *mobj;
            }
            return nullptr;
         }

         if (objlnk)
            *objlnk = lnk;
         return obj;
      }

      if (h1 || gr || scatter) {
         TIter fiter(h1 ? h1->GetListOfFunctions() : (gr ? gr->GetListOfFunctions() : scatter->GetGraph()->GetListOfFunctions()));
         TObject *fobj = nullptr;
         while ((fobj = fiter()) != nullptr)
            if (TString::Hash(&fobj, sizeof(fobj)) == id) {
               if (objpad)
                  *objpad = pad;
               return fobj;
            }
      } else if (obj->InheritsFrom(TPad::Class())) {
         obj = FindPrimitive(sid, idcnt, (TPad *)obj, objlnk, objpad);
         if (objpad && !*objpad)
            *objpad = pad;
         if (obj)
            return obj;
      }
   }

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static method to create TWebCanvas instance
/// Used by plugin manager

TCanvasImp *TWebCanvas::NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   Bool_t readonly = gEnv->GetValue("WebGui.FullCanvas", (Int_t) 1) == 0;

   auto imp = new TWebCanvas(c, name, x, y, width, height, readonly);

   c->fWindowTopX = x;
   c->fWindowTopY = y;
   c->fWindowWidth = width;
   c->fWindowHeight = height;
   if (!gROOT->IsBatch() && (height > 25))
      height -= 25;
   c->fCw = width;
   c->fCh = height;

   return imp;
}

