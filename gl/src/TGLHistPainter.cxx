#include <cstring>
#include <cctype>

#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "TF3.h"

#include "TGLSurfacePainter.h"
#include "TGLHistPainter.h"
#include "TGLLegoPainter.h"
#include "TGLBoxPainter.h"
#include "TGLTF3Painter.h"

ClassImp(TGLHistPainter)

//______________________________________________________________________________
TGLHistPainter::TGLHistPainter(TH1 *hist)
                   : fDefaultPainter(TVirtualHistPainter::HistPainter(hist)),
                     fGLPainter(0),
                     fHist(hist),
                     fF3(0),
                     fStack(0),
                     fPlotType(kGLDefaultPlot)
{
   //ROOT does not use exceptions, so, if default painter's creation failed,
   //fDefaultPainter is 0. In each function, which use it, I have to check pointer first.
}

//______________________________________________________________________________
Int_t TGLHistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
   //Selects plot or axis. 
   //9999 is the magic number, ROOT's classes use in DistancetoPrimitive.
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->DistancetoPrimitive(px, py) : 9999;
   else {

      py -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
      px -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
      const Int_t glContext = gPad->GetGLDevice();

      if (glContext != -1) {
         fGLPainter->SetGLContext(glContext);
         if (!gGLManager->PlotSelected(fGLPainter.get(), px, py))
            gPad->SetSelected(gPad);
      } else {
         Error("DistancetoPrimitive", 
               "Attempt to use TGLHistPainter, while the current pad (gPad) does not support gl");
         gPad->SetSelected(gPad);
      }

      return 0;
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawPanel()
{
   //Default implementation is OK.
   if (fDefaultPainter.get())
      fDefaultPainter->DrawPanel();
}

//______________________________________________________________________________
void TGLHistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute event.
   if (fPlotType == kGLDefaultPlot) {
      if(fDefaultPainter.get())
         fDefaultPainter->ExecuteEvent(event, px, py);
   } else {
      const Int_t glContext = gPad->GetGLDevice();

      if (glContext == -1) {
         Error("ExecuteEvent", 
               "Attempt to use TGLHistPainter, while the current pad (gPad) does not support gl");
         return;
      } else
         fGLPainter->SetGLContext(glContext);

      if (event != kKeyPress) {
         py -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
         px -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
      }

      switch (event) {
      case kButton1Double:
         fGLPainter->ProcessEvent(event, px, py);
         break;
      case kButton1Down :
         fCamera.StartRotation(px, py);
         gGLManager->MarkForDirectCopy(glContext, kTRUE);
         break;
      case kButton1Motion :
         fGLPainter->InvalidateSelection();
         fCamera.RotateCamera(px, py);
         gGLManager->PaintSingleObject(fGLPainter.get());
         break;
      case kButton1Up:
         gGLManager->MarkForDirectCopy(glContext, kFALSE);
         break;
      case kButton2Up:
         gGLManager->MarkForDirectCopy(glContext, kFALSE);
         break;
      case kMouseMotion:
         gPad->SetCursor(kRotate);
         break;
      case 7://kButton2Down://+ shift modifier
         fGLPainter->StartPan(px, py);
         gGLManager->MarkForDirectCopy(glContext, kTRUE);
         break;
      case 8://kButton2Motion://+shift modifier
         gGLManager->PanObject(fGLPainter.get(), px, py);
         gGLManager->PaintSingleObject(fGLPainter.get());
         break;
      case kKeyPress:
      case 5:
      case 6:
         gGLManager->MarkForDirectCopy(glContext, kTRUE);
         if (event == 5 || py == kKey_J || py == kKey_j) {
            fCamera.ZoomIn();
            fGLPainter->InvalidateSelection();
            gGLManager->PaintSingleObject(fGLPainter.get());
         } else if (event == 6 || py == kKey_K || py == kKey_k) {
            fCamera.ZoomOut();
            fGLPainter->InvalidateSelection();
            gGLManager->PaintSingleObject(fGLPainter.get());
         } else if (py == kKey_p || py == kKey_P || py == kKey_S || py == kKey_s) {
            fGLPainter->ProcessEvent(event, px, py);
            gGLManager->PaintSingleObject(fGLPainter.get());
         }
         gGLManager->MarkForDirectCopy(glContext, kFALSE);
         break;
      }
   }
}

//______________________________________________________________________________
void TGLHistPainter::FitPanel()
{
   //Default implementation is OK.
   if (fDefaultPainter.get())
      fDefaultPainter->FitPanel();
}

//______________________________________________________________________________
TList *TGLHistPainter::GetContourList(Double_t contour)const
{
   // Get contour list.
   return fDefaultPainter.get() ? fDefaultPainter->GetContourList(contour) : 0;
}

//______________________________________________________________________________
char *TGLHistPainter::GetObjectInfo(Int_t px, Int_t py)const
{
   //Overrides TObject::GetObjectInfo.
   //Displays the histogram info (bin number, contents, integral up to bin
   //corresponding to cursor position px,py.
   static char *errMsg = "TGLHistPainter::GetObjectInfo: Error in a hist painter\n";
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->GetObjectInfo(px, py) 
                                   : errMsg;
   else
      return gGLManager->GetPlotInfo(fGLPainter.get(), px, py);
}

//______________________________________________________________________________
TList *TGLHistPainter::GetStack()const
{
   // Get stack.
   return fStack;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Int_t x, Int_t y)
{
   //Returns kTRUE if the cell ix, iy is inside one of the graphical cuts.
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->IsInside(x, y) : kFALSE;
   
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Double_t x, Double_t y)
{
   //Returns kTRUE if the cell x, y is inside one of the graphical cuts.
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->IsInside(x, y) : kFALSE;
   
   return kFALSE;
}

//______________________________________________________________________________
void TGLHistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
   // Paint statistics.
   if (fDefaultPainter.get()) 
      fDefaultPainter->PaintStat(dostat, fit);
}

//______________________________________________________________________________
void TGLHistPainter::ProcessMessage(const char *m, const TObject *o)
{
   // Process message.
   if (!std::strcmp(m, "SetF3"))
      fF3 = (TF3 *)o;//static_cast, followed by const_cast
   
   if (fDefaultPainter.get())
      fDefaultPainter->ProcessMessage(m, o);
}

//______________________________________________________________________________
void TGLHistPainter::SetHistogram(TH1 *h)
{
   // Set histogram.
   fHist = h;

   if (fDefaultPainter.get())
      fDefaultPainter->SetHistogram(h);
}

//______________________________________________________________________________
void TGLHistPainter::SetStack(TList *s)
{
   // Set stack.
   fStack = s;

   if (fDefaultPainter.get())
      fDefaultPainter->SetStack(s);
}

//______________________________________________________________________________
Int_t TGLHistPainter::MakeCuts(char *o)
{
   // Make cuts.
   if (fPlotType == kGLDefaultPlot && fDefaultPainter.get())
      return fDefaultPainter->MakeCuts(o);

   return 0;
}

struct TGLHistPainter::PlotOption_t {
   EGLPlotType  fPlotType;
   EGLCoordType fCoordType;
   Bool_t       fBackBox;
   Bool_t       fFrontBox;
   Bool_t       fLogX;
   Bool_t       fLogY;
   Bool_t       fLogZ;
};

//______________________________________________________________________________
void TGLHistPainter::Paint(Option_t *o)
{
   //Final-overrider for TObject::Paint.
   TString option(o);
   option.ToLower();

   const Ssiz_t glPos = option.Index("gl");
   if (glPos != kNPOS)
      option.Remove(glPos, 2);
   else {
      gPad->SetCopyGLDevice(kFALSE);
      if (fDefaultPainter.get())
         fDefaultPainter->Paint(o);//option.Data());
      return;
   }

   CreatePainter(ParsePaintOption(option), option);

   if (fPlotType == kGLDefaultPlot) {
      gPad->SetCopyGLDevice(kFALSE);

      if (fDefaultPainter.get())
         fDefaultPainter->Paint(option.Data());
   } else {
      Int_t glContext = gPad->GetGLDevice();

      if (glContext != -1) {
         gPad->SetCopyGLDevice(kTRUE);
         fGLPainter->SetGLContext(glContext);
         if (fGLPainter->InitGeometry())
            gGLManager->PaintSingleObject(fGLPainter.get());
      }
   }
}

//______________________________________________________________________________
TGLHistPainter::PlotOption_t 
TGLHistPainter::ParsePaintOption(const TString &option)const
{
   //In principle, we can have several conflicting options: "lego surf pol sph",
   //but only one will be selected, which one - depends on parsing order in this function.
   PlotOption_t parsedOption = {kGLDefaultPlot, kGLCartesian, kFALSE, kFALSE, gPad->GetLogx(), 
                                gPad->GetLogy(), gPad->GetLogz()};
   //Check coordinate system type.
   if (option.Index("pol") != kNPOS)
      parsedOption.fCoordType = kGLPolar;
   if (option.Index("cyl") != kNPOS)
      parsedOption.fCoordType = kGLCylindrical;
   if (option.Index("sph") != kNPOS)
      parsedOption.fCoordType = kGLSpherical;
   //Define plot type
   if (option.Index("lego") != kNPOS)
      fStack ? parsedOption.fPlotType = kGLStackPlot : parsedOption.fPlotType = kGLLegoPlot;
   if (option.Index("surf") != kNPOS)
      parsedOption.fPlotType = kGLSurfacePlot;
   if (option.Index("tf3") != kNPOS)
      parsedOption.fPlotType = kGLTF3Plot;
   if (option.Index("box") != kNPOS)
      parsedOption.fPlotType = kGLBoxPlot;

   return parsedOption;
}

//______________________________________________________________________________
void TGLHistPainter::CreatePainter(const PlotOption_t &option, const TString &addOption)
{
   // Create painter.
   if (option.fPlotType != fPlotType)
      fGLPainter.reset(0);

   if (option.fPlotType == kGLLegoPlot) {
      if (!fGLPainter.get())
         fGLPainter.reset(new TGLLegoPainter(fHist, &fCamera, &fCoord));
   } else if (option.fPlotType == kGLSurfacePlot) {
      if (!fGLPainter.get())
         fGLPainter.reset(new TGLSurfacePainter(fHist, &fCamera, &fCoord));
   } else if (option.fPlotType == kGLBoxPlot) {
      if (!fGLPainter.get())
         fGLPainter.reset(new TGLBoxPainter(fHist, &fCamera, &fCoord));
   } else if (option.fPlotType == kGLTF3Plot) {
      if (!fGLPainter.get())
         fGLPainter.reset(new TGLTF3Painter(fF3, fHist, &fCamera, &fCoord));
   }

   if (fGLPainter.get()) {
      fPlotType = option.fPlotType;
      fCoord.SetXLog(gPad->GetLogx());
      fCoord.SetYLog(gPad->GetLogy());
      fCoord.SetZLog(gPad->GetLogz());
      fCoord.SetCoordType(option.fCoordType);

      if (gPad->GetFrameFillColor() != kWhite)
         fGLPainter->SetFrameColor(gROOT->GetColor(gPad->GetFrameFillColor()));

      fGLPainter->SetPadColor(gROOT->GetColor(gPad->GetFillColor()));
      fGLPainter->AddOption(addOption);
   } else
      fPlotType = kGLDefaultPlot;
}

//______________________________________________________________________________
void TGLHistPainter::SetShowProjection(const char *, Int_t)
{
   // Set show projection.
}
