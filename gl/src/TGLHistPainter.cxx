#include <cstring>

#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "TF3.h"

#include "TGLSurfacePainter.h"
#include "TGLHistPainter.h"
#include "TGLLegoPainter.h"
#include "TGLBoxPainter.h"
#include "TGLTF3Painter.h"
#include "TGLParametric.h"

ClassImp(TGLHistPainter)

/*
   Some TGLHistPainter's functions are useless - I have to define them
   to override pure-virtual functions from TVirtualHistPainter:
      GetContourList
      IsInside
      MakeCuts
      SetShowProjection
*/

//______________________________________________________________________________
TGLHistPainter::TGLHistPainter(TH1 *hist)
                   : fDefaultPainter(TVirtualHistPainter::HistPainter(hist)),
                     fEq(0),
                     fHist(hist),
                     fF3(0),
                     fStack(0),
                     fPlotType(kGLDefaultPlot)//THistPainter
{
   //ROOT does not use exceptions, so, if default painter's creation failed,
   //fDefaultPainter is 0. In each function, which use it, I have to check the pointer first.
}

//______________________________________________________________________________
TGLHistPainter::TGLHistPainter(TGLParametricEquation *equation)
                   : fEq(equation),
                     fHist(0),
                     fF3(0),
                     fStack(0),
                     fPlotType(kGLParametricPlot)//THistPainter
{
   //This ctor creates gl-parametric plot's painter.
   fGLPainter.reset(new TGLParametricPlot(equation, &fCamera));
}

//______________________________________________________________________________
Int_t TGLHistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
   //Selects plot or axis. 
   //9999 is the magic number, ROOT's classes use in DistancetoPrimitive.
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->DistancetoPrimitive(px, py) : 9999;
   else {
      //Adjust px and py - canvas can have several pads inside, so we need to convert
      //the from canvas' system into pad's.
      py -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
      px -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
      //One hist can be appended to several pads,
      //the current pad should have valid OpenGL context.
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
   //Default implementation is OK 
   //This function is called from a context menu 
   //after right click on a plot's area. Opens window
   //("panel") with several controls.
   if (fDefaultPainter.get())
      fDefaultPainter->DrawPanel();
}

//______________________________________________________________________________
void TGLHistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   //Execute event.
   //Events are: mouse events in a plot's area, 
   //key presses (while mouse cursor is in plot's area).
   //"Event execution" means one of the following actions:
   //1. Rotation.
   //2. Panning.
   //3. Zoom changing.
   //4. Moving dynamic profile.
   //5. Plot specific events - for example, 's' or 'S' key press for TF3.
   if (fPlotType == kGLDefaultPlot) {
      if(fDefaultPainter.get())
         fDefaultPainter->ExecuteEvent(event, px, py);
   } else {
      //One hist can be appended to several pads,
      //the current pad should have valid OpenGL context.
      const Int_t glContext = gPad->GetGLDevice();

      if (glContext == -1) {
         Error("ExecuteEvent", 
               "Attempt to use TGLHistPainter, while the current pad (gPad) does not support gl");
         return;
      } else
         fGLPainter->SetGLContext(glContext);

      if (event != kKeyPress) {
         //Adjust px and py - canvas can have several pads inside, so we need to convert
         //the from canvas' system into pad's. If it was a key press event,
         //px and py ARE NOT coordinates.
         py -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
         px -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
      }

      switch (event) {
      case kButton1Double:
         //Left double click removes dynamic sections, user created (if plot type supports sections).
         fGLPainter->ProcessEvent(event, px, py);
         break;
      case kButton1Down:
         //Left mouse down in a plot area starts rotation.
         if (!fGLPainter->CutAxisSelected())
            fCamera.StartRotation(px, py);
         else
            fGLPainter->StartPan(px, py);
         //During rotation, usual TCanvas/TPad machinery (CopyPixmap/Flush/UpdateWindow/etc.)
         //is skipped - I use "bit blasting" functions to copy picture directly onto window.
         gGLManager->MarkForDirectCopy(glContext, kTRUE);
         break;
      case kButton1Motion:
         //Rotation invalidates "selection buffer" 
         // - (color-to-object map, previously read from gl-buffer).
         fGLPainter->InvalidateSelection();
         if (fGLPainter->CutAxisSelected())
            gGLManager->PanObject(fGLPainter.get(), px, py);
         else
            fCamera.RotateCamera(px, py);
         //Draw modified scene onto canvas' window.
         gGLManager->PaintSingleObject(fGLPainter.get());
         break;
      case kButton1Up:
      case kButton2Up:
         gGLManager->MarkForDirectCopy(glContext, kFALSE);
         break;
      case kMouseMotion:
         gPad->SetCursor(kRotate);
         break;
      case 7://kButton1Down + shift modifier
         //The current version of ROOT does not
         //have enumerators for button events + key modifiers,
         //so I use hardcoded literals. :(
         //With left mouse button down and shift pressed
         //we can move plot as the whole or move
         //plot's parts - dynamic sections.
         fGLPainter->StartPan(px, py);
         gGLManager->MarkForDirectCopy(glContext, kTRUE);
         break;
      case 8://kButton1Motion + shift modifier
         gGLManager->PanObject(fGLPainter.get(), px, py);
         gGLManager->PaintSingleObject(fGLPainter.get());
         break;
      case kKeyPress:
      case 5:
      case 6:
         //5, 6 are mouse wheel events (see comment about literals above).
         //'p'/'P' - specific events processed by TGLSurfacePainter,
         //'s'/'S' - specific events processed by TGLTF3Painter,
         //'c'/'C' - turn on/off box cut.
         gGLManager->MarkForDirectCopy(glContext, kTRUE);
         if (event == 5 || py == kKey_J || py == kKey_j) {
            fCamera.ZoomIn();
            fGLPainter->InvalidateSelection();
            gGLManager->PaintSingleObject(fGLPainter.get());
         } else if (event == 6 || py == kKey_K || py == kKey_k) {
            fCamera.ZoomOut();
            fGLPainter->InvalidateSelection();
            gGLManager->PaintSingleObject(fGLPainter.get());
         } else if (py == kKey_p || py == kKey_P || py == kKey_S || py == kKey_s
                    || py == kKey_c || py == kKey_C || py == kKey_x || py == kKey_X
                    || py == kKey_y || py == kKey_Y || py == kKey_z || py == kKey_Z
                    || py == kKey_w || py == kKey_W || py == kKey_l || py == kKey_L) 
         {
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
   //This function is called from a context menu 
   //after right click on a plot's area. Opens window
   //("panel") with several controls.
   if (fDefaultPainter.get())
      fDefaultPainter->FitPanel();
}

//______________________________________________________________________________
TList *TGLHistPainter::GetContourList(Double_t contour)const
{
   //Get contour list.
   //I do not use this function. Contours are implemented in
   //a completely different way by gl-painters.
   return fDefaultPainter.get() ? fDefaultPainter->GetContourList(contour) : 0;
}

//______________________________________________________________________________
char *TGLHistPainter::GetObjectInfo(Int_t px, Int_t py)const
{
   //Overrides TObject::GetObjectInfo.
   //For lego info is: bin numbers (i, j), bin content.
   //For TF2 info is: x,y,z 3d surface-point for 2d screen-point under cursor
   //(this can work incorrectly now, because of wrong code in TF2).
   //For TF3 no info now.
   //For box info is: bin numbers (i, j, k), bin content.
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
   //I do not use this function anywhere, this is a "default implementation".
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->IsInside(x, y) : kFALSE;
   
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Double_t x, Double_t y)
{
   //Returns kTRUE if the cell x, y is inside one of the graphical cuts.
   //I do not use this function anywhere, this is a "default implementation".
   if (fPlotType == kGLDefaultPlot)
      return fDefaultPainter.get() ? fDefaultPainter->IsInside(x, y) : kFALSE;
   
   return kFALSE;
}

//______________________________________________________________________________
void TGLHistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
   //Paint statistics.
   //This does not work on windows.
   if (fDefaultPainter.get()) 
      fDefaultPainter->PaintStat(dostat, fit);
}

//______________________________________________________________________________
void TGLHistPainter::ProcessMessage(const char *m, const TObject *o)
{
   // Process message.
   if (!std::strcmp(m, "SetF3"))
      fF3 = (TF3 *)o;
   
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
   else if (fPlotType != kGLParametricPlot){
      gPad->SetCopyGLDevice(kFALSE);
      if (fDefaultPainter.get())
         fDefaultPainter->Paint(o);//option.Data());
      return;
   }

   if (fPlotType != kGLParametricPlot)
      CreatePainter(ParsePaintOption(option), option);

   if (fPlotType == kGLDefaultPlot) {
      //In case of default plot pad 
      //should not copy gl-buffer (it will be simply black)
      gPad->SetCopyGLDevice(kFALSE);

      if (fDefaultPainter.get())
         fDefaultPainter->Paint(option.Data());
   } else {
      Int_t glContext = gPad->GetGLDevice();

      if (glContext != -1) {
         //With gl-plot, pad should copy
         //gl-buffer into the final pad/canvas pixmap/DIB.
         gPad->SetCopyGLDevice(kTRUE);
         fGLPainter->SetGLContext(glContext);
         if (gPad->GetFrameFillColor() != kWhite)
            fGLPainter->SetFrameColor(gROOT->GetColor(gPad->GetFrameFillColor()));  
         fGLPainter->SetPadColor(gROOT->GetColor(gPad->GetFillColor()));
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
      fGLPainter->AddOption(addOption);
   } else
      fPlotType = kGLDefaultPlot;
}

//______________________________________________________________________________
void TGLHistPainter::SetShowProjection(const char *, Int_t)
{
   // Set show projection.
}
