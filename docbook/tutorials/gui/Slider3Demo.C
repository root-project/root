
// Simple macro showing capabilities of triple slider
//Authors: Bertrand Bellenot, Ilka Antcheva
   
#include "TGButton.h"
#include "TRootEmbeddedCanvas.h"
#include "TGLayout.h"
#include "TF1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TGTextEntry.h"
#include "TGTripleSlider.h"

enum ETestCommandIdentifiers {
   HId1,
   HId2,
   HId3,
   HCId1,
   HCId2,

   HSId1
};

class TTripleSliderDemo : public TGMainFrame {

private:
   TRootEmbeddedCanvas *fCanvas;
   TGLayoutHints       *fLcan;
   TF1                 *fFitFcn;
   TGHorizontalFrame   *fHframe0, *fHframe1, *fHframe2;
   TGLayoutHints       *fBly, *fBfly1, *fBfly2, *fBfly3;
   TGTripleHSlider     *fHslider1;
   TGTextEntry         *fTeh1, *fTeh2, *fTeh3;
   TGTextBuffer        *fTbh1, *fTbh2, *fTbh3;
   TGCheckButton       *fCheck1, *fCheck2;

public:
   TTripleSliderDemo();
   virtual ~TTripleSliderDemo();

   void CloseWindow();
   void DoText(const char *text);
   void DoSlider();
   void HandleButtons();

   ClassDef(TTripleSliderDemo, 0)
};

//______________________________________________________________________________
TTripleSliderDemo::TTripleSliderDemo() : TGMainFrame(gClient->GetRoot(), 100, 100)
{

   char buf[32];
   SetCleanup(kDeepCleanup);
   // Create an embedded canvas and add to the main frame, centered in x and y
   // and with 30 pixel margins all around
   fCanvas = new TRootEmbeddedCanvas("Canvas", this, 600, 400);
   fLcan = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 10, 10, 10, 10);
   AddFrame(fCanvas, fLcan);
   fCanvas->GetCanvas()->SetFillColor(33);
   fCanvas->GetCanvas()->SetFrameFillColor(41);
   fCanvas->GetCanvas()->SetBorderMode(0);
   fCanvas->GetCanvas()->SetGrid();
   fCanvas->GetCanvas()->SetLogy();

   fHframe0 = new TGHorizontalFrame(this, 0, 0, 0);

   fCheck1 = new TGCheckButton(fHframe0, "&Constrained", HCId1);
   fCheck2 = new TGCheckButton(fHframe0, "&Relative", HCId2);
   fCheck1->SetState(kButtonUp);
   fCheck2->SetState(kButtonUp);
   fCheck1->SetToolTipText("Pointer position constrained to slider sides");
   fCheck2->SetToolTipText("Pointer position relative to slider position");

   fHframe0->Resize(200, 50);

   fHframe1 = new TGHorizontalFrame(this, 0, 0, 0);

   fHslider1 = new TGTripleHSlider(fHframe1, 190, kDoubleScaleBoth, HSId1,
                                   kHorizontalFrame,
                                   GetDefaultFrameBackground(),
                                   kFALSE, kFALSE, kFALSE, kFALSE);
   fHslider1->Connect("PointerPositionChanged()", "TTripleSliderDemo", 
                      this, "DoSlider()");
   fHslider1->Connect("PositionChanged()", "TTripleSliderDemo", 
                      this, "DoSlider()");
   fHslider1->SetRange(0.05,5.0);

   fHframe1->Resize(200, 25);

   fHframe2 = new TGHorizontalFrame(this, 0, 0, 0);

   fTeh1 = new TGTextEntry(fHframe2, fTbh1 = new TGTextBuffer(5), HId1);
   fTeh2 = new TGTextEntry(fHframe2, fTbh2 = new TGTextBuffer(5), HId2);
   fTeh3 = new TGTextEntry(fHframe2, fTbh3 = new TGTextBuffer(5), HId3);

   fTeh1->SetToolTipText("Minimum (left) Value of Slider");
   fTeh2->SetToolTipText("Pointer Position Value");
   fTeh3->SetToolTipText("Maximum (right) Value of Slider");

   fTbh1->AddText(0, "0.0");
   fTbh2->AddText(0, "0.0");
   fTbh3->AddText(0, "0.0");

   fTeh1->Connect("TextChanged(char*)", "TTripleSliderDemo", this,
                  "DoText(char*)");
   fTeh2->Connect("TextChanged(char*)", "TTripleSliderDemo", this,
                  "DoText(char*)");
   fTeh3->Connect("TextChanged(char*)", "TTripleSliderDemo", this,
                  "DoText(char*)");

   fCheck1->Connect("Clicked()", "TTripleSliderDemo", this,
                    "HandleButtons()");
   fCheck2->Connect("Clicked()", "TTripleSliderDemo", this,
                    "HandleButtons()");

   fHframe2->Resize(100, 25);

   //--- layout for buttons: top align, equally expand horizontally
   fBly = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 5, 5, 5, 5);

   //--- layout for the frame: place at bottom, right aligned
   fBfly1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX, 5, 5, 5, 5);
   fBfly2 = new TGLayoutHints(kLHintsTop | kLHintsLeft,    5, 5, 5, 5);
   fBfly3 = new TGLayoutHints(kLHintsTop | kLHintsRight,   5, 5, 5, 5);

   fHframe0->AddFrame(fCheck1, fBfly2);
   fHframe0->AddFrame(fCheck2, fBfly2);
   fHframe1->AddFrame(fHslider1, fBly);
   fHframe2->AddFrame(fTeh1, fBfly2);
   fHframe2->AddFrame(fTeh2, fBfly1);
   fHframe2->AddFrame(fTeh3, fBfly3);

   AddFrame(fHframe0, fBly);
   AddFrame(fHframe1, fBly);
   AddFrame(fHframe2, fBly);

   // Set main frame name, map sub windows (buttons), initialize layout
   // algorithm via Resize() and map main frame
   SetWindowName("Triple Slider Demo");
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   fFitFcn = new TF1("fFitFcn", "TMath::LogNormal(x, [0], [1], [2])", 0, 5);
   fFitFcn->SetRange(0.0, 2.5);
   fFitFcn->SetParameters(1.0, 0, 1);
   fFitFcn->SetMinimum(1.0e-3);
   fFitFcn->SetMaximum(10.0);
   fFitFcn->SetLineColor(kRed);
   fFitFcn->SetLineWidth(1);
   fFitFcn->Draw();

   fHslider1->SetPosition(0.05,2.5);
   fHslider1->SetPointerPosition(1.0);

   sprintf(buf, "%.3f", fHslider1->GetMinPosition());
   fTbh1->Clear();
   fTbh1->AddText(0, buf);
   sprintf(buf, "%.3f", fHslider1->GetPointerPosition());
   fTbh2->Clear();
   fTbh2->AddText(0, buf);
   sprintf(buf, "%.3f", fHslider1->GetMaxPosition());
   fTbh3->Clear();
   fTbh3->AddText(0, buf);
}

//______________________________________________________________________________
TTripleSliderDemo::~TTripleSliderDemo()
{
   // Clean up

   Cleanup();
}

//______________________________________________________________________________
void TTripleSliderDemo::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

//______________________________________________________________________________
void TTripleSliderDemo::DoText(const char * /*text*/)
{
   // Handle text entry widgets.

   TGTextEntry *te = (TGTextEntry *) gTQSender;
   Int_t id = te->WidgetId();

   switch (id) {
      case HId1:
         fHslider1->SetPosition(atof(fTbh1->GetString()),
                                fHslider1->GetMaxPosition());
         break;
      case HId2:
         fHslider1->SetPointerPosition(atof(fTbh2->GetString()));
         break;
      case HId3:
         fHslider1->SetPosition(fHslider1->GetMinPosition(),
                                atof(fTbh1->GetString()));
         break;
      default:
         break;
   }
   fFitFcn->SetParameters(fHslider1->GetPointerPosition(), 0, 1);
   fFitFcn->SetRange(fHslider1->GetMinPosition()-0.05,
                     fHslider1->GetMaxPosition());
   fFitFcn->Draw();
   fCanvas->GetCanvas()->Modified();
   fCanvas->GetCanvas()->Update();
}

//______________________________________________________________________________
void TTripleSliderDemo::DoSlider()
{
   // Handle slider widgets.

   char buf[32];

   sprintf(buf, "%.3f", fHslider1->GetMinPosition());
   fTbh1->Clear();
   fTbh1->AddText(0, buf);
   fTeh1->SetCursorPosition(fTeh1->GetCursorPosition());
   fTeh1->Deselect();
   gClient->NeedRedraw(fTeh1);

   sprintf(buf, "%.3f", fHslider1->GetPointerPosition());
   fTbh2->Clear();
   fTbh2->AddText(0, buf);
   fTeh2->SetCursorPosition(fTeh2->GetCursorPosition());
   fTeh2->Deselect();
   gClient->NeedRedraw(fTeh2);

   sprintf(buf, "%.3f", fHslider1->GetMaxPosition());
   fTbh3->Clear();
   fTbh3->AddText(0, buf);
   fTeh3->SetCursorPosition(fTeh3->GetCursorPosition());
   fTeh3->Deselect();
   gClient->NeedRedraw(fTeh3);

   fFitFcn->SetParameters(fHslider1->GetPointerPosition(), 0, 1);
   fFitFcn->SetRange(fHslider1->GetMinPosition()-0.05,
                     fHslider1->GetMaxPosition());
   fFitFcn->Draw();
   fCanvas->GetCanvas()->Modified();
   fCanvas->GetCanvas()->Update();
}

//______________________________________________________________________________
void TTripleSliderDemo::HandleButtons()
{
   // Handle different buttons.

   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();

   switch (id) {
      case HCId1:
         fHslider1->SetConstrained(fCheck1->GetState());
         break;
      case HCId2:
         fHslider1->SetRelative(fCheck2->GetState());
         break;
      default:
         break;
   }
}


void Slider3Demo()
{
   new TTripleSliderDemo();
}

