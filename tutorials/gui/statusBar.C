//
// Author: Ilka Antcheva   1/12/2006

// This macro gives an example of how to create a status bar 
// related to an embedded canvas that shows the info of the selected object 
// exactly as the status bar of any canvas window
// To run it do either:
// .x statusBar.C
// .x statusBar.C++

#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TFrame.h>
#include <TRootEmbeddedCanvas.h>
#include <TGStatusBar.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TRandom.h>
#include <TGraph.h>
#include <TAxis.h>


class MyMainFrame : public TGMainFrame {

private:
   TRootEmbeddedCanvas  *fEcan;
   TGStatusBar          *fStatusBar;
   
public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();
   void DoExit();
   void DoDraw();
   void SetStatusText(const char *txt, Int_t pi);
   void EventInfo(Int_t event, Int_t px, Int_t py, TObject *selected);
   
   ClassDef(MyMainFrame, 0)
};

void MyMainFrame::DoDraw()
{
   // Draw something in the canvas

   Printf("Slot DoDraw()");

   TCanvas *c1 = fEcan->GetCanvas();
   c1->SetFillColor(42);
   c1->SetGrid();
   const Int_t n = 20;
   Double_t x[n], y[n];
   for (Int_t i=0;i<n;i++) {
     x[i] = i*0.1;
     y[i] = 10*sin(x[i]+0.2);
     printf(" i %i %f %f \n",i,x[i],y[i]);
   }
   TGraph *gr = new TGraph(n,x,y);
   gr->SetLineColor(2);
   gr->SetLineWidth(4);
   gr->SetMarkerColor(4);
   gr->SetMarkerStyle(21);
   gr->SetTitle("a simple graph");
   gr->GetXaxis()->SetTitle("X title");
   gr->GetYaxis()->SetTitle("Y title");
   gr->Draw("ACP");
   
   // TCanvas::Update() draws the frame, after which it can be changed
   c1->Update();
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderSize(12);
   c1->Modified();
   c1->Update();
}

void MyMainFrame::DoExit()
{
   printf("Exit application...");
   gApplication->Terminate(0);
}

void MyMainFrame::SetStatusText(const char *txt, Int_t pi)
{
   // Set text in status bar.
   fStatusBar->SetText(txt,pi);
}

void MyMainFrame::EventInfo(Int_t event, Int_t px, Int_t py, TObject *selected)
{
//  Writes the event status in the status bar parts

   const char *text0, *text1, *text3;
   char text2[50];
   text0 = selected->GetTitle();
   SetStatusText(text0,0);
   text1 = selected->GetName();
   SetStatusText(text1,1);
   if (event == kKeyPress)
      sprintf(text2, "%c", (char) px);
   else
      sprintf(text2, "%d,%d", px, py);
   SetStatusText(text2,2);
   text3 = selected->GetObjectInfo(px,py);
   SetStatusText(text3,3);
}

MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
   TGMainFrame(p, w, h)
{
   // Create the embedded canvas
   fEcan = new TRootEmbeddedCanvas(0,this,500,400);
   Int_t wid = fEcan->GetCanvasWindowId();
   TCanvas *myc = new TCanvas("MyCanvas", 10,10,wid);
   fEcan->AdoptCanvas(myc);
   myc->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)","MyMainFrame",this, 
               "EventInfo(Int_t,Int_t,Int_t,TObject*)");

   AddFrame(fEcan, new TGLayoutHints(kLHintsTop | kLHintsLeft | 
                                     kLHintsExpandX  | kLHintsExpandY,0,0,1,1));
   // status bar
   Int_t parts[] = {45, 15, 10, 30};
   fStatusBar = new TGStatusBar(this, 50, 10, kVerticalFrame);
   fStatusBar->SetParts(parts, 4);
   fStatusBar->Draw3DCorner(kFALSE);
   AddFrame(fStatusBar, new TGLayoutHints(kLHintsExpandX, 0, 0, 10, 0));
   
   // Create a horizontal frame containing two buttons
   TGHorizontalFrame *hframe = new TGHorizontalFrame(this, 200, 40);
  
   TGTextButton *draw = new TGTextButton(hframe, "&Draw");
   draw->Connect("Clicked()", "MyMainFrame", this, "DoDraw()");
   hframe->AddFrame(draw, new TGLayoutHints(kLHintsCenterX, 5, 5, 3, 4));
   TGTextButton *exit = new TGTextButton(hframe, "&Exit ");
   exit->Connect("Pressed()", "MyMainFrame", this, "DoExit()");
   hframe->AddFrame(exit, new TGLayoutHints(kLHintsCenterX, 5, 5, 3, 4));

   AddFrame(hframe, new TGLayoutHints(kLHintsCenterX, 2, 2, 2, 2));

   // Set a name to the main frame   
   SetWindowName("Embedded Canvas Status Info");
   MapSubwindows();

   // Initialize the layout algorithm via Resize()
   Resize(GetDefaultSize());

   // Map main frame
   MapWindow();
}


MyMainFrame::~MyMainFrame()
{
   // Clean up main frame...
   Cleanup();
   delete fEcan;
}


void statusBar()
{
   // Popup the GUI...
   new MyMainFrame(gClient->GetRoot(), 200, 200);
}
