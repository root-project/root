//
// Author: Ilka Antcheva   1/12/2006

// This macro gives an example of how to create different kind of labels
// and the possibility to enable/disable them.
// To run it do either:
// .x labels.C
// .x labels.C++

#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGLabel.h>
#include <TGResourcePool.h>

class MyMainFrame : public TGMainFrame {

private:
   TGLabel       *fLbl1, *fLbl2, *fLbl3;
   TGTextButton  *fToggle;
public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();
   void DoExit();
   void DoSwitch();

   ClassDef(MyMainFrame, 0)
};

void MyMainFrame::DoSwitch()
{
   if (fLbl1->IsDisabled()) {
   printf("Enabled labels\n");
      fLbl1->Enable();
      fLbl2->Enable();
      fLbl3->Enable();
   } else {
   printf("Disabled labels\n");
      fLbl1->Disable();
      fLbl2->Disable();
      fLbl3->Disable();
   }
}

void MyMainFrame::DoExit()
{
   Printf("Slot DoExit()");
   gApplication->Terminate(0);
}

MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
  TGMainFrame(p, w, h)
{

   // label + horizontal line
    TGGC *fTextGC;
    const TGFont *font = gClient->GetFont("-*-times-bold-r-*-*-18-*-*-*-*-*-*-*");
    if (!font)
       font = gClient->GetResourcePool()->GetDefaultFont();
    FontStruct_t labelfont = font->GetFontStruct();
    GCValues_t   gval;
    gval.fMask = kGCBackground | kGCFont | kGCForeground;
    gval.fFont = font->GetFontHandle();
    gClient->GetColorByName("red", gval.fBackground);
    fTextGC = gClient->GetGC(&gval, kTRUE);
   
   
   ULong_t bcolor, ycolor, rcolor;
   gClient->GetColorByName("yellow", ycolor);
   gClient->GetColorByName("red", rcolor);
   gClient->GetColorByName("blue", bcolor);

   // Create a main frame
   fLbl1 = new TGLabel(this, "HasBackground", fTextGC->GetGC(), labelfont, kChildFrame, bcolor);
   AddFrame(fLbl1, new TGLayoutHints(kLHintsCenterX, 5, 5, 3, 4));
   fLbl1->SetTextColor(ycolor);


   fLbl2 = new TGLabel(this, "Has Own Font + FColor", fTextGC->GetGC(), labelfont);
   AddFrame(fLbl2,  new TGLayoutHints(kLHintsCenterX, 5, 5, 3, 4));
   fLbl2->SetTextColor(rcolor);

   fLbl3 = new TGLabel(this, "Normal Label");
   AddFrame(fLbl3,  new TGLayoutHints(kLHintsCenterX, 5, 5, 3, 4));

   // Create a horizontal frame containing two buttons
   TGTextButton *toggle = new TGTextButton(this, "&Toggle Labels");
   toggle->Connect("Clicked()", "MyMainFrame", this, "DoSwitch()");
   toggle->SetToolTipText("Click on the button to toggle label's state (enable/disable)");
   AddFrame(toggle, new TGLayoutHints(kLHintsExpandX, 5, 5, 3, 4));
   TGTextButton *exit = new TGTextButton(this, "&Exit ");
   exit->Connect("Pressed()", "MyMainFrame", this, "DoExit()");
   AddFrame(exit, new TGLayoutHints(kLHintsExpandX, 5, 5, 3, 4));

   // Set a name to the main frame
   SetWindowName("Labels");
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
}

void guilabels()
{
   // Popup the GUI...
   new MyMainFrame(0, 200, 200);
}
