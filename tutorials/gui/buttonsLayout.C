//
// Author: Ilka Antcheva   1/12/2006

// This macro gives an example of different buttons' layout.
// To run it do either:
// .x buttonsLayout.C
// .x buttonsLayout.C++

#include <TGClient.h>
#include <TGButton.h>

class MyMainFrame : public TGMainFrame {

private:
   TGTextButton *test, *draw, *help, *ok, *cancel, *exit;

public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();

   ClassDef(MyMainFrame, 0)
};


MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
  TGMainFrame(p, w, h)
{
   // Create a container frames containing buttons

   // one button is resized up to the parent width.
   // Note! this width should be fixed!
   TGVerticalFrame *hframe1 = new TGVerticalFrame(this, 170, 50, kFixedWidth);
   test = new TGTextButton(hframe1, "&Test ");
   // to take whole space we need to use kLHintsExpandX layout hints
   hframe1->AddFrame(test, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             2, 0, 2, 2));
   AddFrame(hframe1, new TGLayoutHints(kLHintsCenterX, 2, 2, 5, 1));

   // two buttons are resized up to the parent width.
   // Note! this width should be fixed!
   TGCompositeFrame *cframe1 = new TGCompositeFrame(this, 170, 20,
                                             kHorizontalFrame | kFixedWidth);
   draw = new TGTextButton(cframe1, "&Draw");
   // to share whole parent space we need to use kLHintsExpandX layout hints
   cframe1->AddFrame(draw, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             2, 2, 2, 2));

   // button background will be set to yellow
   ULong_t yellow;
   gClient->GetColorByName("yellow", yellow);
   help = new TGTextButton(cframe1, "&Help");
   help->ChangeBackground(yellow);
   cframe1->AddFrame(help, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             2, 2, 2, 2));
   AddFrame(cframe1, new TGLayoutHints(kLHintsCenterX, 2, 2, 5, 1));

   // three buttons are resized up to the parent width.
   // Note! this width should be fixed!
   TGCompositeFrame *cframe2 = new TGCompositeFrame(this, 170, 20,
                                             kHorizontalFrame | kFixedWidth);
   ok = new TGTextButton(cframe2, "OK");
   // to share whole parent space we need to use kLHintsExpandX layout hints
   cframe2->AddFrame(ok, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                           3, 2, 2, 2));

   cancel = new TGTextButton(cframe2, "Cancel ");
   cframe2->AddFrame(cancel, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                               3, 2, 2, 2));

   exit = new TGTextButton(cframe2, "&Exit ","gApplication->Terminate(0)");
   cframe2->AddFrame(exit, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             2, 0, 2, 2));

   AddFrame(cframe2, new TGLayoutHints(kLHintsCenterX, 2, 2, 5, 1));

   SetWindowName("Buttons' Layout");

   // gives min/max window size + a step of x,y incrementing
   // between the given sizes
   SetWMSizeHints(200, 80, 320, 320, 1, 1);
   MapSubwindows();
   // important for layout algorithm
   Resize(GetDefaultSize());
   MapWindow();
}


MyMainFrame::~MyMainFrame()
{
   // Clean up all widgets, frames and layouthints that were used
   Cleanup();
}

void buttonsLayout()
{
   // Popup the GUI...
   new MyMainFrame(gClient->GetRoot(), 350, 80);
}
