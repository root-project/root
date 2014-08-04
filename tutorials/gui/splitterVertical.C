//
// Author: Ilka Antcheva   1/12/2006

// This macro gives an example of how to create a vertical splitter
// To run it do either:
// .x splitterVertical.C
// .x splitterVertical.C++

#include <TGClient.h>
#include <TGButton.h>
#include <TGLabel.h>
#include <TGFrame.h>
#include <TGLayout.h>
#include <TGSplitter.h>


class MyMainFrame : public TGMainFrame {

public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();
   void     DoSave();
   void     CloseWindow();

   ClassDef(MyMainFrame, 0)
};

//______________________________________________________________________________
void MyMainFrame::DoSave()
{
  Printf("Save in progress...");
  SaveSource("","");
}

//______________________________________________________________________________
MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
  TGMainFrame(p, w, h)
{
   // Create vertical splitter

   TGHorizontalFrame *fHf = new TGHorizontalFrame(this, 50, 50);

   TGVerticalFrame *fV1 = new TGVerticalFrame(fHf, 10, 10, kFixedWidth);
   TGVerticalFrame *fV2 = new TGVerticalFrame(fHf, 10, 10);
   TGCompositeFrame *fFleft = new TGCompositeFrame(fV1, 10, 10, kSunkenFrame);
   TGCompositeFrame *fFright = new TGCompositeFrame(fV2, 10, 10, kSunkenFrame);

   TGLabel *fLleft = new TGLabel(fFleft, "Left Frame");
   TGLabel *fLright = new TGLabel(fFright, "Right Frame");

   fFleft->AddFrame(fLleft, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
                                              3, 0, 0, 0));
   fFright->AddFrame(fLright, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
                                                3, 0, 0, 0));

   fV1->AddFrame(fFleft, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
                                           0, 0, 5, 10));
   fV2->AddFrame(fFright, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
                                            0, 0, 5, 10));

   fV1->Resize(fFleft->GetDefaultWidth()+20, fV1->GetDefaultHeight());
   fV2->Resize(fFright->GetDefaultWidth(), fV1->GetDefaultHeight());
   fHf->AddFrame(fV1, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   TGVSplitter *splitter = new TGVSplitter(fHf,2,2);
   splitter->SetFrame(fV1, kTRUE);
   fHf->AddFrame(splitter, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   fHf->AddFrame(fV2, new TGLayoutHints(kLHintsRight | kLHintsExpandX |
                                        kLHintsExpandY));
   AddFrame(fHf, new TGLayoutHints(kLHintsRight | kLHintsExpandX |
                                   kLHintsExpandY));

   // button frame
   TGVerticalFrame *vframe = new TGVerticalFrame(this, 10, 10);
   TGCompositeFrame *cframe2 = new TGCompositeFrame(vframe, 170, 20,
                                             kHorizontalFrame | kFixedWidth);
   TGTextButton *save = new TGTextButton(cframe2, "&Save");
   cframe2->AddFrame(save, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             3, 2, 2, 2));
   save->Connect("Clicked()", "MyMainFrame", this, "DoSave()");
   save->SetToolTipText("Click on the button to save the application as C++ macro");

   TGTextButton *exit = new TGTextButton(cframe2, "&Exit ","gApplication->Terminate(0)");
   cframe2->AddFrame(exit, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             2, 0, 2, 2));
   vframe->AddFrame(cframe2, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 1));
   AddFrame(vframe, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 1));

   // What to clean up in destructor
   SetCleanup(kDeepCleanup);

   // Set a name to the main frame
   SetWindowName("Vertical Splitter");
   SetWMSizeHints(350, 200, 600, 400, 0, 0);
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
}


//______________________________________________________________________________
MyMainFrame::~MyMainFrame()
{
   // Clean up all widgets, frames and layouthints that were used
   Cleanup();
}

//______________________________________________________________________________
void MyMainFrame::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

void splitterVertical()
{
   // Popup the GUI...
   new MyMainFrame(gClient->GetRoot(), 350, 200);
}
