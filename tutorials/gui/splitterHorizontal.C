//
// Author: Ilka Antcheva   1/12/2006

// This macro gives an example of how to create a horizontal splitter
// To run it do either:
// .x splitterHorizontal.C
// .x splitterHorizontal.C++

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

void MyMainFrame::DoSave()
{
//------      TGMainFrame::SaveSource()       --------
  Printf("Save in progress...");
  SaveSource("","");
}

MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
  TGMainFrame(p, w, h)
{
   // Create horizontal splitter
   TGVerticalFrame *fVf = new TGVerticalFrame(this, 10, 10);

   TGHorizontalFrame *fH1 = new TGHorizontalFrame(fVf, 10, 50, kFixedHeight);
   TGHorizontalFrame *fH2 = new TGHorizontalFrame(fVf, 10, 10);
   TGCompositeFrame *fFtop = new TGCompositeFrame(fH1, 10, 10, kSunkenFrame);
   TGCompositeFrame *fFbottom = new TGCompositeFrame(fH2, 10, 10, kSunkenFrame);

   TGLabel *fLtop = new TGLabel(fFtop, "Top Frame");
   TGLabel *fLbottom = new TGLabel(fFbottom, "Bottom Frame");

   fFtop->AddFrame(fLtop, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
                                            3, 0, 0, 0));
   fFbottom->AddFrame(fLbottom, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
                                                  3, 0, 0, 0));

   fH1->AddFrame(fFtop, new TGLayoutHints(kLHintsTop | kLHintsExpandY |
                                          kLHintsExpandX, 0, 0, 1, 2));
   fH2->AddFrame(fFbottom, new TGLayoutHints(kLHintsTop | kLHintsExpandY |
                                             kLHintsExpandX, 0, 0, 1, 2));

   fH1->Resize(fFtop->GetDefaultWidth(), fH1->GetDefaultHeight()+20);
   fH2->Resize(fFbottom->GetDefaultWidth(), fH2->GetDefaultHeight()+20);
   fVf->AddFrame(fH1, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   TGHSplitter *hsplitter = new TGHSplitter(fVf,2,2);
   hsplitter->SetFrame(fH1, kTRUE);
   fVf->AddFrame(hsplitter, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   fVf->AddFrame(fH2, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // button frame
   TGVerticalFrame *hframe = new TGVerticalFrame(this, 10, 10);
   TGCompositeFrame *cframe2 = new TGCompositeFrame(hframe, 170, 50,
                                             kHorizontalFrame | kFixedWidth);
   TGTextButton *save = new TGTextButton(cframe2, "&Save");
   cframe2->AddFrame(save, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             3, 2, 2, 2));
   save->Connect("Clicked()", "MyMainFrame", this, "DoSave()");
   save->SetToolTipText("Click on the button to save the application as C++ macro");

   TGTextButton *exit = new TGTextButton(cframe2, "&Exit ","gApplication->Terminate(0)");
   cframe2->AddFrame(exit, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                             2, 0, 2, 2));
   hframe->AddFrame(cframe2, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 1));

   AddFrame(fVf, new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY));
   AddFrame(hframe, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 1));

   // What to clean up in dtor
   SetCleanup(kDeepCleanup);

   // Set a name to the main frame
   SetWindowName("Horizontal Splitter");
   SetWMSizeHints(300, 250, 600, 600, 0, 0);
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
}

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

void splitterHorizontal()
{
   // Popup the GUI...

   new MyMainFrame(gClient->GetRoot(), 300, 250);
}
