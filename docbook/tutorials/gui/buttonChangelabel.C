//
// Author: Ilka Antcheva   1/12/2006

// This macro gives an example for changing text button labels anytime
// the Start or Pause buttons are clicked.
// To run it do either:
// .x buttonChangelabel.C
// .x buttonChangelabel.C++

#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>

class MyMainFrame : public TGMainFrame {

private:
   TGCompositeFrame *fCframe;
   TGTextButton     *fStart, *fPause, *fExit;
   Bool_t            start, pause;

public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();
   // slots
   void ChangeStartLabel();
   void ChangePauseLabel();

   ClassDef(MyMainFrame, 0)
};

void MyMainFrame::ChangeStartLabel()
{
  // Slot connected to the Clicked() signal. 
  // It will toggle labels "Start" and "Stop".
  
  fStart->SetState(kButtonDown);
  if (!start) {
     fStart->SetText("&Stop");
     start = kTRUE;
  } else {
     fStart->SetText("&Start");
     start = kFALSE;
  }
  fStart->SetState(kButtonUp);
}

void MyMainFrame::ChangePauseLabel()
{
  // Slot connected to the Clicked() signal. 
  // It will toggle labels "Resume" and "Pause".
  
  fPause->SetState(kButtonDown);
  if (!pause) {
     fPause->SetText("&Resume");
     pause = kTRUE;
  } else {
     fPause->SetText("&Pause");
     pause = kFALSE;
  }
  fPause->SetState(kButtonUp);
}

MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
  TGMainFrame(p, w, h)
{
   // Create a horizontal frame containing buttons
   fCframe = new TGCompositeFrame(this, 170, 20, kHorizontalFrame|kFixedWidth);
   
   fStart = new TGTextButton(fCframe, "&Start");
   fStart->Connect("Clicked()", "MyMainFrame", this, "ChangeStartLabel()");
   fCframe->AddFrame(fStart, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 
                                               3, 2, 2, 2));
   fStart->SetToolTipText("Click to toggle the button label (Start/Stop)");
   start = kFALSE;
   
   fPause = new TGTextButton(fCframe, "&Pause");
   fPause->Connect("Clicked()", "MyMainFrame", this, "ChangePauseLabel()");
   fPause->SetToolTipText("Click to toggle the button label (Pause/Resume)");
   fCframe->AddFrame(fPause, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                               3, 2, 2, 2));
   pause = kFALSE;
   
   AddFrame(fCframe, new TGLayoutHints(kLHintsCenterX, 2, 2, 5, 1));

   fExit = new TGTextButton(this, "&Exit ","gApplication->Terminate(0)");
   AddFrame(fExit, new TGLayoutHints(kLHintsTop | kLHintsExpandX,5,5,2,2));
   
   SetWindowName("Change Labels");
   
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
}


MyMainFrame::~MyMainFrame()
{
   // Clean up all widgets, frames and layouthints that were used
   fCframe->Cleanup();
   Cleanup();
}


void buttonChangelabel()
{
   // Popup the GUI...
   new MyMainFrame(gClient->GetRoot(), 350, 80);
}
