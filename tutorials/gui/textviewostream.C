// Author: Bertrand Bellenot  06/01/2015
//
// This macro gives an example of how to use the TGTextViewostream widget.
// Simply type a command in the "Command" text entry, then the output is redirected to 
// theTGTextViewostream
//
// To run it do either:
// .x textviewostream.C
// .x textviewostream.C++

#include "TGButton.h"
#include "TGButtonGroup.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGTextViewStream.h"
#include "TApplication.h"
#include "TGFrame.h"
#include "TSystem.h"

////////////////////////////////////////////////////////////////////////////////
class TextViewMainFrame : public TGMainFrame
{
protected:
   TGTextButton      *fReset, *fExit;
   TGTextViewostream *fTextView;
   TGVerticalFrame   *fContents;
   TGHorizontalFrame *fButtons, *fCommandFrame;
   TGTextEntry       *fCommand;

public:
   TextViewMainFrame();
   virtual ~TextViewMainFrame() {}
   void Reset();
   void HandleReturn();

   ClassDef(TextViewMainFrame, 0)
};


//______________________________________________________________________________
TextViewMainFrame::TextViewMainFrame() : TGMainFrame(gClient->GetRoot())
{
   // Main test window.

   SetCleanup(kDeepCleanup);

   // Contents
   fContents = new TGVerticalFrame(this);
   fButtons = new TGHorizontalFrame(fContents);

   // TextView
   fTextView = new TGTextViewostream(fContents, 500, 300);
   fContents->AddFrame(fTextView, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 5, 5, 5, 0));

   fCommandFrame = new TGHorizontalFrame(fContents);
   fCommand = new TGTextEntry(fCommandFrame, (const char *)"", 20);
   fCommand->Connect("ReturnPressed()", "TextViewMainFrame", this, "HandleReturn()");
   fCommandFrame->AddFrame(new TGLabel(fCommandFrame, "Command: "),
                           new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 5, 5, 5, 5));
   fCommandFrame->AddFrame(fCommand, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));
   fContents->AddFrame(fCommandFrame, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));

   // The button for test
   fReset = new TGTextButton(fButtons, "&Reset");
   fReset->SetToolTipText("Press to clear the command entry\nand the TGTextView", 200);
   fReset->Connect("Clicked()", "TextViewMainFrame", this, "Reset()");
   fButtons->AddFrame(fReset, new TGLayoutHints(kLHintsExpandX | kLHintsTop, 5, 5, 5, 5));

   fExit = new TGTextButton(fButtons, "&Exit");
   fExit->SetToolTipText("Terminate the application", 200);
   fButtons->AddFrame(fExit, new TGLayoutHints(kLHintsExpandX | kLHintsTop, 5, 5, 5, 5));
   fExit->Connect("Pressed()", "TApplication", gApplication, "Terminate()");

   fContents->AddFrame(fButtons, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 0, 0));
   Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");
   DontCallClose();

   AddFrame(fContents, new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY));
   MapSubwindows();
   Resize(GetDefaultSize());

   SetWindowName("TGTextView Demo");
   MapRaised();
}

//______________________________________________________________________________
void TextViewMainFrame::Reset()
{
   fCommand->Clear();
   fTextView->Clear();
}

//______________________________________________________________________________
void TextViewMainFrame::HandleReturn()
{
   std::string line;
   std::string command = fCommand->GetText();
   *fTextView << gSystem->GetFromPipe(command.c_str()).Data() << std::endl;
   fTextView->ShowBottom();
   fCommand->Clear();
}

//______________________________________________________________________________
void textviewostream()
{
   // Main program.

   new TextViewMainFrame();
}


