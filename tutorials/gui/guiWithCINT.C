/// \file
/// \ingroup tutorial_gui
/// A simple example of entering  CINT commands and having the CINT output in a ROOT GUI application window.
/// An editable combo box is used as a CINT prompt, a text view widget displays the command output.
///
/// \macro_code
///
/// \author Ilka Antcheva   06/07/2007

#include <iostream>
#include <TApplication.h>
#include <TRint.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TGTextEntry.h>
#include <TGTextView.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TGLayout.h>
#include <TGWindow.h>
#include <TGLabel.h>
#include <TString.h>
#include <TGComboBox.h>
#include <Getline.h>

class IDList {

private:
   Int_t fID;  //create widget Id(s)

public:
   IDList() : fID(0) {}
   ~IDList() {}
   Int_t GetUnID(void) { return ++fID; }
};

class MyApplication : public TGMainFrame {

private:
   TGTextButton        *fExit;
   IDList               fIDs;
   TGComboBox          *fComboCmd;   // CINT command combobox
   TGTextBuffer        *fCommandBuf; // text buffer in use
   TGTextEntry         *fCommand;    // text entry for CINT commands
   TGTextView          *fTextView;   // display CINT output
   TString              fName;       // name of temp created file
public:
   MyApplication(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyApplication();

   void DoExit();
   void DoEnteredCommand();

   ClassDef(MyApplication, 0)
};

MyApplication::MyApplication(const TGWindow *p, UInt_t w, UInt_t h)
   : TGMainFrame(p, w, h)
{
   SetCleanup(kDeepCleanup);

   Connect("CloseWindow()", "MyApplication", this, "DoExit()");
   DontCallClose();

   TGHorizontalFrame *fHL2 = new TGHorizontalFrame(this, 70, 100);
   AddFrame(fHL2, new TGLayoutHints(kLHintsNormal, 5, 5, 5, 5));
   TGLabel *fInlabel = new TGLabel(fHL2, "CINT Prompt:");
   fHL2->AddFrame(fInlabel, new TGLayoutHints(kLHintsCenterY));

   TGLabel *fOutlabel = new TGLabel(this, "Output Window:");
   AddFrame(fOutlabel);

   fCommandBuf = new TGTextBuffer(256);
   fComboCmd = new TGComboBox(fHL2, "", fIDs.GetUnID());
   fCommand = fComboCmd->GetTextEntry();
   fComboCmd->Resize(450, fCommand->GetDefaultHeight());
   fHL2->AddFrame(fComboCmd, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 20,0,0,0));

   TString hist(Form("%s/.root_hist", gSystem->UnixPathName(gSystem->HomeDirectory())));
   FILE *fhist = fopen(hist.Data(), "rt");
   if (fhist) {
      char histline[256];
      while (fgets(histline, 256, fhist)) {
         histline[strlen(histline)-1] = 0; // remove trailing "\n"
         fComboCmd->InsertEntry(histline, 0, -1);
      }
      fclose(fhist);
   }

   Pixel_t backpxl;
   gClient->GetColorByName("#c0c0c0", backpxl);
   fTextView = new TGTextView(this, 500, 94, fIDs.GetUnID(), kFixedWidth | kFixedHeight);
   fTextView->SetBackground(backpxl);
   AddFrame(fTextView, new TGLayoutHints(kLHintsExpandX));
   TGHorizontalFrame *fHL3 = new TGHorizontalFrame(this, 70, 150, kFixedWidth);
   fExit = new TGTextButton(fHL3, "&Exit", fIDs.GetUnID());
   fExit->Connect("Clicked()", "MyApplication", this, "DoExit()");
   fHL3->AddFrame(fExit, new TGLayoutHints(kLHintsExpandX));
   AddFrame(fHL3, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, 1, 1, 1, 1));

   SetWindowName("GUI with CINT Input/Output");
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
   fCommand->Connect("ReturnPressed()", "MyApplication", this, "DoEnteredCommand()");
   fName = Form("%soutput.log", gSystem->WorkingDirectory());
};

MyApplication::~MyApplication()
{
   // Destructor.

   Cleanup();
}

void MyApplication::DoExit()
{
   // Close application window.

   gSystem->Unlink(fName.Data());
   gApplication->Terminate();
}

void MyApplication::DoEnteredCommand()
{
   // Execute the CINT command after the ENTER key was pressed.

   const char *command = fCommand->GetTitle();
   TString prompt;

   if (strlen(command)) {
      // form temporary file path
      prompt = ((TRint*)gROOT->GetApplication())->GetPrompt();
      FILE *cintout = fopen(fName.Data(), "a+t");
      if (cintout) {
         fputs(Form("%s%s\n",prompt.Data(), command), cintout);
         fclose(cintout);
      }
      gSystem->RedirectOutput(fName.Data(), "a");
      gROOT->ProcessLine(command);
      fComboCmd->InsertEntry(command, 0, fIDs.GetUnID());
      Gl_histadd((char *)command);
      gSystem->RedirectOutput(0);
      fTextView->LoadFile(fName.Data());
      if (fTextView->ReturnLineCount() > 10)
         fTextView->SetVsbPosition(fTextView->ReturnLineCount());
      fCommand->Clear();
   } else {
      printf("No command entered\n");
   }
   fTextView->ShowBottom();
}

void guiWithCINT()
{
   new MyApplication(gClient->GetRoot(),600,300);
}

