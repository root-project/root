// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   26/09/2007

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGCommandPlugin
    \ingroup guiwidgets

Class used to redirect the command line input/output.

*/

#include "TROOT.h"
#include "TSystem.h"
#include "TRint.h"
#include "TApplication.h"
#include "TGClient.h"
#include "TGLabel.h"
#include "TGFrame.h"
#include "TGLayout.h"
#include "TGComboBox.h"
#include "TGTextView.h"
#include "TGTextEntry.h"
#include "TInterpreter.h"
#include "Getline.h"
#include "KeySymbols.h"

#include "TGCommandPlugin.h"
#include <vector>
#include <string>

ClassImp(TGCommandPlugin);

////////////////////////////////////////////////////////////////////////////////
/// TGCommandPlugin Constructor.

TGCommandPlugin::TGCommandPlugin(const TGWindow *p, UInt_t w, UInt_t h) :
      TGMainFrame(p, w, h)
{
   SetCleanup(kDeepCleanup);
   fHistAdd = kFALSE;
   fPos = 0;
   fTempString = "";
   fHf = new TGHorizontalFrame(this, 100, 20);
   fComboCmd   = new TGComboBox(fHf, "", 1);
   fCommand    = fComboCmd->GetTextEntry();
   fCommandBuf = fCommand->GetBuffer();
   fComboCmd->Resize(200, fCommand->GetDefaultHeight());
   fHf->AddFrame(fComboCmd, new TGLayoutHints(kLHintsCenterY |
                 kLHintsRight | kLHintsExpandX, 5, 5, 1, 1));
   fHf->AddFrame(fLabel = new TGLabel(fHf, "Command (local):"),
                 new TGLayoutHints(kLHintsCenterY | kLHintsRight,
                 5, 5, 1, 1));
   AddFrame(fHf, new TGLayoutHints(kLHintsLeft | kLHintsTop |
            kLHintsExpandX, 3, 3, 3, 3));
   fCommand->Connect("ReturnPressed()", "TGCommandPlugin", this,
                     "HandleCommand()");
   fCommand->Connect("CursorOutUp()", "TGCommandPlugin", this,
                     "HandleArrows(=kKey_Up)");
   fCommand->Connect("CursorOutDown()", "TGCommandPlugin", this,
                     "HandleArrows(=kKey_Down)");
   fCommand->Connect("TabPressed()", "TGCommandPlugin", this,
                     "HandleTab()");
   fCommand->Connect("TextChanged(const char *)", "TGCommandPlugin", this,
                     "HandleTextChanged(const char *)");
   fStatus = new TGTextView(this, 10, 100, 1);
   if (gClient->GetStyle() < 2) {
      Pixel_t pxl;
      gClient->GetColorByName("#a0a0a0", pxl);
      fStatus->SetSelectBack(pxl);
      fStatus->SetSelectFore(TGFrame::GetWhitePixel());
   }
   AddFrame(fStatus, new TGLayoutHints(kLHintsLeft | kLHintsTop |
            kLHintsExpandX | kLHintsExpandY, 3, 3, 3, 3));
   fPid = gSystem->GetPid();
   TString defhist(Form("%s/.root_hist", gSystem->UnixPathName(
                        gSystem->HomeDirectory())));
   FILE *lunin = fopen(defhist.Data(), "rt");
   if (lunin) {
      ULong_t linecount = 0;
      char histline[256];
      rewind(lunin);
      while (fgets(histline, 256, lunin))
         ++linecount;
      rewind(lunin);
      if (linecount > 500) {
         linecount -= 500;
         while(--linecount > 0)
            if (!fgets(histline, 256, lunin))
               break;
      }
      linecount = 0;
      while (fgets(histline, 256, lunin)) {
         histline[strlen(histline)-1] = 0; // remove trailing "\n"
         fComboCmd->InsertEntry(histline, linecount, -1);
         // limit the history size to 500 lines
         if (++linecount > 500)
            break;
      }
      fclose(lunin);
   }
   fTimer = new TTimer(this, 1000);
   fTimer->Reset();
   fTimer->TurnOn();
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGCommandPlugin::~TGCommandPlugin()
{
   TString pathtmp = TString::Format("%s/command.%d.log",
                                     gSystem->TempDirectory(), fPid);
   gSystem->Unlink(pathtmp);
   fCommand->Disconnect("ReturnPressed()");
   fCommand->Disconnect("CursorOutUp()");
   fCommand->Disconnect("CursorOutDown()");
   fCommand->Disconnect("TabPressed()");
   fCommand->Disconnect("TextChanged(const char *)");
   delete fTimer;
   fTimer = 0;
   Cleanup();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if actual ROOT session is a remote one or a local one.

void TGCommandPlugin::CheckRemote(const char * /*str*/)
{
   Pixel_t pxl;
   TApplication *app = gROOT->GetApplication();
   if (!app->InheritsFrom("TRint"))
      return;
   TString sPrompt = ((TRint*)app)->GetPrompt();
   Int_t end = sPrompt.Index(":root [", 0);
   if (end > 0 && end != kNPOS) {
      // remote session
      sPrompt.Remove(end);
      gClient->GetColorByName("#ff0000", pxl);
      fLabel->SetTextColor(pxl);
      fLabel->SetText(Form("Command (%s):", sPrompt.Data()));
   }
   else {
      // local session
      gClient->GetColorByName("#000000", pxl);
      fLabel->SetTextColor(pxl);
      fLabel->SetText("Command (local):");
   }
   fHf->Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Handle the 'up' and 'down' arrow key events.

void TGCommandPlugin::HandleArrows(Int_t keysym)
{
   Int_t entries = fComboCmd->GetNumberOfEntries();
   switch ((EKeySym)keysym) {
      case kKey_Up:
         if (fPos < entries-1) ++fPos;
         break;
      case kKey_Down:
         if (fPos > 0) --fPos;
         break;
      default:
         break;
   }
   if (fPos > 0) {
      TGTextLBEntry *te = (TGTextLBEntry *)fComboCmd->GetListBox()->GetEntry(entries-fPos);
      if (te) {
         fCommand->SetText(te->GetText()->GetString(), kFALSE);
      }
   } else {
      if (fTempString.Length() > 0)
         fCommand->SetText(fTempString.Data(), kFALSE);
      else
         fCommand->Clear();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle command line from the "command" combo box.

void TGCommandPlugin::HandleCommand()
{
   const char *string = fCommandBuf->GetString();
   if (strlen(string) > 0) {
      // form temporary file path
      TString sPrompt = "root []";
      TString pathtmp = TString::Format("%s/command.%d.log",
                                        gSystem->TempDirectory(), fPid);
      TApplication *app = gROOT->GetApplication();
      if (app->InheritsFrom("TRint"))
         sPrompt = ((TRint*)gROOT->GetApplication())->GetPrompt();
      FILE *lunout = fopen(pathtmp.Data(), "a+t");
      if (lunout) {
         fputs(Form("%s%s\n",sPrompt.Data(), string), lunout);
         fclose(lunout);
      }
      gSystem->RedirectOutput(pathtmp.Data(), "a");
      gApplication->SetBit(TApplication::kProcessRemotely);
      gROOT->ProcessLine(string);
      Int_t entries = fComboCmd->GetNumberOfEntries();
      fComboCmd->InsertEntry(string, entries, -1);
      fPos = 0;
      if (app->InheritsFrom("TRint") || fHistAdd)
         Gl_histadd((char *)string);
      gSystem->RedirectOutput(0);
      fStatus->LoadFile(pathtmp.Data());
      fStatus->ShowBottom();
      CheckRemote(string);
      fCommand->Clear();
      fTempString.Clear();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle the 'TAB' key events.

void TGCommandPlugin::HandleTab()
{
   std::string line = fCommandBuf->GetString();
   std::vector<std::string> result;
   size_t cur = line.length();
   gInterpreter->CodeComplete(line, cur, result);
   if (result.size() == 1) {
      // when there is only one result, complete the command line input
      std::string found = result[0];
      std::string what = line;
      size_t colon = line.find_last_of("::");
      if (colon != std::string::npos)
         what = line.substr(colon+2);
      size_t pos = found.find(what) + what.length();
      std::string suffix = found.substr(pos);
      fCommand->AppendText(suffix.c_str());
   } else {
      // otherwise print all results
      std::string prompt = gInterpreter->GetPrompt();
      if (prompt.find("root") == std::string::npos)
         prompt = "root []";
      prompt += " ";
      prompt += line;
      fStatus->AddLine(prompt.c_str());
      fStatus->ShowBottom();
      for (auto& res : result) {
         fStatus->AddLine(res.c_str());
         fStatus->ShowBottom();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle the text changed events.

void TGCommandPlugin::HandleTextChanged(const char *text)
{
   fTempString = text;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle timer event.

Bool_t TGCommandPlugin::HandleTimer(TTimer *t)
{
   if ((fTimer == 0) || (t != fTimer)) return kTRUE;
   CheckRemote("");
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Let user stop the internal timer when there is no need to check for remote.

void TGCommandPlugin::StopTimer()
{
   fTimer->TurnOff();
}

////////////////////////////////////////////////////////////////////////////////
/// The function SetHistAdd() is needed for a standalone TApplication to log the
/// TGCommandPlugin commands into a ROOT history file.
/// However, this function has no effect if the user does not explictly set on
/// his standalone application the name of the ROOT history file.
/// To log into the default ROOT history file, call this on the user-side of the
/// code:
///    Gl_histinit(gEnv->GetValue("Rint.History", gSystem->HomeDirectory()));
/// Otherwise, replace the argument of Gl_histinit with a text file name you want
/// to use for application-specific logging.

void TGCommandPlugin::SetHistAdd(Bool_t add)
{
   fHistAdd = add;
}
