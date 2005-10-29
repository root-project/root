// @(#)root/treeviewer:$Name:  $:$Id: TSessionViewer.cxx
// Author: Marek Biskup, Jakub Madejczyk, Bertrand Bellenot 10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionViewer                                                       //
//                                                                      //
// Widget used to manage PROOF or local sessions, PROOF connections,    //
// queries construction and results handling.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TApplication.h"
#include "TSystem.h"
#include "TGFileDialog.h"
#include "TBrowser.h"
#include "TGButton.h"
#include "TGLayout.h"
#include "TGListTree.h"
#include "TGCanvas.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGTableLayout.h"
#include "TGComboBox.h"
#include "TGSplitter.h"
#include "TGProgressBar.h"
#include "TGListView.h"
#include "TGMsgBox.h"
#include "TGMenu.h"
#include "TGStatusBar.h"
#include "TGIcon.h"
#include "TChain.h"
#include "TDSet.h"
#include "TVirtualProof.h"
#include "TRandom.h"
#include "TSessionViewer.h"
#include "TSessionLogView.h"
#include "TQueryResult.h"
#include "TGTextView.h"
#include "TGMenu.h"
#include "TGToolBar.h"
#include "TGTab.h"
#include "TRootEmbeddedCanvas.h"
#include "TCanvas.h"
#include "TGMimeTypes.h"
#include "TInterpreter.h"
#include "TContextMenu.h"
#include "TG3DLine.h"
#include "TSessionDialogs.h"
#include "TEnv.h"
#include "TH1.h"
#include "TH2.h"
#ifdef WIN32
#include "TWin32SplashThread.h"
#endif

TSessionViewer *gSessionViewer = 0;

ClassImp(TQueryDescription)
ClassImp(TSessionDescription)
ClassImp(TSessionServerFrame)
ClassImp(TSessionFrame)
ClassImp(TSessionQueryFrame)
ClassImp(TSessionOutputFrame)
ClassImp(TSessionInputFrame)
ClassImp(TSessionViewer)

const char *xpm_names[] = {
    "monitor01.xpm",
    "monitor02.xpm",
    "monitor03.xpm",
    "monitor04.xpm",
    0
};

const char *conftypes[] = {
   "Config files",  "*.conf",
   "All files",     "*",
    0,               0
};

const char *pkgtypes[] = {
   "Package files", "*.par",
   "All files",     "*",
    0,               0
};

const char *kFeedbackHistos[] = {
   "PROOF_PacketsHist",
   "PROOF_EventsHist",
   "PROOF_NodeHist",
   "PROOF_LatencyHist",
   "PROOF_ProcTimeHist",
   "PROOF_CpuTimeHist",
   0
};

const char* const kPROOF_GuiConfFile = ".proofservers.conf";
const char* const kSession_RedirectFile = ".templog";
const char* const kSession_RedirectCmd = ".tempcmd";
char const kPROOF_GuiConfFileSeparator = '\t';

// Menu command id's
enum ESessionViewerCommands {
   kFileLoadLibrary,
   kFileCloseViewer,
   kFileQuit,

   kSessionConnect,
   kSessionDisconnect,
   kSessionCleanup,
   kSessionBrowse,
   kSessionShowStatus,

   kQueryNew,
   kQueryEdit,
   kQueryDelete,
   kQuerySubmit,
   kQueryStartViewer,

   kOptionsStatsHist,
   kOptionsStatsTrace,
   kOptionsSlaveStatsTrace,
   kOptionsFeedback,

   kHelpAbout
};


////////////////////////////////////////////////////////////////////////////////
// Server Frame

//______________________________________________________________________________
TSessionServerFrame::TSessionServerFrame(TGWindow* p, Int_t w, Int_t h) :
   TGCompositeFrame(p, w, h), fFrmNewServer(0), fTxtName(0), fTxtAddress(0),
      fTxtConfig(0), fTxtUsrName(0), fViewer(0)
{
   // Constructor
}

//______________________________________________________________________________
TSessionServerFrame::~TSessionServerFrame()
{
   // Destructor
   Cleanup();
}

//______________________________________________________________________________
void TSessionServerFrame::Build(TSessionViewer *gui)
{
   // Build server configuration frame

   SetLayoutManager(new TGVerticalLayout(this));
   TGCompositeFrame *tmp;
   TGButton* btnTmp;

   SetCleanup(kDeepCleanup);

   fViewer = gui;
   fFrmNewServer = new TGGroupFrame(this, "New Server");
   fFrmNewServer->SetCleanup(kDeepCleanup);

   AddFrame(fFrmNewServer, new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 2));

   fFrmNewServer->SetLayoutManager(new TGMatrixLayout(fFrmNewServer, 0, 2, 8));

   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "Connection Name:"),
                           new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));
   fFrmNewServer->AddFrame(fTxtName = new TGTextEntry(fFrmNewServer,
                           (const char *)0, 1), new TGLayoutHints());
   fTxtName->Resize(156, fTxtName->GetDefaultHeight());
   fTxtName->Associate(this);
   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "Server name:"),
                           new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));
   fFrmNewServer->AddFrame(fTxtAddress = new TGTextEntry(fFrmNewServer,
                           (const char *)0, 2), new TGLayoutHints());
   fTxtAddress->Resize(156, fTxtAddress->GetDefaultHeight());
   fTxtAddress->Associate(this);
   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "Port (default: 1093):"),
                           new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));
   fFrmNewServer->AddFrame(fNumPort = new TGNumberEntry(fFrmNewServer, 1093, 5,
            3, TGNumberFormat::kNESInteger,TGNumberFormat::kNEANonNegative,
            TGNumberFormat::kNELLimitMinMax, 0, 65535),new TGLayoutHints());
   fNumPort->Associate(this);
   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "Configuration File:"),
                           new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));
   fFrmNewServer->AddFrame(fTxtConfig = new TGTextEntry(fFrmNewServer,
                           (const char *)0, 4), new TGLayoutHints());
   fTxtConfig->Resize(156, fTxtConfig->GetDefaultHeight());
   fTxtConfig->Associate(this);
   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "Log Level:"),
                           new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));

   fFrmNewServer->AddFrame(fLogLevel = new TGNumberEntry(fFrmNewServer, 0, 5, 5,
                           TGNumberFormat::kNESInteger,
                           TGNumberFormat::kNEANonNegative,
                           TGNumberFormat::kNELLimitMinMax, 0, 5),
                           new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));
   fLogLevel->Associate(this);

   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "User Name:"),
            new TGLayoutHints(kLHintsLeft, 3, 3, 3, 3));
   fFrmNewServer->AddFrame(fTxtUsrName = new TGTextEntry(fFrmNewServer,
                           (const char *)0, 6), new TGLayoutHints());
   fTxtUsrName->Resize(156, fTxtUsrName->GetDefaultHeight());
   fTxtUsrName->Associate(this);

   fFrmNewServer->AddFrame(new TGLabel(fFrmNewServer, "Process mode :"),
            new TGLayoutHints(kLHintsLeft | kLHintsBottom | kLHintsExpandX,
            3, 3, 3, 3));
   fFrmNewServer->AddFrame(fSync = new TGCheckButton(fFrmNewServer,
      "&Synchronous"), new TGLayoutHints(kLHintsLeft | kLHintsBottom |
      kLHintsExpandX, 3, 3, 3, 3));
   fSync->SetToolTipText("Default Process Mode");
   fSync->SetState(kButtonDown);

   AddFrame(tmp = new TGCompositeFrame(this, 140, 10, kHorizontalFrame),
                       new TGLayoutHints(kLHintsLeft | kLHintsExpandX));
   tmp->SetCleanup(kDeepCleanup);
   tmp->AddFrame(btnTmp = new TGTextButton(tmp, "     Add     "),
                 new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 3, 3, 3, 3));
   tmp->Resize(155, btnTmp->GetDefaultHeight());
   btnTmp->Connect("Clicked()", "TSessionServerFrame", this,
                   "OnBtnAddClicked()");
   tmp->AddFrame(btnTmp = new TGTextButton(tmp, "   Connect   "),
                 new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 3, 3, 3, 3));
   tmp->Resize(155, btnTmp->GetDefaultHeight());
   btnTmp->Connect("Clicked()", "TSessionServerFrame", this,
                   "OnBtnConnectClicked()");

   AddFrame(tmp = new TGCompositeFrame(this, 140, 20, kHorizontalFrame),
                       new TGLayoutHints(kLHintsLeft | kLHintsExpandX));
   tmp->SetCleanup(kDeepCleanup);
   tmp->AddFrame(btnTmp = new TGTextButton(tmp, "  New server  "),
                 new TGLayoutHints(kLHintsLeft | kLHintsBottom |
                 kLHintsExpandX, 3, 3, 15, 3));
   btnTmp->Connect("Clicked()", "TSessionServerFrame", this,
                   "OnBtnNewServerClicked()");
   tmp->AddFrame(btnTmp = new TGTextButton(tmp, "    Delete    "),
                 new TGLayoutHints(kLHintsLeft | kLHintsBottom |
                 kLHintsExpandX, 3, 3, 15, 3));
   btnTmp->Connect("Clicked()", "TSessionServerFrame", this,
                   "OnBtnDeleteClicked()");
   fTxtConfig->Connect("DoubleClicked()", "TSessionServerFrame", this,
                       "OnConfigFileClicked()");
}

//______________________________________________________________________________
Bool_t TSessionServerFrame::HandleExpose(Event_t * /*event*/)
{
   // Handle expose event
   fTxtName->SelectAll();
   fTxtName->SetFocus();
   return kTRUE;
}

//______________________________________________________________________________
void TSessionServerFrame::OnConfigFileClicked()
{
   // Browse for configuration files

   // do nothing if connection in progress
   if (fViewer->IsBusy())
      return;
   TGFileInfo fi;
   fi.fFileTypes = conftypes;
   new TGFileDialog(fClient->GetRoot(), fViewer, kFDOpen, &fi);
   if (!fi.fFilename) return;
   fTxtConfig->SetText(gSystem->BaseName(fi.fFilename));
}

//______________________________________________________________________________
void TSessionServerFrame::OnBtnDeleteClicked()
{
   // Delete selected session configuration

   // do nothing if connection in progress
   if (fViewer->IsBusy())
      return;
   TString name(fTxtName->GetText());
   TIter next(fViewer->GetSessions());
   TSessionDescription *desc = 0;
   // browse list of session descriptions
   while ((desc = (TSessionDescription *)next())) {
      // name match
      if (desc->fName == name) {
         // if local session, just display message
         if ((name.CompareTo("Local", TString::kIgnoreCase) == 0) &&
             (desc->fLocal)) {
            Int_t retval;
            new TGMsgBox(fClient->GetRoot(), this, "Error Deleting Session",
                         "Deleting Local Sessions is not allowed !",
                         kMBIconExclamation,kMBOk,&retval);
            break;
         }
         // if connected, first disconnect
         if (desc->fConnected)
            desc->fProof->Close();
         // ask for confirmation
         TString m;
         m.Form("Are you sure to delete the server \"%s\"",
                desc->fName.Data());
         Int_t result;
         new TGMsgBox(fClient->GetRoot(), this, "", m.Data(), 0,
                      kMBOk | kMBCancel, &result);
         // if confirmed, delete it
         if (result == kMBOk) {
            // remove the Proof session from gROOT list of Proofs
            if (desc->fProof)
               gROOT->GetListOfProofs()->Remove(desc->fProof);
            // remove it from our sessions list
            fViewer->GetSessions()->Remove((TObject *)desc);
            // update configuration file
            WriteConfigFile(kPROOF_GuiConfFile, fViewer->GetSessions());
            // rebuilds tree viewer with updated list
            fViewer->BuildSessionHierarchy(fViewer->GetSessions());
            // update viewer with new selected session
            TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
            fViewer->OnListTreeClicked(item, 1, 0, 0);
         }
         break;
      }
   }
}

//______________________________________________________________________________
void TSessionServerFrame::OnBtnConnectClicked()
{
   // Connect to selected server

   // do nothing if connection in progress
   if (fViewer->IsBusy())
      return;

   // set flag busy
   fViewer->SetBusy();
   // avoid input events in list tree while connecting
   fViewer->GetSessionHierarchy()->RemoveInput(kPointerMotionMask |
         kEnterWindowMask | kLeaveWindowMask | kKeyPressMask);
   gVirtualX->GrabButton(fViewer->GetSessionHierarchy()->GetId(), kAnyButton,
         kAnyModifier, kButtonPressMask | kButtonReleaseMask, kNone, kNone, kFALSE);
   // set watch cursor to indicate connection in progress
   gVirtualX->SetCursor(fViewer->GetSessionHierarchy()->GetId(),
         gVirtualX->CreateCursor(kWatch));
   gVirtualX->SetCursor(GetId(),gVirtualX->CreateCursor(kWatch));
   // display connection progress bar in first part of status bar
   fViewer->GetStatusBar()->GetBarPart(0)->ShowFrame(fViewer->GetConnectProg());
   // connect to proof startup message (to update progress bar)
   TQObject::Connect("TProof", "StartupMessage(char *,Bool_t,Int_t,Int_t)",
         "TSessionViewer", fViewer, "StartupMessage(char *,Bool_t,Int_t,Int_t)");
   // collect and set-up configuration
   TString url = fTxtUsrName->GetText();
   url += "@"; url += fTxtAddress->GetText();
   if (fNumPort->GetIntNumber() > 0)
      url += fNumPort->GetIntNumber();
   fViewer->GetActDesc()->fLogLevel = fLogLevel->GetIntNumber();
   if (strlen(fTxtConfig->GetText()) > 1)
      fViewer->GetActDesc()->fConfigFile = TString(fTxtConfig->GetText());
   else
      fViewer->GetActDesc()->fConfigFile = "";
   // connect to Proof server
   fViewer->GetActDesc()->fProof = gROOT->Proof(url,
            fViewer->GetActDesc()->fConfigFile, 0,
            fViewer->GetActDesc()->fLogLevel);
   // check if connected and valid
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      // set log level
      fViewer->GetActDesc()->fProof->SetLogLevel(fViewer->GetActDesc()->fLogLevel);
      // set query type (synch / asynch)
      fViewer->GetActDesc()->fProof->SetQueryType(fViewer->GetActDesc()->fSync ?
                             TVirtualProof::kSync : TVirtualProof::kAsync);
      // set connected flag
      fViewer->GetActDesc()->fConnected = kTRUE;
      // change list tree item picture to connected pixmap
      TGListTreeItem *item = fViewer->GetSessionHierarchy()->FindChildByData(
                             fViewer->GetSessionItem(),fViewer->GetActDesc());
      item->SetPictures(fViewer->GetProofConPict(), fViewer->GetProofConPict());
      // update viewer
      fViewer->OnListTreeClicked(item, 1, 0, 0);
      fClient->NeedRedraw(fViewer->GetSessionHierarchy());
      // connect to progress related signals
      fViewer->GetActDesc()->fProof->Connect("Progress(Long64_t,Long64_t)",
                                 "TSessionQueryFrame", fViewer->GetQueryFrame(),
                                 "Progress(Long64_t,Long64_t)");
      fViewer->GetActDesc()->fProof->Connect("StopProcess(Bool_t)",
                                 "TSessionQueryFrame", fViewer->GetQueryFrame(),
                                 "IndicateStop(Bool_t)");
      fViewer->GetActDesc()->fProof->Connect(
                  "ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)",
                  "TSessionQueryFrame", fViewer->GetQueryFrame(),
                  "ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)");
      // enable timer used for status bar icon's animation
      fViewer->EnableTimer();
      // change status bar right icon to connected pixmap
      fViewer->ChangeRightLogo("monitor01.xpm");
      // do not animate yet
      fViewer->SetChangePic(kFALSE);
      // connect to signal "query result ready"
      fViewer->GetActDesc()->fProof->Connect("QueryResultReady(char *)",
                       "TSessionViewer", fViewer, "QueryResultReady(char *)");
      // display connection information on status bar
      TString msg;
      msg.Form("PROOF Cluster %s ready", fViewer->GetActDesc()->fName.Data());
      fViewer->GetStatusBar()->SetText(msg.Data(), 1);
      fViewer->GetSessionFrame()->ProofInfos();
   }
   // hide connection progress bar from status bar
   fViewer->GetStatusBar()->GetBarPart(0)->HideFrame(fViewer->GetConnectProg());
   // release busy flag
   fViewer->SetBusy(kFALSE);
   // restore cursors and input
   gVirtualX->SetCursor(GetId(), 0);
   gVirtualX->GrabButton(fViewer->GetSessionHierarchy()->GetId(), kAnyButton,
         kAnyModifier, kButtonPressMask | kButtonReleaseMask, kNone, kNone);
   fViewer->GetSessionHierarchy()->AddInput(kPointerMotionMask |
         kEnterWindowMask | kLeaveWindowMask | kKeyPressMask);
   gVirtualX->SetCursor(fViewer->GetSessionHierarchy()->GetId(), 0);
}

//______________________________________________________________________________
void TSessionServerFrame::OnBtnNewServerClicked()
{
   // reset server configuration fields

   // do nothing if connection in progress
   if (fViewer->IsBusy())
      return;
   fTxtName->SetText("");
   fTxtAddress->SetText("");
   fNumPort->SetIntNumber(1093);
   fLogLevel->SetIntNumber(0);
   fTxtUsrName->SetText("");
}

//______________________________________________________________________________
void TSessionServerFrame::OnBtnAddClicked()
{
   // Add new session configuration

   // do nothing if connection in progress
   if (fViewer->IsBusy())
      return;
   TSessionDescription* desc = new TSessionDescription();
   desc->fName = TString(fTxtName->GetText());
   desc->fAddress = TString(fTxtAddress->GetText());
   desc->fPort = fNumPort->GetIntNumber();
   desc->fConnected = kFALSE;
   desc->fLocal = kFALSE;
   desc->fQueries = new TList();
   desc->fActQuery = 0;
   if (strlen(fTxtConfig->GetText()) > 1)
      desc->fConfigFile = TString(fTxtConfig->GetText());
   else
      desc->fConfigFile = "";
   desc->fLogLevel = fLogLevel->GetIntNumber();
   desc->fUserName = TString(fTxtUsrName->GetText());
   desc->fSync = (fSync->GetState() == kButtonDown);
   desc->fProof = 0;
   // add newly created session config to our session list
   fViewer->GetSessions()->Add((TObject *)desc);
   // save into configuration file
   WriteConfigFile(kPROOF_GuiConfFile, fViewer->GetSessions());
   // update list tree with updated session list
   fViewer->BuildSessionHierarchy(fViewer->GetSessions());
}

//______________________________________________________________________________
void TSessionServerFrame::Update(TSessionDescription* desc)
{
   // update session configuration fields

   fTxtName->SetText(desc->fName);
   fTxtAddress->SetText(desc->fAddress);
   fNumPort->SetIntNumber(desc->fPort);
   fLogLevel->SetIntNumber(desc->fLogLevel);

   if (desc->fConfigFile.Length() > 1) {
      fTxtConfig->SetText(desc->fConfigFile);
   }
   else {
      fTxtConfig->SetText("");
   }
   fTxtUsrName->SetText(desc->fUserName);
}

//______________________________________________________________________________
Bool_t TSessionServerFrame::WriteConfigFile(const TString &filePath, TList *vec)
{
   // write proof sessions configuration file ($(HOME)/.proofservers.conf)

   char line[2048];
   char c = kPROOF_GuiConfFileSeparator;
   // set full path to $(HOME)/.proofservers.conf
   TString homefilePath(gSystem->UnixPathName(gSystem->HomeDirectory()));
   homefilePath.Append('/');
   homefilePath.Append(filePath);
   FILE* f = fopen(homefilePath.Data(), "w");
   if (!f) {
      Error("WriteConfigFile", "Cannot open the config file %s for writing",
            filePath.Data());
      return kFALSE;
   }
   // iterator on list of sessions config
   TIter next(vec);
   TSessionDescription *desc = 0;
   while ((desc = (TSessionDescription *)next())) {
      sprintf(line, "%s%c%s%c%d%c%s%c", desc->fName.Data(), c,
              desc->fAddress.Data(), c, desc->fPort, c,
              desc->fConfigFile.Data(), c);
      sprintf(line, "%s%s%c%d%c%s%c%s%c%s%c%d",line, "loglevel", c,
              desc->fLogLevel, c, "user", c, desc->fUserName.Data(),
              c, "sync", c, desc->fSync);
      sprintf(line,"%s\n", line);
      // write in file
      if (fprintf(f, line) == 0) {
         Error("WriteConfigFile", "Error writing to the config file");
         fclose(f);
         return kFALSE;
      }
   }
   fclose(f);
   return kTRUE;
}

//______________________________________________________________________________
TList *TSessionServerFrame::ReadConfigFile(const TString &filePath)
{
   // read proof sessions configuration file ($(HOME)/.proofservers.conf)

   TList *vec = new TList;
   TString homefilePath(gSystem->UnixPathName(gSystem->HomeDirectory()));
   homefilePath.Append('/');
   homefilePath.Append(filePath);
   FILE* f = fopen(homefilePath.Data(), "r");
   if (!f) {
      if (gDebug > 0)
         Info("ReadConfigFile", "Cannot open the config file %s", filePath.Data());
      return vec;
   }
   char line[2048];
   while (fgets(line, sizeof(line), f)) {
      Int_t len = strlen(line);
      if (len > 0 && line[len-1] == '\n')
         line[len-- -1] = '\0';
      if (line[0] == '#' || line[0] == '\0')
         continue;         // skip comment and empty lines
      char* parts[10];
      Int_t noParts = 0;
      parts[noParts++] = line;
      // count number of parts (fields)
      for (int i = 0; i < len && noParts < 10; i++) {
         if (line[i] == kPROOF_GuiConfFileSeparator) {
            parts[noParts++] = &line[i + 1];
            line[i] = '\0';
         }
      }
      if (noParts < 8) {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (1)");
         continue;
      }
      Int_t port, loglevel, sync;
      // read port number
      if (sscanf(parts[2], "%d", &port) != 1) {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (2)");
         continue;
      }
      // read log level
      if (strcmp(parts[4], "loglevel") != 0) {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (3)");
         continue;
      }
      if (sscanf(parts[5], "%d", &loglevel) != 1) {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (4)");
         continue;
      }
      // build session description
      TSessionDescription *proofDesc = new TSessionDescription();
      proofDesc->fName = TString(parts[0]);
      proofDesc->fAddress = TString(parts[1]);
      proofDesc->fPort = port;
      proofDesc->fConfigFile = TString(parts[3]);
      proofDesc->fLogLevel = loglevel;
      proofDesc->fConnected = kFALSE;
      proofDesc->fLocal = kFALSE;
      proofDesc->fQueries = new TList();
      proofDesc->fActQuery = 0;
      proofDesc->fProof = 0;
      // read synch flag
      if (strcmp(parts[8], "sync") != 0) {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (5)");
         continue;
      }
      if (sscanf(parts[9], "%d", &sync) != 1) {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (6)");
         continue;
      }
      proofDesc->fSync = (Bool_t)sync;
      // read user name
      if (strcmp(parts[6], "user") == 0) {
         if (noParts != 10) {
            Error("ReadConfigFile",  "PROOF Servers config file corrupted; skipping (7)");
            delete proofDesc;
            continue;
         }
         proofDesc->fUserName = TString(parts[7]);
      }
      else {
         Error("ReadConfigFile", "PROOF Servers config file corrupted; skipping (8)");
         delete proofDesc;
         continue;
      }
      // add session description to our session list
      vec->Add((TObject *)proofDesc);
   }
   fclose(f);
   return vec;
}

//______________________________________________________________________________
Bool_t TSessionServerFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process messages for session server frame
   // used to navigate between text entry fields

   switch (GET_MSG(msg)) {
      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_ENTER:
            case kTE_TAB:
               switch (parm1) {
                  case 1: // session name
                     fTxtAddress->SelectAll();
                     fTxtAddress->SetFocus();
                     break;
                  case 2: // server address
                     fNumPort->GetNumberEntry()->SelectAll();
                     fNumPort->GetNumberEntry()->SetFocus();
                     break;
                  case 3: // port number
                     fTxtConfig->SelectAll();
                     fTxtConfig->SetFocus();
                     break;
                  case 4: // configuration file
                     fLogLevel->GetNumberEntry()->SelectAll();
                     fLogLevel->GetNumberEntry()->SetFocus();
                     break;
                  case 5: // log level
                     fTxtUsrName->SelectAll();
                     fTxtUsrName->SetFocus();
                     break;
                  case 6: // user name
                     fTxtName->SelectAll();
                     fTxtName->SetFocus();
                     break;
               }
               break;

            default:
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
// Session Frame

//______________________________________________________________________________
TSessionFrame::TSessionFrame(TGWindow* p, Int_t w, Int_t h) :
   TGCompositeFrame(p, w, h)
{
   // Constructor
   fPackages = 0;
}

//______________________________________________________________________________
TSessionFrame::~TSessionFrame()
{
   // Destructor
   Cleanup();
}

//______________________________________________________________________________
void TSessionFrame::Build(TSessionViewer *gui)
{
   // build session frame

   SetLayoutManager(new TGVerticalLayout(this));
   SetCleanup(kDeepCleanup);
   fViewer  = gui;
   Int_t i,j;

   // main session tab
   fTab = new TGTab(this, 200, 200);
   AddFrame(fTab, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
            kLHintsExpandY, 2, 2, 2, 2));

   // add "Status" tab element
   TGCompositeFrame *tf = fTab->AddTab("Status");
   fFA = new TGCompositeFrame(tf, 100, 100, kVerticalFrame);
   tf->AddFrame(fFA, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                kLHintsExpandX | kLHintsExpandY));

   // add first session information line
   fInfoLine[0] = new TGLabel(fFA, " ");
   fFA->AddFrame(fInfoLine[0], new TGLayoutHints(kLHintsCenterX |
                 kLHintsExpandX, 5, 5, 15, 5));

   TGCompositeFrame* frmInfos = new TGHorizontalFrame(fFA, 350, 100);
   frmInfos->SetLayoutManager(new TGTableLayout(frmInfos, 6, 2));

   // add session information lines
   j = 0;
   for (i=0;i<11;i+=2) {
      fInfoLine[i+1] = new TGLabel(frmInfos, " ");
      frmInfos->AddFrame(fInfoLine[i+1], new TGTableLayoutHints(0, 1, j, j+1,
         kLHintsLeft | kLHintsCenterY, 5, 5, 5, 5));
      fInfoLine[i+2] = new TGLabel(frmInfos, " ");
      frmInfos->AddFrame(fInfoLine[i+2], new TGTableLayoutHints(1, 2, j, j+1,
         kLHintsLeft | kLHintsCenterY, 5, 5, 5, 5));
      j++;
   }
   fFA->AddFrame(frmInfos, new TGLayoutHints(kLHintsLeft | kLHintsTop |
               kLHintsExpandX  | kLHintsExpandY, 5, 5, 5, 5));

   // add "new query" and "get queries" buttons
   TGCompositeFrame* frmBut1 = new TGHorizontalFrame(fFA, 350, 100);
   frmBut1->SetCleanup(kDeepCleanup);
   frmBut1->AddFrame(fBtnNewQuery = new TGTextButton(frmBut1, "New Query..."),
      new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 5, 5, 5));
   frmBut1->AddFrame(fBtnGetQueries = new TGTextButton(frmBut1, " Get Queries  "),
       new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 5, 5, 5));
   fFA->AddFrame(frmBut1, new TGLayoutHints(kLHintsLeft | kLHintsBottom | kLHintsExpandX));

   // add "disconnect" and "show log" buttons
   TGCompositeFrame* frmBut0 = new TGHorizontalFrame(fFA, 350, 100);
   frmBut0->SetCleanup(kDeepCleanup);
   frmBut0->AddFrame(fBtnDisconnect = new TGTextButton(frmBut0,
      " Disconnect "),new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 5, 5, 5));
   fBtnShowLog = new TGTextButton(frmBut0, "Show log...");
   frmBut0->AddFrame(fBtnShowLog, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 5, 5, 5));
   fFA->AddFrame(frmBut0, new TGLayoutHints(kLHintsLeft | kLHintsBottom | kLHintsExpandX));

   // add "Commands" tab element
   tf = fTab->AddTab("Commands");
   fFC = new TGCompositeFrame(tf, 100, 100, kVerticalFrame);
   tf->AddFrame(fFC, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                kLHintsExpandX | kLHintsExpandY));

   // add comand line label and text entry
   TGCompositeFrame* frmCmd = new TGHorizontalFrame(fFC, 350, 100);
   frmCmd->SetCleanup(kDeepCleanup);
   frmCmd->AddFrame(new TGLabel(frmCmd, "Command Line :"),
      new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 5, 5, 15, 5));
   fCommandBuf = new TGTextBuffer(120);
   frmCmd->AddFrame(fCommandTxt = new TGTextEntry(frmCmd,
      fCommandBuf ),new TGLayoutHints(kLHintsLeft | kLHintsCenterY |
      kLHintsExpandX, 5, 5, 15, 5));
   fFC->AddFrame(frmCmd, new TGLayoutHints(kLHintsExpandX, 5, 5, 10, 5));
   // connect command line text entry to "return pressed" signal
   fCommandTxt->Connect("ReturnPressed()", "TSessionFrame", this,
                           "OnCommandLine()");

   // check box for option "clear view"
   fClearCheck = new TGCheckButton(fFC, "Clear view after each command");
   fFC->AddFrame(fClearCheck,new TGLayoutHints(kLHintsLeft | kLHintsTop,
                 10, 5, 5, 5));
   fClearCheck->SetState(kButtonUp);
   // add text view for redirected output
   fFC->AddFrame(new TGLabel(fFC, "Output :"),
      new TGLayoutHints(kLHintsLeft | kLHintsTop, 10, 5, 5, 5));
   fInfoTextView = new TGTextView(fFC, 330, 150, "", kSunkenFrame |
                                  kDoubleBorder);
   fFC->AddFrame(fInfoTextView, new TGLayoutHints(kLHintsLeft |
      kLHintsTop | kLHintsExpandX | kLHintsExpandY, 10, 10, 5, 5));

   // add "Packages" tab element
   tf = fTab->AddTab("Packages");
   fFB = new TGCompositeFrame(tf, 100, 100, kVerticalFrame);
   tf->AddFrame(fFB, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                kLHintsExpandX | kLHintsExpandY));

   // new frame containing packages listbox and control buttons
   TGCompositeFrame* frmcanvas = new TGHorizontalFrame(fFB, 350, 100);

   // packages listbox
   fLBPackages = new TGListBox(frmcanvas);
   fLBPackages->Resize(80,150);
   fLBPackages->SetMultipleSelections(kFALSE);
   frmcanvas->AddFrame(fLBPackages, new TGLayoutHints(kLHintsExpandX |
            kLHintsExpandY, 4, 4, 4, 4));
   // control buttons frame
   TGCompositeFrame* frmBut2 = new TGVerticalFrame(frmcanvas, 150, 100);

   fChkMulti = new TGCheckButton(frmBut2, "Multiple Selection");
   frmBut2->AddFrame(fChkMulti, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));

   fBtnAdd = new TGTextButton(frmBut2, "        Add...         ");
   frmBut2->AddFrame(fBtnAdd,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fBtnRemove = new TGTextButton(frmBut2, "Remove");
   frmBut2->AddFrame(fBtnRemove,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fBtnUp = new TGTextButton(frmBut2, "Move Up");
   frmBut2->AddFrame(fBtnUp,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fBtnDown = new TGTextButton(frmBut2, "Move Down");
   frmBut2->AddFrame(fBtnDown,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   frmcanvas->AddFrame(frmBut2, new TGLayoutHints(kLHintsLeft | kLHintsCenterY |
            kLHintsExpandY));
   fFB->AddFrame(frmcanvas, new TGLayoutHints(kLHintsLeft | kLHintsTop |
                 kLHintsExpandX | kLHintsExpandY));

   TGCompositeFrame* frmBtn = new TGHorizontalFrame(fFB, 300, 100);
   frmBtn->SetCleanup(kDeepCleanup);
   frmBtn->AddFrame(fBtnUpload = new TGTextButton(frmBtn,
      "     Upload      "), new TGLayoutHints(kLHintsLeft | kLHintsExpandX |
      kLHintsCenterY, 5, 5, 5, 5));
   frmBtn->AddFrame(fBtnEnable = new TGTextButton(frmBtn,
      "     Enable      "), new TGLayoutHints(kLHintsLeft | kLHintsExpandX |
      kLHintsCenterY, 5, 5, 5, 5));
   frmBtn->AddFrame(fBtnDisable = new TGTextButton(frmBtn,
      "     Disable     "), new TGLayoutHints(kLHintsLeft | kLHintsExpandX |
      kLHintsCenterY, 5, 5, 5, 5));
   frmBtn->AddFrame(fBtnClear = new TGTextButton(frmBtn,
      "      Clear      "), new TGLayoutHints(kLHintsLeft | kLHintsExpandX |
      kLHintsCenterY, 5, 5, 5, 5));
   fFB->AddFrame(frmBtn, new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 0));

   TGCompositeFrame* frmBtn3 = new TGHorizontalFrame(fFB, 300, 100);
   frmBtn3->SetCleanup(kDeepCleanup);
   fBtnShow = new TGTextButton(frmBtn3, "Show packages");
   frmBtn3->AddFrame(fBtnShow,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fBtnShowEnabled = new TGTextButton(frmBtn3, "Show Enabled");
   frmBtn3->AddFrame(fBtnShowEnabled,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fFB->AddFrame(frmBtn3, new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 0));

   fChkEnable = new TGCheckButton(fFB, "Enable at session startup");
   fFB->AddFrame(fChkEnable, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));
   // Disable it for now (until implemented)
   fChkEnable->SetEnabled(kFALSE);

   // add "Options" tab element
   tf = fTab->AddTab("Options");
   fFD = new TGCompositeFrame(tf, 100, 100, kVerticalFrame);
   tf->AddFrame(fFD, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                kLHintsExpandX | kLHintsExpandY));

   // add Log Level label and text entry
   TGCompositeFrame* frmLog = new TGHorizontalFrame(fFD, 300, 100, kFixedWidth);
   frmLog->SetCleanup(kDeepCleanup);
   frmLog->AddFrame(fApplyLogLevel = new TGTextButton(frmLog,
      "        Apply        "), new TGLayoutHints(kLHintsRight |
      kLHintsCenterY, 10, 5, 5, 5));
   fLogLevel = new TGNumberEntry(frmLog, 0, 5, 5, TGNumberFormat::kNESInteger,
      TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 0, 5);
   frmLog->AddFrame(fLogLevel, new TGLayoutHints(kLHintsRight |
      kLHintsCenterY, 5, 5, 5, 5));
   frmLog->AddFrame(new TGLabel(frmLog, "Log Level :"),
      new TGLayoutHints(kLHintsRight | kLHintsCenterY, 5, 5, 5, 5));
   fFD->AddFrame(frmLog, new TGLayoutHints(kLHintsLeft, 5, 5, 15, 5));

   // add Parallel Nodes label and text entry
   TGCompositeFrame* frmPar = new TGHorizontalFrame(fFD, 300, 100, kFixedWidth);
   frmPar->SetCleanup(kDeepCleanup);
   frmPar->AddFrame(fApplyParallel = new TGTextButton(frmPar,
      "        Apply        "), new TGLayoutHints(kLHintsRight |
      kLHintsCenterY, 10, 5, 5, 5));
   fTxtParallel = new TGTextEntry(frmPar);
   fTxtParallel->SetAlignment(kTextRight);
   fTxtParallel->SetText("99999");
   fTxtParallel->Resize(fLogLevel->GetDefaultWidth(), fTxtParallel->GetDefaultHeight());
   frmPar->AddFrame(fTxtParallel, new TGLayoutHints(kLHintsRight |
      kLHintsCenterY, 5, 5, 5, 5));
   frmPar->AddFrame(new TGLabel(frmPar, "Set Parallel Nodes :"),
      new TGLayoutHints(kLHintsRight | kLHintsCenterY, 5, 5, 5, 5));
   fFD->AddFrame(frmPar, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));

   // connect button actions to functions
   fBtnDisconnect->Connect("Clicked()", "TSessionFrame", this,
                           "OnBtnDisconnectClicked()");
   fBtnShowLog->Connect("Clicked()", "TSessionFrame", this,
                        "OnBtnShowLogClicked()");
   fBtnNewQuery->Connect("Clicked()", "TSessionFrame", this,
                         "OnBtnNewQueryClicked()");
   fBtnGetQueries->Connect("Clicked()", "TSessionFrame", this,
                           "OnBtnGetQueriesClicked()");

   fChkMulti->Connect("Toggled(Bool_t)", "TSessionFrame", this,
                           "OnMultipleSelection(Bool_t)");
   fBtnAdd->Connect("Clicked()", "TSessionFrame", this,
                           "OnBtnAddClicked()");
   fBtnRemove->Connect("Clicked()", "TSessionFrame", this,
                           "OnBtnRemoveClicked()");
   fBtnUp->Connect("Clicked()", "TSessionFrame", this,
                           "OnBtnUpClicked()");
   fBtnDown->Connect("Clicked()", "TSessionFrame", this,
                           "OnBtnDownClicked()");
   fApplyLogLevel->Connect("Clicked()", "TSessionFrame", this,
                           "OnApplyLogLevel()");
   fApplyParallel->Connect("Clicked()", "TSessionFrame", this,
                           "OnApplyParallel()");
   fBtnUpload->Connect("Clicked()", "TSessionFrame", this,
                           "OnUploadPackages()");
   fBtnEnable->Connect("Clicked()", "TSessionFrame", this,
                           "OnEnablePackages()");
   fBtnDisable->Connect("Clicked()", "TSessionFrame", this,
                           "OnDisablePackages()");
   fBtnClear->Connect("Clicked()", "TSessionFrame", this,
                           "OnClearPackages()");
   fBtnShowEnabled->Connect("Clicked()", "TSessionViewer", fViewer,
                           "ShowEnabledPackages()");
   fBtnShow->Connect("Clicked()", "TSessionViewer", fViewer,
                           "ShowPackages()");
}

//______________________________________________________________________________
void TSessionFrame::ProofInfos()
{
   // Display informations on current session

   char buf[256];

   // if local session
   if (fViewer->GetActDesc()->fLocal) {
      sprintf(buf, "*** Local Session on %s ***", gSystem->HostName());
      fInfoLine[0]->SetText(buf);
      UserGroup_t *userGroup = gSystem->GetUserInfo();
      fInfoLine[1]->SetText("User :");
      sprintf(buf, "%s", userGroup->fRealName.Data());
      fInfoLine[2]->SetText(buf);
      fInfoLine[3]->SetText("Working directory :");
      sprintf(buf, "%s", gSystem->WorkingDirectory());
      fInfoLine[4]->SetText(buf);
      fInfoLine[5]->SetText(" ");
      fInfoLine[6]->SetText(" ");
      fInfoLine[7]->SetText(" ");
      fInfoLine[8]->SetText(" ");
      fInfoLine[9]->SetText(" ");
      fInfoLine[10]->SetText(" ");
      fInfoLine[11]->SetText(" ");
      fInfoLine[12]->SetText(" ");
      delete userGroup;
      Layout();
      Resize(GetDefaultSize());
      return;
   }
   // return if not a valid Proof session
   if (!fViewer->GetActDesc()->fProof ||
       !fViewer->GetActDesc()->fProof->IsValid())
       return;

   if (!fViewer->GetActDesc()->fProof->IsMaster()) {
      if (fViewer->GetActDesc()->fProof->IsParallel())
         sprintf(buf,"*** Connected to %s (parallel mode, %d workers) ***",
                fViewer->GetActDesc()->fProof->GetMaster(),
                fViewer->GetActDesc()->fProof->GetParallel());
      else
         sprintf(buf, "*** Connected to %s (sequential mode) ***",
                fViewer->GetActDesc()->fProof->GetMaster());
      fInfoLine[0]->SetText(buf);
      fInfoLine[1]->SetText("Port number : ");
      sprintf(buf, "%d", fViewer->GetActDesc()->fProof->GetPort());
      fInfoLine[2]->SetText(buf);
      fInfoLine[3]->SetText("User : ");
      sprintf(buf, "%s", fViewer->GetActDesc()->fProof->GetUser());
      fInfoLine[4]->SetText(buf);
      fInfoLine[5]->SetText("Client protocol version : ");
      sprintf(buf, "%d", fViewer->GetActDesc()->fProof->GetClientProtocol());
      fInfoLine[6]->SetText(buf);
      fInfoLine[7]->SetText("Remote protocol version : ");
      sprintf(buf, "%d", fViewer->GetActDesc()->fProof->GetRemoteProtocol());
      fInfoLine[8]->SetText(buf);
      fInfoLine[9]->SetText("Log level : ");
      sprintf(buf, "%d", fViewer->GetActDesc()->fProof->GetLogLevel());
      fInfoLine[10]->SetText(buf);
      fInfoLine[11]->SetText("Session unique tag : ");
      sprintf(buf, "%s", fViewer->GetActDesc()->fProof->IsValid() ?
         fViewer->GetActDesc()->fProof->GetSessionTag() : " ");
      fInfoLine[12]->SetText(buf);
   }
   else {
      if (fViewer->GetActDesc()->fProof->IsParallel())
         sprintf(buf,"*** Master server %s (parallel mode, %d workers) ***",
                fViewer->GetActDesc()->fProof->GetMaster(),
                fViewer->GetActDesc()->fProof->GetParallel());
      else
         sprintf(buf, "*** Master server %s (sequential mode) ***",
                fViewer->GetActDesc()->fProof->GetMaster());
      fInfoLine[0]->SetText(buf);
      fInfoLine[1]->SetText("Port number : ");
      sprintf(buf, "%d", fViewer->GetActDesc()->fProof->GetPort());
      fInfoLine[2]->SetText(buf);
      fInfoLine[3]->SetText("User : ");
      sprintf(buf, "%s", fViewer->GetActDesc()->fProof->GetUser());
      fInfoLine[4]->SetText(buf);
      fInfoLine[5]->SetText("Protocol version : ");
      sprintf(buf, "%d", fViewer->GetActDesc()->fProof->GetClientProtocol());
      fInfoLine[6]->SetText(buf);
      fInfoLine[7]->SetText("Image name : ");
      sprintf(buf, "%s",fViewer->GetActDesc()->fProof->GetImage());
      fInfoLine[8]->SetText(buf);
      fInfoLine[9]->SetText("Config directory : ");
      sprintf(buf, "%s", fViewer->GetActDesc()->fProof->GetConfDir());
      fInfoLine[10]->SetText(buf);
      fInfoLine[11]->SetText("Config file : ");
      sprintf(buf, "%s", fViewer->GetActDesc()->fProof->GetConfFile());
      fInfoLine[12]->SetText(buf);
   }
   Layout();
   Resize(GetDefaultSize());
}

//______________________________________________________________________________
void TSessionFrame::OnApplyLogLevel()
{
   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, set log level
   if (fViewer->GetActDesc()->fProof &&
      fViewer->GetActDesc()->fProof->IsValid()) {
      fViewer->GetActDesc()->fLogLevel = fLogLevel->GetIntNumber();
      fViewer->GetActDesc()->fProof->SetLogLevel(fViewer->GetActDesc()->fLogLevel);
   }
}

//______________________________________________________________________________
void TSessionFrame::OnApplyParallel()
{
   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, set parallel slaves
   if (fViewer->GetActDesc()->fProof &&
      fViewer->GetActDesc()->fProof->IsValid()) {
      Int_t nodes = atoi(fTxtParallel->GetText());
      fViewer->GetActDesc()->fProof->SetParallel(nodes);
   }
}

//______________________________________________________________________________
void TSessionFrame::OnMultipleSelection(Bool_t on)
{
   fLBPackages->SetMultipleSelections(on);
}

//______________________________________________________________________________
void TSessionFrame::OnUploadPackages()
{
   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, upload packages
   if (fViewer->GetActDesc()->fProof &&
      fViewer->GetActDesc()->fProof->IsValid()) {
      TObject *obj;
      TList selected;
      fLBPackages->GetSelectedEntries(&selected);
      TIter next(&selected);
      while ((obj = next())) {
         TString name = obj->GetTitle();
         if (fViewer->GetActDesc()->fProof->UploadPackage(name) != 0)
            Error("Submit", "Upload package failed");
      }
   }
}

//______________________________________________________________________________
void TSessionFrame::OnEnablePackages()
{
   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, enable packages
   if (fViewer->GetActDesc()->fProof &&
      fViewer->GetActDesc()->fProof->IsValid()) {
      TObject *obj;
      TList selected;
      fLBPackages->GetSelectedEntries(&selected);
      TIter next(&selected);
      while ((obj = next())) {
         TString name = obj->GetTitle();
         if (fViewer->GetActDesc()->fProof->EnablePackage(name) != 0)
            Error("Submit", "Enable package failed");
      }
   }
}

//______________________________________________________________________________
void TSessionFrame::OnDisablePackages()
{
   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, disable (clear) packages
   if (fViewer->GetActDesc()->fProof &&
      fViewer->GetActDesc()->fProof->IsValid()) {
      TObject *obj;
      TList selected;
      fLBPackages->GetSelectedEntries(&selected);
      TIter next(&selected);
      while ((obj = next())) {
         TString name = obj->GetTitle();
         if (fViewer->GetActDesc()->fProof->ClearPackage(name) != 0)
            Error("Submit", "Clear package failed");
      }
   }
}

//______________________________________________________________________________
void TSessionFrame::OnClearPackages()
{
   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, clear packages
   if (fViewer->GetActDesc()->fProof &&
      fViewer->GetActDesc()->fProof->IsValid()) {
      if (fViewer->GetActDesc()->fProof->ClearPackages() != 0)
         Error("Submit", "Clear packages failed");
   }
}

//______________________________________________________________________________
void TSessionFrame::OnBtnAddClicked()
{
   if (fViewer->IsBusy())
      return;
   if (fPackages == 0) {
      fPackages = new TList();
   }
   TGFileInfo fi;
   fi.fFileTypes = pkgtypes;
   new TGFileDialog(fClient->GetRoot(), fViewer, kFDOpen, &fi);
   if (!fi.fFilename) return;
   TPackageDescription *package = new TPackageDescription;
   package->fName = fi.fFilename;
   package->fId   = fPackages->GetEntries();
   fPackages->Add((TObject *)package);
   fLBPackages->AddEntry(package->fName, package->fId);
   fLBPackages->Layout();
   fClient->NeedRedraw(fLBPackages);
}

//______________________________________________________________________________
void TSessionFrame::OnBtnRemoveClicked()
{
   TPackageDescription *package;
   Int_t pos = fLBPackages->GetSelected();
   fLBPackages->RemoveEntries(0, fLBPackages->GetNumberOfEntries());
   fPackages->Remove(fPackages->At(pos));
   Int_t id = 0;
   TIter next(fPackages);
   while ((package = (TPackageDescription *)next())) {
      package->fId = id;
      id++;
      fLBPackages->AddEntry(package->fName, package->fId);
   }
   fLBPackages->Layout();
   fClient->NeedRedraw(fLBPackages);
}

//______________________________________________________________________________
void TSessionFrame::OnBtnUpClicked()
{
   TPackageDescription *package;
   Int_t pos = fLBPackages->GetSelected();
   if (pos <= 0) return;
   fLBPackages->RemoveEntries(0, fLBPackages->GetNumberOfEntries());
   package = (TPackageDescription *)fPackages->At(pos);
   fPackages->Remove(fPackages->At(pos));
   package->fId -= 1;
   fPackages->AddAt(package, package->fId);
   Int_t id = 0;
   TIter next(fPackages);
   while ((package = (TPackageDescription *)next())) {
      package->fId = id;
      id++;
      fLBPackages->AddEntry(package->fName, package->fId);
   }
   fLBPackages->Select(pos-1);
   fLBPackages->Layout();
   fClient->NeedRedraw(fLBPackages);
}

//______________________________________________________________________________
void TSessionFrame::OnBtnDownClicked()
{
   TPackageDescription *package;
   Int_t pos = fLBPackages->GetSelected();
   if (pos == -1 || pos == fPackages->GetEntries()-1) return;
   fLBPackages->RemoveEntries(0, fLBPackages->GetNumberOfEntries());
   package = (TPackageDescription *)fPackages->At(pos);
   fPackages->Remove(fPackages->At(pos));
   package->fId += 1;
   fPackages->AddAt(package, package->fId);
   Int_t id = 0;
   TIter next(fPackages);
   while ((package = (TPackageDescription *)next())) {
      package->fId = id;
      id++;
      fLBPackages->AddEntry(package->fName, package->fId);
   }
   fLBPackages->Select(pos+1);
   fLBPackages->Layout();
   fClient->NeedRedraw(fLBPackages);
}

//______________________________________________________________________________
void TSessionFrame::OnBtnDisconnectClicked()
{
   // Disconnect from current Proof session

   // if local session, do nothing
   if (fViewer->GetActDesc()->fLocal) return;
   // if valid Proof session, disconnect (close)
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid())
      fViewer->GetActDesc()->fProof->Close();
   // reset connected flag
   fViewer->GetActDesc()->fConnected = kFALSE;
   // disable animation timer
   fViewer->DisableTimer();
   // change list tree item picture to disconnected pixmap
   TGListTreeItem *item = fViewer->GetSessionHierarchy()->FindChildByData(
                          fViewer->GetSessionItem(), fViewer->GetActDesc());
   item->SetPictures(fViewer->GetProofDisconPict(),
                     fViewer->GetProofDisconPict());
   // update viewer
   fViewer->OnListTreeClicked(fViewer->GetSessionItem(), 1, 0, 0);
   fClient->NeedRedraw(fViewer->GetSessionHierarchy());
   fViewer->GetStatusBar()->SetText("", 1);
}

//______________________________________________________________________________
void TSessionFrame::OnBtnShowLogClicked()
{
   // Show session log

   fViewer->ShowLog(0);
}

//______________________________________________________________________________
void TSessionFrame::OnBtnNewQueryClicked()
{
   // Just call "New Query" Dialog

   TNewQueryDlg *dlg = new TNewQueryDlg(fViewer, 350, 310);
   dlg->Popup();
}

//______________________________________________________________________________
void TSessionFrame::OnBtnGetQueriesClicked()
{
   // Get list of queries from current Proof server

   TList *lqueries = 0;
   TQueryResult *query = 0;
   TQueryDescription *newquery = 0, *lquery = 0;
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      lqueries = fViewer->GetActDesc()->fProof->GetListOfQueries("A");
   }
   if (lqueries) {
      TIter nextp(lqueries);
      // loop over list of queries received from Proof server
      while ((query = (TQueryResult *)nextp())) {
         // create new query description
         newquery = new TQueryDescription();
         newquery->fReference       = Form("%s:%s", query->GetTitle(),
                                      query->GetName());
         // check in our tree if it is already there
         TGListTreeItem *item =
            fViewer->GetSessionHierarchy()->FindChildByData(
                     fViewer->GetSessionItem(), fViewer->GetActDesc());
         // if already there, skip
         if (fViewer->GetSessionHierarchy()->FindChildByName(item,
            newquery->fReference.Data()))
            continue;
         // check also in our query description list
         Bool_t found = kFALSE;
         TIter nextp(fViewer->GetActDesc()->fQueries);
         while ((lquery = (TQueryDescription *)nextp())) {
            if (lquery->fReference.CompareTo(newquery->fReference) == 0) {
               found = kTRUE;
               break;
            }
         }
         if (found) continue;
         // build new query description with infos from Proof
         newquery->fStatus = query->IsFinalized() ?
               TQueryDescription::kSessionQueryFinalized :
               (TQueryDescription::ESessionQueryStatus)query->GetStatus();
         newquery->fSelectorString  = query->GetSelecImp()->GetName();
         newquery->fQueryName       = Form("%s:%s", query->GetTitle(),
                                           query->GetName());
         newquery->fOptions         = query->GetOptions();
         newquery->fEventList       = "";
         newquery->fParFile         = "";
         newquery->fNbFiles         = 0;
         newquery->fNoEntries       = query->GetEntries();
         newquery->fFirstEntry      = query->GetFirst();
         newquery->fResult          = query;
         fViewer->GetActDesc()->fQueries->Add((TObject *)newquery);
         TGListTreeItem *item2 = fViewer->GetSessionHierarchy()->AddItem(item,
                  newquery->fQueryName, fViewer->GetQueryConPict(),
                  fViewer->GetQueryConPict());
         item2->SetUserData(newquery);
         if (query->GetInputList())
            fViewer->GetSessionHierarchy()->AddItem(item2, "InputList");
         if (query->GetInputList())
            fViewer->GetSessionHierarchy()->AddItem(item2, "OutputList");
      }
   }
   // at the end, update list tree
   fClient->NeedRedraw(fViewer->GetSessionHierarchy());
}

//______________________________________________________________________________
void TSessionFrame::OnCommandLine()
{
   // command line handling

   // get command string
   const char *cmd = fCommandTxt->GetText();
   char opt[2];
   // form temporary file path
   TString pathtmp = Form("%s/%s", gSystem->TempDirectory(),
                          kSession_RedirectCmd);
   // if check box "clear view" is checked, open temp file in write mode
   // (overwrite), in append mode otherwise.
   if (fClearCheck->IsOn())
      sprintf(opt, "w");
   else
      sprintf(opt, "a");

   // if valid Proof session, pass the command to Proof
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      // redirect stdout/stderr to temp file
      if (gSystem->RedirectOutput(pathtmp.Data(), opt) != 0) {
         Error("ShowStatus", "stdout/stderr redirection failed; skipping");
         return;
      }
      // execute command line
      fViewer->GetActDesc()->fProof->Exec(cmd);
      // restore back stdout/stderr
      if (gSystem->RedirectOutput(0) != 0) {
         Error("ShowStatus", "stdout/stderr retore failed; skipping");
         return;
      }
      // if check box "clear view" is checked, clear text view
      if (fClearCheck->IsOn())
         fInfoTextView->Clear();
      // load (display) temp file in text view
      fInfoTextView->LoadFile(pathtmp.Data());
      // set focus to "command line" text entry
      fCommandTxt->SetFocus();
   }
   else {
      // if no Proof session, or Proof session not valid,
      // lets execute command line by TApplication

      // redirect stdout/stderr to temp file
      if (gSystem->RedirectOutput(pathtmp.Data(), opt) != 0) {
         Error("ShowStatus", "stdout/stderr redirection failed; skipping");
      }
      // execute command line
      gApplication->ProcessLine(cmd);
      // restore back stdout/stderr
      if (gSystem->RedirectOutput(0) != 0) {
         Error("ShowStatus", "stdout/stderr retore failed; skipping");
      }
      // if check box "clear view" is checked, clear text view
      if (fClearCheck->IsOn())
         fInfoTextView->Clear();
      // load (display) temp file in text view
      fInfoTextView->LoadFile(pathtmp.Data());
      // set focus to "command line" text entry
      fCommandTxt->SetFocus();
   }
   // display bottom of text view
   fInfoTextView->ShowBottom();
}

////////////////////////////////////////////////////////////////////////////////
// Query Frame

//______________________________________________________________________________
TSessionQueryFrame::TSessionQueryFrame(TGWindow* p, Int_t w, Int_t h) :
   TGCompositeFrame(p, w, h)
{
   // Constructor
}

//______________________________________________________________________________
TSessionQueryFrame::~TSessionQueryFrame()
{
   // Destructor
   Cleanup();
}

//______________________________________________________________________________
void TSessionQueryFrame::Build(TSessionViewer *gui)
{
   // build query informations frame

   SetLayoutManager(new TGVerticalLayout(this));
   SetCleanup(kDeepCleanup);
   fFirst = fEntries = fPrevTotal = 0;
   fPrevProcessed = 0;
   fStatus    = kRunning;
   fViewer  = gui;

   // main query tab
   fTab = new TGTab(this, 200, 200);
   AddFrame(fTab, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
            kLHintsExpandY, 2, 2, 2, 2));

   // add "Status" tab element
   TGCompositeFrame *tf = fTab->AddTab("Status");
   fFB = new TGCompositeFrame(tf, 100, 100, kVerticalFrame);
   tf->AddFrame(fFB, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                kLHintsExpandX | kLHintsExpandY));

   // new frame containing control buttons and feedback histos canvas
   TGCompositeFrame* frmcanvas = new TGHorizontalFrame(fFB, 350, 100);
   // control buttons frame
   TGCompositeFrame* frmBut2 = new TGVerticalFrame(frmcanvas, 150, 100);
   fBtnSubmit = new TGTextButton(frmBut2, "        Submit        ");
   frmBut2->AddFrame(fBtnSubmit,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fBtnStop = new TGTextButton(frmBut2, "Stop");
   frmBut2->AddFrame(fBtnStop,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   fBtnAbort = new TGTextButton(frmBut2, "Abort");
   frmBut2->AddFrame(fBtnAbort,new TGLayoutHints(kLHintsCenterY | kLHintsLeft |
            kLHintsExpandX, 5, 5, 5, 5));
   frmcanvas->AddFrame(frmBut2, new TGLayoutHints(kLHintsLeft | kLHintsCenterY |
            kLHintsExpandY));
   // feedback histos embedded canvas
   fECanvas = new TRootEmbeddedCanvas("fECanvas", frmcanvas, 400, 150);
   fStatsCanvas = fECanvas->GetCanvas();
   fStatsCanvas->SetFillColor(10);
   fStatsCanvas->SetBorderMode(0);
   frmcanvas->AddFrame(fECanvas, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
            4, 4, 4, 4));
   fFB->AddFrame(frmcanvas, new TGLayoutHints(kLHintsLeft | kLHintsTop |
                 kLHintsExpandX | kLHintsExpandY));

   // progress infos label
   fLabInfos = new TGLabel(fFB, "                                  ");
   fFB->AddFrame(fLabInfos, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));
   // progress status label
   fLabStatus = new TGLabel(fFB, "                                  ");
   fFB->AddFrame(fLabStatus, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));

   //progress bar
   frmProg = new TGHProgressBar(fFB, TGProgressBar::kFancy, 350 - 20);
   frmProg->ShowPosition();
   frmProg->SetBarColor("green");
   fFB->AddFrame(frmProg, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));
   // total progress infos
   fFB->AddFrame(fTotal = new TGLabel(fFB,
      " Estimated time left : 00:00:00 (--- events of --- processed) "),
             new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));
   // progress rate infos
   fFB->AddFrame(fRate = new TGLabel(fFB,
      " Processing Rate : -- events/sec    "),
            new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));

   // add "Results" tab element
   tf = fTab->AddTab("Results");
   fFC = new TGCompositeFrame(tf, 100, 100, kVerticalFrame);
   tf->AddFrame(fFC, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                kLHintsExpandX | kLHintsExpandY));
   // query result (header) information text view
   fInfoTextView = new TGTextView(fFC, 330, 185, "", kSunkenFrame |
                                  kDoubleBorder);
   fFC->AddFrame(fInfoTextView, new TGLayoutHints(kLHintsTop | kLHintsLeft |
            kLHintsExpandY | kLHintsExpandX, 5, 5, 10, 10));

   // add "Retrieve", "Finalize" and "Show Log" buttons
   TGCompositeFrame* frmBut3 = new TGHorizontalFrame(fFC, 350, 100);
   fBtnRetrieve = new TGTextButton(frmBut3, "Retrieve");
   frmBut3->AddFrame(fBtnRetrieve,new TGLayoutHints(kLHintsTop | kLHintsLeft |
            kLHintsExpandX, 5, 5, 10, 10));
   fBtnFinalize = new TGTextButton(frmBut3, "Finalize");
   frmBut3->AddFrame(fBtnFinalize,new TGLayoutHints(kLHintsTop | kLHintsLeft |
            kLHintsExpandX, 5, 5, 10, 10));
   fBtnShowLog = new TGTextButton(frmBut3, "Show Log");
   frmBut3->AddFrame(fBtnShowLog,new TGLayoutHints(kLHintsTop | kLHintsLeft |
            kLHintsExpandX, 5, 5, 10, 10));
   fFC->AddFrame(frmBut3, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX));

   // connect button actions to functions
   fBtnSubmit->Connect("Clicked()", "TSessionQueryFrame", this,
                       "OnBtnSubmit()");
   fBtnFinalize->Connect("Clicked()", "TSessionQueryFrame", this,
                         "OnBtnFinalize()");
   fBtnStop->Connect("Clicked()", "TSessionQueryFrame", this,
                     "OnBtnStop()");
   fBtnAbort->Connect("Clicked()", "TSessionQueryFrame", this,
                      "OnBtnAbort()");
   fBtnShowLog->Connect("Clicked()", "TSessionQueryFrame", this,
                        "OnBtnShowLog()");
   fBtnRetrieve->Connect("Clicked()", "TSessionQueryFrame", this,
                         "OnBtnRetrieve()");
   Resize(350, 310);
}

//______________________________________________________________________________
void TSessionQueryFrame::Feedback(TList *objs)
{
   // Feedback function connected to Feedback signal
   // Used to update feedback histograms

   // if no actual session, just return
   if (!fViewer->GetActDesc()->fProof)
      return;
   TVirtualProof *sender = dynamic_cast<TVirtualProof*>((TQObject*)gTQSender);
   // if Proof sender match actual session one, update feedback histos
   if (sender && (sender == fViewer->GetActDesc()->fProof))
      UpdateHistos(objs);
}

//______________________________________________________________________________
void TSessionQueryFrame::UpdateHistos(TList *objs)
{
   // Update feedback histograms
   TVirtualPad *save = gPad;
   TObject *o;
   Int_t pos = 1;
   TIter next(objs);
   // loop over object list
   while( (o = next()) ) {
      TString name = o->GetName();
      gPad->SetEditable(kTRUE);
      Int_t i = 0;
      // loop over feedback histo list
      while (kFeedbackHistos[i]) {
         // check if user has selected this histogram in the option menu
         if (fViewer->GetCascadeMenu()->IsEntryChecked(41+i) &&
               name.Contains(kFeedbackHistos[i])) {
            // cd to correct pad and draw histo
            fStatsCanvas->cd(pos);
            if (TH1 *h = dynamic_cast<TH1*>(o)) {
               h->SetStats(0);
               h->SetBarWidth(0.75);
               h->SetBarOffset(0.125);
               h->SetFillColor(9);
               h->DrawCopy("bar");
            }
            else if (TH2 *h2 = dynamic_cast<TH2*>(o)) {
               h2->DrawCopy();
            }
            pos++;
         }
         i++;
      }
      // update canvas
      fStatsCanvas->Modified();
      fStatsCanvas->Update();
   }
   if (save != 0) {
      save->cd();
   } else {
      gPad = 0;
   }
}

//______________________________________________________________________________
void TSessionQueryFrame::Progress(Long64_t total, Long64_t processed)
{
   // Update progress bar and status labels.

   // if no actual session, just return
   if (!fViewer->GetActDesc()->fProof)
      return;
   // if Proof sender does't match actual session one, return
   TVirtualProof *sender = dynamic_cast<TVirtualProof*>((TQObject*)gTQSender);
   if (!sender || (sender != fViewer->GetActDesc()->fProof))
      return;
   static const char *cproc[] = { "running", "done", "STOPPED", "ABORTED" };

   if (total < 0)
      total = fPrevTotal;
   else
      fPrevTotal = total;

   // if no change since last call, just return
   if (fPrevProcessed == processed)
      return;
   char buf[256];

   // Update informations at first call
   if (fEntries != total) {
      sprintf(buf, "PROOF cluster : \"%s\" - %d worker nodes",
           fViewer->GetActDesc()->fProof->GetMaster(),
           fViewer->GetActDesc()->fProof->GetParallel());
      fLabInfos->SetText(buf);

      fEntries = total;
      sprintf(buf, " %d files, %lld events, starting event %lld",
              fFiles, fEntries, fFirst);
      fLabStatus->SetText(buf);
   }

   // compute progress bar position and update
   Float_t pos = Float_t(Double_t(processed * 100)/Double_t(total));
   frmProg->SetPosition(pos);
   // if 100%, stop animation and set icon to "connected"
   if (pos >= 100.0) {
      fViewer->SetChangePic(kFALSE);
      fViewer->ChangeRightLogo("monitor01.xpm");
   }

   // get current time
   fEndTime = gSystem->Now();
   TTime tdiff = fEndTime - fStartTime;
   Float_t eta = 0;
   if (processed)
      eta = ((Float_t)((Long_t)tdiff)*total/Float_t(processed) -
            Long_t(tdiff))/1000.;

   if (processed == total) {
      // finished
      sprintf(buf, " Processed : %lld events in %.1f sec", total, Long_t(tdiff)/1000.);
      fTotal->SetText(buf);
   } else {
      // update status infos
      if (fStatus > kDone) {
         sprintf(buf, " Estimated time left : %.1f sec (%lld events of %lld processed) - %s  ",
                      eta, processed, total, cproc[fStatus]);
      } else {
         sprintf(buf, " Estimated time left : %.1f sec (%lld events of %lld processed)        ",
                      eta, processed, total);
      }
      fTotal->SetText(buf);
      sprintf(buf, " Processing Rate : %.1f events/sec   ",
              Float_t(processed)/Long_t(tdiff)*1000.);
      fRate->SetText(buf);
   }
   fPrevProcessed = processed;

   fFB->Layout();
}

//______________________________________________________________________________
void TSessionQueryFrame::IndicateStop(Bool_t aborted)
{
   // Indicate that Cancel or Stop was clicked.

   if (aborted == kTRUE) {
      // Aborted
      frmProg->SetBarColor("red");
      fStatus = kAborted;
   }
   else {
      // Stopped
      frmProg->SetBarColor("yellow");
      fStatus = kStopped;
   }
   // disconnect progress related signals
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      fViewer->GetActDesc()->fProof->Disconnect("Progress(Long64_t,Long64_t)",
                                          this, "Progress(Long64_t,Long64_t)");
      fViewer->GetActDesc()->fProof->Disconnect("StopProcess(Bool_t)", this,
                                                "IndicateStop(Bool_t)");
   }
}

//______________________________________________________________________________
void TSessionQueryFrame::ResetProgressDialog(const char * /*selector*/, Int_t files,
                                        Long64_t first, Long64_t entries)
{
   // Reset Progress frame information fields

   char buf[256];
   fFiles         = files > 0 ? files : 0;
   fFirst         = first;
   fEntries       = entries;
   fPrevProcessed = 0;
   fPrevTotal     = 0;
   fStatus        = kRunning;

   frmProg->SetBarColor("green");
   frmProg->Reset();

   sprintf(buf, "%0d files, %0lld events, starting event %0lld",
           fFiles > 0 ? fFiles : 0, fEntries > 0 ? fEntries : 0,
           fFirst >= 0 ? fFirst : 0);
   fLabStatus->SetText(buf);
   // Reconnect the slots
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      fViewer->GetActDesc()->fProof->Connect("Progress(Long64_t,Long64_t)",
                     "TSessionQueryFrame", this, "Progress(Long64_t,Long64_t)");
      fViewer->GetActDesc()->fProof->Connect("StopProcess(Bool_t)",
                     "TSessionQueryFrame", this, "IndicateStop(Bool_t)");
      sprintf(buf, "PROOF cluster : \"%s\" - %d worker nodes",
              fViewer->GetActDesc()->fProof->GetMaster(),
              fViewer->GetActDesc()->fProof->GetParallel());
      fLabInfos->SetText(buf);
   }
   else {
      fLabInfos->SetText("");
   }
   fFB->Layout();
}

//______________________________________________________________________________
void TSessionQueryFrame::OnBtnFinalize()
{
   // Finalize query

   // check if Proof is valid
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      gPad->SetEditable(kFALSE);
      TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
      if (!item) return;
      TObject *obj = (TObject *)item->GetUserData();
      if (obj->IsA() == TQueryDescription::Class()) {
         // as it can take time, set watch cursor
         gVirtualX->SetCursor(GetId(),gVirtualX->CreateCursor(kWatch));
         TQueryDescription *query = (TQueryDescription *)obj;
         fViewer->GetActDesc()->fProof->Finalize(query->fReference);
         UpdateButtons(query);
         // restore cursor
         gVirtualX->SetCursor(GetId(), 0);
      }
   }
}

//______________________________________________________________________________
void TSessionQueryFrame::OnBtnStop()
{
   // stop processing query

   // check for proof validity
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      fViewer->GetActDesc()->fProof->StopProcess(kFALSE);
   }
   // stop icon animation and set connected icon
   fViewer->ChangeRightLogo("monitor01.xpm");
   fViewer->SetChangePic(kFALSE);
}

//______________________________________________________________________________
void TSessionQueryFrame::OnBtnShowLog()
{
   // Show query log

   TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
   if (!item) return;
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class())
      return;
   TQueryDescription *query = (TQueryDescription *)obj;
   fViewer->ShowLog(query->fReference.Data());
}

//______________________________________________________________________________
void TSessionQueryFrame::OnBtnRetrieve()
{
   // Retrieve query

   // check for proof validity
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
      if (!item) return;
      TObject *obj = (TObject *)item->GetUserData();
      if (obj->IsA() == TQueryDescription::Class()) {
         // as it can take time, set watch cursor
         gVirtualX->SetCursor(GetId(), gVirtualX->CreateCursor(kWatch));
         TQueryDescription *query = (TQueryDescription *)obj;
         fViewer->GetActDesc()->fProof->Retrieve(query->fReference);
         // restore cursor
         gVirtualX->SetCursor(GetId(), 0);
      }
   }
}

//______________________________________________________________________________
void TSessionQueryFrame::OnBtnAbort()
{
   // Abort processing query

   // check for proof validity
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      fViewer->GetActDesc()->fProof->StopProcess(kTRUE);
   }
   // stop icon animation and set connected icon
   fViewer->ChangeRightLogo("monitor01.xpm");
   fViewer->SetChangePic(kFALSE);
}

//______________________________________________________________________________
void TSessionQueryFrame::OnBtnSubmit()
{
   // Submit query

   Long64_t id = 0;
   TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
   if (!item) return;
   // retrieve query description attached to list tree item
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class())
      return;
   TQueryDescription *newquery = (TQueryDescription *)obj;
   // reset progress informations
   ResetProgressDialog(newquery->fSelectorString,
         newquery->fNbFiles, newquery->fFirstEntry, newquery->fNoEntries);
   // set start time
   SetStartTime(gSystem->Now());
   fViewer->GetActDesc()->fNbHistos = 0;
   // check for proof validity
   if (fViewer->GetActDesc()->fProof &&
       fViewer->GetActDesc()->fProof->IsValid()) {
      // set query description status to submitted
      newquery->fStatus = TQueryDescription::kSessionQuerySubmitted;
      // if feedback option selected
      if (fViewer->GetOptionsMenu()->IsEntryChecked(kOptionsFeedback)) {
         Int_t i = 0;
         // browse list of feedback histos and check user's selected ones
         while (kFeedbackHistos[i]) {
            if (fViewer->GetCascadeMenu()->IsEntryChecked(41+i)) {
               fViewer->GetActDesc()->fProof->AddFeedback(kFeedbackHistos[i]);
               fViewer->GetActDesc()->fNbHistos++;
            }
            i++;
         }
         // connect feedback signal
         fViewer->GetActDesc()->fProof->Connect("Feedback(TList *objs)",
                           "TSessionQueryFrame", fViewer->GetQueryFrame(),
                           "Feedback(TList *objs)");
         gROOT->Time();
      }
      else {
         // if feedback option not selected, clear Proof's feedback option
         fViewer->GetActDesc()->fProof->ClearFeedback();
      }
      // set current proof session
      fViewer->GetActDesc()->fProof->cd();
      // check if parameter file has been specified
      if (newquery->fParFile.Length() > 1) {
         const char *packname = newquery->fParFile.Data();
         // upload parameter file
         if (fViewer->GetActDesc()->fProof->UploadPackage(packname) != 0)
            Error("Submit", "Upload package failed");
         // enable parameter file
         if (fViewer->GetActDesc()->fProof->EnablePackage(packname) != 0)
            Error("Submit", "Enable package failed");
      }
      if (newquery->fChain) {
         // Quick FIX just for the demo. Creating a new TDSet causes a memory leak.
         if (newquery->fChain->IsA() == TChain::Class()) {
            // TChain case
            newquery->fStatus = TQueryDescription::kSessionQuerySubmitted;
            TDSet* s = ((TChain *)newquery->fChain)->MakeTDSet();
            gProof = fViewer->GetActDesc()->fProof;
            id = s->Process(newquery->fSelectorString,
                    newquery->fOptions,
                    newquery->fNoEntries > 0 ? newquery->fNoEntries : 1234567890,
                    newquery->fFirstEntry);
//            ((TChain *)newquery->fChain)->SetProof(fViewer->GetActDesc()->fProof);
//            id = ((TChain *)newquery->fChain)->Process(newquery->fSelectorString,
//                    newquery->fOptions,
//                    newquery->fNoEntries > 0 ? newquery->fNoEntries : 1234567890,
//                    newquery->fFirstEntry);
         }
         else if (newquery->fChain->IsA() == TDSet::Class()) {
            // TDSet case
            id = ((TDSet *)newquery->fChain)->Process(newquery->fSelectorString,
                    newquery->fOptions,
                    newquery->fNoEntries,
                    newquery->fFirstEntry);
         }
      }
      // set query reference id to unique identifier
      newquery->fReference= Form("session-%s:q%d",
            fViewer->GetActDesc()->fProof->GetSessionTag(), id);
      // start icon animation
      fViewer->SetChangePic(kTRUE);
   }
   else if (fViewer->GetActDesc()->fLocal) { // local session case
      // if feedback option selected
      if (fViewer->GetOptionsMenu()->IsEntryChecked(kOptionsFeedback)) {
         Int_t i = 0;
         // browse list of feedback histos and check user's selected ones
         while (kFeedbackHistos[i]) {
            if (fViewer->GetCascadeMenu()->IsEntryChecked(41+i)) {
               fViewer->GetActDesc()->fNbHistos++;
            }
            i++;
         }
      }
      if (newquery->fChain) {
         if (newquery->fChain->IsA() == TChain::Class()) {
            // TChain case
            id = ((TChain *)newquery->fChain)->Process(newquery->fSelectorString,
                            newquery->fOptions,
                            newquery->fNoEntries > 0 ? newquery->fNoEntries : 1234567890,
                            newquery->fFirstEntry);
         }
         else if (newquery->fChain->IsA() == TDSet::Class()) {
            // TDSet case
            id = ((TDSet *)newquery->fChain)->Process(newquery->fSelectorString,
                                                      newquery->fOptions,
                                                      newquery->fNoEntries,
                                                      newquery->fFirstEntry);
         }
      }
      // set query reference id to unique identifier
      newquery->fReference = Form("local-session-%s:q%d", newquery->fQueryName.Data(), id);
   }
   // update buttons state
   UpdateButtons(newquery);
}

//______________________________________________________________________________
void TSessionQueryFrame::UpdateButtons(TQueryDescription *desc)
{
   // Update buttons state for the current query status

   TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
   if (!item) return;
   // retrieve query description attached to list tree item
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class())
      return;
   TQueryDescription *query = (TQueryDescription *)obj;
   if (desc != query) return;

   switch (desc->fStatus) {
      case TQueryDescription::kSessionQueryFromProof:
         fBtnSubmit->SetEnabled(kFALSE);
         fBtnFinalize->SetEnabled(kTRUE);
         fBtnStop->SetEnabled(kFALSE);
         fBtnAbort->SetEnabled(kFALSE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kTRUE);
         break;

      case TQueryDescription::kSessionQueryCompleted:
         fBtnSubmit->SetEnabled(kFALSE);
         fBtnFinalize->SetEnabled(kTRUE);
         if (desc->fResult && desc->fResult->IsFinalized())
            fBtnFinalize->SetEnabled(kFALSE);
         fBtnStop->SetEnabled(kFALSE);
         fBtnAbort->SetEnabled(kFALSE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kTRUE);
         break;

      case TQueryDescription::kSessionQueryCreated:
         fBtnSubmit->SetEnabled(kTRUE);
         fBtnFinalize->SetEnabled(kFALSE);
         fBtnStop->SetEnabled(kFALSE);
         fBtnAbort->SetEnabled(kFALSE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kFALSE);
         break;

      case TQueryDescription::kSessionQuerySubmitted:
         fBtnSubmit->SetEnabled(kFALSE);
         fBtnFinalize->SetEnabled(kFALSE);
         fBtnStop->SetEnabled(kTRUE);
         fBtnAbort->SetEnabled(kTRUE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kFALSE);
         break;

      case TQueryDescription::kSessionQueryRunning:
         fBtnSubmit->SetEnabled(kFALSE);
         fBtnFinalize->SetEnabled(kFALSE);
         fBtnStop->SetEnabled(kTRUE);
         fBtnAbort->SetEnabled(kTRUE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kFALSE);
         break;

      case TQueryDescription::kSessionQueryStopped:
         fBtnSubmit->SetEnabled(kFALSE);
         fBtnFinalize->SetEnabled(kTRUE);
         fBtnStop->SetEnabled(kFALSE);
         fBtnAbort->SetEnabled(kFALSE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kFALSE);
         break;

      case TQueryDescription::kSessionQueryAborted:
         fBtnSubmit->SetEnabled(kTRUE);
         fBtnFinalize->SetEnabled(kFALSE);
         fBtnStop->SetEnabled(kFALSE);
         fBtnAbort->SetEnabled(kFALSE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kFALSE);
         break;

      case TQueryDescription::kSessionQueryFinalized:
         fBtnSubmit->SetEnabled(kFALSE);
         fBtnFinalize->SetEnabled(kFALSE);
         fBtnStop->SetEnabled(kFALSE);
         fBtnAbort->SetEnabled(kFALSE);
         fBtnShowLog->SetEnabled(kTRUE);
         fBtnRetrieve->SetEnabled(kFALSE);
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void TSessionQueryFrame::UpdateInfos()
{
   // Update Query information (header) text view

   char buffer[8192];
   const char *qst[] = {"aborted  ", "submitted", "running  ",
                        "stopped  ", "completed"};

   fInfoTextView->Clear();
   if (!fViewer->GetActDesc()->fActQuery ||
       !fViewer->GetActDesc()->fActQuery->fResult) {
      return;
   }
   TQueryResult *result = fViewer->GetActDesc()->fActQuery->fResult;

   // Status label
   Int_t st = (result->GetStatus() > 0 && result->GetStatus() <=
               TQueryResult::kCompleted) ? result->GetStatus() : 0;

   Int_t qry = result->GetSeqNum();

   sprintf(buffer,"------------------------------------------------------\n");
   // Print header
   if (!result->IsDraw()) {
      const char *fin = result->IsFinalized() ? "finalized" : qst[st];
      const char *arc = result->IsArchived() ? "(A)" : "";
      sprintf(buffer,"%s Query No  : %d\n", buffer, qry);
      sprintf(buffer,"%s Ref       : \"%s:%s\"\n", buffer, result->GetTitle(),
              result->GetName());
      sprintf(buffer,"%s Selector  : %s\n", buffer,
              result->GetSelecImp()->GetTitle());
      sprintf(buffer,"%s Status    : %9s%s\n", buffer, fin, arc);
      sprintf(buffer,"%s------------------------------------------------------\n",
              buffer);
   } else {
      sprintf(buffer,"%s Query No  : %d\n", buffer, qry);
      sprintf(buffer,"%s Ref       : \"%s:%s\"\n", buffer, result->GetTitle(),
              result->GetName());
      sprintf(buffer,"%s Selector  : %s\n", buffer,
              result->GetSelecImp()->GetTitle());
      sprintf(buffer,"%s------------------------------------------------------\n",
              buffer);
   }

   // Time information
   Int_t elapsed = (Int_t)(result->GetEndTime().Convert() -
                           result->GetStartTime().Convert());
   sprintf(buffer,"%s Started   : %s\n",buffer,
           result->GetStartTime().AsString());
   sprintf(buffer,"%s Real time : %d sec (CPU time: %.1f sec)\n", buffer, elapsed,
           result->GetUsedCPU());

   // Number of events processed, rate, size
   Double_t rate = 0.0;
   if (result->GetEntries() > -1 && elapsed > 0)
      rate = result->GetEntries() / (Double_t)elapsed ;
   Float_t size = ((Float_t)result->GetBytes())/(1024*1024);
   sprintf(buffer,"%s Processed : %lld events (size: %.3f MBs)\n",buffer,
          result->GetEntries(), size);
   sprintf(buffer,"%s Rate      : %.1f evts/sec\n",buffer, rate);

   // Package information
   if (strlen(result->GetParList()) > 1) {
      sprintf(buffer,"%s Packages  :  %s\n",buffer, result->GetParList());
   }

   // Result information
   TString res = result->GetResultFile();
   if (!result->IsArchived()) {
      Int_t dq = res.Index("queries");
      if (dq > -1) {
         res.Remove(0,res.Index("queries"));
         res.Insert(0,"<PROOF_SandBox>/");
      }
      if (res.BeginsWith("-")) {
         res = (result->GetStatus() == TQueryResult::kAborted) ?
               "not available" : "sent to client";
      }
   }
   if (res.Length() > 1) {
      sprintf(buffer,"%s------------------------------------------------------\n",
              buffer);
      sprintf(buffer,"%s Results   : %s\n",buffer, res.Data());
   }

   if (result->GetOutputList() && result->GetOutputList()->GetSize() > 0) {
      sprintf(buffer,"%s Outlist   : %d objects\n",buffer,
              result->GetOutputList()->GetSize());
      sprintf(buffer,"%s------------------------------------------------------\n",
              buffer);
   }
   fInfoTextView->LoadBuffer(buffer);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Output frame

//______________________________________________________________________________
TSessionOutputFrame::TSessionOutputFrame(TGWindow* p, Int_t w, Int_t h) :
   TGCompositeFrame(p, w, h), fLVContainer(0)
{
   // constructor
}

//______________________________________________________________________________
TSessionOutputFrame::~TSessionOutputFrame()
{
   // destructor
   delete fLVContainer; // this container is inside the TGListView and is not
                        // deleted automatically
   Cleanup();
}

//______________________________________________________________________________
void TSessionOutputFrame::Build(TSessionViewer *gui)
{
   // build query output informations frame

   fViewer = gui;
   SetLayoutManager(new TGVerticalLayout(this));
   SetCleanup(kDeepCleanup);

   // Container of object TGListView
   TGListView *frmListView = new TGListView(this, 340, 190);
   fLVContainer = new TGLVContainer(frmListView, kSunkenFrame,
                  GetWhitePixel());
   fLVContainer->Associate(frmListView);
   fLVContainer->SetCleanup(kDeepCleanup);
   AddFrame(frmListView, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
            4, 4, 4, 4));

   frmListView->Connect("Clicked(TGLVEntry*, Int_t, Int_t, Int_t)",
      "TSessionOutputFrame", this,
      "OnElementClicked(TGLVEntry* ,Int_t, Int_t, Int_t)");
   frmListView->Connect("DoubleClicked(TGLVEntry*, Int_t, Int_t, Int_t)",
      "TSessionOutputFrame", this,
      "OnElementDblClicked(TGLVEntry* ,Int_t, Int_t, Int_t)");
}

//______________________________________________________________________________
void TSessionOutputFrame::OnElementClicked(TGLVEntry* entry, Int_t btn, Int_t x,
                                           Int_t y)
{
   // handle mouse clicks on list view items

   TObject *obj = (TObject *)entry->GetUserData();
   if ((obj) && (btn ==3)) {
      // if right button, popup context menu
      fViewer->GetContextMenu()->Popup(x, y, obj, (TBrowser *)0);
   }
}

//______________________________________________________________________________
void TSessionOutputFrame::OnElementDblClicked(TGLVEntry* entry, Int_t , Int_t, Int_t)
{
   // handle double-clicks on list view items

   char action[512];
   TString act;
   TObject *obj = (TObject *)entry->GetUserData();
   TString ext = obj->GetName();
   gPad->SetEditable(kFALSE);
   // check default action from root.mimes
   if (fClient->GetMimeTypeList()->GetAction(obj->IsA()->GetName(), action)) {
      act = Form("((%s*)0x%lx)%s", obj->IsA()->GetName(), (Long_t)obj, action);
      if (act[0] == '!') {
         act.Remove(0, 1);
         gSystem->Exec(act.Data());
      } else {
         // do not allow browse
         if (!act.Contains("Browse"))
            gROOT->ProcessLine(act.Data());
      }
   }
}

//______________________________________________________________________________
void TSessionOutputFrame::AddObject(TObject *obj)
{
   // add object to output list view

   TGLVEntry *item;
   if (obj) {
      item = new TGLVEntry(fLVContainer, obj->GetName(), obj->IsA()->GetName());
      item->SetUserData(obj);
      fLVContainer->AddItem(item);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Input Frame

//______________________________________________________________________________
TSessionInputFrame::TSessionInputFrame(TGWindow* p, Int_t w, Int_t h) :
   TGCompositeFrame(p, w, h), fLVContainer(0)
{
   // constructor
}

//______________________________________________________________________________
TSessionInputFrame::~TSessionInputFrame()
{
   delete fLVContainer; // this container is inside the TGListView and is not
                        // deleted automatically
   Cleanup();
}

//______________________________________________________________________________
void TSessionInputFrame::Build(TSessionViewer *gui)
{
   // build query input informations frame

   fViewer = gui;
   SetLayoutManager(new TGVerticalLayout(this));
   SetCleanup(kDeepCleanup);

   // Container of object TGListView
   TGListView *frmListView = new TGListView(this, 340, 190);
   fLVContainer = new TGLVContainer(frmListView, kSunkenFrame, GetWhitePixel());
   fLVContainer->Associate(frmListView);
   fLVContainer->SetCleanup(kDeepCleanup);
   AddFrame(frmListView, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
            4, 4, 4, 4));
}

//______________________________________________________________________________
void TSessionInputFrame::AddObject(TObject *obj)
{
   // add object to input list view

   TGLVEntry *item;
   if (obj) {
      item = new TGLVEntry(fLVContainer, obj->GetName(), obj->IsA()->GetName());
      item->SetUserData(obj);
      fLVContainer->AddItem(item);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Session Viewer Main Frame

//______________________________________________________________________________
TSessionViewer::TSessionViewer(const char *name, UInt_t w, UInt_t h) :
   TGMainFrame(gClient->GetRoot(), w, h), fSessionHierarchy(0), fSessionItem(0)
{
   // Main Session viewer constructor

   // only one session viewer allowed
   if (gSessionViewer)
      return;
   Build();
   SetWindowName(name);
   Resize(w, h);
   gSessionViewer = this;
}

//______________________________________________________________________________
TSessionViewer::TSessionViewer(const char *name, Int_t x, Int_t y, UInt_t w,
                              UInt_t h) : TGMainFrame(gClient->GetRoot(), w, h),
                              fSessionHierarchy(0), fSessionItem(0)
{
   // Main Session viewer constructor

   // only one session viewer allowed
   if (gSessionViewer)
      return;
   Build();
   SetWindowName(name);
   Move(x, y);
   Resize(w, h);
   gSessionViewer = this;
}

//______________________________________________________________________________
void TSessionViewer::Build()
{
   // build main session viewer frame and subframes

   char line[120];
   fActDesc = 0;
   fLogWindow = 0;
   fBusy = kFALSE;
   SetCleanup(kDeepCleanup);
   // set minimun size
   SetWMSizeHints(400 + 200, 310+50, 2000, 1000, 1, 1);

   // collect icons
   fLocal = fClient->GetPicture("local_session.xpm");
   fProofCon = fClient->GetPicture("proof_connected.xpm");
   fProofDiscon = fClient->GetPicture("proof_disconnected.xpm");
   fQueryCon = fClient->GetPicture("query_connected.xpm");
   fQueryDiscon = fClient->GetPicture("query_disconnected.xpm");
   fBaseIcon = fClient->GetPicture("proof_base.xpm");

   //--- File menu
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&Load Library...", kFileLoadLibrary);
   fFileMenu->AddEntry("&Close Viewer",    kFileCloseViewer);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Quit ROOT",       kFileQuit);

   //--- Session menu
   fSessionMenu = new TGPopupMenu(gClient->GetRoot());
   fSessionMenu->AddEntry("&Connect...", kSessionConnect);
   fSessionMenu->AddEntry("&Disconnect", kSessionDisconnect);
   fSessionMenu->AddEntry("&Show status",kSessionShowStatus);
   fSessionMenu->AddSeparator();
   fSessionMenu->AddEntry("&Cleanup", kSessionCleanup);

   //--- Query menu
   fQueryMenu = new TGPopupMenu(gClient->GetRoot());
   fQueryMenu->AddEntry("&New...", kQueryNew);
   fQueryMenu->AddEntry("&Edit", kQueryEdit);
   fQueryMenu->AddEntry("&Submit", kQuerySubmit);
   fQueryMenu->AddSeparator();
   fQueryMenu->AddEntry("Start &Viewer", kQueryStartViewer);
   fQueryMenu->AddSeparator();
   fQueryMenu->AddEntry("&Delete", kQueryDelete);

   fCascadeMenu = new TGPopupMenu(fClient->GetRoot());
   Int_t i = 0;
   while (kFeedbackHistos[i]) {
      fCascadeMenu->AddEntry(kFeedbackHistos[i], 41+i);
      i++;
   }
   fCascadeMenu->AddEntry("User defined...", 50);
   // disable it for now (until implemented)
   fCascadeMenu->DisableEntry(50);

   //--- Options menu
   fOptionsMenu = new TGPopupMenu(fClient->GetRoot());
   fOptionsMenu->AddLabel("Performance Monitoring");
   fOptionsMenu->AddSeparator();
   fOptionsMenu->AddEntry("Master &Histos", kOptionsStatsHist);
   fOptionsMenu->AddEntry("&Master Events", kOptionsStatsTrace);
   fOptionsMenu->AddEntry("&Worker Events", kOptionsSlaveStatsTrace);
   fOptionsMenu->AddSeparator();
   fOptionsMenu->AddEntry("Feedback &Active", kOptionsFeedback);
   fOptionsMenu->AddSeparator();
   fOptionsMenu->AddPopup("&Feedback Histos", fCascadeMenu);
   fOptionsMenu->CheckEntry(kOptionsStatsHist);
   fOptionsMenu->CheckEntry(kOptionsFeedback);
   fCascadeMenu->CheckEntry(42);
   gEnv->SetValue("Proof.StatsHist", 1);

   //--- Help menu
   fHelpMenu = new TGPopupMenu(gClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...",  kHelpAbout);

   fFileMenu->Associate(this);
   fSessionMenu->Associate(this);
   fQueryMenu->Associate(this);
   fOptionsMenu->Associate(this);
   fCascadeMenu->Associate(this);
   fHelpMenu->Associate(this);

   //--- create menubar and add popup menus
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame);

   fMenuBar->AddPopup("&File", fFileMenu, new TGLayoutHints(kLHintsTop |
                      kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Session", fSessionMenu, new TGLayoutHints(kLHintsTop |
                      kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Query",  fQueryMenu,  new TGLayoutHints(kLHintsTop |
            kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Options",  fOptionsMenu,  new TGLayoutHints(kLHintsTop |
            kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Help", fHelpMenu, new TGLayoutHints(kLHintsTop |
            kLHintsRight));

   TGHorizontal3DLine *toolBarSep = new TGHorizontal3DLine(this);
   AddFrame(toolBarSep, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   AddFrame(fMenuBar, new TGLayoutHints(kLHintsTop | kLHintsLeft |
            kLHintsExpandX, 0, 0, 1, 1));

   toolBarSep = new TGHorizontal3DLine(this);
   AddFrame(toolBarSep, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   fPopupSrv = new TGPopupMenu(fClient->GetRoot());
   fPopupSrv->AddEntry("Connect",kSessionConnect);
   fPopupSrv->AddEntry("Disconnect",kSessionDisconnect);
   fPopupSrv->AddEntry("Browse",kSessionBrowse);
   fPopupSrv->AddEntry("&Show status",kSessionShowStatus);
   fPopupSrv->AddSeparator();
   fPopupSrv->AddEntry("&Cleanup", kSessionCleanup);
   fPopupSrv->Connect("Activated(Int_t)","TSessionViewer", this,
            "MyHandleMenu(Int_t)");

   fPopupQry = new TGPopupMenu(fClient->GetRoot());
   fPopupQry->AddEntry("Edit",kQueryEdit);
   fPopupQry->AddEntry("Submit",kQuerySubmit);
   fPopupQry->AddSeparator();
   fPopupQry->AddEntry("Start &Viewer", kQueryStartViewer);
   fPopupQry->AddSeparator();
   fPopupQry->AddEntry("Delete",kQueryDelete);
   fPopupQry->Connect("Activated(Int_t)","TSessionViewer", this,
            "MyHandleMenu(Int_t)");

   fPopupSrv->DisableEntry(kSessionDisconnect);
   fPopupSrv->DisableEntry(kSessionCleanup);
   fSessionMenu->DisableEntry(kSessionDisconnect);
   fSessionMenu->DisableEntry(kSessionCleanup);

   //--- Horizontal mother frame -----------------------------------------------
   fHf = new TGHorizontalFrame(this, 10, 10);
   fHf->SetCleanup(kDeepCleanup);

   //--- fV1 -------------------------------------------------------------------
   fV1 = new TGVerticalFrame(fHf, 100, 100, kFixedWidth);
   fV1->SetCleanup(kDeepCleanup);

   fTreeView = new TGCanvas(fV1, 100, 200, kSunkenFrame | kDoubleBorder);
   fV1->AddFrame(fTreeView, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
                 2, 0, 0, 0));
   fSessionHierarchy = new TGListTree(fTreeView, kHorizontalFrame);
   fSessionHierarchy->Connect("Clicked(TGListTreeItem*,Int_t,Int_t,Int_t)",
            "TSessionViewer", this,
            "OnListTreeClicked(TGListTreeItem*, Int_t, Int_t, Int_t)");
   fV1->Resize(fTreeView->GetDefaultWidth()+100, fV1->GetDefaultHeight());

   //--- fV2 -------------------------------------------------------------------
   fV2 = new TGVerticalFrame(fHf, 350, 310);
   fV2->SetCleanup(kDeepCleanup);

   //--- Server Frame ----------------------------------------------------------
   fServerFrame = new TSessionServerFrame(fV2, 350, 310);
   fSessions = fServerFrame->ReadConfigFile(kPROOF_GuiConfFile);
   BuildSessionHierarchy(fSessions);
   fServerFrame->Build(this);
   fV2->AddFrame(fServerFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
                 kLHintsExpandY, 2, 0, 1, 2));

   //--- Session Frame ---------------------------------------------------------
   fSessionFrame = new TSessionFrame(fV2, 350, 310);
   fSessionFrame->Build(this);
   fV2->AddFrame(fSessionFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
                 kLHintsExpandY, 2, 0, 1, 2));

   //--- Query Frame -----------------------------------------------------------
   fQueryFrame = new TSessionQueryFrame(fV2, 350, 310);
   fQueryFrame->Build(this);
   fV2->AddFrame(fQueryFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
                 kLHintsExpandY, 2, 0, 1, 2));

   //--- Output Frame ----------------------------------------------------------
   fOutputFrame = new TSessionOutputFrame(fV2, 350, 310);
   fOutputFrame->Build(this);
   fV2->AddFrame(fOutputFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
                 kLHintsExpandY, 2, 0, 1, 2));

   //--- Input Frame -----------------------------------------------------------
   fInputFrame = new TSessionInputFrame(fV2, 350, 310);
   fInputFrame->Build(this);
   fV2->AddFrame(fInputFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
                 kLHintsExpandY, 2, 0, 1, 2));

   fHf->AddFrame(fV1, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   // add vertical splitter between list tree and frames
   TGVSplitter *splitter = new TGVSplitter(fHf, 4);
   splitter->SetFrame(fV1, kTRUE);
   fHf->AddFrame(splitter,new TGLayoutHints(kLHintsLeft | kLHintsExpandY));
   fHf->AddFrame(new TGVertical3DLine(fHf), new TGLayoutHints(kLHintsLeft |
                 kLHintsExpandY));

   fHf->AddFrame(fV2, new TGLayoutHints(kLHintsRight | kLHintsExpandX |
                 kLHintsExpandY));

   AddFrame(fHf, new TGLayoutHints(kLHintsRight | kLHintsExpandX |
            kLHintsExpandY));

   // if description available, update server infos frame
   if (fActDesc)
      fServerFrame->Update(fActDesc);

   //--- Status Bar ------------------------------------------------------------
   int parts[] = { 36, 49, 15 };
   fStatusBar = new TGStatusBar(this, 10, 10);
   fStatusBar->SetCleanup(kDeepCleanup);
   fStatusBar->SetParts(parts, 3);
   for (int i = 0; i < 3; i++)
      fStatusBar->GetBarPart(i)->SetCleanup(kDeepCleanup);
   AddFrame(fStatusBar, new TGLayoutHints(kLHintsTop | kLHintsLeft |
            kLHintsExpandX, 0, 0, 1, 1));

   // connection icon (animation) and time info
   fStatusBar->SetText("      00:00:00", 2);
   TGCompositeFrame *leftpart = fStatusBar->GetBarPart(2);
   fRightIconPicture = (TGPicture *)fClient->GetPicture("proof_disconnected.xpm");
   fRightIcon = new TGIcon(leftpart, fRightIconPicture,
                    fRightIconPicture->GetWidth(),
                    fRightIconPicture->GetHeight());
   leftpart->AddFrame(fRightIcon, new TGLayoutHints(kLHintsLeft, 2, 0, 0, 0));

   // connection progress bar
   TGCompositeFrame *rightpart = fStatusBar->GetBarPart(0);
   fConnectProg = new TGHProgressBar(rightpart, TGProgressBar::kStandard, 100);
   fConnectProg->ShowPosition();
   fConnectProg->SetBarColor("green");
   rightpart->AddFrame(fConnectProg, new TGLayoutHints(kLHintsExpandX, 1, 1, 1, 1));

   // add user info
   fUserGroup = gSystem->GetUserInfo();
   sprintf(line,"User : %s - %s", fUserGroup->fRealName.Data(),
           fUserGroup->fGroup.Data());
   fStatusBar->SetText(line, 1);

   fTimer = 0;

   // create context menu
   fContextMenu = new TContextMenu("SessionViewerContextMenu") ;

   SetWindowName("ROOT Session Viewer");
   MapSubwindows();
   MapWindow();

   // hide frames
   fStatusBar->GetBarPart(0)->HideFrame(fConnectProg);
   fV2->HideFrame(fSessionFrame);
   fV2->HideFrame(fQueryFrame);
   fV2->HideFrame(fOutputFrame);
   fV2->HideFrame(fInputFrame);
   fActFrame = fServerFrame;
   Resize(GetDefaultSize());
}

//______________________________________________________________________________
TSessionViewer::~TSessionViewer()
{
   // dtor

   Cleanup();
   delete fUserGroup;
   if (gSessionViewer == this)
      gSessionViewer = 0;
}

//______________________________________________________________________________
void TSessionViewer::OnListTreeClicked(TGListTreeItem *entry, Int_t btn,
                                       Int_t x, Int_t y)
{
   // handle mouse clicks in list tree

   TList *objlist;
   TObject *obj;
   TString msg;

   if (entry->GetParent() == 0) {  // PROOF
      // switch frames only if actual one doesn't match
      if (fActFrame != fServerFrame) {
         fV2->HideFrame(fActFrame);
         fV2->ShowFrame(fServerFrame);
         fActFrame = fServerFrame;
      }
   }
   else if (entry->GetParent()->GetParent() == 0) { // Server
      if (entry->GetUserData()) {
         obj = (TObject *)entry->GetUserData();
         if (obj->IsA() != TSessionDescription::Class())
            return;
         // update server frame informations
         fServerFrame->Update((TSessionDescription *)obj);
         fActDesc = (TSessionDescription*)obj;
         // if Proof valid, update connection infos
         if (fActDesc->fProof && fActDesc->fProof->IsValid()) {
            fActDesc->fProof->cd();
            msg.Form("PROOF Cluster %s ready", fActDesc->fName.Data());
         }
         else {
            msg.Form("PROOF Cluster %s not connected", fActDesc->fName.Data());
         }
         fStatusBar->SetText(msg.Data(), 1);
      }
      // local session
      if (fActDesc->fLocal) {
         if (fActFrame != fSessionFrame) {
            fV2->HideFrame(fActFrame);
            fV2->ShowFrame(fSessionFrame);
            fActFrame = fSessionFrame;
         }
         fSessionFrame->GetTab()->SetTab("Status");
         fSessionFrame->GetTab()->HideFrame(
               fSessionFrame->GetTab()->GetTabTab("Options"));
         fSessionFrame->GetTab()->HideFrame(
               fSessionFrame->GetTab()->GetTabTab("Packages"));
      }
      // proof session not connected
      if ((!fActDesc->fLocal) && (!fActDesc->fConnected) &&
           (fActFrame != fServerFrame)) {
         fV2->HideFrame(fActFrame);
         fV2->ShowFrame(fServerFrame);
         fActFrame = fServerFrame;
      }
      // proof session connected
      if ((!fActDesc->fLocal) && (fActDesc->fConnected)) {
         if (fActFrame != fSessionFrame) {
            fV2->HideFrame(fActFrame);
            fV2->ShowFrame(fSessionFrame);
            fActFrame = fSessionFrame;
         }
         fSessionFrame->GetTab()->ShowFrame(
               fSessionFrame->GetTab()->GetTabTab("Options"));
         fSessionFrame->GetTab()->ShowFrame(
               fSessionFrame->GetTab()->GetTabTab("Packages"));
      }
      // update session information frame
      fSessionFrame->ProofInfos();
   }
   else if (entry->GetParent()->GetParent()->GetParent() == 0) { // query
      obj = (TObject *)entry->GetParent()->GetUserData();
      if (obj->IsA() == TSessionDescription::Class()) {
         fActDesc = (TSessionDescription *)obj;
      }
      obj = (TObject *)entry->GetUserData();
      if (obj->IsA() == TQueryDescription::Class()) {
         fActDesc->fActQuery = (TQueryDescription *)obj;
      }
      // update query informations and buttons state
      fQueryFrame->UpdateInfos();
      fQueryFrame->UpdateButtons(fActDesc->fActQuery);
      if (fActFrame != fQueryFrame) {
         fV2->HideFrame(fActFrame);
         fV2->ShowFrame(fQueryFrame);
         fActFrame = fQueryFrame;
      }
      // trick to update feedback histos
      OnCascadeMenu();
   }
   else {   // a list (input, output)
      obj = (TObject *)entry->GetParent()->GetParent()->GetUserData();
      if (obj->IsA() == TSessionDescription::Class()) {
         fActDesc = (TSessionDescription *)obj;
      }
      obj = (TObject *)entry->GetParent()->GetUserData();
      if (obj->IsA() == TQueryDescription::Class()) {
         fActDesc->fActQuery = (TQueryDescription *)obj;
      }
      if (fActDesc->fActQuery) {
         // update input/output list views
         fInputFrame->RemoveAll();
         fOutputFrame->RemoveAll();
         if (fActDesc->fActQuery->fResult) {
            objlist = fActDesc->fActQuery->fResult->GetOutputList();
            if (objlist) {
               TIter nexto(objlist);
               while ((obj = (TObject *) nexto())) {
                  fOutputFrame->AddObject(obj);
               }
            }
            objlist = fActDesc->fActQuery->fResult->GetInputList();
            if (objlist) {
               TIter nexti(objlist);
               while ((obj = (TObject *) nexti())) {
                  fInputFrame->AddObject(obj);
               }
            }
         }
         fInputFrame->Resize();
         fOutputFrame->Resize();
         fClient->NeedRedraw(fOutputFrame->GetLVContainer());
         fClient->NeedRedraw(fInputFrame->GetLVContainer());
      }
      // switch frames
      if (strstr(entry->GetText(),"Output")) {
         if (fActFrame != fOutputFrame) {
            fV2->HideFrame(fActFrame);
            fV2->ShowFrame(fOutputFrame);
            fActFrame = fOutputFrame;
         }
      }
      else if (strstr(entry->GetText(),"Input")) {
         if (fActFrame != fInputFrame) {
            fV2->HideFrame(fActFrame);
            fV2->ShowFrame(fInputFrame);
            fActFrame = fInputFrame;
         }
      }
   }
   if (btn == 3) { // right button
      // place popup menus
      TGListTreeItem *item = fSessionHierarchy->GetSelected();
      if (!item) return;
      obj = (TObject *)item->GetUserData();
      if (obj && obj->IsA() == TQueryDescription::Class()) {
         fPopupQry->PlaceMenu(x, y, 1, 1);
      }
      else if (obj && obj->IsA() == TSessionDescription::Class()) {
         if (!fActDesc->fLocal)
            fPopupSrv->PlaceMenu(x, y, 1, 1);
      }
   }
   // enable / disable menu entries
   if (fActDesc->fConnected) {
      fPopupSrv->DisableEntry(kSessionConnect);
      fPopupSrv->EnableEntry(kSessionDisconnect);
      fPopupSrv->EnableEntry(kSessionCleanup);
      fSessionMenu->DisableEntry(kSessionConnect);
      fSessionMenu->EnableEntry(kSessionDisconnect);
      fSessionMenu->EnableEntry(kSessionCleanup);
   }
   else {
      fPopupSrv->DisableEntry(kSessionDisconnect);
      fPopupSrv->DisableEntry(kSessionCleanup);
      fPopupSrv->EnableEntry(kSessionConnect);
      fSessionMenu->DisableEntry(kSessionDisconnect);
      fSessionMenu->DisableEntry(kSessionCleanup);
      fSessionMenu->EnableEntry(kSessionConnect);
   }
   if (fActDesc->fLocal) {
      fSessionMenu->DisableEntry(kSessionConnect);
      fSessionMenu->DisableEntry(kSessionDisconnect);
      fSessionMenu->DisableEntry(kSessionCleanup);
   }
}

//______________________________________________________________________________
void TSessionViewer::BuildSessionHierarchy(TList *list)
{
   // Read the list of proof servers from the configuration file.
   // Get the list of proof servers and running queries from gROOT.
   // Build the hierarchy.

   // remove list tree entries
   if (fSessionItem)
      fSessionHierarchy->DeleteChildren(fSessionItem);
   else
      fSessionItem = fSessionHierarchy->AddItem(0, "Sessions", fBaseIcon,
            fBaseIcon);
   // add local session description
   TGListTreeItem *item = fSessionHierarchy->AddItem(fSessionItem, "Local",
            fLocal, fLocal);
   fSessionHierarchy->SetToolTipItem(item, "Local Session");
   TSessionDescription *localdesc = new TSessionDescription();
   localdesc->fName = "Local";
   localdesc->fAddress = "Local";
   localdesc->fPort = 0;
   localdesc->fConfigFile = "";
   localdesc->fLogLevel = 0;
   localdesc->fUserName = "";
   localdesc->fQueries = new TList();
   localdesc->fActQuery = 0;
   localdesc->fProof = 0;
   localdesc->fLocal = kTRUE;
   localdesc->fSync = kTRUE;
   localdesc->fNbHistos = 0;
   item->SetUserData(localdesc);

   // get list of proof sessions
   TSeqCollection *proofs = gROOT->GetListOfProofs();
   if (proofs) {
      TIter nextp(proofs);
      TVirtualProof *proof;
      TQueryResult *query;
      TQueryDescription *newquery;
      TSessionDescription *newdesc;
      // loop over existing Proof sessions
      while ((proof = (TVirtualProof *)nextp())) {
         TIter nexts(fSessions);
         TSessionDescription *desc = 0;
         Bool_t found = kFALSE;
         // check if session is already in the list
         while ((desc = (TSessionDescription *)nexts())) {
            if (desc->fProof == proof) {
               desc->fConnected = kTRUE;
               found = kTRUE;
               break;
            }
         }
         if (found) continue;
         // create new session description
         newdesc = new TSessionDescription();
         // and fill informations from Proof session
         newdesc->fName       = proof->GetMaster();
         newdesc->fConfigFile = proof->GetConfFile();
         newdesc->fUserName   = proof->GetUser();
         newdesc->fPort       = proof->GetPort();
         newdesc->fLogLevel   = proof->GetLogLevel();
         newdesc->fAddress    = proof->GetMaster();
         newdesc->fQueries    = new TList();
         newdesc->fActQuery   = 0;
         newdesc->fConnected = kTRUE;
         newdesc->fLocal = kFALSE;
         newdesc->fSync = kFALSE;
         newdesc->fNbHistos = 0;

         // get list of queries and fill list tree
         TIter nextq(proof->GetListOfQueries());
         while ((query = (TQueryResult *)nextp())) {
            newquery = new TQueryDescription();
            newquery->fStatus = query->IsFinalized() ?
                 TQueryDescription::kSessionQueryFinalized :
               (TQueryDescription::ESessionQueryStatus)query->GetStatus();
            newquery->fSelectorString  = query->GetSelecImp()->GetName();
            newquery->fQueryName       = query->GetName();
            newquery->fOptions         = query->GetOptions();
            newquery->fEventList       = "";
            newquery->fParFile         = "";
            newquery->fNbFiles         = 0;
            newquery->fNoEntries       = query->GetEntries();
            newquery->fFirstEntry      = query->GetFirst();
            newquery->fResult          = query;
            newdesc->fQueries->Add((TObject *)newquery);
         }
         // add new session description in list tree
         item = fSessionHierarchy->AddItem(fSessionItem, newdesc->fName.Data(),
                  fProofCon, fProofCon);
         fSessionHierarchy->SetToolTipItem(item, "Proof Session");
         item ->SetUserData(newdesc);
         // and in our session description list
         list->Add(newdesc);
         // set actual description to the last one
         fActDesc = newdesc;
      }
   }

   // loop over session description list and set correct icon
   // ( connected or disconnected )
   TIter next(list);
   TSessionDescription *desc = 0;
   while ((desc = (TSessionDescription *)next())) {
      if (desc->fConnected) {
         item = fSessionHierarchy->AddItem(fSessionItem, desc->fName.Data(),
                  fProofCon, fProofCon);
      }
      else {
         item = fSessionHierarchy->AddItem(fSessionItem, desc->fName.Data(),
                  fProofDiscon, fProofDiscon);
      }
      fSessionHierarchy->SetToolTipItem(item, "Proof Session");
      item->SetUserData(desc);
      fActDesc = desc;
   }
   // update list tree
   fSessionHierarchy->ClearHighlighted();
   fSessionHierarchy->OpenItem(fSessionItem);
   fSessionHierarchy->OpenItem(item);
   fSessionHierarchy->HighlightItem(item);
   fSessionHierarchy->SetSelected(item);
   fClient->NeedRedraw(fSessionHierarchy);
}

//______________________________________________________________________________
void TSessionViewer::CloseWindow()
{
   // close main Session Viewer window

   // clean-up temporary files
   TString pathtmp;
   pathtmp = Form("%s/%s", gSystem->TempDirectory(), kSession_RedirectFile);
   if (!gSystem->AccessPathName(pathtmp)) {
      gSystem->Unlink(pathtmp);
   }
   pathtmp = Form("%s/%s", gSystem->TempDirectory(), kSession_RedirectCmd);
   if (!gSystem->AccessPathName(pathtmp)) {
      gSystem->Unlink(pathtmp);
   }
   // close opened Proof sessions (if any)
   TIter next(fSessions);
   TSessionDescription *desc = 0;
   while ((desc = (TSessionDescription *)next())) {
      if (desc->fProof && desc->fProof->IsValid())
         desc->fProof->Close();
   }
   delete fSessionHierarchy; // this has been put int TGCanvas which isn't a
                             // TGComposite frame and doesn't do cleanups.
   fClient->FreePicture(fLocal);
   fClient->FreePicture(fProofCon);
   fClient->FreePicture(fProofDiscon);
   fClient->FreePicture(fQueryCon);
   fClient->FreePicture(fQueryDiscon);
   fClient->FreePicture(fBaseIcon);
   delete fTimer;
   DeleteWindow();
}

//______________________________________________________________________________
void TSessionViewer::ChangeRightLogo(const char *name)
{
    // Change the right logo ( used for animation )
    fClient->FreePicture(fRightIconPicture);
    fRightIconPicture = (TGPicture *)fClient->GetPicture(name);
    fRightIcon->SetPicture(fRightIconPicture);
}

//______________________________________________________________________________
void TSessionViewer::EnableTimer()
{
   // enable animation timer
   if (!fTimer) fTimer = new TTimer(this, 500);
   fTimer->Reset();
   fTimer->TurnOn();
   time( &fStart );
}

//______________________________________________________________________________
void TSessionViewer::DisableTimer()
{
   // disable animation timer
   if (fTimer)
      fTimer->TurnOff();
   ChangeRightLogo("proof_disconnected.xpm");
}

//______________________________________________________________________________
Bool_t TSessionViewer::HandleTimer(TTimer *)
{
   // handle animation timer
   char line[120];
   struct tm *connected;
   Int_t count = gRandom->Integer(4);
   if (count > 3) {
      count = 0;
   }
   if (fChangePic)
      ChangeRightLogo(xpm_names[count]);
   time( &fElapsed );
   time_t elapsed_time = (time_t)difftime( fElapsed, fStart );
   connected = gmtime( &elapsed_time );
   sprintf(line,"      %02d:%02d:%02d", connected->tm_hour,
           connected->tm_min, connected->tm_sec);
   fStatusBar->SetText(line, 2);
   fTimer->Reset();
   return kTRUE;
}

//______________________________________________________________________________
void TSessionViewer::LogMessage(const char *msg, Bool_t all)
{
   // Load/append a log msg in the log frame, if open

   if (fLogWindow) {
      if (all) {
         // load buffer
         fLogWindow->LoadBuffer(msg);
      } else {
         // append
         fLogWindow->AddBuffer(msg);
      }
   }
}

//______________________________________________________________________________
void TSessionViewer::QueryResultReady(char *query)
{
   // handle signal "query result ready" coming from Proof session

   char strtmp[256];
   sprintf(strtmp,"Query Result Ready for %s\n", query);
   // show information on status bar
   ShowInfo(strtmp);
   TGListTreeItem *item=0, *item2=0;
   TQueryDescription *lquery = 0;
   // loop over actual queries to find which one is ready
   TIter nextp(fActDesc->fQueries);
   while ((lquery = (TQueryDescription *)nextp())) {
      if (lquery->fReference.Contains(query)) {
         // results are ready for this query
         lquery->fResult = fActDesc->fProof->GetQueryResult(query);
         lquery->fStatus = TQueryDescription::kSessionQueryFromProof;
         if (!lquery->fResult)
            break;
         // get query status
         lquery->fStatus = lquery->fResult->IsFinalized() ?
           TQueryDescription::kSessionQueryFinalized :
           (TQueryDescription::ESessionQueryStatus)lquery->fResult->GetStatus();
         // get data set
         if (lquery->fResult->GetDSet())
            lquery->fChain = lquery->fResult->GetDSet();
         item = fSessionHierarchy->FindItemByObj(fSessionItem, fActDesc);
         if (item) {
            item2 = fSessionHierarchy->FindItemByObj(item, lquery);
         }
         if (item2) {
            // add input and output list entries
            if (lquery->fResult->GetInputList())
               if (!fSessionHierarchy->FindChildByName(item2, "InputList"))
                  fSessionHierarchy->AddItem(item2, "InputList");
            if (lquery->fResult->GetInputList())
               if (!fSessionHierarchy->FindChildByName(item2, "OutputList"))
                  fSessionHierarchy->AddItem(item2, "OutputList");
         }
         // update list tree, query frame informations, and buttons state
         fClient->NeedRedraw(fSessionHierarchy);
         fQueryFrame->UpdateInfos();
         fQueryFrame->UpdateButtons(lquery);
         break;
      }
   }
}

//______________________________________________________________________________
void TSessionViewer::CleanupSession()
{
   // clean-up Proof session

   TGListTreeItem *item = fSessionHierarchy->GetSelected();
   if (!item) return;
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class()) return;
   if (!fActDesc->fProof || !fActDesc->fProof->IsValid()) return;
   TQueryDescription *query = (TQueryDescription *)obj;
   TString m;
   m.Form("Are you sure to cleanup the session \"%s::%s\"",
           fActDesc->fAddress.Data(), fActDesc->fName.Data());
   Int_t result;
   new TGMsgBox(fClient->GetRoot(), this, "", m.Data(), 0,
                kMBYes | kMBNo | kMBCancel, &result);
   if (result == kMBYes) {
      // send cleanup request for the session specified by the query reference
      fActDesc->fProof->CleanupSession(query->fReference.Data());
      fSessionHierarchy->DeleteChildren(item->GetParent());
      fSessionFrame->OnBtnGetQueriesClicked();
   }
   // update list tree
   fClient->NeedRedraw(fSessionHierarchy);
}

//______________________________________________________________________________
void TSessionViewer::DeleteQuery()
{
   // delete query from list tree and ask user if he wants do delete it also
   // from server

   TGListTreeItem *item = fSessionHierarchy->GetSelected();
   if (!item) return;
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class()) return;
   TQueryDescription *query = (TQueryDescription *)obj;
   TString m;
   Int_t result = 0;

   if (fActDesc->fProof && fActDesc->fProof->IsValid()) {
      m.Form("Do you want to delete query \"%s\" from server too ?",
               query->fQueryName.Data());
      new TGMsgBox(fClient->GetRoot(), this, "", m.Data(), kMBIconQuestion,
                   kMBYes | kMBNo | kMBCancel, &result);
   }
   else {
      m.Form("Dou you really want to delete query \"%s\" ?",
               query->fQueryName.Data());
      new TGMsgBox(fClient->GetRoot(), this, "", m.Data(), kMBIconQuestion,
                   kMBOk | kMBCancel, &result);
   }
   if (result == kMBYes) {
      fActDesc->fProof->Remove(query->fReference.Data());
      fActDesc->fQueries->Remove((TObject *)query);
      fSessionHierarchy->DeleteItem(item);
      delete query;
   }
   else if (result == kMBNo || result == kMBOk) {
      fActDesc->fQueries->Remove((TObject *)query);
      fSessionHierarchy->DeleteItem(item);
      delete query;
   }
   fClient->NeedRedraw(fSessionHierarchy);
}

//______________________________________________________________________________
void TSessionViewer::EditQuery()
{
   // Edit currently selected query

   TGListTreeItem *item = fSessionHierarchy->GetSelected();
   if (!item) return;
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class()) return;
   TQueryDescription *query = (TQueryDescription *)obj;
   TNewQueryDlg *dlg = new TNewQueryDlg(this, 350, 310, query, kTRUE);
   dlg->Popup();
}

//______________________________________________________________________________
void TSessionViewer::StartViewer()
{
   // Start TreeViewer from selected TChain

   TGListTreeItem *item = fSessionHierarchy->GetSelected();
   if (!item) return;
   TObject *obj = (TObject *)item->GetUserData();
   if (obj->IsA() != TQueryDescription::Class()) return;
   TQueryDescription *query = (TQueryDescription *)obj;
   if (!query->fChain && query->fResult && query->fResult->GetDSet()) {
      query->fChain = query->fResult->GetDSet();
   }
   if (query->fChain->IsA() == TChain::Class())
      ((TChain *)query->fChain)->StartViewer();
   else if (query->fChain->IsA() == TDSet::Class())
      ((TDSet *)query->fChain)->StartViewer();
}

//______________________________________________________________________________
void TSessionViewer::ShowPackages()
{
   Window_t wdummy;
   Int_t  ax, ay;

   if (fActDesc->fLocal) return;
   if (!fActDesc->fProof || !fActDesc->fProof->IsValid())
      return;
   TString pathtmp = Form("%s/%s", gSystem->TempDirectory(),
                          kSession_RedirectFile);
   // redirect stdout/stderr to temp file
   if (gSystem->RedirectOutput(pathtmp.Data(), "w") != 0) {
      Error("ShowStatus", "stdout/stderr redirection failed; skipping");
      return;
   }
   fActDesc->fProof->ShowPackages(kTRUE);
   // restore stdout/stderr
   if (gSystem->RedirectOutput(0) != 0) {
      Error("ShowStatus", "stdout/stderr retore failed; skipping");
      return;
   }
   if (!fLogWindow) {
      fLogWindow = new TSessionLogView(this, 700, 100);
   } else {
      // Clear window
      fLogWindow->Clear();
   }
   fLogWindow->LoadFile(pathtmp.Data());
   gVirtualX->TranslateCoordinates(GetId(),
              fClient->GetDefaultRoot()->GetId(),
              0, 0, ax, ay, wdummy);
   fLogWindow->Move(ax, ay + GetHeight() + 35);
   fLogWindow->Popup();
}

//______________________________________________________________________________
void TSessionViewer::ShowEnabledPackages()
{
   Window_t wdummy;
   Int_t  ax, ay;

   if (fActDesc->fLocal) return;
   if (!fActDesc->fProof || !fActDesc->fProof->IsValid())
      return;
   TString pathtmp = Form("%s/%s", gSystem->TempDirectory(),
                          kSession_RedirectFile);
   // redirect stdout/stderr to temp file
   if (gSystem->RedirectOutput(pathtmp.Data(), "w") != 0) {
      Error("ShowStatus", "stdout/stderr redirection failed; skipping");
      return;
   }
   fActDesc->fProof->ShowEnabledPackages(kTRUE);
   // restore stdout/stderr
   if (gSystem->RedirectOutput(0) != 0) {
      Error("ShowStatus", "stdout/stderr retore failed; skipping");
      return;
   }
   if (!fLogWindow) {
      fLogWindow = new TSessionLogView(this, 700, 100);
   } else {
      // Clear window
      fLogWindow->Clear();
   }
   fLogWindow->LoadFile(pathtmp.Data());
   gVirtualX->TranslateCoordinates(GetId(),
              fClient->GetDefaultRoot()->GetId(),
              0, 0, ax, ay, wdummy);
   fLogWindow->Move(ax, ay + GetHeight() + 35);
   fLogWindow->Popup();
}

//______________________________________________________________________________
void TSessionViewer::ShowLog(const char *queryref)
{
   // Display the content of the temporary log file for queryref

   Window_t wdummy;
   Int_t  ax, ay;

   if (fActDesc->fProof) {
      gVirtualX->SetCursor(GetId(),gVirtualX->CreateCursor(kWatch));
      if (!fLogWindow) {
         fLogWindow = new TSessionLogView(this, 700, 100);
      } else {
         // Clear window
         fLogWindow->Clear();
      }
      fActDesc->fProof->Connect("LogMessage(const char*,Bool_t)",
            "TSessionViewer", this, "LogMessage(const char*,Bool_t)");
      Bool_t logonly = fActDesc->fProof->SendingLogToWindow();
      fActDesc->fProof->SendLogToWindow(kTRUE);
      if (queryref)
         fActDesc->fProof->ShowLog(queryref);
      else
         fActDesc->fProof->ShowLog(0);
      fActDesc->fProof->SendLogToWindow(logonly);
      // set log window position at the bottom of Session Viewer
      gVirtualX->TranslateCoordinates(GetId(),
         fClient->GetDefaultRoot()->GetId(), 0, 0, ax, ay, wdummy);
      fLogWindow->Move(ax, ay + GetHeight() + 35);
      fLogWindow->Popup();
      gVirtualX->SetCursor(GetId(), 0);
   }
}

//______________________________________________________________________________
void TSessionViewer::ShowInfo(const char *txt)
{
   // display text in status bar

   fStatusBar->SetText(txt,0);
   fClient->NeedRedraw(fStatusBar);
   gSystem->ProcessEvents();
}

//______________________________________________________________________________
void TSessionViewer::ShowStatus()
{
   // retrieve and display Proof status

   Window_t wdummy;
   Int_t  ax, ay;

   if (!fActDesc->fProof || !fActDesc->fProof->IsValid())
      return;
   TString pathtmp = Form("%s/%s", gSystem->TempDirectory(),
                          kSession_RedirectFile);
   // redirect stdout/stderr to temp file
   if (gSystem->RedirectOutput(pathtmp.Data(), "w") != 0) {
      Error("ShowStatus", "stdout/stderr redirection failed; skipping");
      return;
   }
   fActDesc->fProof->GetStatus();
   // restore stdout/stderr
   if (gSystem->RedirectOutput(0) != 0) {
      Error("ShowStatus", "stdout/stderr retore failed; skipping");
      return;
   }
   if (!fLogWindow) {
      fLogWindow = new TSessionLogView(this, 700, 100);
   } else {
      // Clear window
      fLogWindow->Clear();
   }
   fLogWindow->LoadFile(pathtmp.Data());
   gVirtualX->TranslateCoordinates(GetId(),
              fClient->GetDefaultRoot()->GetId(),
              0, 0, ax, ay, wdummy);
   fLogWindow->Move(ax, ay + GetHeight() + 35);
   fLogWindow->Popup();
}

//______________________________________________________________________________
void TSessionViewer::StartupMessage(char *msg, Bool_t, Int_t done, Int_t total)
{
   // Handle startup message (connection progress) coming from Proof session

   Float_t pos = Float_t(Double_t(done * 100)/Double_t(total));
   fConnectProg->SetPosition(pos);
   fStatusBar->SetText(msg, 1);
}

//______________________________________________________________________________
void TSessionViewer::MyHandleMenu(Int_t id)
{
   // Handle session viewer custom popup menus

   switch (id) {

      case kSessionConnect:
         fServerFrame->OnBtnConnectClicked();
         break;

      case kSessionDisconnect:
         fSessionFrame->OnBtnDisconnectClicked();
         break;

      case kSessionCleanup:
         CleanupSession();
         break;

      case kSessionBrowse:
         if (fActDesc->fProof && fActDesc->fProof->IsValid()) {
            TBrowser *b = new TBrowser();
            fActDesc->fProof->Browse(b);
         }
         break;

      case kSessionShowStatus:
         ShowStatus();
         break;

      case kQueryEdit:
         EditQuery();
         break;

      case kQueryDelete:
         DeleteQuery();
         break;

      case kQueryStartViewer:
         StartViewer();
         break;

      case kQuerySubmit:
         fQueryFrame->OnBtnSubmit();
         break;
   }
}

//______________________________________________________________________________
void TSessionViewer::OnCascadeMenu()
{
   // Handle feedback histograms configuration menu

   if (!fActDesc || !fActDesc->fActQuery) return;
   fActDesc->fNbHistos = 0;
   Int_t i = 0;
   // loop over feedback histo list
   while (kFeedbackHistos[i]) {
      // check if user has selected this histogram in the option menu
      if (fCascadeMenu->IsEntryChecked(41+i))
         fActDesc->fNbHistos++;
      i++;
   }
   // divide stats canvas by number of selected feedback histos
   fQueryFrame->GetStatsCanvas()->cd();
   fQueryFrame->GetStatsCanvas()->Clear();
   if (fActDesc->fNbHistos == 4)
      fQueryFrame->GetStatsCanvas()->Divide(2, 2);
   else if (fActDesc->fNbHistos > 4)
      fQueryFrame->GetStatsCanvas()->Divide(3, 2);
   else
      fQueryFrame->GetStatsCanvas()->Divide(fActDesc->fNbHistos, 1);
   // if actual query has results, update feedback histos
   if (fActDesc->fActQuery && fActDesc->fActQuery->fResult) {
      if (fActDesc->fActQuery->fResult->GetOutputList()) {
         fQueryFrame->UpdateHistos(fActDesc->fActQuery->fResult->GetOutputList());
      }
   }
}
//______________________________________________________________________________
Bool_t TSessionViewer::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle messages send to the TSessionViewer object. E.g. all menu entries
   // messages.
   TNewQueryDlg *dlg;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_MENU:
               switch (parm1) {

                  case kFileCloseViewer:
                     CloseWindow();
                     break;

                  case kFileLoadLibrary:
                     break;

                  case kFileQuit:
                     CloseWindow();
                     gApplication->Terminate(0);
                     break;

                  case kSessionCleanup:
                     CleanupSession();
                     break;

                  case kSessionConnect:
                     fServerFrame->OnBtnConnectClicked();
                     break;

                  case kSessionDisconnect:
                     fSessionFrame->OnBtnDisconnectClicked();
                     break;

                  case kSessionShowStatus:
                     ShowStatus();
                     break;

                  case kQueryNew:
                     dlg = new TNewQueryDlg(this, 350, 310);
                     dlg->Popup();
                     break;

                  case kQueryEdit:
                     EditQuery();
                     break;

                  case kQueryDelete:
                     DeleteQuery();
                     break;

                  case kQueryStartViewer:
                     StartViewer();
                     break;

                  case kQuerySubmit:
                     fQueryFrame->OnBtnSubmit();
                     break;

                  case kOptionsStatsHist:
                     if(fOptionsMenu->IsEntryChecked(kOptionsStatsHist)) {
                        fOptionsMenu->UnCheckEntry(kOptionsStatsHist);
                        gEnv->SetValue("Proof.StatsHist", 0);
                     }
                     else {
                        fOptionsMenu->CheckEntry(kOptionsStatsHist);
                        gEnv->SetValue("Proof.StatsHist", 1);
                     }
                     break;

                  case kOptionsStatsTrace:
                     if(fOptionsMenu->IsEntryChecked(kOptionsStatsTrace)) {
                        fOptionsMenu->UnCheckEntry(kOptionsStatsTrace);
                        gEnv->SetValue("Proof.StatsTrace", 0);
                     }
                     else {
                        fOptionsMenu->CheckEntry(kOptionsStatsTrace);
                        gEnv->SetValue("Proof.StatsTrace", 1);
                     }
                     break;

                  case kOptionsSlaveStatsTrace:
                     if(fOptionsMenu->IsEntryChecked(kOptionsSlaveStatsTrace)) {
                        fOptionsMenu->UnCheckEntry(kOptionsSlaveStatsTrace);
                        gEnv->SetValue("Proof.SlaveStatsTrace", 0);
                     }
                     else {
                        fOptionsMenu->CheckEntry(kOptionsSlaveStatsTrace);
                        gEnv->SetValue("Proof.SlaveStatsTrace", 1);
                     }
                     break;

                  case kOptionsFeedback:
                     if(fOptionsMenu->IsEntryChecked(kOptionsFeedback)) {
                        fOptionsMenu->UnCheckEntry(kOptionsFeedback);
                     }
                     else {
                        fOptionsMenu->CheckEntry(kOptionsFeedback);
                     }
                     break;

                  case 41:
                  case 42:
                  case 43:
                  case 44:
                  case 45:
                  case 46:
                     if (fCascadeMenu->IsEntryChecked(parm1)) {
                        fCascadeMenu->UnCheckEntry(parm1);
                     }
                     else {
                        fCascadeMenu->CheckEntry(parm1);
                     }
                     OnCascadeMenu();
                     break;

                  case 50:
                     if (fCascadeMenu->IsEntryChecked(parm1)) {
                        fCascadeMenu->UnCheckEntry(parm1);
                     }
                     else {
                        fCascadeMenu->CheckEntry(parm1);
                     }
                     OnCascadeMenu();
                     break;

                  case kHelpAbout:
                     {
#ifdef R__UNIX
                        TString rootx;
# ifdef ROOTBINDIR
                        rootx = ROOTBINDIR;
# else
                        rootx = gSystem->Getenv("ROOTSYS");
                        if (!rootx.IsNull()) rootx += "/bin";
# endif
                        rootx += "/root -a &";
                        gSystem->Exec(rootx);
#else
#ifdef WIN32
                        new TWin32SplashThread(kTRUE);
#else
                        char str[32];
                        sprintf(str, "About ROOT %s...", gROOT->GetVersion());
                        TRootHelpDialog *hd = new TRootHelpDialog(this, str, 600, 400);
                        hd->SetText(gHelpAbout);
                        hd->Popup();
#endif
#endif
                     }
                     break;

                  default:
                     break;
               }
            default:
               break;
         }
      default:
         break;
   }

   return kTRUE;
}

