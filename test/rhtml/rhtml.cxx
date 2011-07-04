// @(#)root/test/rhtml/:$Id$
// Author: Bertrand Bellenot   09/05/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TApplication.h"
#include "TSystem.h"
#include "TGMenu.h"
#include "TGComboBox.h"
#include "TGFrame.h"
#include "TGButton.h"
#include "TGTextBuffer.h"
#include "TGTextEntry.h"
#include "TGStatusBar.h"
#include "TGFileDialog.h"
#include "TFile.h"
#include "TBrowser.h"
#include "TGHtml.h"
#include "TString.h"
#include "TUrl.h"
#include "TSocket.h"
#include "Riostream.h"
#include "rhtml.h"

#ifdef WIN32
#include "TWin32SplashThread.h"
#endif

enum EMyMessageTypes {
   M_FILE_OPEN,
   M_FILE_BROWSE,
   M_FILE_EXIT,
   M_FAVORITES_ADD,
   M_TOOLS_CLEARHIST,
   M_HELP_ABOUT
};

const char *filetypes[] = { 
   "HTML files",    "*.html",
   "All files",     "*",
    0,               0 
};

const char *HtmlError[] = {
"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3c.org/TR/1999/REC-html401-19991224/loose.dtd\"> ",
"<HTML><HEAD><TITLE>RHTML cannot display the webpage</TITLE> ",
"<META http-equiv=Content-Type content=\"text/html; charset=UTF-8\"></HEAD> ",
"<BODY> ",
"<TABLE cellSpacing=0 cellPadding=0 width=730 border=0> ",
"  <TBODY> ",
"  <TR> ",
"    <TD id=infoIconAlign vAlign=top align=left width=60 rowSpan=2> ",
"    <IMG src=\"info.gif\"> ",
"    </TD> ",
"    <TD id=mainTitleAlign vAlign=center align=left width=*> ",
"      <H1 id=mainTitle>RHTML cannot display the webpage</H1></TD></TR> ",
"  <TR> ",
"    <TD class=errorCodeAndDivider id=errorCodeAlign align=right>&nbsp;  ",
"      <DIV class=divider></DIV></TD></TR> ",
"  <TR> ",
"      <UL> ",
"      </UL> ",
"    <TD>&nbsp; </TD> ",
"    <TD id=MostLikelyAlign vAlign=top align=left> ",
"      <H3 id=likelyCauses>Most likely causes:</H3> ",
"      <UL> ",
"        <LI id=causeNotConnected>You are not connected to the Internet.  ",
"        <LI id=causeSiteProblem>The website is encountering problems.  ",
"        <LI id=causeErrorInAddress>There might be a typing error in the address.  ",
"        <LI id=causeOtherError>  ",
"        </LI></UL></TD></TR> ",
"  <TR> ",
"    <TD id=infoBlockAlign vAlign=top align=right>&nbsp; </TD> ",
"    <TD id=moreInformationAlign vAlign=center align=left> ",
"      <H4> ",
"      <TABLE> ",
"        <TBODY> ",
"        <TR> ",
"          <TD vAlign=top><SPAN id=moreInfoContainer></SPAN><ID  ",
"            id=moreInformation>More information</ID> ",
"      </TD></TR></TBODY></TABLE></H4> ",
"      <DIV class=infoBlock id=infoBlockID> ",
"      <P><ID id=errorExpl1>This problem can be caused by a variety of issues,  ",
"      including:</ID>  ",
"      <UL> ",
"        <LI id=errorExpl2>Internet connectivity has been lost.  ",
"        <LI id=errorExpl3>The website is temporarily unavailable.  ",
"        <LI id=errorExpl4>The Domain Name Server (DNS) is not reachable.  ",
"        <LI id=errorExpl5>The Domain Name Server (DNS) does not have a listing  ",
"        for the website's domain.  ",
"      <P></P> ",
"      <P></P></DIV></TD></TR></TBODY></TABLE></BODY></HTML> ",
0
};

//______________________________________________________________________________
TGHtmlBrowser::TGHtmlBrowser(const char *filename, const TGWindow *p, UInt_t w, UInt_t h) 
             : TGMainFrame(p, w, h)
{
   // TGHtmlBrowser constructor.

   SetCleanup(kDeepCleanup);
   fNbFavorites = 1000;
   fMenuBar = new TGMenuBar(this, 35, 50, kHorizontalFrame);

   fMenuFile = new TGPopupMenu(gClient->GetRoot());
   fMenuFile->AddEntry(" &Open...            Ctrl+O", M_FILE_OPEN, 0,  
                       gClient->GetPicture("bld_open.png"));
   fMenuFile->AddEntry(" &Browse...         Ctrl+B", M_FILE_BROWSE);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry(" E&xit                   Ctrl+Q", M_FILE_EXIT, 0, 
                       gClient->GetPicture("bld_exit.png"));
   fMenuFile->Associate(this);

   fMenuFavorites = new TGPopupMenu(gClient->GetRoot());
   fMenuFavorites->AddEntry("&Add to Favorites", M_FAVORITES_ADD, 0, 
                            gClient->GetPicture("bld_plus.png"));
   fMenuFavorites->AddSeparator();
   fMenuFavorites->AddEntry("http://root.cern.ch/drupal/", fNbFavorites++, 0, 
                            gClient->GetPicture("htmlfile.gif"));
   fMenuFavorites->Associate(this);

   fMenuTools = new TGPopupMenu(gClient->GetRoot());
   fMenuTools->AddEntry("&Clear History", M_TOOLS_CLEARHIST, 0,
                        gClient->GetPicture("ed_delete.png"));
   fMenuTools->Associate(this);

   fMenuHelp = new TGPopupMenu(gClient->GetRoot());
   fMenuHelp->AddEntry(" &About...", M_HELP_ABOUT, 0, gClient->GetPicture("about.xpm"));
   fMenuHelp->Associate(this);

   fMenuBar->AddPopup("&File", fMenuFile, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Favorites", fMenuFavorites, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Tools", fMenuTools, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup("&Help", fMenuHelp, new TGLayoutHints(kLHintsTop | kLHintsRight));

   AddFrame(fMenuBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   // vertical frame
   fVerticalFrame = new TGVerticalFrame(this,727,600,kVerticalFrame);

   fHorizontalFrame = new TGHorizontalFrame(fVerticalFrame,727,600);

   fBack = new TGPictureButton(fHorizontalFrame,gClient->GetPicture("GoBack.gif"));
   fBack->SetStyle(gClient->GetStyle());
   fBack->SetToolTipText("Go Back");
   fHorizontalFrame->AddFrame(fBack, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterY,2,2,2,2));
   fBack->Connect("Clicked()", "TGHtmlBrowser", this, "Back()");

   fForward = new TGPictureButton(fHorizontalFrame,gClient->GetPicture("GoForward.gif"));
   fForward->SetStyle(gClient->GetStyle());
   fForward->SetToolTipText("Go Forward");
   fHorizontalFrame->AddFrame(fForward, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterY,2,2,2,2));
   fForward->Connect("Clicked()", "TGHtmlBrowser", this, "Forward()");

   fReload = new TGPictureButton(fHorizontalFrame,gClient->GetPicture("ReloadPage.gif"));
   fReload->SetStyle(gClient->GetStyle());
   fReload->SetToolTipText("Reload Page");
   fHorizontalFrame->AddFrame(fReload, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterY,2,2,2,2));
   fReload->Connect("Clicked()", "TGHtmlBrowser", this, "Reload()");

   fStop = new TGPictureButton(fHorizontalFrame,gClient->GetPicture("StopLoading.gif"));
   fStop->SetStyle(gClient->GetStyle());
   fStop->SetToolTipText("Stop Loading");
   fHorizontalFrame->AddFrame(fStop, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterY,2,2,2,2));
   fStop->Connect("Clicked()", "TGHtmlBrowser", this, "Stop()");

   fHome = new TGPictureButton(fHorizontalFrame,gClient->GetPicture("GoHome.gif"));
   fHome->SetStyle(gClient->GetStyle());
   fHome->SetToolTipText("Go to ROOT HomePage\n  (http://root.cern.ch)");
   fHorizontalFrame->AddFrame(fHome, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterY,2,2,2,2));
   fHome->Connect("Clicked()", "TGHtmlBrowser", this, "Selected(=\"http://root.cern.ch/drupal/\")");

   // combo box
   fURLBuf   = new TGTextBuffer(256);
   fComboBox = new TGComboBox(fHorizontalFrame, "");
   fURL      = fComboBox->GetTextEntry();
   fURLBuf   = fURL->GetBuffer();
   fComboBox->Resize(200, fURL->GetDefaultHeight());
   fURL->Connect("ReturnPressed()", "TGHtmlBrowser", this, "URLChanged()");

   fComboBox->AddEntry(filename,1);
   fURL->SetText(filename);

   fComboBox->Select(0);
   fComboBox->Connect("Selected(char *)", "TGHtmlBrowser", this, "Selected(char *)");

   fHorizontalFrame->AddFrame(fComboBox, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX,2,2,2,2));

   fVerticalFrame->AddFrame(fHorizontalFrame, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX,2,2,2,2));

   // embedded canvas
   fHtml = new TGHtml(fVerticalFrame, 10, 10, -1);
   fVerticalFrame->AddFrame(fHtml, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX | kLHintsExpandY,2,2,2,2));

   AddFrame(fVerticalFrame, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX | kLHintsExpandY,2,2,2,2));

   // status bar
   fStatusBar = new TGStatusBar(this,100,20);
   Int_t partsusBar[] = {75,25};
   fStatusBar->SetParts(partsusBar,2);
   AddFrame(fStatusBar, new TGLayoutHints(kLHintsBottom | kLHintsExpandX));

   fHtml->Connect("MouseOver(char *)", "TGHtmlBrowser", this, "MouseOver(char *)");
   fHtml->Connect("MouseDown(char *)", "TGHtmlBrowser", this, "MouseDown(char *)");

   Selected(filename);

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
   Resize(w, h);
}

//______________________________________________________________________________
void TGHtmlBrowser::CloseWindow()
{
   // Close TGHtmlBrowser window.

   Cleanup();
   delete this;
   gApplication->Terminate(0);
}

//______________________________________________________________________________
static char *ReadRemote(const char *url)
{
   // Read (open) remote files.

   static char *buf = 0;
   TUrl fUrl(url);

   TString msg = "GET ";
   msg += fUrl.GetProtocol();
   msg += "://";
   msg += fUrl.GetHost();
   msg += ":";
   msg += fUrl.GetPort();
   msg += "/";
   msg += fUrl.GetFile();
   msg += "\r\n";

   TString uri(url);
   if (!uri.BeginsWith("http://"))
      return 0;
   TSocket s(fUrl.GetHost(), fUrl.GetPort());
   if (!s.IsValid())
      return 0;
   if (s.SendRaw(msg.Data(), msg.Length()) == -1)
      return 0;
   Int_t size = 1024*1024;
   buf = (char *)calloc(size, sizeof(char));
   if (s.RecvRaw(buf, size) == -1) {
      free(buf);
      return 0;
   }
   return buf;
}

//______________________________________________________________________________
void TGHtmlBrowser::Selected(const char *uri)
{
   // Open (browse) selected URL.

   char *buf = 0;
   FILE *f;

   TString surl(gSystem->UnixPathName(uri));
   if (!surl.BeginsWith("http://") && !surl.BeginsWith("file://"))
      surl.Prepend("file://");
   if (surl.EndsWith(".root")) {
      gVirtualX->SetCursor(fHtml->GetId(), gVirtualX->CreateCursor(kWatch));
      TFile *f = TFile::Open(surl.Data());
      if (f && !f->IsZombie()) {
         f->Browse(new TBrowser());
      }
      gVirtualX->SetCursor(fHtml->GetId(), gVirtualX->CreateCursor(kPointer));
      return;
   }
   gVirtualX->SetCursor(fHtml->GetId(), gVirtualX->CreateCursor(kWatch));
   TUrl url(surl.Data());
   if ((!strcmp(url.GetProtocol(), "http"))) {
      buf = ReadRemote(url.GetUrl());
      if (buf) {
         fHtml->Clear();
         fHtml->Layout();
         fHtml->SetBaseUri(url.GetUrl());
         fHtml->ParseText(buf);
         free(buf);
         fURL->SetText(surl.Data());
         if (!fComboBox->FindEntry(surl.Data()))
            fComboBox->AddEntry(surl.Data(), fComboBox->GetNumberOfEntries()+1);
      }
      else {
         fHtml->Clear();
         fHtml->Layout();
         fHtml->SetBaseUri("");
         for (int i=0; HtmlError[i]; i++) {
            fHtml->ParseText((char *)HtmlError[i]);
         }
      }
   }
   else {
      f = fopen(url.GetFile(), "r");
      if (f) {
         fHtml->Clear();
         fHtml->Layout();
         fHtml->SetBaseUri("");
         buf = (char *)calloc(4096, sizeof(char));
         while (fgets(buf, 4096, f)) {
            fHtml->ParseText(buf);
         }
         free(buf);
         fclose(f);
         fURL->SetText(surl.Data());
         if (!fComboBox->FindEntry(surl.Data()))
            fComboBox->AddEntry(surl.Data(), fComboBox->GetNumberOfEntries()+1);
      }
      else {
         fHtml->Clear();
         fHtml->Layout();
         fHtml->SetBaseUri("");
         for (int i=0; HtmlError[i]; i++) {
            fHtml->ParseText((char *)HtmlError[i]);
         }
      }
   }
   gVirtualX->SetCursor(fHtml->GetId(), gVirtualX->CreateCursor(kPointer));
   fHtml->Layout();
   SetWindowName(Form("%s - RHTML",surl.Data()));
}

//______________________________________________________________________________
void TGHtmlBrowser::URLChanged()
{
   // URL combobox has changed.

   const char *string = fURL->GetText();
   if (string) {
      Selected(StrDup(gSystem->UnixPathName(string)));
   }
}

//______________________________________________________________________________
void TGHtmlBrowser::Back()
{
   // Handle "Back" navigation button.

   Int_t index = 0;
   const char *string = fURL->GetText();
   TGLBEntry * lbe1 = fComboBox->FindEntry(string); 
   if (lbe1)
      index = lbe1->EntryId(); 
   if (index > 0) {
      fComboBox->Select(index - 1, kTRUE);
      TGTextLBEntry *entry = (TGTextLBEntry *)fComboBox->GetSelectedEntry();
      if (entry) {
         const char *string = entry->GetTitle();
         if (string)
            Selected(string);
      }
   }
}

//______________________________________________________________________________
void TGHtmlBrowser::Forward()
{
   // Handle "Forward" navigation button.

   Int_t index = 0;
   const char *string = fURL->GetText();
   TGLBEntry * lbe1 = fComboBox->FindEntry(string);
   if (lbe1)
      index = lbe1->EntryId();
   if (index < fComboBox->GetNumberOfEntries()) {
      fComboBox->Select(index + 1, kTRUE);
      TGTextLBEntry *entry = (TGTextLBEntry *)fComboBox->GetSelectedEntry();
      if (entry) {
         const char *string = entry->GetTitle();
         if (string)
            Selected(string);
      }
   }
}

//______________________________________________________________________________
void TGHtmlBrowser::Reload()
{
   // Handle "Reload" navigation button.

   const char *string = fURL->GetText();
   if (string)
      Selected(string);
}

//______________________________________________________________________________
void TGHtmlBrowser::Stop()
{
   // Handle "Reload" navigation button.

}

//______________________________________________________________________________
void TGHtmlBrowser::MouseOver(char *url)
{
   // Handle "MouseOver" TGHtml signal.

   fStatusBar->SetText(url, 0);
}

//______________________________________________________________________________
void TGHtmlBrowser::MouseDown(char *url)
{
   // Handle "MouseDown" TGHtml signal.

   Selected(url);
}

//______________________________________________________________________________
Bool_t TGHtmlBrowser::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process Events.

   switch (GET_MSG(msg)) {
   case kC_COMMAND:
      {
         switch (GET_SUBMSG(msg)) {

            case kCM_MENU:
            case kCM_BUTTON:

               switch(parm1) {

                  case M_FILE_EXIT:
                     CloseWindow();
                     break;

                  case M_FILE_OPEN:
                     {
                        static TString dir(".");
                        TGFileInfo fi;
                        fi.fFileTypes = filetypes;
                        fi.fIniDir    = StrDup(dir);
                        new TGFileDialog(fClient->GetRoot(), this,
                                         kFDOpen, &fi);
                        dir = fi.fIniDir;
                        if (fi.fFilename) {
                           Selected(StrDup(Form("file://%s",
                              gSystem->UnixPathName(fi.fFilename))));
                        }
                     }
                     break;

                  case M_FAVORITES_ADD:
                     fMenuFavorites->AddEntry(Form("%s",
                           fURL->GetText()), fNbFavorites++, 0, 
                           gClient->GetPicture("htmlfile.gif"));
                     break;

                  case M_TOOLS_CLEARHIST:
                     fComboBox->RemoveEntries(1,fComboBox->GetNumberOfEntries());
                     break;

                  case M_FILE_BROWSE:
                     new TBrowser();
                     break;
                  
                  case M_HELP_ABOUT:
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
                        TRootHelpDialog *hd = new TRootHelpDialog(this, str,
                                                                  600, 400);
                        hd->SetText(gHelpAbout);
                        hd->Popup();
#endif
#endif
                     }
                     break;

                  default:
                     {
                        if (parm1 < 1000) break;
                        TGMenuEntry *entry = fMenuFavorites->GetEntry(parm1);
                        if (!entry) break;
                        const char *shortcut = entry->GetName();
                        if (shortcut)
                           Selected(shortcut);
                     }
                     break;
               }
               break;
         }
         break;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // Main application.
   
   TApplication theApp("App", &argc, argv);
   new TGHtmlBrowser("http://root.cern.ch/drupal/");
   theApp.Run();
   return 0;
}
