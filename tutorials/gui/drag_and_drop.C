/// \file
/// \ingroup tutorial_gui
/// This tutorial illustrates how to use drag and drop within ROOT.
/// Select a list tree item with a mouse press, drag it (move the mouse while keeping the mouse button pressed)
/// and release the mouse button in any pad inside the canvas or in the top list tree item ("Base").
/// When the button is released, the selected data is "dropped" at that location,
/// displaying the object in the canvas or adding (copying) it in the list tree.
///
/// \macro_code
///
/// \author Bertrand Bellenot

#include "TROOT.h"
#include "TApplication.h"
#include "TSystem.h"
#include "TGFrame.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "TGFileDialog.h"
#include "TBrowser.h"
#include "TRootEmbeddedCanvas.h"
#include "TRootHelpDialog.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF2.h"
#include "TGraph.h"
#include "TImage.h"
#include "TRandom.h"
#include "TGMsgBox.h"
#include "TGPicture.h"
#include "TGListTree.h"
#include "TObjString.h"
#include "TMessage.h"
#include "TTimer.h"
#include "TGDNDManager.h"
#include <cmath>

const char gHelpDND[] = "\
                     Drag & Drop (DnD)\n\
Drag and Drop support is implemented on Linux via Xdnd, the\n\
drag and drop protocol for X window system, and on Windows\n\
via the Clipboard.\n\
Users can selects something in ROOT with a mouse press, drags\n\
it (moves the mouse while keeping the mouse button pressed) and\n\
releases the mouse button somewhere else. When the button is\n\
released the selected data is \"dropped\" at that location. This\n\
way, a histogram from an opened ROOT file in the browser can be\n\
dragged to any TCanvas. A script file from the browser can be\n\
dropped to a TGTextView or TGTextEdit widget in TGTextEditor.\n\
On Linux, it is possible to drag objects between ROOT and an\n\
external application. For example to drag a macro file from the\n\
ROOT browser to the Kate editor. On Windows, drag and drop works\n\
only within a single ROOT application, but it is possible to drag\n\
from the Windows Explorer to ROOT (e.g. a picture file to a canvas\n\
or a text file to a text editor).\n\
";

const char gReadyMsg[] = "Ready. You can drag list tree items to any \
pad in the canvas, or to the \"Base\" folder of the list tree itself...";

//----------------------------------------------------------------------

class DNDMainFrame : public TGMainFrame {

protected:
   TRootEmbeddedCanvas  *fEc;          // embedded canvas
   TGTextButton         *fButtonExit;  // "Exit" text button
   TGMenuBar            *fMenuBar;     // main menu bar
   TGPopupMenu          *fMenuFile;    // "File" popup menu entry
   TGPopupMenu          *fMenuHelp;    // "Help" popup menu entry
   TCanvas              *fCanvas;      // canvas
   TGListTree           *fListTree;    // left list tree
   TGListTreeItem       *fBaseLTI;     // base (root) list tree item
   TGLabel              *fStatus;      // label used to display status
   TGraph               *fGraph;       // TGraph object
   TH1F                 *fHist1D;      // 1D histogram
   TH2F                 *fHist2D;      // 2D histogram

public:
   DNDMainFrame(const TGWindow *p, int w, int h);
   virtual ~DNDMainFrame();

   void              DoCloseWindow();
   void              HandleMenu(Int_t);
   TObject          *GetObject(const char *obj);
   void              DataDropped(TGListTreeItem* item, TDNDData* data);
   void              ResetStatus();

   //ClassDef(DNDMainFrame, 0); // Mainframe for Drag and Drop demo
};

enum EMyMessageTypes {
   M_FILE_OPEN,
   M_FILE_BROWSE,
   M_FILE_NEWCANVAS,
   M_FILE_CLOSEWIN,
   M_FILE_EXIT,
   M_HELP_ABOUT
};

const char *dnd_types[] = {
   "ROOT files",    "*.root",
   "ROOT macros",   "*.C",
   "All files",     "*",
    0,               0
};

static Atom_t gRootObj  = kNone;

//______________________________________________________________________________
DNDMainFrame::DNDMainFrame(const TGWindow *p, int w, int h) :
   TGMainFrame(p, w, h), fGraph(0), fHist1D(0), fHist2D(0)

{
   // Constructor.

   SetCleanup(kDeepCleanup);
   const TGPicture *pic = 0;
   TGListTreeItem *item;
   fMenuBar = new TGMenuBar(this, 35, 50, kHorizontalFrame);

   fMenuFile = new TGPopupMenu(gClient->GetRoot());
   fMenuFile->AddEntry(" &Open...\tCtrl+O", M_FILE_OPEN, 0,
                       gClient->GetPicture("bld_open.png"));
   fMenuFile->AddEntry(" &Browse...\tCtrl+B", M_FILE_BROWSE);
   fMenuFile->AddEntry(" &New Canvas\tCtrl+N", M_FILE_NEWCANVAS);
   fMenuFile->AddEntry(" &Close Window\tCtrl+W", M_FILE_CLOSEWIN);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry(" E&xit\tCtrl+Q", M_FILE_EXIT, 0,
                       gClient->GetPicture("bld_exit.png"));
   fMenuFile->Connect("Activated(Int_t)", "DNDMainFrame", this,
                      "HandleMenu(Int_t)");

   fMenuHelp = new TGPopupMenu(gClient->GetRoot());
   fMenuHelp->AddEntry(" &About...", M_HELP_ABOUT, 0,
                       gClient->GetPicture("about.xpm"));
   fMenuHelp->Connect("Activated(Int_t)", "DNDMainFrame", this,
                      "HandleMenu(Int_t)");

   fMenuBar->AddPopup("&File", fMenuFile, new TGLayoutHints(kLHintsTop|kLHintsLeft,
                                                            0, 4, 0, 0));

   fMenuBar->AddPopup("&Help", fMenuHelp, new TGLayoutHints(kLHintsTop|kLHintsRight));

   AddFrame(fMenuBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 2, 5));

   TGHorizontalFrame *hfrm = new TGHorizontalFrame(this, 10, 10);
   TGCanvas *canvas = new TGCanvas(hfrm, 150, 100);
   fListTree = new TGListTree(canvas, kHorizontalFrame);
   fListTree->Associate(this);
   fEc = new TRootEmbeddedCanvas("glec", hfrm, 550, 350);
   hfrm->AddFrame(canvas, new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 5, 5));
   hfrm->AddFrame(fEc, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   AddFrame(hfrm, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   fEc->SetDNDTarget(kTRUE);
   fCanvas = fEc->GetCanvas();
   fCanvas->Divide(3, 2);
   fCanvas->SetBorderMode(0);
   fBaseLTI = fListTree->AddItem(0, "Base");

   TGHorizontalFrame *hf = new TGHorizontalFrame(this, 10, 10);

   fStatus = new TGLabel(hf, new TGHotString(gReadyMsg));
   fStatus->SetTextJustify(kTextLeft);
   fStatus->SetTextColor(0x0000ff);
   hf->AddFrame(fStatus, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY,
                10, 10, 10, 10));

   fButtonExit = new TGTextButton(hf, "        &Exit...        ", 3);
   fButtonExit->Resize(fButtonExit->GetDefaultSize());
   fButtonExit->SetToolTipText("Exit Application (ROOT)");
   fButtonExit->Connect("Clicked()" , "TApplication", gApplication,
                        "Terminate()");
   hf->AddFrame(fButtonExit, new TGLayoutHints(kLHintsCenterY | kLHintsRight,
                                               10, 10, 10, 10));

   AddFrame(hf, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 5, 5, 5, 5));

   gRootObj  = gVirtualX->InternAtom("application/root", kFALSE);

   TGraph *gr = (TGraph *)GetObject("Graph");
   pic = gClient->GetPicture("f1_t.xpm");
   item = fListTree->AddItem(fBaseLTI, gr->GetName(), gr, pic, pic);
   fListTree->SetToolTipItem(item, "Simple Graph");
   item->SetDNDSource(kTRUE);

   TH1F *hpx = (TH1F *)GetObject("1D Hist");
   pic = gClient->GetPicture("h1_t.xpm");
   item = fListTree->AddItem(fBaseLTI, hpx->GetName(), hpx, pic, pic);
   fListTree->SetToolTipItem(item, "1D Histogram");
   item->SetDNDSource(kTRUE);

   TH2F *h2 = (TH2F *)GetObject("2D Hist");
   pic = gClient->GetPicture("h2_t.xpm");
   item = fListTree->AddItem(fBaseLTI, h2->GetName(), h2, pic, pic);
   fListTree->SetToolTipItem(item, "2D Histogram");
   item->SetDNDSource(kTRUE);

   TString rootsys(gSystem->UnixPathName(gSystem->Getenv("ROOTSYS")));
#ifdef R__WIN32
   // remove the drive letter (e.g. "C:/") from $ROOTSYS, if any
   if (rootsys[1] == ':' && rootsys[2] == '/')
      rootsys.Remove(0, 3);
#endif
   TString link = TString::Format("/%s/tutorials/image/rose512.jpg",
                                  rootsys.Data());
   if (!gSystem->AccessPathName(link.Data(), kReadPermission)) {
      TImage *img = TImage::Open(link.Data());
      if (img) {
         // create a 16x16 icon from the original picture
         img->Scale(16, 16);
         pic = gClient->GetPicturePool()->GetPicture("rose512", img->GetPixmap(),
                                                     img->GetMask());
         delete img;
      }
      else pic = gClient->GetPicture("psp_t.xpm");
      link.Prepend("file://");
      TObjString *ostr = new TObjString(link.Data());
      item = fListTree->AddItem(fBaseLTI, "Rose", ostr, pic, pic);
      fListTree->SetToolTipItem(item, link.Data());
      item->SetDNDSource(kTRUE);
   }

   // open the base list tree item and allow to drop into it
   fListTree->OpenItem(fBaseLTI);
   fListTree->GetFirstItem()->SetDNDTarget(kTRUE);

   // connect the DataDropped signal to be able to handle it
   fListTree->Connect("DataDropped(TGListTreeItem*, TDNDData*)", "DNDMainFrame",
                      this, "DataDropped(TGListTreeItem*,TDNDData*)");

   SetWindowName("ROOT DND Demo Application");
   MapSubwindows();
   Resize(GetDefaultSize());
   Connect("CloseWindow()", "DNDMainFrame", this, "DoCloseWindow()");
   DontCallClose(); // to avoid double deletions.
}

//______________________________________________________________________________
DNDMainFrame::~DNDMainFrame()
{
   // Destructor. Doesnt't do much here.
}

//______________________________________________________________________________
void DNDMainFrame::DoCloseWindow()
{
   // Do some cleanup, disconnect signals and then really close the main window.

   if (fGraph) { delete fGraph; fGraph = 0; }
   if (fHist1D) { delete fHist1D; fHist1D = 0; }
   if (fHist2D) { delete fHist2D; fHist2D = 0; }
   fMenuFile->Disconnect("Activated(Int_t)", this, "HandleMenu(Int_t)");
   fMenuHelp->Disconnect("Activated(Int_t)", this, "HandleMenu(Int_t)");
   fButtonExit->Disconnect("Clicked()" , this, "CloseWindow()");
   fListTree->Disconnect("DataDropped(TGListTreeItem*, TDNDData*)", this,
                         "DataDropped(TGListTreeItem*,TDNDData*)");
   delete fListTree;
   CloseWindow();
}

//______________________________________________________________________________
void DNDMainFrame::DataDropped(TGListTreeItem *, TDNDData *data)
{
   // Handle the drop event in the TGListTree. This will just create a new
   // list tree item and copy the received data into it.

   fStatus->SetTextColor(0xff0000);
   fStatus->ChangeText("I received data!!!");
   if (data) {
      const TGPicture *pic = 0;
      TGListTreeItem *itm = 0;
      char tmp[1000];
      if (data->fDataType == gRootObj) {
         TBufferFile buf(TBuffer::kRead, data->fDataLength, (void *)data->fData);
         buf.SetReadMode();
         TObject *obj = (TObject *)buf.ReadObjectAny(TObject::Class());
         sprintf(tmp, "Received DND data : Type = \"%s\"; Length = %d bytes;",
                 obj->ClassName(), data->fDataLength);
         if (obj->InheritsFrom("TGraph"))
            pic = gClient->GetPicture("f1_t.xpm");
         else if (obj->InheritsFrom("TH2F"))
            pic = gClient->GetPicture("h2_t.xpm");
         else if (obj->InheritsFrom("TH1F"))
            pic = gClient->GetPicture("h1_t.xpm");
         itm = fListTree->AddItem(fBaseLTI, obj->GetName(), obj, pic, pic);
         fListTree->SetToolTipItem(itm, obj->GetName());
      }
      else {
         sprintf(tmp, "Received DND data: \"%s\"", (char *)data->fData);
         TObjString *ostr = new TObjString((char *)data->fData);
         TImage *img1 = TImage::Open("doc_t.xpm");
         TImage *img2 = TImage::Open("slink_t.xpm");
         if (img1 && img2) {
            img1->Merge(img2);
            pic = gClient->GetPicturePool()->GetPicture("doc_lnk", img1->GetPixmap(),
                                                        img1->GetMask());
            delete img2;
            delete img1;
         }
         else pic = gClient->GetPicture("doc_t.xpm");
         itm = fListTree->AddItem(fBaseLTI, "Link...", ostr, pic, pic);
         fListTree->SetToolTipItem(itm, (const char *)data->fData);
      }
      if (itm) itm->SetDNDSource(kTRUE);
      fStatus->ChangeText(tmp);
   }
   TTimer::SingleShot(3000, "DNDMainFrame", this, "ResetStatus()");
}

//______________________________________________________________________________
TObject *DNDMainFrame::GetObject(const char *obj)
{
   // Return the object specified in argument. If the object doesn't exist yet,
   // it is firt created.

   if (!strcmp(obj, "Graph")) {
      if (fGraph == 0) {
         const Int_t n = 20;
         Double_t x[n], y[n];
         for (Int_t i=0;i<n;i++) {
           x[i] = i*0.1;
           y[i] = 10*sin(x[i]+0.2);
         }
         fGraph = new TGraph(n, x, y);
      }
      return fGraph;
   }
   else if (!strcmp(obj, "1D Hist")) {
      if (fHist1D == 0) {
         fHist1D = new TH1F("1D Hist","This is the px distribution",100,-4,4);
         Float_t px, py;
         for ( Int_t i=0; i<10000; i++) {
            gRandom->Rannor(px, py);
            fHist1D->Fill(px);
         }
      }
      return fHist1D;
   }
   else if (!strcmp(obj, "2D Hist")) {
      if (fHist2D == 0) {
         Double_t params[] = {
            130,-1.4,1.8,1.5,1, 150,2,0.5,-2,0.5, 3600,-2,0.7,-3,0.3
         };
         TF2 *f2 = new TF2("f2","xygaus + xygaus(5) + xylandau(10)",
                           -4, 4, -4, 4);
         f2->SetParameters(params);
         fHist2D = new TH2F("2D Hist","xygaus+xygaus(5)+xylandau(10)",
                            20, -4, 4, 20, -4, 4);
         fHist2D->FillRandom("f2",40000);
      }
      return fHist2D;
   }
   return 0;
}

//______________________________________________________________________________
void DNDMainFrame::HandleMenu(Int_t menu_id)
{
   // Handle menu events.

   TRootHelpDialog *hd;
   static TString dir(".");
   TGFileInfo fi;
   fi.fFileTypes = dnd_types;
   fi.SetIniDir(dir);

   switch (menu_id) {
      case M_FILE_EXIT:
         // close the window and quit application
         DoCloseWindow();
         gApplication->Terminate(0);
         break;
      case M_FILE_OPEN:
         new TGFileDialog(gClient->GetRoot(), this, kFDOpen, &fi);
         dir = fi.fIniDir;
         // doesn't do much, but can be used to open a root file...
         break;
      case M_FILE_BROWSE:
         // start a root object browser
         new TBrowser();
         break;
      case M_FILE_NEWCANVAS:
         // open a root canvas
         gROOT->MakeDefCanvas();
         break;
      case M_FILE_CLOSEWIN:
         DoCloseWindow();
         break;
      case M_HELP_ABOUT:
         hd = new TRootHelpDialog(this, "About Drag and Drop...", 550, 250);
         hd->SetText(gHelpDND);
         hd->Popup();
         break;
   }
}

//______________________________________________________________________________
void DNDMainFrame::ResetStatus()
{
   // Restore the original text of the status label and its original color.

   fStatus->SetTextColor(0x0000ff);
   fStatus->ChangeText(gReadyMsg);
}

//------------------------------------------------------------------------------
void drag_and_drop()
{
   // Main function (entry point)

   DNDMainFrame *mainWindow = new DNDMainFrame(gClient->GetRoot(), 700, 400);
   mainWindow->MapWindow();
}

