// @(#)root/gui:$Name:  $:$Id: TRootBrowser.cxx,v 1.12 2001/07/05 16:50:12 rdm Exp $
// Author: Fons Rademakers   27/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootBrowser                                                         //
//                                                                      //
// This class creates a ROOT object browser (looking like Windows       //
// Explorer). The widgets used are the new native ROOT GUI widgets.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootBrowser.h"
#include "TRootApplication.h"
#include "TGCanvas.h"
#include "TGMenu.h"
#include "TGFileDialog.h"
#include "TGStatusBar.h"
#include "TGFSComboBox.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGListView.h"
#include "TGListTree.h"
#include "TGToolBar.h"
#include "TGSplitter.h"
#include "TG3DLine.h"
#include "TGFSContainer.h"
#include "TGMimeTypes.h"
#include "TRootHelpDialog.h"

#include "TROOT.h"
#include "TEnv.h"
#include "TBrowser.h"
#include "TApplication.h"
#include "TFile.h"
#include "TKey.h"
#include "TKeyMapFile.h"
#include "TClass.h"
#include "TContextMenu.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include "TInterpreter.h"

#include "HelpText.h"


// Browser menu command ids
enum ERootBrowserCommands {
   kFileNewBrowser,
   kFileNewCanvas,
   kFileOpen,
   kFileSave,
   kFileSaveAs,
   kFilePrint,
   kFileCloseBrowser,
   kFileQuit,

   kViewToolBar,
   kViewStatusBar,
   kViewLargeIcons,
   kViewSmallIcons,
   kViewList,
   kViewDetails,
   kViewLineUp,
   kViewRefresh,

   kViewArrangeByName,     // Arrange submenu
   kViewArrangeByType,
   kViewArrangeBySize,
   kViewArrangeByDate,
   kViewArrangeAuto,
   kViewGroupLV,

   kOptionShowCycles,

   kOneLevelUp,            // One level up toolbar button
   kFSComboBox,            // File system combobox in toolbar

   kHelpAbout,
   kHelpOnBrowser,
   kHelpOnCanvas,
   kHelpOnMenus,
   kHelpOnGraphicsEd,
   kHelpOnObjects,
   kHelpOnPS
};


//----- Struct for default icons

struct DefaultIcon_t {
   const char      *fPicnamePrefix;
   const TGPicture *fIcon[2];
};

#if 0
static DefaultIcon_t gDefaultIcon[] = {
   { "folder",  { 0, 0 } },
   { "app",     { 0, 0 } },
   { "doc",     { 0, 0 } },
   { "slink",   { 0, 0 } },
   { "histo",   { 0, 0 } },
   { "object",  { 0, 0 } }
};
#endif


//----- Toolbar stuff...

static ToolBarData_t gToolBarData[] = {
   { "tb_uplevel.xpm",   "Up One Level",   kFALSE, kOneLevelUp, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "tb_bigicons.xpm",  "Large Icons",    kTRUE,  kViewLargeIcons, 0 },
   { "tb_smicons.xpm",   "Small Icons",    kTRUE,  kViewSmallIcons, 0 },
   { "tb_list.xpm",      "List",           kTRUE,  kViewList, 0 },
   { "tb_details.xpm",   "Details",        kTRUE,  kViewDetails, 0 },
   { 0,                  0,                kFALSE, 0, 0 }
};


//----- TGFileDialog file types

static const char *gOpenTypes[] = { "ROOT files",   "*.root",
                                    "All files",    "*",
                                    0,              0 };

//----- Special ROOT object item (this are items in the icon box, see
//----- TRootIconBox)

class TRootObjItem : public TGFileItem {
public:
   TRootObjItem(const TGWindow *p, const TGPicture *bpic,
                const TGPicture *spic, TGString *name,
                TObject *obj, TClass *cl, EListViewMode viewMode);
};

//______________________________________________________________________________
TRootObjItem::TRootObjItem(const TGWindow *p, const TGPicture *bpic,
                           const TGPicture *spic, TGString *name,
                           TObject *obj, TClass *, EListViewMode viewMode) :
   TGFileItem(p, bpic, 0, spic, 0, name, 0, 0, 0, 0, viewMode)
{
   // Create an icon box item.

   delete [] fSubnames;
   fSubnames = new TGString* [2];

   fSubnames[0] = new TGString(obj->GetTitle());

   fSubnames[1] = 0;

   int i;
   for (i = 0; fSubnames[i] != 0; ++i)
      ;
   fCtw = new int[i];
   for (i = 0; fSubnames[i] != 0; ++i)
      fCtw[i] = gVirtualX->TextWidth(fFontStruct, fSubnames[i]->GetString(),
                                     fSubnames[i]->GetLength());
}

class TRootIconBox;
class TRootIconList : public TList {

private:
   TRootIconBox    *fIconBox; //
   const TGPicture *fPic;     // list view icon
   TString          fName;    // the name of the list

public:
   TRootIconList(TRootIconBox* box = 0);
   ~TRootIconList();
   const char       *GetName() const { return fName.Data(); }
   void              UpdateName();
   const char       *GetTitle() const { return "ListView Container"; }
   Bool_t            IsFolder() const { return kFALSE; }
   void              Browse(TBrowser *b);
   const TGPicture  *GetPicture() const { return fPic; }
};

//______________________________________________________________________________
TRootIconList::TRootIconList(TRootIconBox* box)
{
   // ctor

   fPic = gClient->GetPicture("listview.xpm");
   fIconBox = box;
   fName = "empty";
}

//______________________________________________________________________________
TRootIconList::~TRootIconList()
{
   // dtor

   gClient->FreePicture(fPic);
}

//______________________________________________________________________________
void TRootIconList::UpdateName()
{
   // composite name

   if (!First()) return;

   if (fSize==1) {
      fName = First()->GetName();
      return;
   }

   fName = First()->GetName();
   fName += "-";
   fName += Last()->GetName();
}

//----- Special ROOT object container (this is the icon box on the
//----- right side of the browser)

class TRootIconBox : public TGFileContainer {
friend class TRootIconList;

private:
   Bool_t           fCheckHeaders;   // if true check headers
   TRootIconList   *fCurrentList;    //
   TRootObjItem    *fCurrentItem;    //
   Bool_t           fGrouped;        //
   TString          fCachedPicName;  //
   TList           *fGarbage;        // garbage for  TRootIconList's
   Int_t            fGroupSize;   // the total number of items when icon box switched to "global view" mode
   TGString        *fCurrentName;    //
   const TGPicture *fLargeCachedPic;  //
   const TGPicture *fSmallCachedPic;  //
   Bool_t           fWasGrouped;

public:
   TRootIconBox(const TGWindow *p, TGListView *lv, UInt_t w, UInt_t h,
                UInt_t options = kSunkenFrame,
                ULong_t back = fgDefaultFrameBackground);

   virtual ~TRootIconBox();

   void   AddObjItem(const char *name, TObject *obj, TClass *cl);
   void   GetObjPictures(const TGPicture **pic, const TGPicture **spic,
                         TObject *obj, const char *name);
   void   SetObjHeaders();
   void   Refresh();
   void   RemoveAll();
   void   SetGroupSize(Int_t siz) { fGroupSize = siz; }
   Int_t  GetGroupSize() const { return fGroupSize; }
   Bool_t WasGrouped() const { return fWasGrouped; }
};

//______________________________________________________________________________
TRootIconBox::TRootIconBox(const TGWindow *p, TGListView *lv, UInt_t w,
                           UInt_t h, UInt_t options, ULong_t back) :
   TGFileContainer(p, w, h, options, back)
{
   // Create iconbox containing ROOT objects in browser.

   fListView = lv;
   fCheckHeaders = kTRUE;
   fTotal = 0;
   fGarbage = new TList();
   fCurrentList = 0;
   fCurrentItem = 0;
   fGrouped = kFALSE;
   fGroupSize = 1000;
   fCurrentName = 0;
   fWasGrouped = kFALSE;

   // Don't use timer HERE (timer is set in TBrowser).
   delete fRefresh;
   fRefresh = 0;
}

//______________________________________________________________________________
TRootIconBox::~TRootIconBox()
{
   // dtor

   fGarbage->Delete();
   delete fGarbage;
}

//______________________________________________________________________________
void TRootIconBox::GetObjPictures(const TGPicture **pic, const TGPicture **spic,
                                  TObject *obj, const char *name)
{
   // Retrieve icons associated with class "name". Association is made
   // via the user's ~/.root.mimes file or via $ROOTSYS/etc/root.mimes.

   if(fCachedPicName==name) {
      *pic = fLargeCachedPic;
      *spic = fSmallCachedPic;
      return;
   }

   *pic = fClient->GetMimeTypeList()->GetIcon(name, kFALSE);

   if (*pic == 0) {
      if (obj->IsFolder())
         *pic = fFolder_s;
      else
         *pic = fDoc_s;
   }
   fLargeCachedPic = *pic;

   *spic = fClient->GetMimeTypeList()->GetIcon(name, kTRUE);

   if (*spic == 0) {
      if (obj->IsFolder())
         *spic = fFolder_t;
      else
         *spic = fDoc_t;
   }
   fSmallCachedPic = *spic;
   fCachedPicName = name;
}

//______________________________________________________________________________
void TRootIconBox::AddObjItem(const char *name, TObject *obj, TClass *cl)
{
   // Add object to iconbox. Class is used to get the associated icons
   // via the mime file (see GetObjPictures()).

   if (!cl) return;
   TRootBrowser *browser = (TRootBrowser *)fListView->GetParent()->GetParent()->GetParent();

   TGFileItem *fi;
   fWasGrouped = kFALSE;

   if (obj->IsA() == TSystemFile::Class() ||
       obj->IsA() == TSystemDirectory::Class()) {
      if (fCheckHeaders) {
         if (strcmp(fListView->GetHeader(1), "Attributes"))
            fListView->SetDefaultHeaders();
         fCheckHeaders = kFALSE;
      }
      fi = AddFile(name);
      if (fi) fi->SetUserData(obj);
      return;
   }

   if (!fCurrentList) {
      fCurrentList = new TRootIconList(this);
      fGarbage->Add(fCurrentList);
   }

   fCurrentList->Add(obj);
   fCurrentList->UpdateName();

   if (fGrouped && fCurrentItem && (fCurrentList->GetSize()>1)) {
      fCurrentName->SetString(fCurrentList->GetName());
   }

   //
   if ((fCurrentList->GetSize()<fGroupSize) && !fGrouped) {
      const TGPicture *pic = 0;
      const TGPicture *spic = 0;

      GetObjPictures(&pic, &spic, obj, obj->GetIconName() ?
                     obj->GetIconName() : cl->GetName());

      if (fCheckHeaders) {
         if (strcmp(fListView->GetHeader(1), "Title"))
            SetObjHeaders();
         fCheckHeaders = kFALSE;
      }

      fi = new TRootObjItem(this, pic, spic, new TGString(name), obj, cl, fViewMode);

      fi->SetUserData(obj);
      AddItem(fi);
      fTotal++;
      return;
   }

   if (fGrouped && (fCurrentList->GetSize()==1)) {
      fCurrentName = new TGString(fCurrentList->GetName());
      fCurrentItem = new TRootObjItem(this, fCurrentList->GetPicture(), fCurrentList->GetPicture(),
                                      fCurrentName,fCurrentList, TList::Class(), fViewMode);
      fCurrentItem->SetUserData(fCurrentList);
      AddItem(fCurrentItem);
      fTotal = fList->GetSize();
      return;
   }

   if ((fCurrentList->GetSize()==fGroupSize) && !fGrouped) {
      browser->SetViewMode(kLVLargeIcons,kTRUE);
      fGrouped = kTRUE;
      fList->Delete();
      fCurrentName = new TGString(fCurrentList->GetName());
      fi = new TRootObjItem(this, fCurrentList->GetPicture(), fCurrentList->GetPicture(),
                            fCurrentName, fCurrentList, TList::Class(), fViewMode);
      fi->SetUserData(fCurrentList);
      AddItem(fi);

      fCurrentList = new TRootIconList(this);
      fGarbage->Add(fCurrentList);
      fTotal = 1;
      return;
   }

   if ((fCurrentList->GetSize()==fGroupSize) && fGrouped) {
      fCurrentList = new TRootIconList(this);
      fGarbage->Add(fCurrentList);
      return;
   }
}

//______________________________________________________________________________
void TRootIconList::Browse(TBrowser *b)
{
   // browse icon list

   if (!fIconBox) return;

   gVirtualX->SetCursor(fIconBox->fId, gVirtualX->CreateCursor(kWatch));
   gVirtualX->Update();

   TObject *obj;
   TGFileItem *fi;
   const TGPicture *pic = 0;
   const TGPicture *spic = 0;
   TClass *cl;
   const char *name;

   fIconBox->RemoveAll();
   TObjLink *lnk = FirstLink();

   while (lnk) {
      obj = lnk->GetObject();
      lnk = lnk->Next();
      cl = obj->IsA();
      name = obj->GetName();

      fIconBox->GetObjPictures(&pic, &spic, obj, obj->GetIconName() ?
                               obj->GetIconName() : cl->GetName());

      fi = new TRootObjItem((const TGWindow*)fIconBox, pic, spic, new TGString(name),
                             obj, cl, (EListViewMode)fIconBox->GetViewMode());
      fi->SetUserData(obj);
      fIconBox->AddItem(fi);
      fIconBox->fTotal++;
   }
   fIconBox->fGarbage->Remove(this);
   fIconBox->fGarbage->Delete();
   fIconBox->fGarbage->Add(this);
   fIconBox->Refresh();
   fIconBox->fWasGrouped = kTRUE;
   gVirtualX->SetCursor(fIconBox->fId, kNone);
}

//______________________________________________________________________________
void TRootIconBox::SetObjHeaders()
{
   // Set list box headers used to display detailed object iformation.
   // Currently this is only "Name" and "Title".

   fListView->SetHeaders(2);
   fListView->SetHeader("Name",  kTextLeft, kTextLeft, 0);
   fListView->SetHeader("Title", kTextLeft, kTextLeft, 1);
}

//______________________________________________________________________________
void TRootIconBox::Refresh()
{
   // Sort icons, and send message to browser with number of objects
   // in box.

   // This automatically calls layout
   Sort(fSortType);

   // Make TRootBrowser display total objects in status bar
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),fTotal, fSelected);
}

//______________________________________________________________________________
void TRootIconBox::RemoveAll()
{
   // Remove all items from icon box

   fCheckHeaders = kTRUE;
   TGFileContainer::RemoveAll();
   fGrouped = kFALSE;
   fCurrentItem = 0;
   fCurrentList = 0;
}



ClassImp(TRootBrowser)

//______________________________________________________________________________
TRootBrowser::TRootBrowser(TBrowser *b, const char *name, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height), TBrowserImp(b)
{
   // Create browser with a specified width and height.

   CreateBrowser(name);

   Resize(width, height);
   Show();
}

//______________________________________________________________________________
TRootBrowser::TRootBrowser(TBrowser *b, const char *name, Int_t x, Int_t y,
                           UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height), TBrowserImp(b)
{
   // Create browser with a specified width and height and at position x, y.

   CreateBrowser(name);

   MoveResize(x, y, width, height);
   SetWMPosition(x, y);
   Show();
}

//______________________________________________________________________________
TRootBrowser::~TRootBrowser()
{
   // Browser destructor.

   UnmapWindow();
   delete fToolBarSep;
   delete fToolBar;
   delete fFSComboBox;
   delete fStatusBar;
   delete fV1;
   delete fV2;
   delete fLbl1;
   delete fLbl2;
   delete fHf;
   delete fTreeHdr;
   delete fListHdr;
   delete fIconBox;
   delete fListView;
   delete fLt;
   delete fTreeView;

   delete fMenuBar;
   delete fFileMenu;
   delete fViewMenu;
   delete fOptionMenu;
   delete fHelpMenu;
   delete fSortMenu;

   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fComboLayout;
   delete fBarLayout;

   if (fWidgets) fWidgets->Delete();
   delete fWidgets;
}

//______________________________________________________________________________
void TRootBrowser::CreateBrowser(const char *name)
{
   // Create the actual canvas.

   fWidgets = new TList;

   // Create menus
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&New Browser",        kFileNewBrowser);
   fFileMenu->AddEntry("New Canvas",          kFileNewCanvas);
   fFileMenu->AddEntry("&Open...",            kFileOpen);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Save",               kFileSave);
   fFileMenu->AddEntry("Save As...",          kFileSaveAs);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Print...",           kFilePrint);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Close Browser",      kFileCloseBrowser);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Quit ROOT",          kFileQuit);

   //fFileMenu->DefaultEntry(kFileNewCanvas);
   fFileMenu->DisableEntry(kFileSave);
   fFileMenu->DisableEntry(kFileSaveAs);
   fFileMenu->DisableEntry(kFilePrint);

   fSortMenu = new TGPopupMenu(fClient->GetRoot());
   fSortMenu->AddEntry("By &Name",            kViewArrangeByName);
   fSortMenu->AddEntry("By &Type",            kViewArrangeByType);
   fSortMenu->AddEntry("By &Size",            kViewArrangeBySize);
   fSortMenu->AddEntry("By &Date",            kViewArrangeByDate);
   fSortMenu->AddSeparator();
   fSortMenu->AddEntry("&Auto Arrange",       kViewArrangeAuto);

   fSortMenu->CheckEntry(kViewArrangeAuto);

   fViewMenu = new TGPopupMenu(fClient->GetRoot());
   fViewMenu->AddEntry("&Toolbar",            kViewToolBar);
   fViewMenu->AddEntry("Status &Bar",         kViewStatusBar);
   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("Lar&ge Icons",        kViewLargeIcons);
   fViewMenu->AddEntry("S&mall Icons",        kViewSmallIcons);
   fViewMenu->AddEntry("&List",               kViewList);
   fViewMenu->AddEntry("&Details",            kViewDetails);
   fViewMenu->AddSeparator();
   fViewMenu->AddPopup("Arrange &Icons",      fSortMenu);
   fViewMenu->AddEntry("Lin&e up Icons",      kViewLineUp);
   fViewMenu->AddEntry("&Group Icons",        kViewGroupLV);

   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("&Refresh",            kViewRefresh);

   fViewMenu->CheckEntry(kViewToolBar);
   fViewMenu->CheckEntry(kViewStatusBar);

   fOptionMenu = new TGPopupMenu(fClient->GetRoot());
   fOptionMenu->AddEntry("&Show Cycles",        kOptionShowCycles);

   fHelpMenu = new TGPopupMenu(fClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...",        kHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help On Browser...",    kHelpOnBrowser);
   fHelpMenu->AddEntry("Help On Canvas...",     kHelpOnCanvas);
   fHelpMenu->AddEntry("Help On Menus...",      kHelpOnMenus);
   fHelpMenu->AddEntry("Help On Graphics Editor...", kHelpOnGraphicsEd);
   fHelpMenu->AddEntry("Help On Objects...",    kHelpOnObjects);
   fHelpMenu->AddEntry("Help On PostScript...", kHelpOnPS);

   // This main frame will process the menu commands
   fFileMenu->Associate(this);
   fViewMenu->Associate(this);
   fSortMenu->Associate(this);
   fOptionMenu->Associate(this);
   fHelpMenu->Associate(this);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File",    fFileMenu,    fMenuBarItemLayout);
   fMenuBar->AddPopup("&View",    fViewMenu,    fMenuBarItemLayout);
   fMenuBar->AddPopup("&Options", fOptionMenu,  fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);

   AddFrame(fMenuBar, fMenuBarLayout);

   // Create toolbar and separator

   fToolBarSep = new TGHorizontal3DLine(this);
   fToolBar = new TGToolBar(this, 60, 20, kHorizontalFrame);
   fFSComboBox = new TGFSComboBox(fToolBar, kFSComboBox);

   fComboLayout = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 0, 0, 2, 0);
   fToolBar->AddFrame(fFSComboBox, fComboLayout);
   fFSComboBox->Resize(150, fFSComboBox->GetDefaultHeight());
   fFSComboBox->Associate(this);

   int spacing = 8;

   for (int i = 0; gToolBarData[i].fPixmap; i++) {
      if (strlen(gToolBarData[i].fPixmap) == 0) {
         spacing = 8;
         continue;
      }
      fToolBar->AddButton(this, &gToolBarData[i], spacing);
      spacing = 0;
   }

   fBarLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);
   AddFrame(fToolBarSep, fBarLayout);
   AddFrame(fToolBar, fBarLayout);

   // Create panes

   fHf = new TGHorizontalFrame(this, 10, 10);

   fV1 = new TGVerticalFrame(fHf, 10, 10, kFixedWidth);
   fV2 = new TGVerticalFrame(fHf, 10, 10);
   fTreeHdr = new TGCompositeFrame(fV1, 10, 10, kSunkenFrame);
   fListHdr = new TGCompositeFrame(fV2, 10, 10, kSunkenFrame);

   fLbl1 = new TGLabel(fTreeHdr, "All Folders");
   fLbl2 = new TGLabel(fListHdr, "Contents of \".\"");

   TGLayoutHints *lo;

   lo = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 0, 0, 0);
   fWidgets->Add(lo);
   fTreeHdr->AddFrame(fLbl1, lo);
   fListHdr->AddFrame(fLbl2, lo);

   lo = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 1, 2);
   fWidgets->Add(lo);
   fV1->AddFrame(fTreeHdr, lo);
   fV2->AddFrame(fListHdr, lo);

   fV1->Resize(fTreeHdr->GetDefaultWidth()+100, fV1->GetDefaultHeight());

   lo = new TGLayoutHints(kLHintsLeft | kLHintsExpandY);
   fWidgets->Add(lo);
   fHf->AddFrame(fV1, lo);

   TGVSplitter *splitter = new TGVSplitter(fHf);
   splitter->SetFrame(fV1, kTRUE);
   lo = new TGLayoutHints(kLHintsLeft | kLHintsExpandY);
   fWidgets->Add(splitter);
   fWidgets->Add(lo);
   fHf->AddFrame(splitter, lo);

   lo = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY);
   fWidgets->Add(lo);
   fHf->AddFrame(fV2, lo);

   // Create tree
   fTreeView = new TGCanvas(fV1, 10, 10, kSunkenFrame | kDoubleBorder);
   fLt = new TGListTree(fTreeView->GetViewPort(), 10, 10, kHorizontalFrame,
                        fgWhitePixel);
   fLt->Associate(this);
   fLt->SetAutoTips();
   fTreeView->SetContainer(fLt);
   fLt->SetCanvas(fTreeView);

   lo = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fWidgets->Add(lo);
   fV1->AddFrame(fTreeView, lo);

   // Create list view (icon box)

   fListView = new TGListView(fV2, 520, 250);
   fIconBox = new TRootIconBox(fListView->GetViewPort(), fListView, 520, 250,
                               kHorizontalFrame, fgWhitePixel);
   fIconBox->Associate(this);

   TString gv = gEnv->GetValue("Browser.GroupView","1000");
   Int_t igv = atoi(gv.Data());

   if (igv>10) {
      fViewMenu->CheckEntry(kViewGroupLV);
      fIconBox->SetGroupSize(igv);
   }


   fListView->GetViewPort()->SetBackgroundColor(fgWhitePixel);
   fListView->SetContainer(fIconBox);

   // reuse lo from "create tree"
   fV2->AddFrame(fListView, lo);

   AddFrame(fHf, lo);

   // Statusbar

   int parts[] = { 25, 75 };
   fStatusBar = new TGStatusBar(this, 60, 10);
   fStatusBar->SetParts(parts, 2);
   lo = new TGLayoutHints(kLHintsBottom | kLHintsExpandX, 0, 0, 3, 0);
   AddFrame(fStatusBar, lo);

   // Misc

   SetWindowName(name);
   SetIconName(name);
   SetIconPixmap("rootdb_s.xpm");
   SetClassHints("Browser", "Browser");

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);

   fWaitCursor = gVirtualX->CreateCursor(kWatch);
   fListLevel = 0;
   fTreeLock  = kFALSE;

   MapSubwindows();

   SetDefaults();

   // we need to use GetDefaultSize() to initialize the layout algorithm...
   Resize(GetDefaultSize());
}

//______________________________________________________________________________
void TRootBrowser::Add(TObject *obj, const char *name)
{
   // Add items to the browser. This function has to be called
   // by the Browse() member function of objects when they are
   // called by a browser.

   if (obj) {
      if (!name) name = obj->GetName();

      AddToBox(obj, name);

      if (obj->IsFolder())
         AddToTree(obj, name);
   }
}

//______________________________________________________________________________
void TRootBrowser::AddToBox(TObject *obj, const char *name)
{
   // Add items to the iconbox of the browser.

   if (obj) {
      if (!name) name = obj->GetName() ? obj->GetName() : "NoName";
      //const char *titlePtr = obj->GetTitle() ? obj->GetTitle() : " ";

      TClass *objClass = 0;

      if (obj->IsA() == TKey::Class())
         objClass = gROOT->GetClass(((TKey *)obj)->GetClassName());
      else if (obj->IsA() == TKeyMapFile::Class())
         objClass = gROOT->GetClass(((TKeyMapFile *)obj)->GetTitle());
      else
         objClass = obj->IsA();

      fIconBox->AddObjItem(name, obj, objClass);
   }
}

//______________________________________________________________________________
void TRootBrowser::AddToTree(TObject *obj, const char *name)
{
   // Add items to the current TGListTree of the browser.

   if (obj && !fTreeLock) {
      if (!name) name = obj->GetName();

      fLt->AddItem(fListLevel, name, obj);
   }
}

//______________________________________________________________________________
void TRootBrowser::BrowseObj(TObject *obj)
{
   // Browse object. This, in turn, will trigger the calling of
   // TRootBrowser::Add() which will fill the IconBox and the tree.

   fIconBox->RemoveAll();
   obj->Browse(fBrowser);
   fIconBox->Refresh();
   if (fBrowser)
      fBrowser->SetRefreshFlag(kFALSE);
}

//______________________________________________________________________________
void TRootBrowser::CloseWindow()
{
   // In case window is closed via WM we get here.

   gInterpreter->DeleteGlobal(fBrowser);
   delete fBrowser;  // this in turn will delete this object
}

//______________________________________________________________________________
void TRootBrowser::DisplayTotal(Int_t total, Int_t selected)
{
   // Display in statusbar total number of objects and number of
   // selected objects in IconBox.

   char tmp[64];
   const char *fmt;

   if (selected)
      fmt = "%d Object%s, %d selected.";
   else
      fmt = "%d Object%s.";

   sprintf(tmp, fmt, total, (total == 1) ? "" : "s", selected);
   fStatusBar->SetText(tmp, 0);
}

//______________________________________________________________________________
void TRootBrowser::DisplayDirectory()
{
   // Display current directory in second label, fLbl2.

   char *p, path[1024];

   fLt->GetPathnameFromItem(fListLevel, path, 3);
   p = path;
   while (*p && *(p+1) == '/') ++p;
   if (strlen(p) == 0)
      fLbl2->SetText(new TGString("Contents of \".\""));
   else
      fLbl2->SetText(new TGString(Form("Contents of \"%s\"", p)));
   fListHdr->Layout();

   // Get full pathname for FS combobox (previously truncated to 3 levels deep)
   fLt->GetPathnameFromItem(fListLevel, path);
   p = path;
   while (*p && *(p+1) == '/') ++p;
   fFSComboBox->Update(p);
}

//______________________________________________________________________________
void TRootBrowser::ExecuteDefaultAction(TObject *obj)
{
   // Execute default action for selected object (action is specified
   // in the $HOME/.root.mimes or $ROOTSYS/etc/root.mimes file.

   char action[512];

   // Special case for file system objects...
   if (obj->IsA() == TSystemFile::Class()) {
      if (fClient->GetMimeTypeList()->GetAction(obj->GetName(), action)) {
         TString act = action;
         act.ReplaceAll("%s", obj->GetName());
         if (act[0] == '!') {
            act.Remove(0,1);
            gSystem->Exec(act.Data());
         } else
            gApplication->ProcessLine(act.Data());
      }
      return;
   }

   // For other objects the default action is still hard coded in
   // their Browse() member function.

}

//______________________________________________________________________________
Bool_t TRootBrowser::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Handle menu and other command generated by the user.

   TRootHelpDialog *hd;

   switch (GET_MSG(msg)) {

      case kC_COMMAND:

         switch (GET_SUBMSG(msg)) {

            case kCM_BUTTON:
            case kCM_MENU:

               switch (parm1) {
                  // Handle File menu items...
                  case kFileNewBrowser:
                     new TBrowser("Browser", "ROOT Object Browser");
                     break;
                  case kFileNewCanvas:
                     gROOT->GetMakeDefCanvas()();
                     break;
                  case kFileOpen:
                     {
                        static TString dir(".");
                        TGFileInfo fi;
                        fi.fFileTypes = gOpenTypes;
                        fi.fIniDir    = StrDup(dir);
                        new TGFileDialog(fClient->GetRoot(), this, kFDOpen,&fi);
                        if (!fi.fFilename) return kTRUE;
                        dir = fi.fIniDir;
                        new TFile(fi.fFilename, "update");
                     }
                     break;
                  case kFileSave:
                  case kFileSaveAs:
                     break;
                  case kFilePrint:
                     break;
                  case kFileCloseBrowser:
                     SendCloseMessage();
                     break;
                  case kFileQuit:
                     gApplication->Terminate(0);
                     break;

                  // Handle View menu items...
                  case kViewToolBar:
                     if (fViewMenu->IsEntryChecked(kViewToolBar))
                        ShowToolBar(kFALSE);
                     else
                        ShowToolBar();
                     break;
                  case kViewStatusBar:
                     if (fViewMenu->IsEntryChecked(kViewStatusBar))
                        ShowStatusBar(kFALSE);
                     else
                        ShowStatusBar();
                     break;
                  case kViewLargeIcons:
                  case kViewSmallIcons:
                  case kViewList:
                  case kViewDetails:
                     SetViewMode((Int_t)parm1);
                     break;
                  case kViewArrangeByName:
                  case kViewArrangeByType:
                  case kViewArrangeBySize:
                  case kViewArrangeByDate:
                     SetSortMode((Int_t)parm1);
                     break;
                  case kViewLineUp:
                     break;
                  case kViewRefresh:
                     Refresh(kTRUE);
                     break;
                  case kViewGroupLV:
                     if (!fViewMenu->IsEntryChecked(kViewGroupLV)) {
                        fViewMenu->CheckEntry(kViewGroupLV);
                        TString gv = gEnv->GetValue("Browser.GroupView","1000");
                        Int_t igv = atoi(gv.Data());

                        if (igv>10) {
                           fIconBox->SetGroupSize(igv);
                        }
                     } else {
                        fViewMenu->UnCheckEntry(kViewGroupLV);
                        fIconBox->SetGroupSize(10000000); // very large value
                     }
                     break;

                  // Handle Option menu items...
                  case kOptionShowCycles:
                     printf("Currently the browser always shows all cycles\n");
                     break;

                  // Handle toolbar button...
                  case kOneLevelUp:
                     if (fListLevel && fIconBox->WasGrouped()) {
                        gVirtualX->SetCursor(fIconBox->GetId(),fWaitCursor);
                        gVirtualX->Update();
                        TObject *obj = (TObject *) fListLevel->GetUserData();
                        if (fListLevel->IsActive()) BrowseObj(obj);
                        gVirtualX->SetCursor(fIconBox->GetId(),kNone);
                        break;
                     }
                     if (fListLevel)
                        fListLevel = fListLevel->GetParent();
                     fLt->ClearHighlighted();
                     fLt->HighlightItem(fListLevel);
                     fClient->NeedRedraw(fLt);
                     DisplayDirectory();
                     Refresh(kTRUE);
                     break;

                  // Handle Help menu items...
                  case kHelpAbout:
                     // coming soon
                     {
                        char str[32];
                        sprintf(str, "About ROOT %s...", gROOT->GetVersion());
                        hd = new TRootHelpDialog(this, str, 600, 400);
                        hd->SetText(gHelpAbout);
                        hd->Popup();
                     }
                     break;
                  case kHelpOnCanvas:
                     hd = new TRootHelpDialog(this, "Help on Canvas...", 600, 400);
                     hd->SetText(gHelpCanvas);
                     hd->Popup();
                     break;
                  case kHelpOnMenus:
                     hd = new TRootHelpDialog(this, "Help on Menus...", 600, 400);
                     hd->SetText(gHelpPullDownMenus);
                     hd->Popup();
                     break;
                  case kHelpOnGraphicsEd:
                     hd = new TRootHelpDialog(this, "Help on Graphics Editor...", 600, 400);
                     hd->SetText(gHelpGraphicsEditor);
                     hd->Popup();
                     break;
                  case kHelpOnBrowser:
                     hd = new TRootHelpDialog(this, "Help on Browser...", 600, 400);
                     hd->SetText(gHelpBrowser);
                     hd->Popup();
                     break;
                  case kHelpOnObjects:
                     hd = new TRootHelpDialog(this, "Help on Objects...", 600, 400);
                     hd->SetText(gHelpObjects);
                     hd->Popup();
                     break;
                  case kHelpOnPS:
                     hd = new TRootHelpDialog(this, "Help on PostScript...", 600, 400);
                     hd->SetText(gHelpPostscript);
                     hd->Popup();
                     break;
               }
            case kCM_COMBOBOX:
               if (parm1 == kFSComboBox) {
                  TGTreeLBEntry *e = (TGTreeLBEntry *) fFSComboBox->GetSelectedEntry();
                  if (e) {
                     fListLevel = fLt->FindItemByPathname(e->GetPath()->GetString());
                     fLt->ClearHighlighted();
                     fLt->HighlightItem(fListLevel);
                     fClient->NeedRedraw(fLt);
                     DisplayDirectory();
                     Refresh(kTRUE);
                  }
               }
               break;
            default:
               break;
         }
         break;

      case kC_LISTTREE:
         switch (GET_SUBMSG(msg)) {

            case kCT_ITEMCLICK:
               if (parm1 == kButton1 || parm1 == kButton3) {
                  TGListTreeItem *item;
                  if ((item = fLt->GetSelected()) != 0 &&
                      item != fListLevel) {
                     ListTreeHighlight(item);
                     fClient->NeedRedraw(fLt);
                  }
                  if (item && parm1 == kButton3) {
                     Int_t x = (Int_t)(parm2 & 0xffff);
                     Int_t y = (Int_t)((parm2 >> 16) & 0xffff);
                     TObject *obj = (TObject *) item->GetUserData();
                     if (obj) fBrowser->GetContextMenu()->Popup(x, y, obj, fBrowser);
                  }
               }
               break;
            case kCT_ITEMDBLCLICK:
               if (parm1 == kButton1) {
                  if (fLt->GetSelected() != 0) {
                     fClient->NeedRedraw(fLt);
                  }
               }
               break;
            default:
               break;
         }
         break;

      case kC_CONTAINER:
         switch (GET_SUBMSG(msg)) {

            case kCT_ITEMCLICK:
               if (fIconBox->NumSelected() == 1) {
                  // display title of selected object
                  TGFileItem *item;
                  void *p = 0;
                  if ((item = (TGFileItem *) fIconBox->GetNextSelected(&p)) != 0) {
                     TObject *obj = (TObject *) item->GetUserData();
                     fStatusBar->SetText(obj->GetTitle(), 1);
                  }
               }
               if (parm1 == kButton3) {
                  // show context menu for selected object
                  if (fIconBox->NumSelected() == 1) {
                     void *p = 0;
                     TGFileItem *item;
                     if ((item = (TGFileItem *) fIconBox->GetNextSelected(&p)) != 0) {
                        Int_t x = (Int_t)(parm2 & 0xffff);
                        Int_t y = (Int_t)((parm2 >> 16) & 0xffff);
                        TObject *obj = (TObject *)item->GetUserData();
                        if (obj) {
                           if (obj->IsA() == TKey::Class())
                              obj = gROOT->FindObject((char *) obj->GetName());
                           fBrowser->GetContextMenu()->Popup(x, y, obj);
                        }
                     }
                  }
               }
               break;
            case kCT_ITEMDBLCLICK:
               if (parm1 == kButton1) {
                  if (fIconBox->NumSelected() == 1) {
                     void *p = 0;
                     TGFileItem *item;
                     if ((item = (TGFileItem *) fIconBox->GetNextSelected(&p)) != 0)
                        IconBoxAction((TObject *)item->GetUserData());
                  }
               }
               break;
            case kCT_SELCHANGED:
               DisplayTotal((Int_t)parm1, (Int_t)parm2);
               break;
            default:
               break;
         }
         break;

      default:
         break;
   }
   fClient->NeedRedraw(fIconBox);
   return kTRUE;
}

//______________________________________________________________________________
void TRootBrowser::Chdir(TGListTreeItem *item)
{
   // Make object associated with item the current directory.

   if (item) {
      TGListTreeItem *i = item;
      TString dir;
      while (i) {
         TObject *obj = (TObject*) i->GetUserData();
         if (obj) {
            if (obj->IsA() == TDirectory::Class()) {
               dir = "/" + dir;
               dir = obj->GetName() + dir;
            }
            if (obj->IsA() == TFile::Class()) {
               dir = ":/" + dir;
               dir = obj->GetName() + dir;
            }
            if (obj->IsA() == TKey::Class()) {
               if (strcmp(((TKey*)obj)->GetClassName(), TDirectory::Class()->GetName()) == 0) {
                  dir = "/" + dir;
                  dir = obj->GetName() + dir;
               }
            }
         }
         i = i->GetParent();
      }
      if (gDirectory && dir.Length()) gDirectory->cd(dir.Data());
   }
}

//______________________________________________________________________________
void TRootBrowser::ListTreeHighlight(TGListTreeItem *item)
{
   // Open tree item and list in iconbox its contents.

   fListLevel = item;

   if (fListLevel) {
      DisplayDirectory();
      TObject *obj = (TObject *) fListLevel->GetUserData();

      if (obj) {
         gVirtualX->SetCursor(fId, fWaitCursor);
         gVirtualX->Update();

         if (obj->IsA() == TKey::Class()) {
            if (strcmp(((TKey*)obj)->GetClassName(), TDirectory::Class()->GetName()) == 0)
               Chdir(fListLevel->GetParent());

            //TObject *k_obj = gDirectory->Get(obj->GetName());
            TObject *k_obj = gROOT->FindObject(obj->GetName());

            if (k_obj) {
               TGListTreeItem *parent = fListLevel->GetParent();
               fLt->DeleteItem(fListLevel);
               TGListTreeItem *itm = fLt->AddItem(parent, k_obj->GetName());
               if (itm) {
                  itm->SetUserData(k_obj);
                  fListLevel = itm;
                  obj = k_obj;
               } else {
                  fListLevel = parent;
               }
               fLt->HighlightItem(fListLevel);
            }
         } else if (obj->IsA() == TDirectory::Class())
            Chdir(fListLevel->GetParent());

         // New TGListTree::AddItem does not allow duplicates if user data
         // is also specified. So no need to delete children.
         //if (fListLevel->GetFirstChild())
         //   fLt->DeleteChildren(fListLevel);

         if (fListLevel->IsActive())
            BrowseObj(obj);

         gVirtualX->SetCursor(fId, kNone);
      }
   }
}

//______________________________________________________________________________
void TRootBrowser::IconBoxAction(TObject *obj)
{
   // Default action when double clicking on icon.

   if (obj) {
      Bool_t useLock = kTRUE;

      gVirtualX->SetCursor(fId, fWaitCursor);
      gVirtualX->Update();

      if (obj->IsFolder()) {
         fIconBox->RemoveAll();
         TGListTreeItem *itm = 0;

         if (fListLevel) {
            fLt->OpenItem(fListLevel);
            itm = fListLevel->GetFirstChild();
         } else
            itm = fLt->GetFirstItem();

         while (itm && (itm->GetUserData() != obj))
            itm = itm->GetNextSibling();

         if (!itm && fListLevel) {
            itm = fLt->AddItem(fListLevel, obj->GetName());
            if (itm) itm->SetUserData(obj);
         }

         if (itm) {
            fListLevel = itm;
            DisplayDirectory();
            TObject *kobj = (TObject *) itm->GetUserData();
            if (kobj->IsA() == TKey::Class()) {
               Chdir(fListLevel->GetParent());

               kobj = gROOT->FindObject(kobj->GetName());

               if (kobj) {
                  TGListTreeItem *parent = fListLevel->GetParent();
                  fLt->DeleteItem(fListLevel);
                  TGListTreeItem *kitem = fLt->AddItem(parent, kobj->GetName());
                  if (kitem) {
                     obj = kobj;
                     useLock = kFALSE;
                     kitem->SetUserData(kobj);
                     fListLevel = kitem;
                  } else
                     fListLevel = parent;
               }
            }
            fLt->ClearHighlighted();
            fLt->HighlightItem(fListLevel);
            fClient->NeedRedraw(fLt);
         }
      }

      if (useLock) fTreeLock = kTRUE;
      obj->Browse(fBrowser);
      if (useLock) fTreeLock = kFALSE;

      Chdir(fListLevel);

      if (obj->IsFolder()) {
         fIconBox->Refresh();
         if (fBrowser)
            fBrowser->SetRefreshFlag(kFALSE);
      } else {
         Refresh();
      }

      gVirtualX->SetCursor(fId, kNone);
   }
}

//______________________________________________________________________________
void TRootBrowser::RecursiveRemove(TObject *obj)
{
   // Recursively remove object from browser.

   //gVirtualX->SetCursor(fId, fWaitCursor);
   //gVirtualX->Update();

   // don't delete fIconBox items here (it's status will be updated
   // via TBrowser::Refresh() which should be called once all objects have
   // been removed.
   //fIconBox->RemoveItemWithData(obj);
   //fIconBox->Refresh();
   fLt->RecursiveDeleteItem(fLt->GetFirstItem(), obj);

   //gVirtualX->SetCursor(fId, kNone);
}

//______________________________________________________________________________
void TRootBrowser::Refresh(Bool_t force)
{
   // Refresh the browser contents.

   if ( ((fBrowser && fBrowser->GetRefreshFlag()) || force) &&
         fIconBox->NumItems()<fIconBox->GetGroupSize() ) {
      gVirtualX->SetCursor(fId, fWaitCursor);
      gVirtualX->Update();

      // Refresh the IconBox
      if (fListLevel) {
         fIconBox->RemoveAll();

         TObject *obj = (TObject *) fListLevel->GetUserData();

         if (obj) {
            fTreeLock = kTRUE;
            BrowseObj(obj);
            fTreeLock = kFALSE;
         }
      } else
         BrowseObj(gROOT);

      fClient->NeedRedraw(fLt);
      gVirtualX->SetCursor(fId, kNone);
   }
}

//______________________________________________________________________________
void TRootBrowser::ShowToolBar(Bool_t show)
{
   // Show or hide toolbar.

   if (show) {
      ShowFrame(fToolBar);
      ShowFrame(fToolBarSep);
      fViewMenu->CheckEntry(kViewToolBar);
   } else {
      HideFrame(fToolBar);
      HideFrame(fToolBarSep);
      fViewMenu->UnCheckEntry(kViewToolBar);
   }
}

//______________________________________________________________________________
void TRootBrowser::ShowStatusBar(Bool_t show)
{
   // Show or hide statusbar.

   if (show) {
      ShowFrame(fStatusBar);
      fViewMenu->CheckEntry(kViewStatusBar);
   } else {
      HideFrame(fStatusBar);
      fViewMenu->UnCheckEntry(kViewStatusBar);
   }
}

//______________________________________________________________________________
void TRootBrowser::SetDefaults(const char *iconStyle, const char *sortBy)
{
   // Set defaults depending on settings in the user's .rootrc.

   const char *opt;

   // IconStyle: big, small, list, details
   if (iconStyle)
      opt = iconStyle;
   else
      opt = gEnv->GetValue("Browser.IconStyle", "small");
   if (!strcasecmp(opt, "big"))
      SetViewMode(kViewLargeIcons, kTRUE);
   else if (!strcasecmp(opt, "small"))
      SetViewMode(kViewSmallIcons, kTRUE);
   else if (!strcasecmp(opt, "list"))
      SetViewMode(kViewList, kTRUE);
   else if (!strcasecmp(opt, "details"))
      SetViewMode(kViewDetails, kTRUE);
   else
      SetViewMode(kViewSmallIcons, kTRUE);

   // SortBy: name, type, size, date
   if (sortBy)
      opt = sortBy;
   else
      opt = gEnv->GetValue("Browser.SortBy", "name");
   if (!strcasecmp(opt, "name"))
      SetSortMode(kViewArrangeByName);
   else if (!strcasecmp(opt, "type"))
      SetSortMode(kViewArrangeByType);
   else if (!strcasecmp(opt, "size"))
      SetSortMode(kViewArrangeBySize);
   else if (!strcasecmp(opt, "date"))
      SetSortMode(kViewArrangeByDate);
   else
      SetSortMode(kViewArrangeByName);
}

//______________________________________________________________________________
void TRootBrowser::SetViewMode(Int_t new_mode, Bool_t force)
{
   // Set iconbox's view mode and update menu and toolbar buttons accordingly.

   int i, bnum;
   EListViewMode lv;

   if (force || (fViewMode != new_mode)) {

      switch (new_mode) {
         default:
            if (!force)
               return;
            else
               new_mode = kViewLargeIcons;
         case kViewLargeIcons:
            bnum = 2;
            lv = kLVLargeIcons;
            break;
         case kViewSmallIcons:
            bnum = 3;
            lv = kLVSmallIcons;
            break;
         case kViewList:
            bnum = 4;
            lv = kLVList;
            break;
         case kViewDetails:
            bnum = 5;
            lv = kLVDetails;
            break;
      }

      fViewMode = new_mode;
      fViewMenu->RCheckEntry(fViewMode, kViewLargeIcons, kViewDetails);

      for (i = 2; i <= 5; ++i)
         gToolBarData[i].fButton->SetState((i == bnum) ? kButtonEngaged : kButtonUp);

      fListView->SetViewMode(lv);
      fIconBox->AdjustPosition();
   }
}

//______________________________________________________________________________
void TRootBrowser::SetSortMode(Int_t new_mode)
{
   // Set iconbox's sort mode and update menu radio buttons accordingly.

   EFSSortMode smode;

   switch (new_mode) {
      default:
         new_mode = kViewArrangeByName;
      case kViewArrangeByName:
         smode = kSortByName;
         break;
      case kViewArrangeByType:
         smode = kSortByType;
         break;
      case kViewArrangeBySize:
         smode = kSortBySize;
         break;
      case kViewArrangeByDate:
         smode = kSortByDate;
         break;
   }

   fSortMode = new_mode;
   fSortMenu->RCheckEntry(fSortMode, kViewArrangeByName, kViewArrangeByDate);

   fIconBox->Sort(smode);
}
