// @(#)root/gui:$Name:  $:$Id: TGFileDialog.cxx,v 1.26 2006/04/24 13:59:21 antcheva Exp $
// Author: Fons Rademakers   20/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFileDialog                                                         //
//                                                                      //
// This class creates a file selection dialog. It contains a combo box  //
// to select the desired directory. A listview with the different       //
// files in the current directory and a combo box with which you can    //
// select a filter (on file extensions).                                //
// When creating a file dialog one passes a pointer to a TGFileInfo     //
// object. In this object you can set the fFileTypes and fIniDir to     //
// specify the list of file types for the filter combo box and the      //
// initial directory. When the TGFileDialog ctor returns the selected   //
// file name can be found in the TGFileInfo::fFilename field and the    //
// selected directory in TGFileInfo::fIniDir. The fFilename and         //
// fIniDir are deleted by the TGFileInfo dtor.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGFileDialog.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGTextEntry.h"
#include "TGComboBox.h"
#include "TGListView.h"
#include "TGFSContainer.h"
#include "TGFSComboBox.h"
#include "TGMsgBox.h"
#include "TSystem.h"

#include <sys/stat.h>

enum EFileFialog {
   kIDF_CDUP,
   kIDF_NEW_FOLDER,
   kIDF_LIST,
   kIDF_DETAILS,
   kIDF_OVERWRITE,
   kIDF_FSLB,
   kIDF_FTYPESLB,
   kIDF_OK,
   kIDF_CANCEL
};

static const char *gDefTypes[] = { "All files",     "*",
                                   "ROOT files",    "*.root",
                                   "ROOT macros",   "*.C",
                                    0,               0 };

static TGFileInfo gInfo;


ClassImp(TGFileDialog)

//______________________________________________________________________________
TGFileInfo::TGFileInfo(const TGFileInfo& fi) : 
    fFilename(fi.fFilename),
    fIniDir(fi.fIniDir),
    fFileTypes(fi.fFileTypes),
    fFileTypeIdx(fi.fFileTypeIdx),
    fOverwrite(fi.fOverwrite)
{ }

//______________________________________________________________________________
TGFileInfo& TGFileInfo::operator=(const TGFileInfo& fi) 
{
  if(this!=&fi) {
    fFilename=fi.fFilename;
    fIniDir=fi.fIniDir;
    fFileTypes=fi.fFileTypes;
    fFileTypeIdx=fi.fFileTypeIdx;
    fOverwrite=fi.fOverwrite;
  } return *this;
}

//______________________________________________________________________________
TGFileDialog::TGFileDialog(const TGWindow *p, const TGWindow *main,
                           EFileDialogMode dlg_type, TGFileInfo *file_info) :
   TGTransientFrame(p, main, 10, 10, kVerticalFrame)
{
   // Create a file selection dialog. Depending on the dlg_type it can be
   // used for opening or saving a file.

   int i;

   if (!p && !main) {
      MakeZombie();
      return;
   }
   if (!file_info) {
      Error("TGFileDialog", "file_info argument not set");
      fFileInfo = &gInfo;
      if (fFileInfo->fIniDir) {
         delete [] fFileInfo->fIniDir;
         fFileInfo->fIniDir = 0;
      }
      if (fFileInfo->fFilename) {
         delete [] fFileInfo->fFilename;
         fFileInfo->fFilename = 0;
      }
      fFileInfo->fFileTypeIdx = 0;
   } else
      fFileInfo = file_info;

   if (!fFileInfo->fFileTypes)
      fFileInfo->fFileTypes = gDefTypes;

   if (!fFileInfo->fIniDir)
      fFileInfo->fIniDir = StrDup(".");

   TGHorizontalFrame *fHtop = new TGHorizontalFrame(this, 10, 10);

   //--- top toolbar elements

   TGLabel *fLookin = new TGLabel(fHtop, new TGHotString((dlg_type == kFDSave)
                                                  ? "S&ave in:" : "&Look in:"));
   fTreeLB = new TGFSComboBox(fHtop, kIDF_FSLB);
   fTreeLB->Associate(this);

   fPcdup = fClient->GetPicture("tb_uplevel.xpm");
   fPnewf = fClient->GetPicture("tb_newfolder.xpm");
   fPlist = fClient->GetPicture("tb_list.xpm");
   fPdetails = fClient->GetPicture("tb_details.xpm");

   if (!(fPcdup && fPnewf && fPlist && fPdetails))
      Error("TGFileDialog", "missing toolbar pixmap(s).\n");

   fCdup    = new TGPictureButton(fHtop, fPcdup, kIDF_CDUP);
   fNewf    = new TGPictureButton(fHtop, fPnewf, kIDF_NEW_FOLDER);
   fList    = new TGPictureButton(fHtop, fPlist, kIDF_LIST);
   fDetails = new TGPictureButton(fHtop, fPdetails, kIDF_DETAILS);

   fCdup->SetToolTipText("Up One Level");
   fNewf->SetToolTipText("Create New Folder");
   fList->SetToolTipText("List");
   fDetails->SetToolTipText("Details");

   fCdup->Associate(this);
   fNewf->Associate(this);
   fList->Associate(this);
   fDetails->Associate(this);

   fList->AllowStayDown(kTRUE);
   fDetails->AllowStayDown(kTRUE);

   fTreeLB->Resize(200, fTreeLB->GetDefaultHeight());

   fHtop->AddFrame(fLookin, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 5, 2, 2));
   fHtop->AddFrame(fTreeLB, new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 3, 0, 2, 2));
   fHtop->AddFrame(fCdup, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 0, 2, 2));
   fHtop->AddFrame(fNewf, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 0, 2, 2));
   fHtop->AddFrame(fList, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 0, 2, 2));
   fHtop->AddFrame(fDetails, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 0, 8, 2, 2));

   if (dlg_type == kFDSave) {
      fOverWR = new TGCheckButton(fHtop, "&Overwrite", kIDF_OVERWRITE);
      fOverWR->SetToolTipText("Overwrite a file without displaying a message if selected.");
      fHtop->AddFrame(fOverWR, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
      fOverWR->SetOn(fFileInfo->fOverwrite);
   } else fOverWR = 0;
   AddFrame(fHtop, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 4, 4, 3, 1));

   //--- file view

   fFv = new TGListView(this, 400, 161);

   fFc = new TGFileContainer(fFv->GetViewPort(),
                             10, 10, kHorizontalFrame, fgWhitePixel);
   fFc->Associate(this);

   fFv->GetViewPort()->SetBackgroundColor(fgWhitePixel);
   fFv->SetContainer(fFc);
   fFv->SetViewMode(kLVList);

   fFc->SetFilter(fFileInfo->fFileTypes[fFileInfo->fFileTypeIdx+1]);
   fFc->Sort(kSortByType);
   fFc->ChangeDirectory(fFileInfo->fIniDir);
   fTreeLB->Update(fFc->GetDirectory());

   fList->SetState(kButtonEngaged);

   AddFrame(fFv, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 4, 4, 3, 1));

   //--- file name and types

   TGHorizontalFrame *fHf = new TGHorizontalFrame(this, 10, 10);

   TGVerticalFrame *fVf = new TGVerticalFrame(fHf, 10, 10);

   TGHorizontalFrame *fHfname = new TGHorizontalFrame(fVf, 10, 10);

   TGLabel *fLfname = new TGLabel(fHfname, new TGHotString("File &name:"));
   fTbfname = new TGTextBuffer(1034);
   fName = new TGTextEntry(fHfname, fTbfname);
   fName->Resize(230, fName->GetDefaultHeight());
   fName->Associate(this);

   fHfname->AddFrame(fLfname, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 5, 2, 2));
   fHfname->AddFrame(fName, new TGLayoutHints(kLHintsRight | kLHintsCenterY, 0, 20, 2, 2));

   fVf->AddFrame(fHfname, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX));

   TGHorizontalFrame *fHftype = new TGHorizontalFrame(fVf, 10, 10);

   TGLabel *fLftypes = new TGLabel(fHftype, new TGHotString("Files of &type:"));
   fTypes = new TGComboBox(fHftype, kIDF_FTYPESLB);
   fTypes->Associate(this);
   fTypes->Resize(230, fName->GetDefaultHeight());

   char s[64];
   for (i = 0; fFileInfo->fFileTypes[i] != 0; i += 2) {
      sprintf(s, "%s (%s)", fFileInfo->fFileTypes[i], fFileInfo->fFileTypes[i+1]);
      fTypes->AddEntry(s, i);
   }
   fTypes->Select(fFileInfo->fFileTypeIdx);

   // Show all items in combobox without scrollbar
   //TGDimension fw = fTypes->GetListBox()->GetContainer()->GetDefaultSize();
   //fTypes->GetListBox()->Resize(fw.fWidth, fw.fHeight);

   if (fFileInfo->fFilename && fFileInfo->fFilename[0])
      fTbfname->AddText(0, fFileInfo->fFilename);
   else
      fTbfname->Clear();

   fHftype->AddFrame(fLftypes, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 5, 2, 2));
   fHftype->AddFrame(fTypes, new TGLayoutHints(kLHintsRight | kLHintsCenterY, 0, 20, 2, 2));

   fVf->AddFrame(fHftype, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX));

   fHf->AddFrame(fVf, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX));

   //--- Open/Save and Cancel buttons

   TGVerticalFrame *fVbf = new TGVerticalFrame(fHf, 10, 10, kFixedWidth);

   fOk = new TGTextButton(fVbf, new TGHotString((dlg_type == kFDSave)
                                                 ? "&Save" : "&Open"), kIDF_OK);
   fCancel = new TGTextButton(fVbf, new TGHotString("Cancel"), kIDF_CANCEL);

   fOk->Associate(this);
   fCancel->Associate(this);

   fVbf->AddFrame(fOk, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 2, 2));
   fVbf->AddFrame(fCancel, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 2, 2));

   UInt_t width = TMath::Max(fOk->GetDefaultWidth(), fCancel->GetDefaultWidth()) + 20;
   fVbf->Resize(width + 20, fVbf->GetDefaultHeight());

   fHf->AddFrame(fVbf, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));

   AddFrame(fHf, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 4, 4, 3, 1));
   SetEditDisabled(kEditDisable);

   MapSubwindows();

   TGDimension size = GetDefaultSize();

   Resize(size);

   //---- position relative to the parent's window

   CenterOnParent();

   //---- make the message box non-resizable

   SetWMSize(size.fWidth, size.fHeight);
   SetWMSizeHints(size.fWidth, size.fHeight, size.fWidth, size.fHeight, 0, 0);

   const char *wname = (dlg_type == kFDSave) ? "Save As..." : "Open";
   SetWindowName(wname);
   SetIconName(wname);
   SetClassHints("FileDialog", "FileDialog");

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                              kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll |  kMWMFuncResize    | kMWMFuncMaximize |
                              kMWMFuncMinimize,
               kMWMInputModeless);

   MapWindow();
   fFc->DisplayDirectory();
   fClient->WaitFor(this);
}

//______________________________________________________________________________
TGFileDialog::TGFileDialog(const TGFileDialog& fd) : 
  TGTransientFrame(fd),
  fTbfname(fd.fTbfname),
  fName(fd.fName),
  fTypes(fd.fTypes),
  fTreeLB(fd.fTreeLB),
  fCdup(fd.fCdup),
  fNewf(fd.fNewf),
  fList(fd.fList),
  fDetails(fd.fDetails),
  fOverWR(fd.fOverWR),
  fPcdup(fd.fPcdup),
  fPnewf(fd.fPnewf),
  fPlist(fd.fPlist),
  fPdetails(fd.fPdetails),
  fOk(fd.fOk),
  fCancel(fd.fCancel),
  fFv(fd.fFv),
  fFc(fd.fFc),
  fFileInfo(fd.fFileInfo)
{ }

//______________________________________________________________________________
TGFileDialog& TGFileDialog::operator=(const TGFileDialog& fd) 
{
  if(this!=&fd) {
    TGTransientFrame::operator=(fd);
    fTbfname=fd.fTbfname;
    fName=fd.fName;
    fTypes=fd.fTypes;
    fTreeLB=fd.fTreeLB;
    fCdup=fd.fCdup;
    fNewf=fd.fNewf;
    fList=fd.fList;
    fDetails=fd.fDetails;
    fOverWR=fd.fOverWR;
    fPcdup=fd.fPcdup;
    fPnewf=fd.fPnewf;
    fPlist=fd.fPlist;
    fPdetails=fd.fPdetails;
    fOk=fd.fOk;
    fCancel=fd.fCancel;
    fFv=fd.fFv;
    fFc=fd.fFc;
    fFileInfo=fd.fFileInfo;
  } return *this;
}

//______________________________________________________________________________
TGFileDialog::~TGFileDialog()
{
   // Delete file dialog.

   if (IsZombie()) return;
   delete fOk; delete fCancel; delete fOverWR;
   delete fName;
   delete fTypes;
   delete fFv; delete fFc;

   delete fCdup; delete fNewf; delete fList; delete fDetails;
   fClient->FreePicture(fPcdup);
   fClient->FreePicture(fPnewf);
   fClient->FreePicture(fPlist);
   fClient->FreePicture(fPdetails);

   delete fTreeLB;
   
}

//______________________________________________________________________________
void TGFileDialog::CloseWindow()
{
   // Close file dialog.

   fFileInfo->fFilename = 0;
   DeleteWindow();
}

//______________________________________________________________________________
Bool_t TGFileDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process messages generated by the user input in the file dialog.

   TGTreeLBEntry *e;
   TGTextLBEntry *te;
   TGFileItem *f;
   void *p = 0;
   const char *txt;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case kIDF_OK:
                     // same code as under kTE_ENTER
                     if (fTbfname->GetTextLength() == 0) {
                        txt = "Please provide file name or use \"Cancel\"";
                        new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                                     "Missing File Name", txt, kMBIconExclamation,
                                     kMBOk);
                        return kTRUE;
                     } else if (!gSystem->AccessPathName(fTbfname->GetString(), kFileExists) &&
                                !strcmp(fOk->GetTitle(), "Save") &&
                                (!(fOverWR->GetState() == kButtonDown))) {
                        Int_t ret;
                        txt = Form("File name %s already exists, OK to overwrite it?",
                                   fTbfname->GetString());
                        new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                                     "File Name Exist", txt, kMBIconExclamation,
                                     kMBYes | kMBNo, &ret);
                        if (ret == kMBNo)
                           return kTRUE;
                     }
                     fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                    fTbfname->GetString());
                     if (fOverWR && (fOverWR->GetState() == kButtonDown))
                        fFileInfo->fOverwrite = kTRUE;
                     else
                        fFileInfo->fOverwrite = kFALSE;
                     DeleteWindow();
                     break;

                  case kIDF_CANCEL:
                     fFileInfo->fFilename = 0;
                     DeleteWindow();
                     break;

                  case kIDF_CDUP:
                     fFc->ChangeDirectory("..");
                     fTreeLB->Update(fFc->GetDirectory());
                     if (fFileInfo->fIniDir) delete [] fFileInfo->fIniDir;
                     fFileInfo->fIniDir = StrDup(fFc->GetDirectory());
                     break;

                  case kIDF_NEW_FOLDER:
                     Warning("ProcessMessage", "new folder not yet implemented");
                     break;

                  case kIDF_LIST:
                     fFv->SetViewMode(kLVList);
                     fDetails->SetState(kButtonUp);
                     break;

                  case kIDF_DETAILS:
                     fFv->SetViewMode(kLVDetails);
                     fList->SetState(kButtonUp);
                     break;
               }
               break;

            case kCM_COMBOBOX:
               switch (parm1) {
                  case kIDF_FSLB:
                     e = (TGTreeLBEntry *) fTreeLB->GetSelectedEntry();
                     if (e) {
                        fFc->ChangeDirectory(e->GetPath()->GetString());
                        fTreeLB->Update(fFc->GetDirectory());
                        if (fFileInfo->fIniDir) delete [] fFileInfo->fIniDir;
                        fFileInfo->fIniDir = StrDup(fFc->GetDirectory());
                     }
                     break;

                  case kIDF_FTYPESLB:
                     te = (TGTextLBEntry *) fTypes->GetSelectedEntry();
                     if (te) {
                        //fTbfname->Clear();
                        //fTbfname->AddText(0, fFileInfo->fFileTypes[te->EntryId()+1]);
                        fFileInfo->fFileTypeIdx = te->EntryId();
                        fFc->SetFilter(fFileInfo->fFileTypes[fFileInfo->fFileTypeIdx+1]);
                        fFc->DisplayDirectory();
                        fClient->NeedRedraw(fName);
                     }
                     break;
               }
               break;

            default:
               break;
         } // switch(GET_SUBMSG(msg))
         break;

      case kC_CONTAINER:
         switch (GET_SUBMSG(msg)) {
            case kCT_ITEMCLICK:
               if (parm1 == kButton1) {
                  if (fFc->NumSelected() == 1) {
                     TGLVEntry *e = (TGLVEntry *) fFc->GetNextSelected(&p);
                     fTbfname->Clear();
                     fTbfname->AddText(0, e->GetItemName()->GetString());
                     fClient->NeedRedraw(fName);
                  }
               }
               break;

            case kCT_ITEMDBLCLICK:

               if (parm1 == kButton1) {
                  if (fFc->NumSelected() == 1) {
                     f = (TGFileItem *) fFc->GetNextSelected(&p);
                     if (R_ISDIR(f->GetType())) {
                        fFc->ChangeDirectory(f->GetItemName()->GetString());
                        fTreeLB->Update(fFc->GetDirectory());
                        if (fFileInfo->fIniDir) delete [] fFileInfo->fIniDir;
                        fFileInfo->fIniDir = StrDup(fFc->GetDirectory());
                     } else {
                        if (!strcmp(fOk->GetTitle(), "Save") && 
                            (!(fOverWR->GetState() == kButtonDown))) {
                           
                           Int_t ret;
                           txt = Form("File name %s already exists, OK to overwrite it?",
                                      fTbfname->GetString());
                           new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                                        "File Name Exist", txt, kMBIconExclamation,
                                        kMBYes | kMBNo, &ret);
                           if (ret == kMBNo)
                              return kTRUE;
                        }
                        fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                       fTbfname->GetString());
                        if (fOverWR && (fOverWR->GetState() == kButtonDown))
                           fFileInfo->fOverwrite = kTRUE;
                        else
                           fFileInfo->fOverwrite = kFALSE;

                        DeleteWindow();
                     }
                  }
               }

               break;

            default:
               break;

         } // switch(GET_SUBMSG(msg))
         break;

      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_ENTER:
               // same code as under kIDF_OK
               if (fTbfname->GetTextLength() == 0) {
                  const char *txt = "Please provide file name or use \"Cancel\"";
                  new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                               "Missing File Name", txt, kMBIconExclamation,
                               kMBOk);
                  return kTRUE;
               } else if (!gSystem->AccessPathName(fTbfname->GetString(), kFileExists) &&
                          !strcmp(fOk->GetTitle(), "Save") &&
                          (!(fOverWR->GetState() == kButtonDown))) {
                  Int_t ret;
                  txt = Form("File name %s already exists, OK to overwrite it?",
                             fTbfname->GetString());
                  new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                               "File Name Exist", txt, kMBIconExclamation,
                               kMBYes | kMBNo, &ret);
                  if (ret == kMBNo)
                     return kTRUE;
               }
               fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                              fTbfname->GetString());
               if (fOverWR && (fOverWR->GetState() == kButtonDown))
                  fFileInfo->fOverwrite = kTRUE;
               else
                  fFileInfo->fOverwrite = kFALSE;
               DeleteWindow();
               break;

            default:
               break;
         }
         break;

      default:
         break;

   } // switch(GET_MSG(msg))

   fClient->NeedRedraw(fFc);
   return kTRUE;
}
