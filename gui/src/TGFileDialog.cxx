// @(#)root/gui:$Name:  $:$Id: TGFileDialog.cxx,v 1.7 2002/06/12 17:56:25 rdm Exp $
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

#ifndef WIN32
#include <sys/stat.h>
#endif

#ifdef GDK_WIN32
#include <sys/stat.h>
#endif

enum {
   kIDF_CDUP,
   kIDF_NEW_FOLDER,
   kIDF_LIST,
   kIDF_DETAILS,
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
TGFileDialog::TGFileDialog(const TGWindow *p, const TGWindow *main,
                           EFileDialogMode dlg_type, TGFileInfo *file_info) :
   TGTransientFrame(p, main, 10, 10, kVerticalFrame)
{
   // Create a file selection dialog. Depending on the dlg_type it can be
   // used for opening or saving a file.

   int i;

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
   } else
      fFileInfo = file_info;

   if (!fFileInfo->fFileTypes)
      fFileInfo->fFileTypes = gDefTypes;

   if (!fFileInfo->fIniDir)
      fFileInfo->fIniDir = StrDup(".");

   fHtop = new TGHorizontalFrame(this, 10, 10);

   fLmain = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 4, 4, 3, 1);

   //--- top toolbar elements

   fLookin = new TGLabel(fHtop, new TGHotString((dlg_type == kFDSave)
                         ? "S&ave in" : "&Look in:"));
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

   fLhl = new TGLayoutHints(kLHintsLeft | kLHintsCenterY,  5,  5, 2, 2);
   fLht = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 10,  0, 2, 2);
   fLb1 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY,  8,  0, 2, 2);
   fLb2 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY,  0, 15, 2, 2);

   fTreeLB->Resize(200, fTreeLB->GetDefaultHeight());

   fHtop->AddFrame(fLookin, fLhl);
   fHtop->AddFrame(fTreeLB, fLht);
   fHtop->AddFrame(fCdup, fLb1);
   fHtop->AddFrame(fNewf, fLb1);
   fHtop->AddFrame(fList, fLb1);
   fHtop->AddFrame(fDetails, fLb2);

   AddFrame(fHtop, fLmain);

   //--- file view

   fFv = new TGListView(this, 400, 161);

   fFc = new TGFileContainer(fFv->GetViewPort(),
                             10, 10, kHorizontalFrame, fgWhitePixel);
   fFc->Associate(this);

   fFv->GetViewPort()->SetBackgroundColor(fgWhitePixel);
   fFv->SetContainer(fFc);
   fFv->SetViewMode(kLVList);

   fFc->SetFilter(fFileInfo->fFileTypes[1]);
   fFc->Sort(kSortByType);
   fFc->ChangeDirectory(fFileInfo->fIniDir);
   fTreeLB->Update(fFc->GetDirectory());

   fList->SetState(kButtonEngaged);

   AddFrame(fFv, fLmain);

   //--- file name and types

   fHf = new TGHorizontalFrame(this, 10, 10);

   fVf = new TGVerticalFrame(fHf, 10, 10);
   fLvf = new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX);

   fHfname = new TGHorizontalFrame(fVf, 10, 10);

   fLfname = new TGLabel(fHfname, new TGHotString("File &name:"));
   fTbfname = new TGTextBuffer(1034);
   fName = new TGTextEntry(fHfname, fTbfname);
   fName->Resize(220, fName->GetDefaultHeight());
   fName->Associate(this);

   fLht1 = new TGLayoutHints(kLHintsRight | kLHintsCenterY, 0, 20, 2, 2);

   fHfname->AddFrame(fLfname, fLhl);
   fHfname->AddFrame(fName, fLht1);

   fVf->AddFrame(fHfname, fLvf);

   fHftype = new TGHorizontalFrame(fVf, 10, 10);

   fLftypes = new TGLabel(fHftype, new TGHotString("Files of &type:"));
   fTypes = new TGComboBox(fHftype, kIDF_FTYPESLB);
   fTypes->Associate(this);
   fTypes->Resize(220, fName->GetDefaultHeight());

   char s[64];
   for (i = 0; fFileInfo->fFileTypes[i] != 0; i += 2) {
      sprintf(s, "%s (%s)", fFileInfo->fFileTypes[i], fFileInfo->fFileTypes[i+1]);
      fTypes->AddEntry(s, i);
   }
   fTypes->Select(0);

   fTbfname->Clear();
   //fTbfname->AddText(0, fFileInfo->fFileTypes[1]);

   fHftype->AddFrame(fLftypes, fLhl);
   fHftype->AddFrame(fTypes, fLht1);

   fVf->AddFrame(fHftype, fLvf);

   fHf->AddFrame(fVf, fLvf);

   //--- Open/Save and Cancel buttons

   fVbf = new TGVerticalFrame(fHf, 10, 10, kFixedWidth);
   fLvbf = new TGLayoutHints(kLHintsLeft | kLHintsCenterY);

   fLb = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 2, 2);

   fOk = new TGTextButton(fVbf, new TGHotString((dlg_type == kFDSave)
                                                 ? "&Save" : "&Open"), kIDF_OK);
   fCancel = new TGTextButton(fVbf, new TGHotString("Cancel"), kIDF_CANCEL);

   fOk->Associate(this);
   fCancel->Associate(this);

   fVbf->AddFrame(fOk, fLb);
   fVbf->AddFrame(fCancel, fLb);

   UInt_t width = TMath::Max(fOk->GetDefaultWidth(), fCancel->GetDefaultWidth()) + 20;
   fVbf->Resize(width, fVbf->GetDefaultHeight());

   fHf->AddFrame(fVbf, fLvbf);

   AddFrame(fHf, fLmain);

   MapSubwindows();

   TGDimension size = GetDefaultSize();

   Resize(size);

   //---- position relative to the parent's window

   if (main) {
      int      ax, ay;
      Window_t wdummy;
      gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(),
                        (Int_t)(((TGFrame *) main)->GetWidth() - fWidth) >> 1,
                        (Int_t)(((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                        ax, ay, wdummy);
      if (ax < 0) ax = 10;
      if (ay < 0) ay = 10;
      Move(ax, ay);
      SetWMPosition(ax, ay);
   }

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
   fClient->WaitFor(this);
}

//______________________________________________________________________________
TGFileDialog::~TGFileDialog()
{
   // Delete file dialog.

   delete fOk; delete fCancel;
   delete fVbf;
   delete fLb; delete fLvbf;

   delete fLfname; delete fName;
   delete fHfname;

   delete fLftypes; delete fTypes;
   delete fHftype;

   delete fVf;
   delete fHf;

   delete fFv; delete fFc;

   delete fCdup; delete fNewf; delete fList; delete fDetails;
   fClient->FreePicture(fPcdup);
   fClient->FreePicture(fPnewf);
   fClient->FreePicture(fPlist);
   fClient->FreePicture(fPdetails);

   delete fTreeLB; delete fLookin;

   delete fLb1; delete fLb2; delete fLhl;
   delete fLht; delete fLht1; delete fLvf;

   delete fHtop;
   delete fLmain;
}

//______________________________________________________________________________
void TGFileDialog::CloseWindow()
{
   // Close file dialog.

   fFileInfo->fFilename = 0;
   delete this;
}

//______________________________________________________________________________
Bool_t TGFileDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   TGTreeLBEntry *e;
   TGTextLBEntry *te;
   TGFileItem *f;
   void *p = 0;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case kIDF_OK:
                     // same code as under kTE_ENTER
                     if (fTbfname->GetTextLength() == 0) {
                        const char *txt = "Please provide file name or use \"Cancel\"";
                        new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                                     "Missing File Name", txt, kMBIconExclamation,
                                     kMBOk);
                        return kTRUE;
                     }
#ifndef WIN32
                     if (fFc->NumSelected() == 1) {
                        f = (TGFileItem *) fFc->GetNextSelected(&p);
                        if (S_ISDIR(f->GetType())) {
                           fFc->ChangeDirectory(f->GetItemName()->GetString());
                           fTreeLB->Update(fFc->GetDirectory());
                           if (fFileInfo->fIniDir) delete [] fFileInfo->fIniDir;
                           fFileInfo->fIniDir = StrDup(fFc->GetDirectory());
                        } else {
                           fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                          fTbfname->GetString());
                           delete this;
                        }
                     }
#else
#ifdef GDK_WIN32
                     if (fFc->NumSelected() == 1) {
                        f = (TGFileItem *) fFc->GetNextSelected(&p);
                        if ((f->GetType()) & _S_IFDIR) {
                           fFc->ChangeDirectory(f->GetItemName()->GetString());
                           fTreeLB->Update(fFc->GetDirectory());
                        } else {
                           fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                          fTbfname->GetString());
                           delete this;
                        }
                     }
#endif
#endif
                     break;

                  case kIDF_CANCEL:
                     fFileInfo->fFilename = 0;
                     delete this;
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
                        fFc->SetFilter(fFileInfo->fFileTypes[te->EntryId()+1]);
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
#ifndef WIN32
               if (parm1 == kButton1) {
                  if (fFc->NumSelected() == 1) {
                     f = (TGFileItem *) fFc->GetNextSelected(&p);
                     if (S_ISDIR(f->GetType())) {
                        fFc->ChangeDirectory(f->GetItemName()->GetString());
                        fTreeLB->Update(fFc->GetDirectory());
                        if (fFileInfo->fIniDir) delete [] fFileInfo->fIniDir;
                        fFileInfo->fIniDir = StrDup(fFc->GetDirectory());
                     } else {
                        fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                       fTbfname->GetString());
                        delete this;
                     }
                  }
               }
#else
#ifdef GDK_WIN32
               if (parm1 == kButton1) {
                  if (fFc->NumSelected() == 1) {
                     f = (TGFileItem *) fFc->GetNextSelected(&p);
                     if ((f->GetType()) & _S_IFDIR) {
                        fFc->ChangeDirectory(f->GetItemName()->GetString());
                        fTreeLB->Update(fFc->GetDirectory());
                     } else {
                        fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                       fTbfname->GetString());
                        delete this;
                     }
                  }
               }
#endif
#endif
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
               }
#ifndef WIN32
               if (fFc->NumSelected() == 1) {
                  f = (TGFileItem *) fFc->GetNextSelected(&p);
                  if (S_ISDIR(f->GetType())) {
                     fFc->ChangeDirectory(f->GetItemName()->GetString());
                     fTreeLB->Update(fFc->GetDirectory());
                     if (fFileInfo->fIniDir) delete [] fFileInfo->fIniDir;
                     fFileInfo->fIniDir = StrDup(fFc->GetDirectory());
                  } else {
                     fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                    fTbfname->GetString());
                     delete this;
                  }
               }
#else
#ifdef GDK_WIN32
               if (fFc->NumSelected() == 1) {
                  f = (TGFileItem *) fFc->GetNextSelected(&p);
                  if ((f->GetType()) & _S_IFDIR) {
                     fFc->ChangeDirectory(f->GetItemName()->GetString());
                     fTreeLB->Update(fFc->GetDirectory());
                  } else {
                     fFileInfo->fFilename = gSystem->ConcatFileName(fFc->GetDirectory(),
                                                                    fTbfname->GetString());
                     delete this;
                  }
               }
#endif
#endif
               break;

            default:
               break;
         }
         break;

      default:
         break;

   } // switch(GET_MSG(msg))

   return kTRUE;
}
