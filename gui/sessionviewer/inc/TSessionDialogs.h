// @(#)root/sessionviewer:$Id$
// Author: Marek Biskup, Jakub Madejczyk, Bertrand Bellenot 10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSessionDialogs
#define ROOT_TSessionDialogs


#include "TSessionViewer.h"

class TList;
class TSeqCollection;
class TChain;
class TDSet;
class TGTextEntry;
class TGTextButton;
class TGTextBuffer;
class TGCheckButton;
class TGLabel;
class TGListView;
class TGPicture;
class TGFileContainer;

//////////////////////////////////////////////////////////////////////////
// New Chain Dialog
//////////////////////////////////////////////////////////////////////////

class TNewChainDlg : public TGTransientFrame {

private:
   TGFileContainer      *fContents;       // macro files container
   TGListView           *fListView;       // memory objects list view
   TGLVContainer        *fLVContainer;    // and its container
   TGTextBuffer         *fNameBuf;        // buffer for dataset name
   TGTextEntry          *fName;           // dataset name text entry
   TGTextButton         *fOkButton;       // ok button
   TGTextButton         *fCancelButton;   // cancel button
   TSeqCollection       *fChains;         // collection of datasets
   TObject              *fChain;          // actual TDSet or TChain

public:
   TNewChainDlg(const TGWindow *p=0, const TGWindow *main=0);
   virtual ~TNewChainDlg();

   void         UpdateList();
   virtual void OnDoubleClick(TGLVEntry*,Int_t);
   virtual void DisplayDirectory(const TString &fname);
   void         OnElementClicked(TGLVEntry* entry, Int_t btn);
   void         OnElementSelected(TObject *obj); //*SIGNAL*

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void CloseWindow();

   ClassDef(TNewChainDlg, 0) // New chain dialog
};

//////////////////////////////////////////////////////////////////////////
// New Query Dialog
//////////////////////////////////////////////////////////////////////////

class TNewQueryDlg : public TGTransientFrame {

private:
   Bool_t             fEditMode;       // kTRUE if used to edit existing query
   Bool_t             fModified;       // kTRUE if settings have changed
   TGCompositeFrame  *fFrmNewQuery;    // top (main) frame
   TGCompositeFrame  *fFrmMore;        // options frame
   TGTextButton      *fBtnMore;        // "more >>" / "less <<" button
   TGTextButton      *fBtnClose;       // close button
   TGTextButton      *fBtnSave;        // save button
   TGTextButton      *fBtnSubmit;      // save & submit button

   TGTextEntry       *fTxtQueryName;   // query name text entry
   TGTextEntry       *fTxtChain;       // chain name text entry
   TGTextEntry       *fTxtSelector;    // selector name text entry
   TGTextEntry       *fTxtOptions;     // options text entry
   TGNumberEntry     *fNumEntries;     // number of entries selector
   TGNumberEntry     *fNumFirstEntry;  // first entry selector
   TGTextEntry       *fTxtEventList;   // event list text entry
   TSessionViewer    *fViewer;         // pointer on main viewer
   TQueryDescription *fQuery;          // query description class
   TObject           *fChain;          // actual TChain

public:
   TNewQueryDlg(TSessionViewer *gui, Int_t Width, Int_t Height,
                   TQueryDescription *query = 0, Bool_t editmode = kFALSE);
   virtual ~TNewQueryDlg();
   void     Build(TSessionViewer *gui);
   void     OnNewQueryMore();
   void     OnBrowseChain();
   void     OnBrowseSelector();
   void     OnBrowseEventList();
   void     OnBtnSaveClicked();
   void     OnBtnCloseClicked();
   void     OnBtnSubmitClicked();
   void     OnElementSelected(TObject *obj);
   void     CloseWindow();
   void     Popup();
   void     SettingsChanged();
   void     UpdateFields(TQueryDescription *desc);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TNewQueryDlg, 0) // New query dialog
};

//////////////////////////////////////////////////////////////////////////
// Upload DataSet Dialog
//////////////////////////////////////////////////////////////////////////

class TUploadDataSetDlg : public TGTransientFrame {

private:
   Bool_t                fUploading;
   TList                *fSkippedFiles;   // List of skipped files
   TGTextEntry          *fDSetName;       // dataset name text entry
   TGTextEntry          *fDestinationURL; // destination URL text entry
   TGTextEntry          *fLocationURL;    // location URL text entry
   TGListView           *fListView;       // dataset files list view
   TGLVContainer        *fLVContainer;    // and its container
   TGTextButton         *fAddButton;      // Add >> button
   TGTextButton         *fBrowseButton;   // Browse... button
   TGTextButton         *fRemoveButton;   // Remove button
   TGTextButton         *fClearButton;    // Clear button
   TGCheckButton        *fOverwriteDSet;  // overwrite DataSet
   TGCheckButton        *fOverwriteFiles; // overwrite All Files
   TGCheckButton        *fAppendFiles;    // append files
   TGTextButton         *fUploadButton;   // Upload button
   TGTextButton         *fCloseDlgButton; // Close Dialog button
   TSessionViewer       *fViewer;         // pointer on main viewer

public:
   TUploadDataSetDlg(TSessionViewer *gui, Int_t w, Int_t h);
   virtual ~TUploadDataSetDlg();

   virtual void   CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void           AddFiles(const char *fileName);
   void           AddFiles(TList *fileList);
   void           BrowseFiles();
   void           ClearFiles();
   void           RemoveFile();
   void           UploadDataSet();
   void           OnOverwriteDataset(Bool_t on);
   void           OnOverwriteFiles(Bool_t on);
   void           OnAppendFiles(Bool_t on);

   ClassDef(TUploadDataSetDlg, 0) // New query dialog
};

#endif
