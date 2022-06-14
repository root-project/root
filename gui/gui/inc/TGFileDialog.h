// @(#)root/gui:$Id$
// Author: Fons Rademakers   20/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGFileDialog
#define ROOT_TGFileDialog


#include "TGFrame.h"


enum EFileDialogMode {
   kFDOpen,
   kFDSave,
   kDOpen,
   kDSave
};


class TGTextBuffer;
class TGTextEntry;
class TGComboBox;
class TGPictureButton;
class TGTextButton;
class TGCheckButton;
class TGListView;
class TGFileContainer;
class TGFSComboBox;


class TGFileInfo {

private:
   TGFileInfo(const TGFileInfo&) = delete;
   TGFileInfo& operator=(const TGFileInfo&) = delete;

public:
   char         *fFilename{nullptr};            ///< selected file name
   char         *fIniDir{nullptr};              ///< on input: initial directory, on output: new directory
   const char  **fFileTypes{nullptr};           ///< file types used to filter selectable files
   Int_t         fFileTypeIdx{0};               ///< selected file type, index in fFileTypes
   Bool_t        fOverwrite{kFALSE};            ///< if true overwrite the file with existing name on save
   Bool_t        fMultipleSelection{kFALSE};    ///< if true, allow multiple file selection
   TList        *fFileNamesList{nullptr};       ///< list of selected file names

   TGFileInfo() = default;
   ~TGFileInfo();

   void SetFilename(const char *fname);
   void SetIniDir(const char *inidir);
   void DeleteFileNamesList();

   void SetMultipleSelection(Bool_t option);
};


class TGFileDialog : public TGTransientFrame {

protected:
   TGTextBuffer      *fTbfname;  ///< text buffer of file name
   TGTextEntry       *fName;     ///< file name text entry
   TGComboBox        *fTypes;    ///< file type combo box
   TGFSComboBox      *fTreeLB;   ///< file system path combo box
   TGPictureButton   *fCdup;     ///< top toolbar button
   TGPictureButton   *fNewf;     ///< top toolbar button
   TGPictureButton   *fList;     ///< top toolbar button
   TGPictureButton   *fDetails;  ///< top toolbar button
   TGCheckButton     *fCheckB;   ///< set on/off file overwriting for Open dialog
                                 ///< OR set on/off multiple file selection for SaveAs dialog
   const TGPicture   *fPcdup;    ///< picture for fCdup
   const TGPicture   *fPnewf;    ///< picture for fNewf
   const TGPicture   *fPlist;    ///< picture for fList
   const TGPicture   *fPdetails; ///< picture for fDetails
   TGTextButton      *fOk;       ///< ok button
   TGTextButton      *fCancel;   ///< cancel button
   TGListView        *fFv;       ///< file list view
   TGFileContainer   *fFc;       ///< file list view container (containing the files)
   TGFileInfo        *fFileInfo; ///< file info passed to this dialog
   EFileDialogMode    fDlgType;  ///< the dialog type passed

private:
   TGFileDialog(const TGFileDialog&) = delete;
   TGFileDialog& operator=(const TGFileDialog&) = delete;

public:
   TGFileDialog(const TGWindow *p = nullptr, const TGWindow *main = nullptr,
                EFileDialogMode dlg_type = kFDOpen, TGFileInfo *file_info = nullptr);
   virtual ~TGFileDialog();

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   void CloseWindow() override;

   ClassDefOverride(TGFileDialog,0)  //File selection dialog
};

#endif
