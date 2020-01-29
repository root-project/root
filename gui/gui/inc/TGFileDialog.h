// @(#)root/gui:$Id$
// Author: Fons Rademakers   20/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGFileDialog
#define ROOT_TGFileDialog

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

#include "TGFrame.h"


enum EFileDialogMode {
   kFDOpen,
   kFDSave
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
   TGFileInfo(const TGFileInfo&) = delete;              // not implemented
   TGFileInfo& operator=(const TGFileInfo&) = delete;   // not implemented

public:
   char         *fFilename{nullptr};            // selected file name
   char         *fIniDir{nullptr};              // on input: initial directory, on output: new directory
   const char  **fFileTypes{nullptr};           // file types used to filter selectable files
   Int_t         fFileTypeIdx{0};               // selected file type, index in fFileTypes
   Bool_t        fOverwrite{kFALSE};            // if true overwrite the file with existing name on save
   Bool_t        fMultipleSelection{kFALSE};    // if true, allow multiple file selection
   TList        *fFileNamesList{nullptr};       // list of selected file names

   TGFileInfo() = default;
   ~TGFileInfo();

   void SetFilename(const char *fname);
   void SetIniDir(const char *inidir);
   void DeleteFileNamesList();

   void SetMultipleSelection(Bool_t option);
};


class TGFileDialog : public TGTransientFrame {

protected:
   TGTextBuffer      *fTbfname;  // text buffer of file name
   TGTextEntry       *fName;     // file name text entry
   TGComboBox        *fTypes;    // file type combo box
   TGFSComboBox      *fTreeLB;   // file system path combo box
   TGPictureButton   *fCdup;     // top toolbar button
   TGPictureButton   *fNewf;     // top toolbar button
   TGPictureButton   *fList;     // top toolbar button
   TGPictureButton   *fDetails;  // top toolbar button
   TGCheckButton     *fCheckB;   // set on/off file overwriting for Open dialog
                                 // OR set on/off multiple file selection for SaveAs dialog
   const TGPicture   *fPcdup;    // picture for fCdup
   const TGPicture   *fPnewf;    // picture for fNewf
   const TGPicture   *fPlist;    // picture for fList
   const TGPicture   *fPdetails; // picture for fDetails
   TGTextButton      *fOk;       // ok button
   TGTextButton      *fCancel;   // cancel button
   TGListView        *fFv;       // file list view
   TGFileContainer   *fFc;       // file list view container (containing the files)
   TGFileInfo        *fFileInfo; // file info passed to this dialog

private:
   TGFileDialog(const TGFileDialog&);              // not implemented
   TGFileDialog& operator=(const TGFileDialog&);   // not implemented

public:
   TGFileDialog(const TGWindow *p = 0, const TGWindow *main = 0,
                EFileDialogMode dlg_type = kFDOpen, TGFileInfo *file_info = 0);
   virtual ~TGFileDialog();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void CloseWindow();

   ClassDef(TGFileDialog,0)  //File selection dialog
};

#endif
