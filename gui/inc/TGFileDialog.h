// @(#)root/gui:$Name:  $:$Id: TGFileDialog.h,v 1.2 2001/05/02 11:45:46 rdm Exp $
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

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


enum EFileDialogMode {
   kFDOpen,
   kFDSave
};


class TGLabel;
class TGTextBuffer;
class TGTextEntry;
class TGComboBox;
class TGPictureButton;
class TGTextButton;
class TGListView;
class TGFileContainer;
class TGFSComboBox;


class TGFileInfo {
public:
   char        *fFilename;    // selected file name
   char        *fIniDir;      // on input: initial directory, on output: new directory
   const char **fFileTypes;   // file types used to filter selectable files
   TGFileInfo() : fFilename(0), fIniDir(0), fFileTypes(0) { }
   ~TGFileInfo() { delete [] fFilename; delete [] fIniDir; }
};


class TGFileDialog : public TGTransientFrame {

protected:
   TGLabel           *fLookin;   // "Save in" or "Look in" label
   TGLabel           *fLfname;
   TGLabel           *fLftypes;
   TGTextBuffer      *fTbfname;
   TGTextEntry       *fName;
   TGComboBox        *fTypes;
   TGFSComboBox      *fTreeLB;
   TGPictureButton   *fCdup;     // top toolbar button
   TGPictureButton   *fNewf;     // top toolbar button
   TGPictureButton   *fList;     // top toolbar button
   TGPictureButton   *fDetails;  // top toolbar button
   const TGPicture   *fPcdup;    // picture for fCdup
   const TGPicture   *fPnewf;    // picture for fNewf
   const TGPicture   *fPlist;    // picture for fList
   const TGPicture   *fPdetails; // picture for fDetails
   TGTextButton      *fOk;       // ok button
   TGTextButton      *fCancel;   // cancel button
   TGListView        *fFv;       // file list view
   TGFileContainer   *fFc;       // file list view container (containing the files)
   TGHorizontalFrame *fHtop;     // main frame directly in popup
   TGHorizontalFrame *fHf;       // frame for file name and types
   TGHorizontalFrame *fHfname;
   TGHorizontalFrame *fHftype;
   TGVerticalFrame   *fVf;
   TGVerticalFrame   *fVbf;
   TGLayoutHints     *fLmain;
   TGLayoutHints     *fLhl;
   TGLayoutHints     *fLb1;
   TGLayoutHints     *fLb2;
   TGLayoutHints     *fLvf;
   TGLayoutHints     *fLvbf;
   TGLayoutHints     *fLb;
   TGLayoutHints     *fLht;
   TGLayoutHints     *fLht1;
   TGFileInfo        *fFileInfo;

public:
   TGFileDialog(const TGWindow *p, const TGWindow *main,
                EFileDialogMode dlg_type, TGFileInfo *file_info);
   virtual ~TGFileDialog();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void CloseWindow();

   ClassDef(TGFileDialog,0)  //File selection dialog
};

#endif
