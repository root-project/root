// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   19/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFSContainer
#define ROOT_TGFSContainer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFileIcon, TGFileEntry, TGFSContainer                               //
//                                                                      //
// Utility classes used by the file selection dialog (TGFileDialog).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGListView
#include "TGListView.h"
#endif
#ifndef ROOT_TGIcon
#include "TGIcon.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


//----- file sort mode
enum EFSSortMode {
   kSortByName,
   kSortByType,
   kSortBySize,
   kSortByDate
};


class TRegexp;
class TGPicture;
class TGFileContainer;



class TViewUpdateTimer : public TTimer {

private:
   TGFileContainer   *fContainer;

public:
   TViewUpdateTimer(TGFileContainer *t, Long_t ms) : TTimer(ms, kTRUE) { fContainer = t; }
   Bool_t Notify();
};



class TGFileIcon : public TGIcon {

protected:
   const TGPicture *fLpic;

   virtual void DoRedraw();

public:
   TGFileIcon(const TGWindow *p, const TGPicture *pic, const TGPicture *lpic,
              UInt_t options = kChildFrame, ULong_t back = fgWhitePixel) :
      TGIcon(p, pic, 0, 0, options, back) { fLpic = lpic; }
};



class TGFileItem : public TGLVEntry {

protected:
   const TGPicture  *fBlpic;        // big icon
   const TGPicture  *fSlpic;        // small icon
   const TGPicture  *fLcurrent;     // current icon
   Int_t             fType;         // file type
   Int_t             fUid, fGid;    // file uid and gid
   Bool_t            fIsLink;       // true if symbolic link
   ULong_t           fSize;         // file size

   virtual void DoRedraw();

public:
   TGFileItem(const TGWindow *p,
              const TGPicture *bpic, const TGPicture *blpic,
              const TGPicture *spic, const TGPicture *slpic,
              TGString *name, Int_t type, ULong_t size, Int_t uid, Int_t gid,
              EListViewMode viewMode, UInt_t options = kVerticalFrame,
              ULong_t back = fgWhitePixel);

   virtual void SetViewMode(EListViewMode viewMode);

   Bool_t  IsActive() const { return fActive; }
   Bool_t  IsSymLink() const { return fIsLink; }
   Int_t   GetType() const { return fType; }
   ULong_t GetSize() const { return fSize; }
};



class TGFileContainer : public TGLVContainer {

friend class TGFSFrameElement;

protected:
   EFSSortMode       fSortType;       // sorting mode of contents
   TRegexp          *fFilter;         // file filter
   TViewUpdateTimer *fRefresh;        // refresh timer
   ULong_t           fMtime;          // directory modification time
   TString           fDirectory;      // current directory
   const TGPicture  *fFolder_t;       // small folder icon
   const TGPicture  *fFolder_s;       // big folder icon
   const TGPicture  *fApp_t;          // small application icon
   const TGPicture  *fApp_s;          // big application icon
   const TGPicture  *fDoc_t;          // small document icon
   const TGPicture  *fDoc_s;          // big document icon
   const TGPicture  *fSlink_t;        // small symbolic link icon
   const TGPicture  *fSlink_s;        // big symbolic link icon

   void CreateFileList();

public:
   TGFileContainer(const TGWindow *p, UInt_t w, UInt_t h,
                   UInt_t options = kSunkenFrame,
                   ULong_t back = fgDefaultFrameBackground);
   virtual ~TGFileContainer();

   virtual Bool_t HandleTimer(TTimer *t);

   TGFileItem *AddFile(const char *name);
   void AddFrame(TGFrame *f, TGLayoutHints *l = 0);
   void Sort(EFSSortMode sortType);
   void SetFilter(const char *filter);
   void ChangeDirectory(const char *path);
   void DisplayDirectory();

   const char *GetDirectory() const { return fDirectory.Data(); }

   void GetFilePictures(const TGPicture **pic, const TGPicture **lpic,
                        Int_t file_type, Bool_t is_link, const char *ext,
                        Bool_t small);

   ClassDef(TGFileContainer,0)  // Container containing file system objects
};

#endif
