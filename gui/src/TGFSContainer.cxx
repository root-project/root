// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   19/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFileIcon, TGFileEntry, TGFSContainer                               //
//                                                                      //
// Utility classes used by the file selection dialog (TGFSDialog).      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef WIN32
#ifndef R__LYNXOS
#include <sys/stat.h>
#endif
#include <pwd.h>
#ifndef __VMS
#include <grp.h>
#endif
#endif

#include "TGFSContainer.h"
#include "TGPicture.h"
#include "TGMsgBox.h"
#include "TGMimeTypes.h"
#include "TRegexp.h"
#include "TList.h"
#include "TSystem.h"


ClassImp(TGFileContainer)


//______________________________________________________________________________
class TGFSFrameElement : public TGFrameElement {
public:
   TGFileContainer  *fContainer;

   Bool_t IsSortable() const { return kTRUE; }
   Int_t  Compare(TObject *obj);
};

//______________________________________________________________________________
Int_t TGFSFrameElement::Compare(TObject *obj)
{
   // Sort frame elements in file selection list view container.

  int type1, type2;

  TGFileItem *f1 = (TGFileItem *) fFrame;
  TGFileItem *f2 = (TGFileItem *) ((TGFrameElement *) obj)->fFrame;

  switch (fContainer->fSortType) {
     default:
     case kSortByName:
        return strcmp(f1->GetItemName()->GetString(), f2->GetItemName()->GetString());

     case kSortByType:
        //--- this is not exactly what I want...

        type1 = f1->GetType();
        type2 = f2->GetType();

        //--- use posix macros

#ifndef WIN32
        if (S_ISDIR(type1)) type1 = 1;
        #if defined(S_IFLNK)
        else if ((type1 & S_IFMT) == S_IFLNK) type1 = 2;
        #endif
        #if defined(S_IFSOCK)
        else if ((type1 & S_IFMT) == S_IFSOCK) type1 = 3;
        #endif
        else if (S_ISFIFO(type1)) type1 = 4;
        else if (S_ISREG(type1) && (type1 & S_IXUSR)) type1 = 5;
        else type1 = 6;

        if (S_ISDIR(type2)) type2 = 1;
        #if defined(S_IFLNK)
        else if ((type2 & S_IFMT) == S_IFLNK) type2 = 2;
        #endif
        #if defined(S_IFSOCK)
        else if ((type2 & S_IFMT) == S_IFSOCK) type2 = 3;
        #endif
        else if (S_ISFIFO(type2)) type2 = 4;
        else if (S_ISREG(type2) && (type2 & S_IXUSR)) type2 = 5;
        else type2 = 6;
#else
        Error("Compare", "not yet implemented for Win32");
#endif
        if (type1 < type2) return -1;
        if (type1 > type2) return 1;
        return strcmp(f1->GetItemName()->GetString(), f2->GetItemName()->GetString());

     case kSortBySize:
        if (f1->GetSize() < f2->GetSize()) return -1;
        if (f1->GetSize() > f2->GetSize()) return 1;
        return strcmp(f1->GetItemName()->GetString(), f2->GetItemName()->GetString());
  }
}


//______________________________________________________________________________
Bool_t TViewUpdateTimer::Notify()
{
   fContainer->HandleTimer(0);
   Reset();
   return kFALSE;
}


//______________________________________________________________________________
void TGFileIcon::DoRedraw()
{
   // Draw icon.

   TGIcon::DoRedraw();
   if (fLpic) fLpic->Draw(fId, fgBckgndGC, 0, 0);
}


//______________________________________________________________________________
TGFileItem::TGFileItem(const TGWindow *p,
                       const TGPicture *bpic, const TGPicture *blpic,
                       const TGPicture *spic, const TGPicture *slpic,
                       TGString *name, Int_t type, ULong_t size, Int_t uid,
                       Int_t gid, EListViewMode viewMode, UInt_t options,
                       ULong_t back) :
   TGLVEntry(p, bpic, spic, name, 0, viewMode, options, back)
{
   // Create a list view item.

   char tmp[256];
   ULong_t fsize, bsize;

   fLcurrent =
   fBlpic = blpic;
   fSlpic = slpic;

   fViewMode = (EListViewMode) -1;
   SetViewMode(viewMode);

   fType = type;
   fSize = size;
   fUid  = uid;
   fGid  = gid;

   // FIXME: hack...
   fIsLink = (blpic != 0);

   fSubnames = new TGString* [5];

#ifndef WIN32
   // file type
   sprintf(tmp, "%c%c%c%c%c%c%c%c%c%c",
                (fIsLink ?
                 'l' :
                 (S_ISREG(type) ?
                  '-' :
                  (S_ISDIR(type) ?
                   'd' :
                    (S_ISCHR(type) ?
                     'c' :
                     (S_ISBLK(type) ?
                      'b' :
                      (S_ISFIFO(type) ?
                       'p' :
                       (S_ISSOCK(type) ?
                        's' : '?' ))))))),
                 ((type & S_IRUSR) ? 'r' : '-'),
                 ((type & S_IWUSR) ? 'w' : '-'),
                 ((type & S_ISUID) ? 's' : ((type & S_IXUSR) ? 'x' : '-')),
                 ((type & S_IRGRP) ? 'r' : '-'),
                 ((type & S_IWGRP) ? 'w' : '-'),
                 ((type & S_ISGID) ? 's' : ((type & S_IXGRP) ? 'x' : '-')),
                 ((type & S_IROTH) ? 'r' : '-'),
                 ((type & S_IWOTH) ? 'w' : '-'),
                 ((type & S_ISVTX) ? 't' : ((type & S_IXOTH) ? 'x' : '-')));
   fSubnames[0] = new TGString(tmp);

   // file size
   fsize = bsize = fSize;
   if (fsize > 1024) {
      fsize /= 1024;
      if (fsize > 1024) {
         // 3.7MB is more informative than just 3MB
         sprintf(tmp, "%ld.%ldM", fsize/1024, (fsize%1024)/103);
      } else {
         sprintf(tmp, "%ld.%ldK", bsize/1024, (bsize%1024)/103);
      }
   } else {
      sprintf(tmp, "%ld", bsize);
   }
   fSubnames[1] = new TGString(tmp);

#ifndef R__VMS
   {
      struct group *grp;
      struct passwd *pwd;
      char   tmp[256];

      pwd = getpwuid(fUid);
      if (pwd) {
         fSubnames[2] = new TGString(pwd->pw_name);
      } else {
         sprintf(tmp, "%d", fUid);
         fSubnames[2] = new TGString(tmp);
      }
      grp = getgrgid(fGid);
      if (grp) {
         fSubnames[3] = new TGString(grp->gr_name);
      } else {
         sprintf(tmp, "%d", fGid);
         fSubnames[3] = new TGString(tmp);
      }
   }
#else
   //***NEED TO FIND A ROUTINE IN VMS THAT DOES THE SAME THING AS GETPWUID
#endif

   fSubnames[4] = 0;

   int i;
   for (i = 0; fSubnames[i] != 0; ++i);
      fCtw = new int[i];
   for (i = 0; fSubnames[i] != 0; ++i)
      fCtw[i] = gVirtualX->TextWidth(fFontStruct, fSubnames[i]->GetString(),
                                             fSubnames[i]->GetLength());
#else
   Error("TGFileItem", "not yet implemented for Win32");
#endif
}

//______________________________________________________________________________
void TGFileItem::SetViewMode(EListViewMode viewMode)
{
   // Set container item view mode.

   TGLVEntry::SetViewMode(viewMode);

   if (viewMode == kLVLargeIcons)
      fLcurrent = fBlpic;
   else
      fLcurrent = fSlpic;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGFileItem::DoRedraw()
{
   // Draw list view container item.

   int ix, iy;

   TGLVEntry::DoRedraw();
   if (!fLcurrent) return;

   if (fViewMode == kLVLargeIcons) {
      ix = (fWidth - fLcurrent->GetWidth()) >> 1;
      iy = 0;
   } else {
      ix = 0;
      iy = (fHeight - fLcurrent->GetHeight()) >> 1;
   }

   fLcurrent->Draw(fId, fNormGC, ix, iy);
}


//______________________________________________________________________________
TGFileContainer::TGFileContainer(const TGWindow *p, UInt_t w, UInt_t h,
                                 UInt_t options, ULong_t back) :
   TGLVContainer(p, w, h, options, back)
{
   // Create a list view container which will hold the contents of
   // the current directory.

   fSortType  = kSortByName;
   fFilter    = 0;
   fDirectory = gSystem->WorkingDirectory();
   fRefresh   = new TViewUpdateTimer(this, 5000);
   gSystem->AddTimer(fRefresh);

   fFolder_s = fClient->GetPicture("folder_s.xpm");
   fFolder_t = fClient->GetPicture("folder_t.xpm");
   fApp_s    = fClient->GetPicture("app_s.xpm");
   fApp_t    = fClient->GetPicture("app_t.xpm");
   fDoc_s    = fClient->GetPicture("doc_s.xpm");
   fDoc_t    = fClient->GetPicture("doc_t.xpm");
   fSlink_s  = fClient->GetPicture("slink_s.xpm");
   fSlink_t  = fClient->GetPicture("slink_t.xpm");

   if (!fFolder_s || !fFolder_t ||
       !fApp_s    || !fApp_t    ||
       !fDoc_s    || !fDoc_t    ||
       !fSlink_s  || !fSlink_t)
      Error("TGFileContainer", "required pixmap(s) missing\n");
}

//______________________________________________________________________________
TGFileContainer::~TGFileContainer()
{
   // Delete list view file container.

   if (fRefresh) delete fRefresh;
   if (fFilter)  delete fFilter;
   fClient->FreePicture(fFolder_s);
   fClient->FreePicture(fFolder_t);
   fClient->FreePicture(fApp_s);
   fClient->FreePicture(fApp_t);
   fClient->FreePicture(fDoc_s);
   fClient->FreePicture(fDoc_t);
   fClient->FreePicture(fSlink_s);
   fClient->FreePicture(fSlink_t);
}

//______________________________________________________________________________
void TGFileContainer::AddFrame(TGFrame *f, TGLayoutHints *l)
{
   // Add frame to the composite frame.

   TGFSFrameElement *nw;

   nw = new TGFSFrameElement;
   nw->fFrame     = f;
   nw->fLayout    = l ? l : fgDefaultHints;
   nw->fState     = 1;
   nw->fContainer = this;
   fList->Add(nw);
}

//______________________________________________________________________________
Bool_t TGFileContainer::HandleTimer(TTimer *)
{
   // Refresh container contents. Check every 5 seconds to see if the
   // directory modification date has changed.

#ifndef WIN32
   struct stat sbuf;

   if (stat(fDirectory.Data(), &sbuf) == 0)
      if (fMtime != (ULong_t)sbuf.st_mtime) DisplayDirectory();
#else
   Error("HandleTImer", "not yet implemented for Win32");
#endif

   return kTRUE;
}

//______________________________________________________________________________
void TGFileContainer::SetFilter(const char *filter)
{
   // Set file selection filter.

   if (fFilter) delete fFilter;
   fFilter = new TRegexp(filter, kTRUE);
}

//______________________________________________________________________________
void TGFileContainer::Sort(EFSSortMode sortType)
{
   // Sort file system list view container according to sortType.

   fSortType = sortType;

   fList->Sort();

   TGCanvas *canvas = (TGCanvas *) this->GetParent()->GetParent();
   canvas->Layout();
}

//______________________________________________________________________________
void TGFileContainer::GetFilePictures(const TGPicture **pic,
             const TGPicture **lpic, Int_t file_type, Bool_t is_link,
             const char *name, Bool_t small)
{
   // Determine the file picture for the given file type.

   *pic = fClient->GetMimeTypeList()->GetIcon(name, small);
   if (*pic == 0) {
      *pic = small ? fDoc_t : fDoc_s;
#ifndef WIN32
      if (S_ISREG(file_type) && (file_type) & S_IXUSR)
         *pic = small ? fApp_t : fApp_s;
      if (S_ISDIR(file_type))
         *pic = small ? fFolder_t : fFolder_s;
#else
      Error("GetFilePictures", "not yet implemented for Win32");
#endif
   }

   if (is_link)
      *lpic = small ? fSlink_t : fSlink_s;
   else
      *lpic = 0;
}

//______________________________________________________________________________
void TGFileContainer::ChangeDirectory(const char *path)
{
   // Change current directory.

   TString savdir = gSystem->WorkingDirectory();
   gSystem->ChangeDirectory(fDirectory.Data());   // so path of ".." will work
   if (gSystem->ChangeDirectory(path)) {
      fDirectory = gSystem->WorkingDirectory();
      gSystem->ChangeDirectory(savdir.Data());
      DisplayDirectory();
   }
}

//______________________________________________________________________________
void TGFileContainer::DisplayDirectory()
{
   // Display the contents of the current directory in the container.
   // This can be used to refresh the contents of the window.

   RemoveAll();
   CreateFileList();

   // This automatically calls layout
   Sort(fSortType);

   // Make TGExplorerMainFrame display total objects in status bar
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
               fTotal, fSelected);

   MapSubwindows();
}

//______________________________________________________________________________
void TGFileContainer::CreateFileList()
{
   // This function creates the file list from current dir.

   TString savdir = gSystem->WorkingDirectory();
   if (!gSystem->ChangeDirectory(fDirectory.Data())) return;

#ifndef WIN32
   struct stat sbuf;
   if (stat(".", &sbuf) == 0) fMtime = sbuf.st_mtime;
#else
   Error("CreateFileList", "not yet implemented for Win32");
   return;
#endif

   void *dirp;
   if ((dirp = gSystem->OpenDirectory(".")) == 0) {
      gSystem->ChangeDirectory(savdir.Data());
      return;
   }

   const char *name;
   while ((name = gSystem->GetDirEntry(dirp)) != 0) {
      if (strcmp(name, ".") && strcmp(name, ".."))
         AddFile(name);
   }
   gSystem->FreeDirectory(dirp);

   gSystem->ChangeDirectory(savdir.Data());
}

//______________________________________________________________________________
TGFileItem *TGFileContainer::AddFile(const char *name)
{
   // Add file in container.

   Bool_t      is_link;
   Int_t       type, uid, gid;
   ULong_t     size;
   TString     filename;
   TGFileItem *item = 0;
   const TGPicture *pic, *lpic, *spic, *slpic;

#ifndef WIN32
   struct stat sbuf;
#else
   Error("AddFile", "not yet implemented for Win32");
   return item;
#endif

   type = 0;
   size = 0;
   uid  = 0;
   gid  = 0;
#ifndef WIN32
#ifndef R__VMS
   is_link = kFALSE;
   if (lstat(name, &sbuf) == 0) {
      is_link = S_ISLNK(sbuf.st_mode);
      type = sbuf.st_mode;
      size = sbuf.st_size;
      uid = sbuf.st_uid;
      gid = sbuf.st_gid;
      if (is_link) {
         if (stat(name, &sbuf) == 0) {
            type = sbuf.st_mode;
            size = sbuf.st_size;
         }
      }
   } else {
      char msg[256];

      sprintf(msg, "Can't read file attributes of \"%s\": %s.",
              name, gSystem->GetError());
      new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                   "Error", msg, kMBIconStop, kMBOk);
      return item;
   }

   filename = name;
   if (S_ISDIR(type) || fFilter == 0 ||
       (fFilter && filename.Index(*fFilter) != kNPOS)) {
      GetFilePictures(&pic, &lpic, type, is_link, name, kFALSE);
      GetFilePictures(&spic, &slpic, type, is_link, name, kTRUE);
      item = new TGFileItem(this, pic, lpic, spic, slpic, new TGString(name),
                            type, size, uid, gid, fViewMode);
      AddItem(item);
      fTotal++;
   }
#endif
#endif
   return item;
}
