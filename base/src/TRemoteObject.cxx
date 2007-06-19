// @(#)root/base:$Name:  $:$Id: TFile.h,v 1.64 2007/05/04 15:14:11 rdm Exp $
// Author: Bertrand Bellenot   19/06/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRemoteObject                                                        //
//                                                                      //
// Describes a Remote Object for the browser.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TApplication.h"
#include "TROOT.h"
#include "TRemoteObject.h"
#include "TSystem.h"
#include "TBrowser.h"
#include "TOrdCollection.h"
#include "TList.h"
#include "TClass.h"

ClassImp(TRemoteObject);

//______________________________________________________________________________
TRemoteObject::TRemoteObject()
{
   // Create a remote object.
}

//______________________________________________________________________________
TRemoteObject::TRemoteObject(const char *name, const char *title,
                             const char *classname) : TNamed(name, title)
{
   // Create a remote object.

   fIsFolder = kFALSE;
   fClassName = classname;
   if ((fClassName == "TSystemDirectory") ||
       (fClassName == "TFile"))
      fIsFolder = kTRUE;
   if (!strcmp(classname, "TSystemDirectory") ||
       !strcmp(classname, "TSystemFile")) {
      gSystem->GetPathInfo(name, fFileStat);
   }
   fRemoteAddress = (Long64_t) this;
}

//______________________________________________________________________________
TRemoteObject::~TRemoteObject()
{
   // Delete system directory object.

}

//______________________________________________________________________________
void TRemoteObject::Browse(TBrowser *b)
{
   // Browse remote object.

   TList *ret;
   TRemoteObject *robj;
   const char *file;

   if (fClassName == "TSystemFile") {
      if (b)
         b->ExecuteDefaultAction(this);
      return;
   }
   if (fClassName == "TKey") {
      if (b->GetRefreshFlag())
         b->SetRefreshFlag(kFALSE);
      gApplication->SetBit(TApplication::kProcessRemotely);
      TObject *obj = (TObject *)gROOT->ProcessLine(Form("((TApplicationServer *)gApplication)->BrowseKey(\"%s\");", GetName()));
      if (b && obj) {
         if (obj->IsA()->GetMethodWithPrototype("SetDirectory", "TDirectory*"))
            gROOT->ProcessLine(Form("((%s *)0x%lx)->SetDirectory(0);", obj->ClassName(), obj));
         obj->Browse(b);
         b->SetRefreshFlag(kTRUE);
      }
   }
   if (fClassName == "TSystemDirectory") {
      if (b->GetRefreshFlag())
         b->SetRefreshFlag(kFALSE);
      gApplication->SetBit(TApplication::kProcessRemotely);
      ret = (TList *)gROOT->ProcessLine(Form("((TApplicationServer *)gApplication)->BrowseDirectory(\"%s\");", GetTitle()));
      if (ret) {
         TIter next(ret);
         while ((robj = (TRemoteObject *)next())) {
            file = robj->GetName();
            if (b->TestBit(TBrowser::kNoHidden) && file[0] == '.' && file[1] != '.' )
               continue;
            b->Add(robj, robj->GetName());
         }
      }
      return;
   }
   if (fClassName == "TFile") {
      if (b->GetRefreshFlag())
         b->SetRefreshFlag(kFALSE);
      gApplication->SetBit(TApplication::kProcessRemotely);
      ret = (TList *)gROOT->ProcessLine(Form("((TApplicationServer *)gApplication)->BrowseFile(\"%s\");", GetName()));
      if (ret) {
         TIter next(ret);
         while ((robj = (TRemoteObject *)next())) {
            file = robj->GetName();
            b->Add(robj, robj->GetName());
         }
      }
      return;
   }
}

//______________________________________________________________________________
static Bool_t IsItDirectory(const char *name, const char *dirfile)
{
   // Check if name is a directory.

   Long64_t size;
   Long_t id, flags, modtime;

   gSystem->ChangeDirectory(dirfile);
   flags = id = size = modtime = 0;
   gSystem->GetPathInfo(name, &id, &size, &flags, &modtime);
   Int_t isdir = (Int_t)flags & 2;

   return isdir ? kTRUE : kFALSE;
}

//______________________________________________________________________________
TList *TRemoteObject::Browse()
{
   // Browse OS system directories.

   // Collections to keep track of all browser objects that have been
   // generated. It's main goal is to prevent the contineous
   // allocations of new objects with the same names during browsing.
   TList *objects  = new TList;

   static Int_t level = 0;
   const char *name = GetTitle();
   TRemoteObject *sdir;
   const char *file;

   if (GetName()[0] == '.' && GetName()[1] == '.')
      SetName(gSystem->BaseName(name));

   void *dir = gSystem->OpenDirectory(name);

   if (!dir)
      return 0;

   while ((file = gSystem->GetDirEntry(dir))) {
      if (IsItDirectory(file, GetTitle())) {
         level++;
         TString sdirpath;
         if (!strcmp(file, "."))
            sdirpath = name;
         else if (!strcmp(file, ".."))
            sdirpath = gSystem->DirName(name);
         else {
            sdirpath =  name;
            if (!sdirpath.EndsWith("/"))
               sdirpath += "/";
            sdirpath += file;
         }
         sdir = new TRemoteObject(file, sdirpath.Data(), "TSystemDirectory");
         objects->Add(sdir);
         level--;
      } else {
         sdir = new TRemoteObject(file, gSystem->WorkingDirectory(), "TSystemFile");
         objects->Add(sdir);
      }
   }
   gSystem->FreeDirectory(dir);
   return objects;
}

//______________________________________________________________________________
Bool_t TRemoteObject::GetFileStat(FileStat_t *buf)
{
   // Get remote file status.

   buf->fDev    = fFileStat.fDev;
   buf->fIno    = fFileStat.fIno;
   buf->fMode   = fFileStat.fMode;
   buf->fUid    = fFileStat.fUid;
   buf->fGid    = fFileStat.fGid;
   buf->fSize   = fFileStat.fSize;
   buf->fMtime  = fFileStat.fMtime;
   buf->fIsLink = fFileStat.fIsLink;
   return kTRUE;
}

//______________________________________________________________________________
void TRemoteObject::Streamer(TBuffer &b)
{
   // Remote object streamer.

   if (b.IsReading()) {
      b >> fFileStat.fDev;
      b >> fFileStat.fIno;
      b >> fFileStat.fMode;
      b >> fFileStat.fUid;
      b >> fFileStat.fGid;
      b >> fFileStat.fSize;
      b >> fFileStat.fMtime;
      b >> fFileStat.fIsLink;
      b >> fIsFolder;
      b >> fRemoteAddress;
      b >> fClassName;
      b >> fKeyObjectName;
      b >> fKeyClassName;
   }
   else {
      b << fFileStat.fDev;
      b << fFileStat.fIno;
      b << fFileStat.fMode;
      b << fFileStat.fUid;
      b << fFileStat.fGid;
      b << fFileStat.fSize;
      b << fFileStat.fMtime;
      b << fFileStat.fIsLink;
      b << fIsFolder;
      b << fRemoteAddress;
      b << fClassName;
      b << fKeyObjectName;
      b << fKeyClassName;
   }
   TNamed::Streamer(b);
}

