// @(#)root/base:$Name:  $:$Id: TFileInfo.cxx,v 1.15 2007/07/20 15:44:57 rdm Exp $
// Author: Andreas-Joachim Peters   20/9/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileInfo                                                            //
//                                                                      //
// Class describing a generic file including meta information.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFileInfo.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TRegexp.h"
#include "TError.h"
#include "TClass.h"


ClassImp(TFileInfo)
ClassImp(TFileInfoMeta)

//______________________________________________________________________________
TFileInfo::TFileInfo(const char *url, Long64_t size, const char *uuid,
   const char *md5, TObject *meta) : fCurrentUrl(0), fUrlList(0), fSize(size),
   fUUID(0), fMD5(0)
{
   // Constructor.

   if (uuid)
      fUUID = new TUUID(uuid);
   else
      fUUID = new TUUID;

   if (md5)
      fMD5 = new TMD5((const UChar_t*)md5);
   else
      fMD5 = new TMD5;

   // Set's the name from the UUID.
   SetName(fUUID->AsString());
   SetTitle("TFileInfo");

   if (url)
      AddUrl(url);

   if (meta)
      AddMetaData(meta);
}

//______________________________________________________________________________
TFileInfo::~TFileInfo()
{
   // Destructor.

   SafeDelete(fMetaDataList);
   SafeDelete(fUUID);
   SafeDelete(fMD5);
   SafeDelete(fUrlList);
}

//______________________________________________________________________________
TUrl *TFileInfo::GetCurrentUrl() const
{
   // Return the current url.

   if (!fCurrentUrl)
      const_cast<TFileInfo*>(this)->ResetUrl();
   return fCurrentUrl;
}

//______________________________________________________________________________
TUrl *TFileInfo::NextUrl()
{
   // Iterator function, start iteration by calling ResetUrl().
   // The first call to NextUrl() will return the 1st element,
   // the seconde the 2nd element etc. Returns 0 in case no more urls.

   if (!fUrlList)
      return 0;

   TUrl *returl = fCurrentUrl;

   if (fCurrentUrl)
      fCurrentUrl = (TUrl*)fUrlList->After(fCurrentUrl);

   return returl;
}

//______________________________________________________________________________
TUrl *TFileInfo::FindByUrl(const char *url)
{
   // Find an element from a URL. Returns 0 if not found.

   TIter nextUrl(fUrlList);
   TUrl *urlelement;

   while  ((urlelement = (TUrl*) nextUrl())) {
      if ( TString(urlelement->GetUrl()) == TString(url) ) {
         return urlelement;
      }
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TFileInfo::AddUrl(const char *url)
{
   // Add a new URL. Returns kTRUE if successful, kFALSE otherwise.

   if (FindByUrl(url))
      return kFALSE;

   if (!fUrlList) {
      fUrlList = new TList;
      fUrlList->SetOwner();
   }

   TUrl *newurl = new TUrl(url, kTRUE);
   // We set the current Url to the first url added
   if (fUrlList->GetSize() == 0)
      fCurrentUrl = newurl;

   fUrlList->Add(newurl);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFileInfo::RemoveUrl(const char *url)
{
   // Remove an URL. Returns kTRUE if successful, kFALSE otherwise.

   TUrl *lurl;
   if ((lurl = FindByUrl(url))) {
      fUrlList->Remove(lurl);
      if (lurl == fCurrentUrl)
         ResetUrl();
      delete lurl;
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TFileInfo::AddMetaData(TObject *meta)
{
   // Add's a meta data object to the file info object. The object will be
   // adopted by the TFileInfo and should not be deleted by the user.
   // Typically objects of class TFileInfoMeta or derivatives should be added,
   // but any class is accepted.
   // Returns kTRUE if successful, kFALSE otherwise.

   if (meta) {
      if (!fMetaDataList) {
         fMetaDataList = new TList;
         fMetaDataList->SetOwner();
      }
      fMetaDataList->Add(meta);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TFileInfo::RemoveMetaData(const char *meta)
{
   // Remove the metadata obeject. If meta is 0 remove all meta data objects.
   // Returns kTRUE if successful, kFALSE otherwise.

   if (fMetaDataList) {
      if (!meta) {
         SafeDelete(fMetaDataList);
         return kTRUE;
      } else {
         TObject *o = fMetaDataList->FindObject(meta);
         if (o) {
            fMetaDataList->Remove(o);
            delete o;
            return kTRUE;
         }
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
TFileInfoMeta *TFileInfo::GetMetaData(const char *meta) const
{
   // Get meta data object with specified name. If meta is 0
   // get first meta data object. Returns 0 in case no
   // suitable meta data object is found.

   if (fMetaDataList) {
      TFileInfoMeta *m;
      if (!meta)
         m = (TFileInfoMeta *) fMetaDataList->First();
      else
         m = (TFileInfoMeta *) fMetaDataList->FindObject(meta);
      if (m) {
         TClass *c = m->IsA();
         return (c && c->InheritsFrom("TFileInfoMeta")) ? m : 0;
      }
   }
   return 0;
}


//______________________________________________________________________________
Int_t TFileInfo::Compare(const TObject *obj) const
{
   // Compare TFileInfo object by their first urls.

   if (this == obj) return 0;
   if (TFileInfo::Class() != obj->IsA()) return -1;
   return (GetFirstUrl()->Compare(((TFileInfo*)obj)->GetFirstUrl()));
}

//______________________________________________________________________________
void TFileInfo::Print(Option_t * /* option */) const
{
   // Print information about this object.

   GetMD5()->Final();
   cout << "UUID: " << GetUUID()->AsString() << "\n"
        << "MD5:  " << GetMD5()->AsString() << "\n"
        << "Size: " << GetSize() << endl;

   TIter next(fUrlList);
   TUrl *u;
   cout << " === URLs ===" << endl;
   while ((u = (TUrl*)next()))
      cout << " URL: " << u->GetUrl() << endl;

   TIter nextm(fMetaDataList);
   TFileInfoMeta *m;
   while ((m = (TFileInfoMeta*) nextm())) {
      cout << " === Meta Data Object ===" << endl;
      m->Print();
   }
}

//______________________________________________________________________________
TList *TFileInfo::CreateListFromFile(const char *file)
{
   // Open the text 'file' and create TList of TFileInfo objects.
   // The 'file' must include one url per line.
   // The function returns a TList of TFileInfos (possibly empty) or
   // 0 if 'file' can not be opened.

   Int_t fileCount = 0;
   ifstream f;
   TList* fileList = 0;
   f.open(gSystem->ExpandPathName(file), ifstream::out);
   if (f.is_open()) {
      fileList = new TList;
      while (f.good()) {
         TString line;
         line.ReadToDelim(f);
         if (!line.IsWhitespace()) {
            fileList->Add(new TFileInfo(line));
            fileCount++;
         }
      }
      f.close();
   } else {
      ::Error("TFileInfo::CreateList", "unable to open file %s", file);
   }
   return fileList;
}

//______________________________________________________________________________
TList *TFileInfo::CreateListMatching(const char *files)
{
   // Find all the files matching 'files' and return a TList of corresponding
   // TFileInfo objects. 'files' can include wildcards after the last slash.
   // If 'files' is the full path of a file, a list with only one element is
   // created. If no files match the selection, 0 is returned

   if (!files || strlen(files) <= 0) {
      ::Info("TFileInfo::CreateListMatching", "input path undefined");
      return 0;
   }

   // Create the list
   TList *fileList = new TList();
   fileList->SetOwner();

   FileStat_t st;
   // If 'files' points to a single file, fill the list and exit
   if (gSystem->GetPathInfo(files, st) == 0) {
      if (R_ISREG(st.fMode)) {
         // Regular, single file
         fileList->Add(new TFileInfo(files));
         return fileList;
      }
   } else {
      void *dataSetDir = gSystem->OpenDirectory(gSystem->DirName(files));
      if (!dataSetDir) {
         // Directory cannot be open
         ::Error("TFileInfo::CreateListMatching",
                 "directory %s cannot be open", gSystem->DirName(files));
      } else {
         const char* ent;
         TString filesExp(gSystem->BaseName(files));
         filesExp.ReplaceAll("*",".*");
         TRegexp rg(filesExp);
         while ((ent = gSystem->GetDirEntry(dataSetDir))) {
            TString entryString(ent);
            if (entryString.Index(rg) != kNPOS) {
               // matching dir entry
               TString fn(Form("%s/%s",gSystem->DirName(files), ent));
               gSystem->GetPathInfo(fn, st);
               if (R_ISREG(st.fMode))
                  // Regular file
                  fileList->Add(new TFileInfo(fn));
            }
         }
         // Close the directory
         gSystem->FreeDirectory(dataSetDir);
      }
   }
   if (fileList->GetSize() == 0) {
      ::Error("TFileInfo::CreateListMatching",
              "no files match your selection, the list was not created");
      delete fileList;
      return 0;
   }
   // Done
   return fileList;
}


//______________________________________________________________________________
TFileInfoMeta::TFileInfoMeta(const char *objName, const char *objClass,
                             const char *dir, Long64_t entries, Long64_t first,
                             Long64_t last)
   : TNamed(objName, objClass), fDirectory(dir), fEntries(entries),
     fFirst(first), fLast(last)
{
   // Create file meta data object.

   if (fDirectory == "")
      fDirectory = "/";

   TClass *c = TClass::GetClass(objClass);
   fIsTree = (c->InheritsFrom("TTree")) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TFileInfoMeta::Print(Option_t * /* option */) const
{
   // Print information about this object.

   cout << " Name:    " << fName << "\n"
        << " Class:   " << fTitle << "\n"
        << " Dir:     " << fDirectory << "\n"
        << " Entries: " << fEntries << "\n"
        << " First:   " << fFirst << "\n"
        << " Last:    " << fLast << endl;
}
