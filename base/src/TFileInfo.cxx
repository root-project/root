// @(#)root/base:$Name:  $:$Id: TFileInfo.cxx,v 1.11 2007/05/03 11:53:23 rdm Exp $
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


ClassImp(TFileInfo)

//______________________________________________________________________________
TFileInfo::TFileInfo(const char *url , Long64_t size, const char *uuid,
   const char *md5, Long64_t entries, Long64_t first, Long64_t last,
   TObject *meta) : fCurrentUrl(0), fUrlList(0), fSize(size), fUUID(0),
   fMD5(0), fEntries(entries), fFirst(first), fLast(last), fMetaDataObject(meta)
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

   fCurrentUrl = 0;
   if (url) {
      fUrlList = new TList();
      fUrlList->SetOwner();
      // TFile Info Constructor
      AddUrl(url);
   }
}

//______________________________________________________________________________
TFileInfo::~TFileInfo()
{
   // Destructor.

   SafeDelete(fMetaDataObject);
   SafeDelete(fUUID);
   SafeDelete(fMD5);
   SafeDelete(fUrlList);
}

//______________________________________________________________________________
TUrl *TFileInfo::NextUrl()
{
   // Iterator Function, init Iteration with ResetUrl().
   // The first Call to NextUrl() will return the 1st element,
   // the seconde the 2nd element aso.

   if (fCurrentUrl && (fCurrentUrl == fUrlList->First())) {
      TUrl *returl = GetCurrentUrl();
      fCurrentUrl = (TUrl*)fUrlList->After((TObject*)fCurrentUrl);
      return returl;
   }

   if (fCurrentUrl)
      fCurrentUrl = (TUrl*)fUrlList->After((TObject*)fCurrentUrl);
   return GetCurrentUrl();
}

//______________________________________________________________________________
TUrl *TFileInfo::FindByUrl(const char *url)
{
   // Find an element from a URL.

   TIter nextUrl(fUrlList);
   TUrl *urlelement;

   while  ( (urlelement = (TUrl*) nextUrl() ) ) {
      if ( TString(urlelement->GetUrl()) == TString(url) ) {
         return urlelement;
      }
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TFileInfo::AddUrl(const char *url)
{
   // Add a new URL
   if (FindByUrl(url)) {
      return kFALSE;
   }

   TUrl *newurl = new TUrl(url);
   // We set the current Url to the first url added
   if (fUrlList->GetSize() == 0) {
      fCurrentUrl = newurl;
   }

   fUrlList->Add( newurl );
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFileInfo::RemoveUrl(const char *url)
{
   // Remove an URL.
   TUrl *lurl;
   if ((lurl=(TUrl*)FindByUrl(url))) {
      fUrlList->Remove((TObject*) lurl);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TFileInfo::AddMetaDataObject(TObject *obj)
{
   // Add's a meta data object to the file info object

   if (obj) {
      if (fMetaDataObject)
         delete fMetaDataObject;
      fMetaDataObject = obj;
   }
}

//______________________________________________________________________________
void TFileInfo::RemoveMetaDataObject()
{
   // Remove all the metadata obejects.

   if (fMetaDataObject) {
      delete fMetaDataObject;
      fMetaDataObject = 0;
   }
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

   cout << "UUID: " << GetUUID()->AsString() << " Size: " << GetSize() << " MD5: " << GetMD5()->AsString() << endl;
   TIter next(fUrlList);
   TObject* obj;

   while ( (obj = next() ) ){
      const char *url = ((TUrl*)obj)->GetUrl();
      cout << " URL: " << url << endl;
   }
}

//______________________________________________________________________________
TList *TFileInfo::CreateList(const char *file)
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
            fileList->Add(new TFileInfo(line.Data()));
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
