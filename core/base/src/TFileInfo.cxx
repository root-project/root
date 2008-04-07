// @(#)root/base:$Id$
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
#include "TClass.h"


ClassImp(TFileInfo)
ClassImp(TFileInfoMeta)

//______________________________________________________________________________
TFileInfo::TFileInfo(const char *url, Long64_t size, const char *uuid,
                     const char *md5, TObject *meta)
   : fCurrentUrl(0), fUrlList(0), fSize(size), fUUID(0), fMD5(0),
     fMetaDataList(0)
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
Bool_t TFileInfo::AddUrl(const char *url, Bool_t infront)
{
   // Add a new URL. If 'infront' is TRUE the new url is pushed at the beginning
   // of the list; otherwise is pushed back.
   // Returns kTRUE if successful, kFALSE otherwise.

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

   if (infront)
      fUrlList->AddFirst(newurl);
   else
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
      if (!meta || strlen(meta) <= 0) {
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
      if (!meta || strlen(meta) <= 0)
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
   TObject *m = 0;   // can be any TObject not only TFileInfoMeta
   while ((m = (TObject*) nextm())) {
      cout << " === Meta Data Object ===" << endl;
      m->Print();
   }
}


//______________________________________________________________________________
TFileInfoMeta::TFileInfoMeta(const char *objPath, const char *objClass,
                             Long64_t entries, Long64_t first, Long64_t last,
                             Long64_t totbytes, Long64_t zipbytes)
              : TNamed(objPath, objClass), fEntries(entries), fFirst(first),
                fLast(last), fTotBytes(totbytes), fZipBytes(zipbytes)
{
   // Create file meta data object.

   TString p = objPath;
   if (!p.BeginsWith("/")) {
      p.Prepend("/");
      SetName(p);
   }

   TClass *c = TClass::GetClass(objClass);
   fIsTree = (c && c->InheritsFrom("TTree")) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
TFileInfoMeta::TFileInfoMeta(const char *objPath, const char *objDir,
                             const char *objClass, Long64_t entries,
                             Long64_t first, Long64_t last,
                             Long64_t totbytes, Long64_t zipbytes)
              : TNamed(objPath, objClass), fEntries(entries), fFirst(first),
                fLast(last), fTotBytes(totbytes), fZipBytes(zipbytes)
{
   // Create file meta data object.

   TString sdir = objDir;
   if (!sdir.BeginsWith("/"))
      sdir.Prepend("/");
   if (!sdir.EndsWith("/"))
      sdir += "/";
   sdir += objPath;
   SetName(sdir);

   TClass *c = TClass::GetClass(objClass);
   fIsTree = (c && c->InheritsFrom("TTree")) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
const char *TFileInfoMeta::GetDirectory() const
{
   // Get the object's directory in the ROOT file.

   return gSystem->DirName(GetName());
}

//______________________________________________________________________________
const char *TFileInfoMeta::GetObject() const
{
   // Get the object name, with path stripped off. For full path
   // use GetName().

   return gSystem->BaseName(GetName());
}

//______________________________________________________________________________
void TFileInfoMeta::Print(Option_t * /* option */) const
{
   // Print information about this object.

   cout << " Name:    " << fName << "\n"
        << " Class:   " << fTitle << "\n"
        << " Entries: " << fEntries << "\n"
        << " First:   " << fFirst << "\n"
        << " Last:    " << fLast << endl;
}
