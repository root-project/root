// @(#)root/base:$Name: v5-11-02 $:$Id: TFileInfo.cxx,v 1.6 2006/03/21 14:53:11 rdm Exp $
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
TFileInfo::TFileInfo(const TFileInfo& fi) :
  TNamed(fi),
  fCurrentUrl(fi.fCurrentUrl),
  fUrlList(fi.fUrlList),
  fSize(fi.fSize),
  fUUID(fi.fUUID),
  fMD5(fi.fMD5),
  fEntries(fi.fEntries),
  fFirst(fi.fFirst),
  fLast(fi.fLast),
  fMetaDataObject(fi.fMetaDataObject)
{ }

//______________________________________________________________________________
TFileInfo& TFileInfo::operator=(const TFileInfo& fi)
{
  if(this!=&fi) {
    TNamed::operator=(fi);
    fCurrentUrl=fi.fCurrentUrl;
    fUrlList=fi.fUrlList;
    fSize=fi.fSize;
    fUUID=fi.fUUID;
    fMD5=fi.fMD5;
    fEntries=fi.fEntries;
    fFirst=fi.fFirst;
    fLast=fi.fLast;
    fMetaDataObject=fi.fMetaDataObject;
  } return *this;
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
