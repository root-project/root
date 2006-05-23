// @(#)root/base:$Name:  $:$Id: TFileInfo.h,v 1.6 2006/05/15 11:01:13 rdm Exp $
// Author: Andreas-Joachim Peters   20/9/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileInfo
#define ROOT_TFileInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileInfo                                                            //
//                                                                      //
// Class describing a generic file including meta information.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_Turl
#include "TUrl.h"
#endif

#ifndef ROOT_TUUID
#include "TUUID.h"
#endif

#ifndef ROOT_TMD5
#include "TMD5.h"
#endif

#ifndef ROOT_TObjString
#include "TObjString.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif


class TFileInfo : public TNamed {

private:
   TUrl            *fCurrentUrl;     //! current URL to access the file, points to one of the UrlList List or 0, if the list end is reached
   TList           *fUrlList;        //-> URL List of the file
   Long64_t         fSize;           // file size
   TUUID           *fUUID;           //-> uuid of the referenced file
   TMD5            *fMD5;            //-> md5 digest of the file

   Long64_t         fEntries;        // entries in case of a ROOT Tree
   Long64_t         fFirst;          // first entry to process
   Long64_t         fLast;           // last  entry to process

   TObject         *fMetaDataObject; //-> generic file meta data object

protected:
   TFileInfo(const TFileInfo&);
   TFileInfo& operator=(const TFileInfo&);

public:
   TFileInfo(const char *url=0, Long64_t size=-1, const char *uuid=0,
             const char *md5=0, Long64_t entries=-1, Long64_t first=-1,
             Long64_t last=-1, TObject *meta=0);

   virtual ~TFileInfo();

   void            ResetUrl() { fCurrentUrl = (TUrl*)fUrlList->First(); }
   TUrl           *NextUrl();
   TUrl           *GetCurrentUrl() const { return ((fCurrentUrl)?fCurrentUrl:0) ; };
   TUrl           *GetFirstUrl() const { return (TUrl*)fUrlList->First(); }

   Long64_t        GetSize() const       { return fSize; }
   Long64_t        GetEntries() const    { return fEntries; }
   Long64_t        GetFirst() const      { return fFirst; }
   Long64_t        GetLast() const       { return fLast; }
   TUUID          *GetUUID() const       { return fUUID; }
   TMD5           *GetMD5() const        { return fMD5; }
   TObject        *GetMetaObject() const { return fMetaDataObject; }

   void            SetFirst(Long64_t first)     { fFirst = first; }
   void            SetLast(Long64_t last)       { fLast = last; }
   void            SetEntries(Long64_t entries) { fEntries = entries; }

   TUrl           *FindByUrl(const char *url);

   Bool_t          AddUrl(const char *url);
   Bool_t          RemoveUrl(const char *url);
   void            AddMetaDataObject(TObject *obj);
   void            RemoveMetaDataObject();

   Bool_t          IsSortable() const { return kTRUE; }
   Int_t           Compare(const TObject *obj) const;

   void            Print(Option_t *options="") const;

   ClassDef(TFileInfo,1)  // Describes generic file info including meta information
};

#endif
