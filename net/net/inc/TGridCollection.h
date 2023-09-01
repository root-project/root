// @(#)root/net:$Id$
// Author: Andreas-Joachim Peters 2005-05-09

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridCollection
#define ROOT_TGridCollection

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridCollection                                                      //
//                                                                      //
// Class which manages collection files on the Grid.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

class TMap;
class TFile;
class TEntryList;
class TList;
class TDSet;
class TGridResult;
class TFileCollection;


class TGridCollection : public TObject {
public:
   TGridCollection() { }
   virtual ~TGridCollection() { }

   virtual void         Reset()
      { MayNotUse("Reset"); }
   virtual TMap        *Next()
      { MayNotUse("Next"); return nullptr;}
   virtual Bool_t       Remove(TMap *)
      { MayNotUse("Remove"); return 0;}
   virtual const char  *GetTURL(const char * /*name*/ = "")
      { MayNotUse("GetTURL"); return nullptr;}
   virtual const char  *GetSURL(const char * /*name*/ = "")
      { MayNotUse("GetSURL"); return nullptr;}
   virtual const char  *GetLFN(const char * /*name*/ = "")
      { MayNotUse("GetLFN"); return nullptr;}
   virtual Long64_t    GetSize(const char * /*name*/ = "")
      { MayNotUse("GetSize"); return -1;}
   virtual Bool_t      IsOnline(const char * /*name*/ = "")
      { MayNotUse("IsOnline"); return 0;}
   virtual Bool_t      IsSelected(const char * /*name*/ = "")
      { MayNotUse("IsSelected"); return 0;}
   virtual void        Status()
      { MayNotUse("Status"); }
   virtual void        SetTag(const char * , const char * , TMap* )
      { MayNotUse("SetTag"); }
   virtual Bool_t      SelectFile(const char *, Int_t /*nstart*/ = -1 , Int_t /*nstop*/ = -1)
      { MayNotUse("SelectFile"); return kFALSE;}
   virtual Bool_t      DeselectFile(const char *, Int_t /*nstart*/ = -1, Int_t /*nstop*/ = -1)
      { MayNotUse("DeselectFile"); return kFALSE;}
   virtual Bool_t      InvertSelection()
      { MayNotUse("InvertSelection"); return kFALSE;}
   virtual Bool_t      DownscaleSelection(UInt_t /* scaler */ = 2)
      { MayNotUse("DownscaleSelection"); return kFALSE;}
   virtual Bool_t      ExportXML(const char *, Bool_t /*selected*/ = kTRUE, Bool_t /*online*/ = kTRUE,
                                 const char * /*name*/ = "ROOT xml", const char * /*comment*/ = "Exported XML")
      { MayNotUse("ExportXML"); return kFALSE;}
   virtual const char* GetExportUrl()
      { MayNotUse("GetExportUrl"); return nullptr;}
   virtual Bool_t      SetExportUrl(const char * /*exporturl*/ = nullptr)
      { MayNotUse("SetExportUrl"); return kFALSE;}
   virtual void         Print(Option_t * = "") const override
      { MayNotUse("Print"); }
   virtual TFile       *OpenFile(const char *)
      { MayNotUse("OpenFile"); return nullptr;}
   virtual TList       *GetFileGroupList() const
      { MayNotUse("GetFileGroupList"); return nullptr;}
   virtual TEntryList  *GetEntryList(const char *)
      { MayNotUse("GetEntryList"); return nullptr;}
   virtual UInt_t       GetNofGroups() const
      { MayNotUse("GetNofGroups"); return 0;}
   virtual UInt_t       GetNofGroupfiles() const
      { MayNotUse("GetNofGroupfiles"); return 0;}
   virtual Bool_t       OverlapCollection(TGridCollection *)
      { MayNotUse("OverlapCollection"); return 0;}
   virtual void         Add(TGridCollection *)
      { MayNotUse("Add");}
   virtual Bool_t       Stage(Bool_t /*bulk*/ = kFALSE, Option_t * /*TFileStager option*/ = "")
      { MayNotUse("Stage"); return kFALSE;}
   virtual Bool_t       Prepare(Bool_t /*bulk*/ = kFALSE)
      { MayNotUse("Prepare"); return kFALSE;}
   virtual Bool_t       CheckIfOnline(Bool_t /*bulk*/ = kFALSE)
      { MayNotUse("CheckIfOnline"); return kFALSE;}
   virtual TDSet       *GetDataset(const char *, const char * , const char *)
      { MayNotUse("GetDataset"); return nullptr;}
   virtual TGridResult *GetGridResult(const char * /*filename*/ = "", Bool_t /*onlyonline*/ = kTRUE , Bool_t /*publicaccess*/ = kFALSE )
      { MayNotUse("GetGridResult"); return nullptr;}
   virtual Bool_t       LookupSUrls(Bool_t /*verbose*/ = kTRUE)
      { MayNotUse("LookupSUrls"); return kFALSE;}
   virtual TList       *GetTagFilterList() const
      { MayNotUse("GetTagFilterList"); return nullptr;}
   virtual void         SetTagFilterList(TList *)
      { MayNotUse("SetTagFilterList");}
   virtual const char* GetCollectionName() const
      { MayNotUse("GetCollectionName"); return nullptr;}
   virtual const char* GetInfoComment() const
      { MayNotUse("GetInfoComment"); return nullptr;}
   virtual TFileCollection* GetFileCollection(const char* /*name*/ = "", const char* /*title*/ = "") const
      { MayNotUse("GetFileCollection"); return nullptr;}

   ClassDefOverride(TGridCollection,1)  // ABC managing collection of files on the Grid
};

#endif
