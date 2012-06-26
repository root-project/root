// @(#)root/alien:$Id$
// Author: Andreas-Joachim Peters 9/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienCollection
#define ROOT_TAlienCollection

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienCollection                                                     //
//                                                                      //
// Class which manages collection of files on AliEn middleware.         //
// The file collection is in the form of an XML file.                   //
//                                                                      //
// The internal list is managed as follows:                             //
// TList* ===> TMap*(file) ===> TMap*(attributes)                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridCollection
#include "TGridCollection.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TFileStager
#include "TFileStager.h"
#endif


class TFileCollection;

class TAlienCollection : public TGridCollection {

private:
   TString      fXmlFile;            // collection XML file
   TList       *fFileGroupList;      //-> list with event file maps
   TIter       *fFileGroupListIter;  //! event file list iterator
   TMap        *fCurrent;            //! current event file map
   UInt_t       fNofGroups;          // number of file groups
   UInt_t       fNofGroupfiles;      // number of files per group
   Bool_t       fHasSUrls;           // defines if SURLs are present in the collection
   Bool_t       fHasSelection;       // defines if the user made some selection on the files
                                     // to be exported for processing
   Bool_t       fHasOnline;          // defines if the collection was checked for the online status
   TString      fLastOutFileName;    // keeps the latest outputfilename produced with GetOutputFileName
   TFileStager *fFileStager;         //! pointer to the file stager object
   TString      fExportUrl;          // defines the url where to store back this collection
   TString      fInfoComment;        // comment in the info section of the XML file
   TString      fCollectionName;     // name of the collection in the collection section of the XML file
   TList       *fTagFilterList;      //-> list of TObjStrings with tags to filter out in export operations

   virtual void ParseXML(UInt_t maxentries);
   Bool_t ExportXML(TFile * file, Bool_t selected, Bool_t online,
                    const char *name, const char *comment);

public:
   TAlienCollection() : fFileGroupList(0), fFileGroupListIter(0), fCurrent(0),
       fNofGroups(0), fNofGroupfiles(0), fHasSUrls(0), fHasSelection(0),
       fHasOnline(0), fFileStager(0), fExportUrl(""), fInfoComment(""),
       fCollectionName("unnamed"), fTagFilterList(0)
      { }
   TAlienCollection(TList *eventlist, UInt_t ngroups = 0,
                    UInt_t ngroupfiles = 0);
   TAlienCollection(const char *localCollectionFile, UInt_t maxentries);

   virtual ~TAlienCollection();

   TFileCollection* GetFileCollection(const char* name = "", const char* title = "") const;

   void        Reset();
   TMap       *Next();
   Bool_t      Remove(TMap * map);
   const char *GetTURL(const char *name = "") ;
   const char *GetSURL(const char *name = "") ;
   const char *GetLFN(const char *name = "") ;
   Long64_t    GetSize(const char *name = "") ;
   Bool_t      IsOnline(const char *name = "") ;
   Bool_t      IsSelected(const char *name = "") ;
   void        Status();
   void        SetTag(const char *tag, const char *value, TMap * tagmap);
   Bool_t      SelectFile(const char *name, Int_t /*start*/ = -1, Int_t /*stop*/ = -1);
   Bool_t      DeselectFile(const char *name, Int_t /*start*/ = -1, Int_t /*stop*/ = -1);
   Bool_t      InvertSelection();
   Bool_t      DownscaleSelection(UInt_t scaler = 2);
   Bool_t      ExportXML(const char *exporturl, Bool_t selected, Bool_t online,
                         const char *name, const char *comment);
   const char *GetExportUrl() {
     if (fExportUrl.Length()) return fExportUrl; else return 0;
   } // return's (if defined) the export url protected:

   Bool_t      SetExportUrl(const char *exporturl = 0);

   void        Print(Option_t * opt) const;
   TFile      *OpenFile(const char *filename) ;

   TEntryList *GetEntryList(const char *name) ;

   TList      *GetFileGroupList() const { return fFileGroupList; }

   UInt_t      GetNofGroups() const { return fNofGroups; }

   UInt_t      GetNofGroupfiles() const { return fNofGroupfiles; }

   Bool_t      OverlapCollection(TGridCollection *comparator);
   void        Add(TGridCollection *addcollection);
   void        AddFast(TGridCollection *addcollection);
   Bool_t      Stage(Bool_t bulk = kFALSE, Option_t* option = "");
   Bool_t      Prepare(Bool_t bulk = kFALSE) { return Stage(bulk,"option=0"); }

   Bool_t      CheckIfOnline(Bool_t bulk = kFALSE);
   TDSet      *GetDataset(const char *type, const char *objname = "*", const char *dir = "/");

   TGridResult *GetGridResult(const char *filename = "",
                              Bool_t onlyonline = kTRUE,
                              Bool_t publicaccess = kFALSE);

   Bool_t      LookupSUrls(Bool_t verbose = kTRUE);

   TList      *GetTagFilterList() const { return fTagFilterList; }

   void        SetTagFilterList(TList * filterlist) { if (fTagFilterList)
     delete fTagFilterList; fTagFilterList = filterlist;
   }

   const char* GetCollectionName() const { return fCollectionName.Data(); }
   const char* GetInfoComment() const { return fInfoComment.Data(); }

   static TGridCollection *Open(const char *collectionurl,
                                UInt_t maxentries = 1000000);
   static TGridCollection *OpenQuery(TGridResult * queryresult,
                                     Bool_t nogrouping = kFALSE);
   static TAlienCollection *OpenAlienCollection(TGridResult * queryresult,
                                             Option_t* option = "");

   const char *GetOutputFileName(const char *infile, Bool_t rename = kTRUE, const char *suffix="root");

   ClassDef(TAlienCollection, 1) // Manages collection of files on AliEn
};

#endif
