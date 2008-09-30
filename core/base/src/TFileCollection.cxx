// @(#)root/base:$Id$
// Author: Gerhard Erich Bruckner, Jan Fiete Grosse-Oetringhaus  04/06/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileCollection                                                      //
//                                                                      //
// Class that contains a list of TFileInfo's and accumulated meta       //
// data information about its entries. This class is used to describe   //
// file sets as stored by Grid file catalogs, by PROOF or any other     //
// collection of TFile names.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFileCollection.h"
#include "THashList.h"
#include "TFileInfo.h"
#include "TIterator.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TSystem.h"
#include "Riostream.h"
#include "TRegexp.h"



ClassImp(TFileCollection)

//______________________________________________________________________________
TFileCollection::TFileCollection(const char *name, const char *title,
                                 const char *textfile, Int_t nfiles, Int_t firstfile)
   : TNamed(name, title), fList(0), fMetaDataList(0),
     fTotalSize(0), fNFiles(0), fNStagedFiles(0), fNCorruptFiles(0)
{
   // TFileCollection constructor. Specify a name and title describing
   // the list. If textfile is specified the file is opened and a
   // TFileCollection is created containing the files in the textfile.

   fList = new THashList();
   fList->SetOwner();

   fMetaDataList = new TList;
   fMetaDataList->SetOwner();

   AddFromFile(textfile, nfiles, firstfile);
}

//______________________________________________________________________________
TFileCollection::~TFileCollection()
{
   // Cleanup.

   delete fList;
   delete fMetaDataList;
}

//______________________________________________________________________________
Int_t TFileCollection::Add(TFileInfo *info)
{
   // Add TFileInfo to the collection.

   if (fList && info) {
      fList->Add(info);
      return 1;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
Int_t TFileCollection::Add(TFileCollection *coll)
{
   // Add content of the TFileCollection to this collection.

   if (fList && coll && coll->GetList()) {
      TList* list = coll->GetList();
      for (Int_t i=0; i<list->GetEntries(); i++) {
         fList->Add((TFileInfo*) list->At(i));
      }
      return 1;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
Int_t TFileCollection::AddFromFile(const char *textfile, Int_t nfiles, Int_t firstfile)
{
   // Add file names contained in the specified text file.
   // The file should contain one url per line; empty lines or lines starting with '#'
   // (commented lines) are ignored.
   // If nfiles > 0 only nfiles files are added, starting from file 'firstfile' (>= 1).
   // The method returns the number of added files.

   if (!fList)
     return 0;

   Int_t nf = 0;
   if (textfile && *textfile) {
      ifstream f;
      f.open(gSystem->ExpandPathName(textfile));
      if (f.is_open()) {
         Bool_t all = (nfiles <= 0) ? kTRUE : kFALSE;
         Int_t ff = (!all && (firstfile < 1)) ? 1 : firstfile;
         Int_t nn = 0;
         while (f.good() && (all || nf < nfiles)) {
            TString line;
            line.ReadToDelim(f);
            // Skip commented or empty lines
            if (!line.IsWhitespace() && !line.BeginsWith("#")) {
               nn++;
               if (all || nn >= ff) {
                  fList->Add(new TFileInfo(line));
                  nf++;
               }
            }
         }
         f.close();
         Update();
      } else
         Error("AddFromFile", "unable to open file %s", textfile);
   }
   return nf;
}

//______________________________________________________________________________
Int_t TFileCollection::Add(const char *dir)
{
   // Add all files matching the specified pattern to the collection.
   // 'dir' can include wildcards after the last slash, which causes all
   // matching files in that directory to be added.
   // If dir is the full path of a file, only one element is added.
   // Return value is the number of added files.

   Int_t nf = 0;

   if (!fList)
      return nf;

   if (!dir || !*dir) {
      Error("AddFromDirectory", "input dir undefined");
      return nf;
   }

   FileStat_t st;
   FileStat_t tmp;
   TString baseDir = gSystem->DirName(dir);
   // if the 'dir' or its base dir exist
   if (gSystem->GetPathInfo(dir, st) == 0 ||
       gSystem->GetPathInfo(baseDir, tmp) == 0) {
      // If 'dir' points to a single file, add to the list and exit
      if (R_ISREG(st.fMode)) {
         // regular, single file
         TFileInfo *info = new TFileInfo(dir);
         info->SetBit(TFileInfo::kStaged);
         Add(info);
         nf++;
         Update();
         return nf;
      } else {
         void *dataSetDir = gSystem->OpenDirectory(gSystem->DirName(dir));
         if (!dataSetDir) {
            // directory cannot be opened
            Error("AddFromDirectory", "directory %s cannot be opened",
                  gSystem->DirName(dir));
         } else {
            const char *ent;
            TString filesExp(gSystem->BaseName(dir));
            filesExp.ReplaceAll("*",".*");
            TRegexp rg(filesExp);
            while ((ent = gSystem->GetDirEntry(dataSetDir))) {
               TString entryString(ent);
               if (entryString.Index(rg) != kNPOS) {
                  // matching dir entry
                  TString fn = gSystem->DirName(dir);
                  fn += "/";
                  fn += ent;
                  gSystem->GetPathInfo(fn, st);
                  if (R_ISREG(st.fMode)) {
                     // regular file
                     TFileInfo *info = new TFileInfo(fn);
                     info->SetBit(TFileInfo::kStaged);
                     Add(info);
                     nf++;
                  }
               }
            }
            // close the directory
            gSystem->FreeDirectory(dataSetDir);
            Update();
         }
      }
   }
   return nf;
}

//______________________________________________________________________________
TFileCollection *TFileCollection::GetStagedSubset()
{
   // Creates a subset of the files that have the kStaged & !kCorrupted bit set.

   if (!fList)
     return 0;

   TFileCollection *subset = new TFileCollection(GetName(), GetTitle());

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*>(iter.Next()))) {
      if (fileInfo->TestBit(TFileInfo::kStaged) && !fileInfo->TestBit(TFileInfo::kCorrupted))
         subset->Add(fileInfo);
   }

   subset->Update();

   return subset;
}

//______________________________________________________________________________
Int_t TFileCollection::Update(Long64_t avgsize)
{
   // Update accumulated information about the elements of the collection
   // (e.g. fTotalSize). If 'avgsize' > 0, use an average file size of 'avgsize'
   // bytes when the size info is not available.
   // Also updates the meta data information by summarizing
   // the meta data of the contained objects.
   // Return -1 in case of any failure, 0 if the total size is exact, 1 if
   // incomplete, 2 if complete but (at least partially) estimated.

   if (!fList)
     return -1;

   Int_t rc = 0;

   fTotalSize = 0;
   fNStagedFiles = 0;
   fNCorruptFiles = 0;
   fMetaDataList->Clear();

   fNFiles = fList->GetEntries();

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*> (iter.Next()))) {

      if (fileInfo->GetSize() > 0) {
         fTotalSize += fileInfo->GetSize();
      } else {
         rc = 1;
         if (avgsize > 0) {
            rc = 2;
            fTotalSize += avgsize;
         }
      }

      if (fileInfo->TestBit(TFileInfo::kStaged) && !fileInfo->TestBit(TFileInfo::kCorrupted)) {
         fNStagedFiles++;

         if (fileInfo->GetMetaDataList()) {
            TIter metaDataIter(fileInfo->GetMetaDataList());
            // other than TFileInfoMeta is also allowed in list
            TObject *obj = 0;
            while ((obj = metaDataIter.Next())) {
               TFileInfoMeta *metaData = dynamic_cast<TFileInfoMeta*>(obj);
               if (!metaData)
                  continue;
               if (!metaData->IsTree())
                  continue;

               // find corresponding entry in TFileCollection's meta data
               TFileInfoMeta *metaDataSum = dynamic_cast<TFileInfoMeta*>(fMetaDataList->FindObject(metaData->GetName()));
               Bool_t newObj = kFALSE;
               if (!metaDataSum) {
                  // create explicitly, there are some values that do not make sense for the sum
                  metaDataSum = new TFileInfoMeta(metaData->GetName(), metaData->GetTitle());
                  fMetaDataList->Add(metaDataSum);
                  newObj = kTRUE;
               }

               // sum the values
               if (newObj)
                  metaDataSum->SetEntries(metaData->GetEntries());
               else
                  metaDataSum->SetEntries(metaDataSum->GetEntries() + metaData->GetEntries());
            }
         }
      }
      if (fileInfo->TestBit(TFileInfo::kCorrupted))
         fNCorruptFiles++;
   }

   // Done
   return rc;
}

//______________________________________________________________________________
void TFileCollection::Print(Option_t *option) const
{
   // Prints the contents of the TFileCollection.
   // If option contains "M": prints meta data entries,
   // if option contains "F": prints all the files in the collection.

   Printf("TFileCollection %s - %s contains: %lld files with a size of"
          " %lld bytes, %.1f %% staged - default tree name: '%s'",
          GetName(), GetTitle(), fNFiles, fTotalSize, GetStagedPercentage(),
          GetDefaultTreeName());

   if (TString(option).Contains("M", TString::kIgnoreCase)) {
      Printf("The files contain the following trees:");

      TIter metaDataIter(fMetaDataList);
      TFileInfoMeta* metaData = 0;
      while ((metaData = dynamic_cast<TFileInfoMeta*>(metaDataIter.Next()))) {
         if (!metaData->IsTree())
            continue;

         Printf("Tree %s: %lld events", metaData->GetName(), metaData->GetEntries());
      }
   }

   if (fList && TString(option).Contains("F", TString::kIgnoreCase)) {
      Printf("The collection contains the following files:");
      fList->Print();
   }
}

//______________________________________________________________________________
void TFileCollection::SetAnchor(const char *anchor)
{
   // Calls TUrl::SetAnchor() for all URLs contained in all TFileInfos.

   if (!fList)
     return;

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*>(iter.Next()))) {
      fileInfo->ResetUrl();
      TUrl *url = 0;
      while ((url = fileInfo->NextUrl()))
         url->SetAnchor(anchor);
      fileInfo->ResetUrl();
   }
}

//______________________________________________________________________________
void TFileCollection::SetBitAll(UInt_t f)
{
   // Set the bit for all TFileInfos

   if (!fList)
     return;

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*>(iter.Next())))
      fileInfo->SetBit(f);
}

//______________________________________________________________________________
void TFileCollection::ResetBitAll(UInt_t f)
{
   // Reset the bit for all TFileInfos

   if (!fList)
     return;

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*>(iter.Next())))
      fileInfo->ResetBit(f);
}

//______________________________________________________________________________
const char *TFileCollection::GetDefaultTreeName() const
{
   // Returns the name of the first tree in the meta data list.
   // Returns 0 in case no trees are found in the meta data list.

   TIter metaDataIter(fMetaDataList);
   TFileInfoMeta *metaData = 0;
   while ((metaData = dynamic_cast<TFileInfoMeta*>(metaDataIter.Next()))) {
      if (!metaData->IsTree())
         continue;
      return metaData->GetName();
   }
   return 0;
}

//______________________________________________________________________________
Long64_t TFileCollection::GetTotalEntries(const char *tree) const
{
   // Returns the number of entries for the specified tree (retrieved from meta data).
   // If tree is not specified, use the default tree name.
   // Returns -1 in case the specified tree is not found.

   if (!tree || !*tree) {
      tree = GetDefaultTreeName();
      if (!tree)
         return -1;
   }

   TFileInfoMeta *metaData = dynamic_cast<TFileInfoMeta*>(fMetaDataList->FindObject(tree));
   if (!metaData)
      return -1;

   return metaData->GetEntries();
}

//______________________________________________________________________________
TFileInfoMeta *TFileCollection::GetMetaData(const char *meta) const
{
   // Returns the meta data object with the soecified meta name.
   // Returns 0 in case specified meta data is not found.

   if (!meta || !*meta)
      return 0;

   return dynamic_cast<TFileInfoMeta*>(fMetaDataList->FindObject(meta));
}

//______________________________________________________________________________
void TFileCollection::SetDefaultMetaData(const char *meta)
{
   // Moves the indicated meta data in the first position, so that
   // it becomes efectively the default.

   TFileInfoMeta *fim = GetMetaData(meta);
   if (fim) {
      fMetaDataList->Remove(fim);
      fMetaDataList->AddFirst(fim);
   }
}

//______________________________________________________________________________
void TFileCollection::RemoveMetaData(const char *meta)
{
   // Removes the indicated meta data object in all TFileInfos and this object
   // If no name is given all metadata is removed

   if (fList) {
      TIter iter(fList);
      TFileInfo *fileInfo = 0;
      while ((fileInfo = dynamic_cast<TFileInfo*>(iter.Next())))
         fileInfo->RemoveMetaData(meta);
   }

   if (meta) {
      TObject* obj = fMetaDataList->FindObject("meta");
      if (obj) {
         fMetaDataList->Remove(obj);
         delete obj;
      }
   } else
      fMetaDataList->Clear();
}

//______________________________________________________________________________
void TFileCollection::Sort()
{
   // Sort the collection.

   if (!fList)
     return;

   fList->Sort();
}

//______________________________________________________________________________
TObjString *TFileCollection::ExportInfo(const char *name)
{
   // Export the relevant info as a string; use 'name' as collection name,
   // if defined, else use GetName().
   // The output object must be destroyed by the caller

   TString treeInfo;
   if (GetDefaultTreeName()) {
      treeInfo = Form(" %s ", GetDefaultTreeName());
      if (treeInfo.Length() < 14)
         treeInfo.Resize(14);
      TFileInfoMeta* meta = GetMetaData(GetDefaultTreeName());
      if (meta)
         treeInfo += Form("| %8lld ", meta->GetEntries());
   } else {
      treeInfo = "        N/A";
   }
   treeInfo.Resize(25);

   // Renormalize the size to kB, MB or GB
   const char *unit[4] = {"kB", "MB", "GB", "TB"};
   Int_t k = 0;
   Long64_t refsz = 1024;
   Long64_t xsz = (Long64_t) (GetTotalSize() / refsz);
   while (xsz > 1024 && k < 3) {
      k++;
      refsz *= 1024;
      xsz = (Long64_t) (GetTotalSize() / refsz);
   }

   // The name
   TString dsname(name);
   if (dsname.IsNull()) dsname = GetName();

   // Create the output string
   TObjString *outs =
      new TObjString(Form("%s| %7lld |%s| %5lld %s |  %3d %%", dsname.Data(),
                     GetNFiles(), treeInfo.Data(), xsz, unit[k],
                     (Int_t)GetStagedPercentage()));
   // Done
   return outs;
}
