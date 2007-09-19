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
#include "TUrl.h"
#include "TSystem.h"
#include "Riostream.h"
#include "TRegexp.h"



ClassImp(TFileCollection)

//______________________________________________________________________________
TFileCollection::TFileCollection(const char *name, const char *title,
                                 const char *textfile)
   : TNamed(name, title), fList(0), fMetaDataList(0),
     fTotalSize(0), fStagedPercentage(0)
{
   // TFileCollection constructor. Specify a name and title describing
   // the list. If textfile is specified the file is opened and a
   // TFileCollection is created containing the files in the textfile.

   fList = new THashList();
   fList->SetOwner();

   fMetaDataList = new TList;
   fMetaDataList->SetOwner();

   AddFromFile(textfile);
}

//______________________________________________________________________________
TFileCollection::~TFileCollection()
{
   // Cleanup.

   delete fList;
   delete fMetaDataList;
}

//______________________________________________________________________________
void TFileCollection::Add(TFileInfo *info)
{
   // Add TFileInfo to the collection.

   fList->Add(info);
}

//______________________________________________________________________________
void TFileCollection::AddFromFile(const char *textfile)
{
   // Add all file names contained in the specified text file.

   if (textfile && *textfile) {
      ifstream f;
      f.open(gSystem->ExpandPathName(textfile), ifstream::out);
      if (f.is_open()) {
         while (f.good()) {
            TString line;
            line.ReadToDelim(f);
            if (!line.IsWhitespace())
               fList->Add(new TFileInfo(line));
         }
         f.close();
         Update();
      } else
         Error("AddFromFile", "unable to open file %s", textfile);
   }
}

//______________________________________________________________________________
void TFileCollection::AddFromDirectory(const char *dir)
{
   // Add all files in the specified directory to the collection. The dir can
   // can include wildcards after the last slash, causing all matching files
   // in that directory to be added. If dir is the full path of a file, only
   // one element is added.

   if (!dir || !*dir) {
      Error("AddFromDirectory", "input dir undefined");
      return;
   }

   FileStat_t st;
   // If 'dir' points to a single file, add to the list and exit
   if (gSystem->GetPathInfo(dir, st) == 0) {
      if (R_ISREG(st.fMode)) {
         // regular, single file
         TFileInfo *info = new TFileInfo(dir);
         info->SetBit(TFileInfo::kStaged);
         Add(info);
         Update();
         return;
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
                  }
               }
            }
            // close the directory
            gSystem->FreeDirectory(dataSetDir);
            Update();
         }
      }
   }
}

//______________________________________________________________________________
TFileCollection *TFileCollection::GetStagedSubset()
{
   // Creates a subset of the files that have the kStaged & !kCorrupted bit set.

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
void TFileCollection::Update()
{
   // Update accumulated information about the elements of the collection
   // (e.g. fTotalSize). Also updates the meta data information by summarizing
   // the meta data of the contained objects.

   fTotalSize = 0;
   fStagedPercentage = 0;

   if (fList->GetEntries() == 0)
      return;

   fMetaDataList->Clear();

   Long64_t stagedFiles = 0;

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*> (iter.Next()))) {
      if (fileInfo->GetSize() > 0)
         fTotalSize += fileInfo->GetSize();

      if (fileInfo->TestBit(TFileInfo::kStaged) && !fileInfo->TestBit(TFileInfo::kCorrupted)) {
         stagedFiles++;

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
   }

   fStagedPercentage = 100.0 * stagedFiles / fList->GetEntries();
}

//______________________________________________________________________________
void TFileCollection::Print(Option_t *option) const
{
   // Prints the contents of the TFileCollection.
   // If option contains "M": prints meta data entries,
   // if option contains "F": prints all the files in the collection.

   Printf("TFileCollection %s - %s contains: %d files with a size of %lld bytes, %.1f %% staged",
          GetName(), GetTitle(), fList->GetEntries(), fTotalSize, fStagedPercentage);

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

   if (TString(option).Contains("F", TString::kIgnoreCase)) {
      Printf("The collection contains the following files:");
      fList->Print();
   }
}

//______________________________________________________________________________
void TFileCollection::SetAnchor(const char *anchor) const
{
   // Calls TUrl::SetAnchor() for all URLs contained in all TFileInfos.

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
void TFileCollection::Sort()
{
   // Sort the collection.

   fList->Sort();
}

//______________________________________________________________________________
Float_t TFileCollection::GetCorruptedPercentage() const
{
   // Returns the percentage of files with the kCorrupted bit set,
   // calculated on-the-fly because it is not supposed to be used often.

   if (fList->GetEntries() == 0)
      return -1;

   Long64_t count = 0;

   TIter iter(fList);
   TFileInfo *fileInfo = 0;
   while ((fileInfo = dynamic_cast<TFileInfo*>(iter.Next()))) {
      if (fileInfo->TestBit(TFileInfo::kCorrupted))
         count++;
   }

   return 100.0 * count / fList->GetEntries();
}
