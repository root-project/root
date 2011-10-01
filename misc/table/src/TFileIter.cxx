// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   01/03/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2001 [BNL] Brookhaven National Laboratory.              *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Class to iterate (read / write ) the events written to TFile.         //
// The event is supposed to assign an unique ID in form of               //
//                                                                       //
//  TKey <event Id> ::= eventName "." run_number "." event_number        //
//                                                                       //
// and stored as the TKey name of the object written                     //
//                                                                       //
//        ///////        //////////      ////////        ///////     //////
//
// void TesTFileIter(){
// // This macros tests the various methods of TFileIter class.
//   gSystem->Load("libTable");
//
//   //First create simple ROOT file
//   TDataSet *ds = new TDataSet("event");
//   TObject *nextObject = 0;
//   TRandom run;
//   TRandom event;
//   {
//     TFileIter outSet("test.root","RECREATE");
//     UInt_t totalEvent = 10;
//     UInt_t runNumber  = 20010301;
//     Int_t i=0;
//     Int_t j=0;
//     for (;j < 10;j++) {
//       for (i = 1;i<totalEvent;i++) {
//         outSet.NextEventPut(ds,UInt_t(i),UInt_t(runNumber+j+10*run.Rndm()-5));
//       }
//     }
//   }
//   printf(" ----------------------> TFile has been created <--------------------\n");
//   TFile *f = new TFile("test.root");
//   TFileIter readObj(f);
//   // the number of the object available directly from "MyDataSet.root"
//   Int_t size = readObj.TotalKeys();
//   printf(" The total number of the objects: %d\n",size);
//
//   //-----------------------------------------------------------------------
//   // Loop over all objects, read them in to memory one by one
//
//   printf(" -- > Loop over all objects, read them in to memory one by one < -- \n");
//   for( readObj = 0; int(readObj) < size; ++readObj){
//       nextObject = *readObj;
//       printf(" %d bytes of the object \"%s\" of class \"%s\" written with TKey \"%s\"  has been read from file\n"
//                ,readObj.GetObjlen()
//                ,nextObject->GetName()
//                ,nextObject->IsA()->GetName()
//                ,(const char *)readObj
//             );
//       delete nextObject;
//  }
// //-----------------------------------------------------------------------
// //  Now loop over all objects in inverse order
//  printf(" -- > Now loop over all objects in inverse order < -- \n");
//  for( readObj = size-1; (int)readObj >= 0; --readObj)
//  {
//       nextObject = *readObj;
//       if (nextObject) {
//          printf(" Object \"%s\" of class \"%s\" written with TKey \"%s\"  has been read from file\n"
//                 ,nextObject->GetName()
//                 , nextObject->IsA()->GetName()
//                 ,(const char *)readObj
//                );
//         delete nextObject;
//      } else {
//        printf("Error reading file by index\n");
//      }
//  }
// //-----------------------------------------------------------------------
// // Loop over the objects starting from the object with the key name "event.02.01"
//   printf(" -- > Loop over the objects starting from the object with the key name \"event.02.01\" < -- \n");
//   for( readObj = "event.02.01"; (const char *)readObj != 0; ++readObj){
//       nextObject = *readObj;
//       printf(" Object \"%s\" of class \"%s\" written with Tkey \"%s\"  has been read from file\n"
//               , nextObject->GetName()
//               , nextObject->IsA()->GetName()
//               , (const char *)readObj
//             );
//       delete nextObject;
//   }
//
//   printf(" -- > Loop over the objects starting from the 86-th object" < -- \n");
//   for( readObj = (const char *)(readObj = 86); (const char *)readObj != 0; ++readObj){
//       nextObject = *readObj;
//       printf(" Object \"%s\" of class \"%s\" written with Tkey \"%s\"  has been read from file\n"
//               , nextObject->GetName()
//               , nextObject->IsA()->GetName()
//               , (const char *)readObj
//             );
//       delete nextObject;
//   }
//
// }
//-----------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////


#include <assert.h>

#include "TEnv.h"
#include "TSystem.h"
#include "TFile.h"
#include "TKey.h"

#include "TFileIter.h"
#include "TDsKey.h"

ClassImp(TFileIter)

//__________________________________________________________________________
TFileIter::TFileIter(TFile *file) : fNestedIterator(0)
         , fRootFile(file)
         , fEventName("event"), fRunNumber(UInt_t(-1)),fEventNumber(UInt_t(-1))
         , fCursorPosition(-1),  fOwnTFile(kFALSE)
{ 
   // Create iterator over all objects from the TFile provided
   Initialize(); 
}

//__________________________________________________________________________
TFileIter::TFileIter(TDirectory *directory) :  fNestedIterator(0)
         , fRootFile(directory)
         , fEventName("event"), fRunNumber(UInt_t(-1)),fEventNumber(UInt_t(-1))
         , fCursorPosition(-1),  fOwnTFile(kFALSE)
{ 
   // Create iterator over all objects from the TDirectory provided
   Initialize(); 
}
//__________________________________________________________________________
TFileIter::TFileIter(const char *name, Option_t *option, const char *ftitle
                     , Int_t compress, Int_t /*netopt*/) : fNestedIterator(0)
                                                         ,fRootFile(0)
                                                         ,fEventName("event"), fRunNumber(UInt_t(-1)) ,fEventNumber(UInt_t(-1))
                                                         ,fCursorPosition(-1), fOwnTFile(kFALSE)
{
   // Open ROOT TFile by the name provided;
   // This TFile is to be deleted by the TFileIter alone
   if (name && name[0]) {
      fOwnTFile = kTRUE;
      // Map a special file system to rfio
      //   /hpss/in2p3.fr/group/atlas/cppm/data/genz
      //   #setenv HPSSIN bnlhpss:/home/atlasgen/evgen
      // #example for castor:   /castor/cern.ch/user/p/paniccia/evgen
      fRootFile = TFile::Open(MapName(name),option,ftitle,compress);
      Initialize();
   }
}

//__________________________________________________________________________
TFileIter::TFileIter(const TFileIter &dst) : TListIter()
          , fNestedIterator(0)
          ,fRootFile(dst.fRootFile),fEventName(dst.fEventName), fRunNumber(dst.fRunNumber)
          ,fEventNumber(dst.fRunNumber),
           fCursorPosition(-1),  fOwnTFile(dst.fOwnTFile)
{
   // Copy ctor can be used with the "read only" files only.
   //the next statement is illegal, spotted by coverity "Dereferencing pointer "this->fRootFile". (Deref happens because this is a virtual function call.)
   //assert(!fRootFile->IsWritable());
   if (fRootFile && fOwnTFile) {
      // Reopen the file
      if (fRootFile->InheritsFrom(TFile::Class())) 
      {
         TFile *thisFile = (TFile *)fRootFile;
         fRootFile = TFile::Open(MapName(fRootFile->GetName())
            ,fRootFile->GetOption()
            ,fRootFile->GetTitle()
            ,thisFile->GetCompressionSettings());
      }
   }

   Initialize();
   // Adjust this iterator position
   SkipObjects(dst.fCursorPosition);
}
//__________________________________________________________________________
TFileIter::~TFileIter()
{
   // TFileIter dtor
   TFileIter *deleteit = fNestedIterator; fNestedIterator = 0;
   delete deleteit;
   if (fRootFile && fOwnTFile ) {  // delete own TFile if any
      if (fRootFile->IsWritable()) fRootFile->Write();
      fRootFile->Close();
      delete fRootFile;
      fRootFile = 0;
   }
}

//__________________________________________________________________________
void TFileIter::Initialize()
{
   //to be documented
   if (fRootFile) {
      fDirection =  kIterForward;
      if (IsOpen()) Reset();
      else  {
         if (fRootFile && fOwnTFile ) delete fRootFile;
         fRootFile = 0;
      }
   }
}
//__________________________________________________________________________
Bool_t  TFileIter::IsOpen() const
{
   // Check whether the associated ROOT TFile was open
   // and TFile object is healthy.

   Bool_t iOpen = kFALSE;
   if (fRootFile && !fRootFile->IsZombie() ) {
      iOpen = kTRUE;
      if (fRootFile->InheritsFrom(TFile::Class()) && !((TFile*)fRootFile)->IsOpen()) 
         iOpen = kFALSE;
   }
   return iOpen;
}

//__________________________________________________________________________
TKey *TFileIter::GetCurrentKey() const
{
  // return the pointer to the current TKey
   
   return ((TFileIter*)this)->SkipObjects(0);
}
//__________________________________________________________________________
Int_t TFileIter::GetDepth() const
{
   // return the current number of the nested subdirectroies;
   //      = 0 - means there is no subdirectories
   return fNestedIterator ? fNestedIterator->GetDepth()+1 : 0;
}

//__________________________________________________________________________
const char *TFileIter::GetKeyName() const
{
   // return the name of the current TKey
   const char *name = 0;
   TKey *key  = GetCurrentKey();
   if (key) name = key->GetName();
   return name;
}
//__________________________________________________________________________
TObject *TFileIter::GetObject() const
{
  // read the object from TFile defined by the current TKey
  //
  // ATTENTION:  memory leak danger !!!
  // ---------
  // This method does create a new object and it is the end-user
  // code responsibility to take care about this object
  // to avoid memory leak.
  //
   return ReadObj(GetCurrentKey());
}
//__________________________________________________________________________
Int_t TFileIter::GetObjlen() const
{
   // Returns the uncompressed length of the current object
   Int_t lenObj = 0;
   TKey *key = GetCurrentKey();
   if (key) lenObj = ((TKey *)key)->GetObjlen();
   return lenObj;
}
//__________________________________________________________________________
Int_t TFileIter::TotalKeys() const
{
  // The total number of the TKey keys in the current TDirectory only
  // Usually this means the total number of different objects
  // those can be read one by one.
  // It does NOT count the nested sub-TDirectory. 
  // It is too costly and it can be abused.
   
   Int_t size = 0;
   if(fList) size +=  fList->GetSize();
   return size;
}
//__________________________________________________________________________
TObject *TFileIter::Next(Int_t  nSkip)
{
  // return the pointer to the object defined by next TKey
  // This method is not recommended. It was done for the sake
  // of the compatibility with TListIter

   SkipObjects(nSkip);
   return GetObject();
}

//__________________________________________________________________________
void TFileIter::PurgeKeys(TList *listOfKeys) 
{
   // Remove the TKey duplication,
   // leave the keys with highest cycle number only
   // Sort if first

   assert(listOfKeys);
   listOfKeys->Sort();
   TObjLink *lnk   = listOfKeys->FirstLink();
   while(lnk) {
      TKey *key = (TKey *)lnk->GetObject();
      Short_t cycle = key->GetCycle(); 
      const char *keyName = key->GetName();
      // Check next object
      lnk = lnk->Next();
      if (lnk) {
         TKey *nextkey = 0;
         TObjLink *lnkThis = lnk;
         while (     lnk
             &&   (nextkey = (TKey *)lnk->GetObject()) 
             &&  !strcmp(nextkey->GetName(), keyName) 
            ) {
            // compare the cycles
            Short_t nextCycle = nextkey->GetCycle() ;
            //printf(" TFileIter::PurgeKeys found new cycle %s :%d : %d\n",
            //      keyName,cycle ,nextCycle);
            assert(cycle != nextCycle);
            TObjLink *lnkNext = lnk->Next();
            if (cycle > nextCycle ) { 
               delete listOfKeys->Remove(lnk);
            } else {
               delete listOfKeys->Remove(lnkThis);
               cycle   = nextCycle;
               lnkThis = lnk;
            }
            lnk = lnkNext;
         }
      }
   }
}

//__________________________________________________________________________
void TFileIter::Reset()
{
   // Reset the status of the iterator
   if (fNestedIterator) { 
      TFileIter *it = fNestedIterator; 
      fNestedIterator=0;
      delete it;
   }
   TListIter::Reset();
   if (!fRootFile->IsWritable()) {
      TList *listOfKeys = fRootFile->GetListOfKeys();
      if (listOfKeys) {
         if (!listOfKeys->IsSorted()) PurgeKeys(listOfKeys);
         fList = listOfKeys;
         if (fDirection == kIterForward) {
            fCursorPosition = 0;
            fCurCursor = fList->FirstLink();
            if (fCurCursor) fCursor = fCurCursor->Next();
         } else {
            fCursorPosition = fList->GetSize()-1;
            fCurCursor = fList->LastLink();
            if (fCurCursor) fCursor = fCurCursor->Prev();
         }
      }
   }
}
//__________________________________________________________________________
void TFileIter::SetCursorPosition(const char *keyNameToFind)
{
   // Find the key by the name provided
   Reset();
   while( (*this != keyNameToFind) && SkipObjects() ) {;}
}
//__________________________________________________________________________
TKey *TFileIter::SkipObjects(Int_t  nSkip)
{
 //
 // Returns the TKey pointer to the nSkip TKey object from the current one
 // nSkip = 0; the state of the iterator is not changed
 //
 // nSkip > 0; iterator skips nSkip objects in the container.
 //            the direction of the iteration is
 //            sign(nSkip)*kIterForward
 //
 // Returns: TKey that can be used to fetch the object from the TDirectory
 //
   TKey *nextObject  = fNestedIterator ? fNestedIterator->SkipObjects(nSkip): 0;
   if (!nextObject) {
      if (fNestedIterator) {
         TFileIter *it = fNestedIterator;
         fNestedIterator = 0;
         delete it;
      }
      Int_t collectionSize = 0;
      if (fList && (collectionSize = fList->GetSize())  ) {
         if (fDirection !=kIterForward) nSkip = -nSkip;
         Int_t newPos = fCursorPosition + nSkip;
         if (0 <= newPos && newPos < collectionSize) {
            do {
               if (fCursorPosition < newPos) {
                  fCursorPosition++;
                  fCurCursor = fCursor;
                  fCursor    = fCursor->Next();
               } else if (fCursorPosition > newPos) {
                  fCursorPosition--;
                  fCurCursor = fCursor;
                  fCursor    = fCursor->Prev();
               }
            } while (fCursorPosition != newPos);
            if (fCurCursor) nextObject = dynamic_cast<TKey *>(fCurCursor->GetObject());
         } else  {
            fCurCursor = fCursor = 0;
            if (newPos < 0) {
               fCursorPosition = -1;
               if (fList) fCursor = fList->FirstLink();
            } else  {
               fCursorPosition = collectionSize;
               if (fList) fCursor = fList->LastLink();
            }
         }
      }
   }
   return nextObject;
}
//__________________________________________________________________________
TKey *TFileIter::NextEventKey(UInt_t eventNumber, UInt_t runNumber, const char *name)
{

   // Return the key that name matches the "event" . "run number" . "event number" schema

   Bool_t reset = kFALSE;
   if (name && name[0] && name[0] != '*') { if (fEventName > name) reset = kTRUE; fEventName   = name; }
   if (runNumber   !=UInt_t(-1) ) { if (fRunNumber > runNumber)     reset = kTRUE; fRunNumber   = runNumber;}
   if (eventNumber !=UInt_t(-1) ) { if (fEventNumber > eventNumber) reset = kTRUE; fEventNumber = eventNumber;}

   if (reset) Reset();
   //   TIter &nextKey = *fKeysIterator;
   TKey *key = 0;
   TDsKey thisKey;
   while ( (key = SkipObjects()) ) {
      if (fDirection==kIterForward) fCursorPosition++;
      else                          fCursorPosition--;
      if ( name && name[0] != '*') {
         thisKey.SetKey(key->GetName());
         if (thisKey.GetName() < name)  continue;
         if (thisKey.GetName() > name) { key = 0; break; }
      }
      // Check "run number"
      if (runNumber != UInt_t(-1)) {
         UInt_t thisRunNumber = thisKey.RunNumber();
         if (thisRunNumber < runNumber) continue;
         if (thisRunNumber < runNumber) { key = 0; break; }
      }
      // Check "event number"
      if (eventNumber != UInt_t(-1)) {
         UInt_t thisEventNumber = thisKey.EventNumber();
         if (thisEventNumber < eventNumber) continue;
         if (thisEventNumber > eventNumber) {key = 0; break; }
      }
      break;
   }
   return key;
}
//__________________________________________________________________________
TObject *TFileIter::NextEventGet(UInt_t eventNumber, UInt_t runNumber, const char *name)
{
  // reads, creates and returns the object by TKey name that matches
  // the "name" ."runNumber" ." eventNumber" schema
  // Attention: This method does create a new TObject and it is the user
  // code responsibility to take care (delete) this object to avoid
  // memory leak.

   return ReadObj(NextEventKey(eventNumber,runNumber,name));
}

//__________________________________________________________________________
TObject *TFileIter::ReadObj(const TKey *key)  const
{
   //Read the next TObject from for the TDirectory by TKey provided
   TObject *obj = 0;
   if (fNestedIterator) obj = fNestedIterator->ReadObj(key);
   else if (key)  {
      obj = ((TKey *)key)->ReadObj();
      if (obj && obj->InheritsFrom(TDirectory::Class()) ) 
      {
         // create the next iteration level.
         assert(!fNestedIterator);
         ((TFileIter*)this)->fNestedIterator = new TFileIter((TDirectory *)obj);
         // FIXME:  needs to set  fDirection if needed 02/11/2007 vf
      }
   }
   return obj;
}

//__________________________________________________________________________
Int_t  TFileIter::NextEventPut(TObject *obj, UInt_t eventNum,  UInt_t runNumber
                              , const char *name)
{
   // Create a special TKey name with obj provided and write it out.

   Int_t wBytes = 0;
   if (obj && IsOpen() && fRootFile->IsWritable()) {
      TDsKey thisKey(runNumber,eventNum);
      if (name && name[0])
         thisKey.SetName(name);
      else
         thisKey.SetName(obj->GetName());

      TDirectory::TContext ctxt(fRootFile); // Store the current directory, cd to fRootFile and at the end of the block restore the current directory.
      
      wBytes = obj->Write(thisKey.GetKey());
      if (fRootFile->InheritsFrom(TFile::Class())) ((TFile*)fRootFile)->Flush();
   }
   return wBytes;
}
//__________________________________________________________________________
TString TFileIter::MapName(const char *name, const char *localSystemKey,const char *mountedFileSystemKey)
{
   // --------------------------------------------------------------------------------------
   // MapName(const char *name, const char *localSystemKey,const char *mountedFileSystemKey)
   // --------------------------------------------------------------------------------------
   // Substitute the logical name with the real one if any
   // 1. add a line into system.rootrc or ~/.rootrc or ./.rootrc
   //
   //  TFileIter.ForeignFileMap  mapFile // the name of the file
                                         // to map the local name
                                         // to the global file service
   //
   //  If this line is omitted then TFileIter class seeks for
   //  the default mapping file in the current directory "io.config"

   // 2. If the "io.config" file found then it defines the mapping as follows:
   //
   //  TFileIter.LocalFileSystem   /castor
   //  TFileIter.MountedFileSystem rfio:/castor

   // If "io.config" doesn't exist then no mapping is to be performed
   // and all file names are treated "as is"

   if ( !localSystemKey)       localSystemKey       = GetLocalFileNameKey();
   if ( !mountedFileSystemKey) mountedFileSystemKey = GetForeignFileSystemKey();
   TString newName = name;
   TString fileMap = gEnv->GetValue(GetResourceName(),GetDefaultMapFileName());
   const char *localName    = 0;
   const char *foreignName  = 0;
   if ( gSystem->AccessPathName(fileMap) == 0 ){
      TEnv myMapResource(fileMap);
      localName    = myMapResource.Defined(localSystemKey) ?
                                    myMapResource.GetValue(localSystemKey,"") : 0;
      foreignName  = myMapResource.Defined(mountedFileSystemKey) ?
                                    myMapResource.GetValue(mountedFileSystemKey,""):0;
   } else {
      localName    = "/castor";      // This is the default CERN name
      foreignName  = "rfio:/castor"; // and it needs "RFIO"
   }
   if (localName && localName[0]
                 && foreignName
                 && foreignName[0]
                 && newName.BeginsWith(localName) )
      newName.Replace(0,strlen(localName),foreignName);
   return newName;
}
