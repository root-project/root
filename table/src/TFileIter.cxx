// $Id: TFileIter.cxx,v 1.5 2001/08/03 11:24:10 brun Exp $
// Author: Valery Fine(fine@bnl.gov)   01/03/2001
// Copyright(c) 2001 [BNL] Brookhaven National Laboratory, Valeri Fine (fine@bnl.gov). All right reserved",
//
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
// void TestFileIter(){
// // This macros tests the various methods of TFileIter class.
//   gSystem->Load("libStar");
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
//   for( readObj = 0; int(readObj) < size; readObj.SkipObjects()){ 
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
//  for( readObj = size-1; (int)readObj >= 0; readObj.SkipObjects(-1))
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
//   for( readObj = "event.02.01"; (const char *)readObj != 0; readObj.SkipObjects()){ 
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
//   for( readObj = (const char *)(readObj = 86); (const char *)readObj != 0; readObj.SkipObjects()){ 
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


#include "TFile.h"
#include "TKey.h"

#include "TFileIter.h"
#include "TDsKey.h"

ClassImp(TFileIter)

//__________________________________________________________________________
TFileIter::TFileIter(TFile *file) : fRootFile(file),
           fEventName("event"), fRunNumber(UInt_t(-1)),fEventNumber(UInt_t(-1)),
           fCursorPosition(-1),  fOwnTFile(kFALSE)
{ Initialize(); }
//__________________________________________________________________________
TFileIter::TFileIter(const char *name, Option_t *option, const char *ftitle
                   , Int_t compress, Int_t /*netopt*/) : fRootFile (0)
{ 
  // Open ROOT TFile by the name provided;
  // This TFile is to be deleted by the TFileIter alone
  if (name && name[0]) {
    fOwnTFile = kTRUE;
    fRootFile = TFile::Open(name,option,ftitle,compress);
    Initialize();
  }
}

//__________________________________________________________________________
TFileIter::~TFileIter() 
{ 
  if (fRootFile && fOwnTFile )
  {  // delete own TFile if any
    if (fRootFile->IsWritable()) fRootFile->Write();
    fRootFile->Close();
    delete fRootFile;
    fRootFile = 0;
  }
}
#if 0
//__________________________________________________________________________
Int_t TFileIter::Copy(TFile *destFile)
{ 
   Int_t nBytes = 0;
   class TCopyKey : public TKey {
     public:
       // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
       TCopyKey(const TKey &src) : TName(src) {
         Mirror(src);  
       }
       // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
       void Mirror(const TKey &src) {
         fVersion = src.fVersion;     //Key version identifier
         fNbytes  = src.fNbytes;      //Number of bytes for the object on file
         fObjlen  = src.fObjlen;      //Length of uncompressed object in bytes
         fDatime  = src.fDatime;      //Date/Time of insertion in file
         fKeylen  = src.fKeylen;      //Number of bytes for the key itself
         fCycle   = src.fCycle;       //Cycle number
         fSeekKey = src.fSeekKey;     //Location of object on file
         fSeekPdir= src.fSeekPdir;    //Location of parent directory on file
         fClassName = src.fClassName; //Object Class name
         fLeft    = src.fLeft;        //Number of bytes left in current segment

         fBuffer = 0;  //Object buffer
         fBufferRef = 0;;     //Pointer to the TBuffer object
       }
       // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
       virtual ~TCopyKey();
       // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
       virtual TObject   ReadObj() {
         fBufferRef = new TBuffer(TBuffer::kRead, fObjlen+fKeylen);
         if (!fBufferRef) {
            Error("ReadObj", "Cannot allocate buffer: fObjlen = %d", fObjlen);
           return 0;
         }
         if (fObjlen > fNbytes-fKeylen) {
           fBuffer = new char[fNbytes];
           ReadFile();                    //Read object structure from file
           memcpy(fBufferRef->Buffer(),fBuffer,fKeylen);
         } else {
            fBuffer = fBufferRef->Buffer();
            ReadFile();                    //Read object structure from file
         }
      }
   };

==
   TKey *cKey = GetCurrentKey();
   if (cKey) {
     TFile *save = gFile;
     TDirectiry *savedir = gDirectrory;
     destFile->cd();
     TCopyKey *key = new TCopyKey(*cKey);

     if (!key->GetSeekKey()) {
       gDirectory->GetListOfKeys()->Remove(key);
       delete key;
     } else {
       gFile->SumBuffer(key->GetObjlen());
       nBytes = key->WriteFile(0);
     }
   }
==
   return nBytes;
}
#endif
//__________________________________________________________________________
void TFileIter::Initialize() 
{ 
  fDirection =  kIterForward;
  if (fRootFile &&  fRootFile->IsOpen() ) Reset();
  else  {
    if (fRootFile) delete fRootFile;
    fRootFile = 0;
  }      
}
//__________________________________________________________________________
TKey *TFileIter::GetCurrentKey() const 
{ 
  // return the pointer to the current TKey
   TKey *key = 0; 
   if (fCurCursor) key = (TKey *)fCurCursor->GetObject(); 
   return key;
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
  // code responsibility to take care about the object returned
  // to avoid memeory leak.
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
  // The total number of the TKey keys in this current TFile
  // Usually this means the total number of different objects
  // thos can be read separately with one "read" operation

  Int_t size = 0; 
  if(fList) size =  fList->GetSize(); 
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
void TFileIter::Reset()
{ 
  // Reset the status of the iterator

  TListIter::Reset();
  if (!fRootFile->IsWritable()) 
  {
    TList *listOfKeys = fRootFile->GetListOfKeys();
    if (listOfKeys) {
      if (!listOfKeys->IsSorted()) listOfKeys->Sort();       
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
  while( (*this != keyNameToFind) && SkipObjects() );
}
//__________________________________________________________________________
TObject *TFileIter::SkipObjects(Int_t  nSkip)
{
 //
 // Returns the pointer to the nSkip object from the current one
 // nSkip = 0; the state of the iterator is not changed
 //
 // nSkip > 0; iterator skips nSkip objects in the container.
 //            the direction of the iteration is 
 //            sign(nSkip)*kIterForward
 //
 TObject *nextObject  = 0;
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
     } while (fCursorPosition != newPos) ;
     if (fCurCursor) nextObject = fCurCursor->GetObject();;
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
 while ( (key = (TKey *)SkipObjects()) ) {
   if (fDirection==kIterForward) fCursorPosition++;
   else                          fCursorPosition--;
   if ( name != "*") {
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
  TObject *obj = 0;
  if (key)  {
     if (fRootFile != gFile) {
       TFileIter &th = *((TFileIter *)this);
       th.SaveFileScope();
       (th.fRootFile)->cd();
     }
     obj = ((TKey *)key)->ReadObj();
     if (fRootFile != gFile) {
        TFileIter &th = *((TFileIter *)this);
        th.RestoreFileScope();
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
  if (obj && fRootFile &&  fRootFile->IsOpen() && fRootFile->IsWritable()) 
  {
    TDsKey thisKey(runNumber,eventNum);
    if (name && name[0]) 
       thisKey.SetName(name);
    else 
       thisKey.SetName(obj->GetName());

    if (fRootFile != gFile) {
      SaveFileScope();
      fRootFile->cd();
    }
      wBytes = obj->Write(thisKey.GetKey());
      fRootFile->Flush();
    if (fRootFile != gFile) {
       RestoreFileScope();
    }
  }
  return wBytes;
}
