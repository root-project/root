// @(#)root/base:$Name:  $:$Id: TKey.cxx,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Rene Brun   28/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  The TKey class includes functions to book space on a file,          //
//   to create I/O buffers, to fill these buffers                       //
//   to compress/uncompress data buffers.                               //
//                                                                      //
//  Before saving (making persistent) an object on a file, a key must   //
//  be created. The key structure contains all the information to       //
//  uniquely identify a persistent object on a file.                    //
//     fNbytes    = number of bytes for the compressed object+key       //
//     fObjlen    = Length of uncompressed object                       //
//     fDatime    = Date/Time when the object was written               //
//     fKeylen    = number of bytes for the key structure               //
//     fCycle     = cycle number of the object                          //
//     fSeekKey   = Address of the object on file (points to fNbytes)   //
//                  This is a redundant information used to cross-check //
//                  the data base integrity.                            //
//     fSeekPdir  = Pointer to the directory supporting this object     //
//     fClassName = Object class name                                   //
//     fName      = name of the object                                  //
//     fTitle     = title of the object                                 //
//                                                                      //
//  The TKey class is used by ROOT to:                                  //
//    - to write an object in the Current Directory                     //
//    - to write a new ntuple buffer                                    //
//                                                                      //
//  The structure of a file is shown in TFile::TFile                    //
//  The structure of a directory is shown in TDirectory::TDirectory     //
//  The TKey class is used by the TBasket class                         //
//     See also TTree                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream.h>

#include "TROOT.h"
#include "TClass.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TKey.h"
#include "TFree.h"
#include "TBrowser.h"
#include "Bytes.h"

extern "C" void R__zip (Int_t cxlevel, Int_t *nin, char *bufin, Int_t *lout, char *bufout, Int_t *nout);
extern "C" void R__unzip(Int_t *nin, UChar_t *bufin, Int_t *lout, char *bufout, Int_t *nout);
const Int_t kMAXFILEBUFFER = 262144;
const Int_t kMAXBUF = 0xffffff;

ClassImp(TKey)

//______________________________________________________________________________
TKey::TKey() : TNamed(), fDatime((UInt_t)0)
{
//*-*-*-*-*-*-*-*-*-*-*TKey default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   fSeekKey    = 0;
   fNbytes     = 0;
   fBuffer     = 0;
   fKeylen     = Sizeof();
   fObjlen     = 0;
   fBufferRef  = 0;
   fCycle      = 0;
   fSeekPdir   = 0;
}

//______________________________________________________________________________
TKey::TKey(Seek_t pointer, Int_t nbytes) : TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a TKey object to read keys*-*-*-*-*-*-*-*
//*-*                      =================================
//  Constructor called by TDirectory::ReadKeys and by TFile::TFile
//  A TKey object is created to read the keys structure itself
//
   fSeekKey    = pointer;
   fNbytes     = nbytes;
   fBuffer     = new char[nbytes];
   fKeylen     = 0;
   fObjlen     = 0;
   fBufferRef  = 0;
   fCycle      = 0;
   fSeekPdir   = 0;
}

//______________________________________________________________________________
TKey::TKey(const char *name, const char *title, TClass *cl, Int_t nbytes)
      : TNamed(name,title)
{
   fClassName  = cl->GetName();
   fNbytes     = 0;
   fBuffer     = 0;
   fKeylen     = Sizeof();
   fObjlen     = nbytes;
   fBufferRef  = 0;
   fCycle      = 0;
   Create(nbytes);
}

//______________________________________________________________________________
TKey::TKey(const TString &name, const TString &title, TClass *cl, Int_t nbytes)
      : TNamed(name,title)
{
   fClassName  = cl->GetName();
   fNbytes     = 0;
   fBuffer     = 0;
   fKeylen     = Sizeof();
   fObjlen     = nbytes;
   fBufferRef  = 0;
   fCycle      = 0;
   Create(nbytes);
}

//______________________________________________________________________________
TKey::TKey(TObject *obj, const char *name, const Int_t bufsize)
     : TNamed(name, obj->GetTitle())
{
//*-*-*-*-*-*-*-*-*-*Create a TKey object and fill output buffer*-*-*-*-*-*-*
//*-*                ===========================================

   Int_t lbuf, nout, noutot, bufmax, nzip;
   fClassName = obj->ClassName();
   fNbytes    = 0;
   fBuffer    = 0;
   fBufferRef = new TBuffer(TBuffer::kWrite, bufsize);
   fCycle = gDirectory->AppendKey(this);
   fObjlen    = 0 ; // RDK: Must initialize before calling Streamer()
   fKeylen    = 0 ; // RDK: Must initialize before calling Streamer()
   fSeekKey   = 0 ; // RDK: Must initialize before calling Streamer()
   fSeekPdir  = 0 ; // RDK: Must initialize before calling Streamer()
   Streamer(*fBufferRef);         //write key itself
   fKeylen    = fBufferRef->Length();
   fBufferRef->MapObject(obj);    //register obj in map in case of self reference
   obj->Streamer(*fBufferRef);    //write object
   lbuf       = fBufferRef->Length();
   fObjlen    = lbuf - fKeylen;

   Int_t cxlevel = gFile->GetCompressionLevel();
   if (cxlevel && fObjlen > 256) {
      if (cxlevel == 2) cxlevel--;
      Int_t nbuffers = fObjlen/kMAXBUF;
      Int_t buflen = TMath::Max(512,fKeylen + fObjlen + 9*nbuffers + 8); //add 8 bytes in case object is placed in a deleted gap
      fBuffer = new char[buflen];
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      char *bufcur = &fBuffer[fKeylen];
      noutot = 0;
      nzip   = 0;
      for (Int_t i=0;i<=nbuffers;i++) {
         if (i == nbuffers) bufmax = fObjlen -nzip;
         else               bufmax = kMAXBUF;
         R__zip(cxlevel, &bufmax, objbuf, &bufmax, bufcur, &nout);
         if (nout == 0) { //this happens when the buffer cannot be compressed
            fBuffer = fBufferRef->Buffer();
            Create(fObjlen);
            fBufferRef->SetBufferOffset(0);
            Streamer(*fBufferRef);         //write key itself again
            return;
         }
         bufcur += nout;
         noutot += nout;
         objbuf += kMAXBUF;
         nzip   += kMAXBUF;
      }
      Create(noutot);
      fBufferRef->SetBufferOffset(0);
      Streamer(*fBufferRef);         //write key itself again
      memcpy(fBuffer,fBufferRef->Buffer(),fKeylen);
      delete fBufferRef; fBufferRef = 0;
   } else {
      fBuffer = fBufferRef->Buffer();
      Create(fObjlen);
      fBufferRef->SetBufferOffset(0);
      Streamer(*fBufferRef);         //write key itself again
   }
}

//______________________________________________________________________________
void TKey::Browse(TBrowser *b)
{
   // Read object from disk and call its Browse() method.
   // If object with same name already exist in memory delete it (like
   // TDirectory::Get() is doing), except when the key references a
   // folder in which case we don't want to re-read the folder object
   // since it might contain new objects not yet saved.

   // check that key points to the current dir
   if (fSeekPdir != gDirectory->GetSeekDir()) {
      Error("Browse"," Key: %s is not in the current directory: %s",GetName(),gDirectory->GetName());
      return;
   }
   TObject *obj = gDirectory->GetList()->FindObject(GetName());
   if (obj && !obj->IsFolder()) {
      if (obj->InheritsFrom(TCollection::Class()))
         obj->Delete();   // delete also collection elements
      delete obj;
      obj = 0;
   }

   if (!obj)
      obj = ReadObj();

   if (b && obj) {
      obj->Browse(b);
      b->SetRefreshFlag(kTRUE);
   }
}

//______________________________________________________________________________
void TKey::Create(Int_t nbytes)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a TKey object *-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ====================

//*-*-------------------find free segment
//*-*                    =================
   Int_t nsize      = nbytes + fKeylen;
   TFree *f1        = (TFree*) gFile->GetListOfFree()->First();
   TFree *bestfree  = f1->GetBestFree(nsize);
   if (bestfree == 0) {
      Error("Create","Cannot allocate %d bytes for ID = %s Title = %s",
            nsize,GetName(),GetTitle());
      return;
   }
   fDatime.Set();
   fSeekKey  = bestfree->GetFirst();
//*-*----------------- Case Add at the end of the file
   if (fSeekKey == gFile->GetEND()) {
      gFile->SetEND(fSeekKey+nsize);
      bestfree->SetFirst(fSeekKey+nsize);
      fLeft   = -1;
      if (!fBuffer) fBuffer = new char[nsize];
   } else {
      fLeft = Int_t(bestfree->GetLast() - fSeekKey - nsize + 1);
   }
//*-*----------------- Case where new object fills exactly a deleted gap
   fNbytes = nsize;
   if (fLeft == 0) {
      if (!fBuffer) {
         fBuffer = new char[nsize];
      }
      gFile->GetListOfFree()->Remove(bestfree);
      delete bestfree;
   }
//*-*----------------- Case where new object is placed in a deleted gap larger than itself
   if (fLeft > 0) {    // found a bigger segment
      if (!fBuffer) {
         fBuffer = new char[nsize+sizeof(Int_t)];
      }
      char *buffer  = fBuffer+nsize;
      Int_t nbytesleft = -fLeft;  // set header of remaining record
      tobuf(buffer, nbytesleft);
      bestfree->SetFirst(fSeekKey+nsize);
   }

   fSeekPdir = gDirectory->GetSeekDir();
}

//______________________________________________________________________________
TKey::~TKey()
{
//*-*-*-*-*-*-*-*-*-*-*TKey default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================

//   delete [] fBuffer; fBuffer = 0;
//   delete fBufferRef; fBufferRef = 0;

   DeleteBuffer();
}

//______________________________________________________________________________
void TKey::Delete(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Delete an object from the file*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============================
// Note: the key is not deleted. You still have to call "delete key".
// This is different from the behaviour of TObject::Delete()!


   if (option && option[0] == 'v') printf("Deleting key: %s at address %d, nbytes = %d\n",GetName(),fSeekKey,fNbytes);
   Seek_t first = fSeekKey;
   Seek_t last  = fSeekKey + fNbytes -1;
   gFile->MakeFree(first, last);  // release space used by this key
   gDirectory->GetListOfKeys()->Remove(this);
}

//______________________________________________________________________________
void TKey::DeleteBuffer()
{
//*-*-*-*-*-*-*-*-*-*-*-*Delete key buffer(s)*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ====================
  if (fBufferRef) {
     delete fBufferRef;
     fBufferRef = 0;
  } else {
     if (fBuffer) delete [] fBuffer;
  }
  fBuffer = 0;
}

//______________________________________________________________________________
Short_t TKey::GetCycle()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Return cycle number associated to this key*-*-*-*
//*-*                      ==========================================
  return ((fCycle >0) ? fCycle : -fCycle);
}

//______________________________________________________________________________
Short_t TKey::GetKeep()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Returns the "KEEP" status*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =========================
  return ((fCycle >0) ? 0 : 1);
}

//______________________________________________________________________________
void TKey::FillBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*-*Encode key header into output buffer-*-*-*-*-*-*-*
//*-*                    ====================================
  tobuf(buffer, fNbytes);
  Version_t version = TKey::Class_Version();
  tobuf(buffer, version);

  tobuf(buffer, fObjlen);
  fDatime.FillBuffer(buffer);
  tobuf(buffer, fKeylen);
  tobuf(buffer, fCycle);
  tobuf(buffer, fSeekKey);
  tobuf(buffer, fSeekPdir);
  fClassName.FillBuffer(buffer);
  fName.FillBuffer(buffer);
  fTitle.FillBuffer(buffer);
}

//______________________________________________________________________________
Bool_t TKey::IsFolder()
{
    Bool_t ret = kFALSE;

    TClass *classPtr = gROOT->GetClass( (const char * ) fClassName );
    if( classPtr ) {
       TObject *obj = ( TObject * ) classPtr->New();
       if( obj ) {
           ret = obj->IsFolder();
           delete obj;
       }
    }

    return( ret );
}

//______________________________________________________________________________
void TKey::Keep()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Set the "KEEP" status*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================
// When the KEEP flag is set to 1 the object cannot be purged
//
  if (fCycle >0)  fCycle = -fCycle;
}

//______________________________________________________________________________
void TKey::ls(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*List Key contents-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================
   IndentLevel();
   cout <<"KEY: "<<fClassName<<"\t"<<GetName()<<";"<<GetCycle()<<"\t"<<GetTitle()<<endl;
}

//______________________________________________________________________________
void TKey::Print(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Print key contents*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==================

   printf("TKey Name = %s, Title = %s, Cycle = %d\n",GetName(),GetTitle(),GetCycle());
}

//______________________________________________________________________________
TObject *TKey::ReadObj()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*To read an object from the file*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//  The object associated to this key is read from the file into memory
//  Once the key structure is read (via Streamer) the class identifier
//  of the object is known.
//  Using the class identifier we find the TClass object for this class.
//  A TClass object contains a full description (i.e. dictionary) of the
//  associated class. In particular the TClass object can create a new
//  object of the class type it describes. This new object now calls its
//  Streamer function to rebuilt itself.
//

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

   // get version of key
   fBufferRef->SetBufferOffset(sizeof(fNbytes));
   Version_t kvers = fBufferRef->ReadVersion();

   fBufferRef->SetBufferOffset(fKeylen);
   TObject *obj = 0;
   TDirectory *cursav = gDirectory;
   TClass *cl = gROOT->GetClass(fClassName.Data());
   if (!cl) {
       Error("ReadObj", "Unknown class %s", fClassName.Data());
       return 0;
   }
   // Create an instance of this class
   obj = (TObject*)cl->New();
   if (!obj) {
      Error("ReadObj", "Cannot create new object of class %s", fClassName.Data());
      return 0;
   }
   if (kvers > 1)
      fBufferRef->MapObject(obj);  //register obj in map to handle self reference

   if (fObjlen > fNbytes-fKeylen) {
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      UChar_t *bufcur = (UChar_t *)&fBuffer[fKeylen];
      Int_t nin, nout, nbuf;
      Int_t noutot = 0;
      while (1) {
         nin  = 9 + ((Int_t)bufcur[3] | ((Int_t)bufcur[4] << 8) | ((Int_t)bufcur[5] << 16));
         nbuf = (Int_t)bufcur[6] | ((Int_t)bufcur[7] << 8) | ((Int_t)bufcur[8] << 16);
         R__unzip(&nin, bufcur, &nbuf, objbuf, &nout);
         if (!nout) break;
         noutot += nout;
         if (noutot >= fObjlen) break;
         bufcur += nin;
         objbuf += nout;
      }
      if (nout) obj->Streamer(*fBufferRef);
      delete [] fBuffer;
   } else {
      obj->Streamer(*fBufferRef);
   }

   if (gROOT->GetForceStyle()) obj->UseCurrentStyle();

   if (obj->IsA() == TDirectory::Class()) {
      TDirectory *dir = (TDirectory*)obj;
      dir->SetName(GetName());
      dir->SetTitle(GetTitle());
      gDirectory->Append(dir);
   }
   delete fBufferRef;
   fBufferRef = 0;
   fBuffer    = 0;
   gDirectory = cursav;

   return obj;
}

//______________________________________________________________________________
Int_t TKey::Read(TObject *obj)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*To read an object from the file*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//  The object associated to this key is read from the file into memory
//  Before invoking this function, obj has been created via the
//  default constructor.
//

   if (!obj) return 0;

   fBufferRef = new TBuffer(TBuffer::kRead, fObjlen+fKeylen);

   if (fVersion > 1)
      fBufferRef->MapObject(obj);  //register obj in map to handle self reference

   if (fObjlen > fNbytes-fKeylen) {
      fBuffer = new char[fNbytes];
      ReadFile();                    //Read object structure from file
      memcpy(fBufferRef->Buffer(),fBuffer,fKeylen);
   } else {
      fBuffer = fBufferRef->Buffer();
      ReadFile();                    //Read object structure from file
   }
   fBufferRef->SetBufferOffset(fKeylen);
   TDirectory *cursav = gDirectory;
   if (fObjlen > fNbytes-fKeylen) {
      char *objbuf    = fBufferRef->Buffer() + fKeylen;
      UChar_t *bufcur = (UChar_t *)&fBuffer[fKeylen];
      Int_t nin = fNbytes-fKeylen;
      Int_t nout;
      R__unzip(&nin, bufcur, &fObjlen, objbuf, &nout);
      if (nout != fObjlen) Error("Read", "fObjlen = %d, nout = %d", fObjlen, nout);
      obj->Streamer(*fBufferRef);
      delete [] fBuffer;
   } else {
      obj->Streamer(*fBufferRef);
   }
   delete fBufferRef;
   fBufferRef = 0;
   fBuffer    = 0;
   gDirectory = cursav;
   return fNbytes;
}

//______________________________________________________________________________
void TKey::ReadBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*-*Decode input buffer-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ===================
   frombuf(buffer, &fNbytes);
   Version_t version;
   frombuf(buffer,&version);
   fVersion = version;
   frombuf(buffer, &fObjlen);
   fDatime.ReadBuffer(buffer);
   frombuf(buffer, &fKeylen);
   frombuf(buffer, &fCycle);
   frombuf(buffer, &fSeekKey);
   frombuf(buffer, &fSeekPdir);
   fClassName.ReadBuffer(buffer);
   fName.ReadBuffer(buffer);
   fTitle.ReadBuffer(buffer);
   if (!gROOT->ReadingBasket()) {
      if (fSeekPdir != gDirectory->GetSeekDir()) gDirectory->AppendKey(this);
   }
}

//______________________________________________________________________________
void TKey::ReadFile()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Read the key structure from the file-*-*-*-*-*-*
//*-*                      ====================================
  Int_t nsize = fNbytes;
  gFile->Seek(fSeekKey);
  for (Int_t i = 0; i < nsize; i += kMAXFILEBUFFER) {
     int nb = kMAXFILEBUFFER;
     if (i+nb > nsize) nb = nsize - i;
     gFile->ReadBuffer(fBuffer+i,nb);
  }
  if (gDebug) {
     cout << "TKey Reading "<<nsize<< " bytes at address "<<fSeekKey<<endl;
  }
}


//______________________________________________________________________________
Int_t TKey::Sizeof() const
{
//*-*-*-*-*-*-*-*Return the size in bytes of the key header structure*-*-*-*
//*-*-*-*-*-*-*-*====================================================
   Int_t nbytes = sizeof fNbytes;
   nbytes      += sizeof(Version_t);
   nbytes      += sizeof fObjlen;
   nbytes      += fDatime.Sizeof();
   nbytes      += sizeof fKeylen;
   nbytes      += sizeof fCycle;
   nbytes      += sizeof fSeekKey;
   nbytes      += sizeof fSeekPdir;
   nbytes      += fClassName.Sizeof();
   nbytes      += fName.Sizeof();
   nbytes      += fTitle.Sizeof();
   return nbytes;
}


//_______________________________________________________________________
void TKey::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      b >> fNbytes;
      b.ReadVersion();  //Version_t v = b.ReadVersion();
      b >> fObjlen;
      fDatime.Streamer(b);
      b >> fKeylen;
      b >> fCycle;
      b >> fSeekKey;
      b >> fSeekPdir;
      fClassName.Streamer(b);
      fName.Streamer(b);
      fTitle.Streamer(b);
   } else {
      b << fNbytes;
      b.WriteVersion(TKey::IsA());
      b << fObjlen;
      if (fDatime.Get() == 0) fDatime.Set();
      fDatime.Streamer(b);
      b << fKeylen;
      b << fCycle;
      b << fSeekKey;
      b << fSeekPdir;
      fClassName.Streamer(b);
      fName.Streamer(b);
      fTitle.Streamer(b);
   }
}

//______________________________________________________________________________
Int_t TKey::WriteFile(Int_t cycle)
{
//*-*-*-*-*-*-*-*-*-*-*Write the encoded object supported by this key*-*-*-*
//*-*                  ==============================================

  Int_t nsize  = fNbytes;
  char *buffer = fBuffer;
  if (cycle) {
     fCycle = cycle;
     FillBuffer(buffer);
     buffer = fBuffer;
  }

  if (fLeft > 0) nsize += sizeof(Int_t);
  gFile->Seek(fSeekKey);
  for (Int_t i=0;i<nsize;i+=kMAXFILEBUFFER) {
     int nb = kMAXFILEBUFFER;
     if( i+nb >nsize) nb= int(nsize -i);
     gFile->WriteBuffer(buffer,nb);
     buffer += nb;
  }
//  gFile->Flush(); Flushing takes too much time.
//                  Let user flush the file when he wants.
  if (gDebug) {
     cout <<"   TKey Writing "<<nsize<< " bytes at address "<<fSeekKey
          <<" for ID= " <<GetName()<<" Title= "<<GetTitle()<<endl;
  }

  DeleteBuffer();
  return nsize;
}
