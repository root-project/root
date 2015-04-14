// @(#)root/io:$Id$
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
//  The TKey class includes functions to book space in a file,          //
//   to create I/O buffers, to fill these buffers,                      //
//   to compress/uncompress data buffers.                               //
//                                                                      //
//  Before saving (making persistent) an object in a file, a key must   //
//  be created. The key structure contains all the information to       //
//  uniquely identify a persistent object in a file.                    //
//     fNbytes    = Number of bytes for the compressed object+key       //
//     fObjlen    = Length of uncompressed object                       //
//     fDatime    = Date/Time when the object was written               //
//     fKeylen    = Number of bytes for the key structure               //
//     fCycle     = Cycle number of the object                          //
//     fSeekKey   = Address of the object on file (points to fNbytes)   //
//                  This is a redundant information used to cross-check //
//                  the data base integrity.                            //
//     fSeekPdir  = Pointer to the directory supporting this object     //
//     fClassName = Object class name                                   //
//     fName      = Name of the object                                  //
//     fTitle     = Title of the object                                 //
//                                                                      //
//  In the 16 highest bits of fSeekPdir is encoded a pid offset.  This  //
//  offset is to be added to the pid index stored in the TRef object    //
//  and the referenced TObject.                                         //
//                                                                      //
//  The TKey class is used by ROOT to:                                  //
//    - to write an object in the current directory                     //
//    - to write a new ntuple buffer                                    //
//                                                                      //
//  The structure of a file is shown in TFile::TFile.                   //
//  The structure of a directory is shown in TDirectoryFile ctor.       //
//  The TKey class is used by the TBasket class.                        //
//  See also TTree.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <atomic>

#include "Riostream.h"
#include "TROOT.h"
#include "TClass.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TKey.h"
#include "TBufferFile.h"
#include "TFree.h"
#include "TBrowser.h"
#include "Bytes.h"
#include "TInterpreter.h"
#include "TError.h"
#include "TVirtualStreamerInfo.h"
#include "TSchemaRuleSet.h"

#include "RZip.h"

const Int_t kTitleMax = 32000;
#if 0
const Int_t kMAXFILEBUFFER = 262144;
#endif

#if !defined(_MSC_VER) || (_MSC_VER>1300)
const ULong64_t kPidOffsetMask = 0xffffffffffffULL;
#else
const ULong64_t kPidOffsetMask = 0xffffffffffffUL;
#endif
const UChar_t kPidOffsetShift = 48;

TString &gTDirectoryString() {
   TTHREAD_TLS_DECL_ARG(TString,gTDirectoryString,"TDirectory");
   return gTDirectoryString;
}
std::atomic<UInt_t> keyAbsNumber{0};

ClassImp(TKey)

//______________________________________________________________________________
TKey::TKey() : TNamed(), fDatime((UInt_t)0)
{
   // TKey default constructor.

   Build(0, "", 0);

   fKeylen     = Sizeof();

   keyAbsNumber++; SetUniqueID(keyAbsNumber);
}

//______________________________________________________________________________
TKey::TKey(TDirectory* motherDir) : TNamed(), fDatime((UInt_t)0)
{
   // TKey default constructor.

   Build(motherDir, "", 0);

   fKeylen     = Sizeof();

   keyAbsNumber++; SetUniqueID(keyAbsNumber);
}

//______________________________________________________________________________
TKey::TKey(TDirectory* motherDir, const TKey &orig, UShort_t pidOffset) : TNamed(), fDatime((UInt_t)0)
{
   // Copy a TKey from its original directory to the new 'motherDir'

   fMotherDir  = motherDir;

   fPidOffset  = orig.fPidOffset + pidOffset;
   fNbytes     = orig.fNbytes;
   fObjlen     = orig.fObjlen;
   fClassName  = orig.fClassName;
   fName       = orig.fName;
   fTitle      = orig.fTitle;

   fCycle      = fMotherDir->AppendKey(this);
   fSeekPdir   = 0;
   fSeekKey    = 0;
   fLeft       = 0;

   fVersion    = TKey::Class_Version();
   Long64_t filepos = GetFile()->GetEND();
   if (filepos > TFile::kStartBigFile || fPidOffset) fVersion += 1000;

   fKeylen     = Sizeof();  // fVersion must be set.

   UInt_t bufferDecOffset = 0;
   UInt_t bufferIncOffset = 0;
   UInt_t alloc = fNbytes + sizeof(Int_t);  // The extra Int_t is for any free space information.
   if (fKeylen < orig.fKeylen) {
      bufferDecOffset = orig.fKeylen - fKeylen;
      fNbytes -= bufferDecOffset;
   } else if (fKeylen > orig.fKeylen) {
      bufferIncOffset = fKeylen - orig.fKeylen;
      alloc += bufferIncOffset;
      fNbytes += bufferIncOffset;
   }

   fBufferRef  = new TBufferFile(TBuffer::kWrite, alloc);
   fBuffer     = fBufferRef->Buffer();

   // Steal the data from the old key.

   TFile* f = orig.GetFile();
   if (f) {
      Int_t nsize = orig.fNbytes;
      f->Seek(orig.fSeekKey);
      if( f->ReadBuffer(fBuffer+bufferIncOffset,nsize) )
      {
         Error("ReadFile", "Failed to read data.");
         return;
      }
      if (gDebug) {
         std::cout << "TKey Reading "<<nsize<< " bytes at address "<<fSeekKey<<std::endl;
      }
   }
   fBuffer += bufferDecOffset; // Reset the buffer to be appropriate for this key.
   Int_t nout = fNbytes - fKeylen;
   Create(nout);
   fBufferRef->SetBufferOffset(bufferDecOffset);
   Streamer(*fBufferRef);         //write key itself again
}

//______________________________________________________________________________
TKey::TKey(Long64_t pointer, Int_t nbytes, TDirectory* motherDir) : TNamed()
{
   // Create a TKey object to read keys.
   // Constructor called by TDirectoryFile::ReadKeys and by TFile::TFile.
   // A TKey object is created to read the keys structure itself.

   Build(motherDir, "", pointer);

   fSeekKey    = pointer;
   fNbytes     = nbytes;
   fBuffer     = new char[nbytes];
   keyAbsNumber++; SetUniqueID(keyAbsNumber);
}

//______________________________________________________________________________
TKey::TKey(const char *name, const char *title, const TClass *cl, Int_t nbytes, TDirectory* motherDir)
      : TNamed(name,title)
{
   // Create a TKey object with the specified name, title for the given class.
   //
   //  WARNING: in name avoid special characters like '^','$','.' that are used
   //  by the regular expression parser (see TRegexp).

   Build(motherDir, cl->GetName(), -1);

   fKeylen     = Sizeof();
   fObjlen     = nbytes;
   Create(nbytes);
}

//______________________________________________________________________________
TKey::TKey(const TString &name, const TString &title, const TClass *cl, Int_t nbytes, TDirectory* motherDir)
      : TNamed(name,title)
{
   // Create a TKey object with the specified name, title for the given class.
   //
   //  WARNING: in name avoid special characters like '^','$','.' that are used
   //  by the regular expression parser (see TRegexp).

   Build(motherDir, cl->GetName(), -1);

   fKeylen     = Sizeof();
   fObjlen     = nbytes;
   Create(nbytes);
}

//______________________________________________________________________________
TKey::TKey(const TObject *obj, const char *name, Int_t bufsize, TDirectory* motherDir)
     : TNamed(name, obj->GetTitle())
{
   // Create a TKey object for a TObject* and fill output buffer
   //
   //  WARNING: in name avoid special characters like '^','$','.' that are used
   //  by the regular expression parser (see TRegexp).

   R__ASSERT(obj);

   if (!obj->IsA()->HasDefaultConstructor()) {
      Warning("TKey", "since %s has no public constructor\n"
              "\twhich can be called without argument, objects of this class\n"
              "\tcan not be read with the current library. You will need to\n"
              "\tadd a default constructor before attempting to read it.",
              obj->ClassName());
   }

   Build(motherDir, obj->ClassName(), -1);

   Int_t lbuf, nout, noutot, bufmax, nzip;
   fBufferRef = new TBufferFile(TBuffer::kWrite, bufsize);
   fBufferRef->SetParent(GetFile());
   fCycle     = fMotherDir->AppendKey(this);

   Streamer(*fBufferRef);         //write key itself
   fKeylen    = fBufferRef->Length();
   fBufferRef->MapObject(obj);    //register obj in map in case of self reference
   ((TObject*)obj)->Streamer(*fBufferRef);    //write object
   lbuf       = fBufferRef->Length();
   fObjlen    = lbuf - fKeylen;

   Int_t cxlevel = GetFile() ? GetFile()->GetCompressionLevel() : 0;
   Int_t cxAlgorithm = GetFile() ? GetFile()->GetCompressionAlgorithm() : 0;
   if (cxlevel > 0 && fObjlen > 256) {
      Int_t nbuffers = 1 + (fObjlen - 1)/kMAXZIPBUF;
      Int_t buflen = TMath::Max(512,fKeylen + fObjlen + 9*nbuffers + 28); //add 28 bytes in case object is placed in a deleted gap
      fBuffer = new char[buflen];
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      char *bufcur = &fBuffer[fKeylen];
      noutot = 0;
      nzip   = 0;
      for (Int_t i = 0; i < nbuffers; ++i) {
         if (i == nbuffers - 1) bufmax = fObjlen - nzip;
         else               bufmax = kMAXZIPBUF;
         R__zipMultipleAlgorithm(cxlevel, &bufmax, objbuf, &bufmax, bufcur, &nout, cxAlgorithm);
         if (nout == 0 || nout >= fObjlen) { //this happens when the buffer cannot be compressed
            fBuffer = fBufferRef->Buffer();
            Create(fObjlen);
            fBufferRef->SetBufferOffset(0);
            Streamer(*fBufferRef);         //write key itself again
            return;
         }
         bufcur += nout;
         noutot += nout;
         objbuf += kMAXZIPBUF;
         nzip   += kMAXZIPBUF;
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
TKey::TKey(const void *obj, const TClass *cl, const char *name, Int_t bufsize, TDirectory* motherDir)
     : TNamed(name, "object title")
{
   // Create a TKey object for any object obj of class cl d and fill
   // output buffer.
   //
   //  WARNING: in name avoid special characters like '^','$','.' that are used
   //  by the regular expression parser (see TRegexp).

   R__ASSERT(obj && cl);

   if (!cl->HasDefaultConstructor()) {
      Warning("TKey", "since %s has no public constructor\n"
              "\twhich can be called without argument, objects of this class\n"
              "\tcan not be read with the current library. You will need to\n"
              "\tadd a default constructor before attempting to read it.",
              cl->GetName());
   }

   TClass *clActual = cl->GetActualClass(obj);
   const void* actualStart;
   if (clActual) {
      const char *temp = (const char*) obj;
      // clActual->GetStreamerInfo();
      Int_t offset = (cl != clActual) ?
                     clActual->GetBaseClassOffset(cl) : 0;
      temp -= offset;
      actualStart = temp;
   } else {
      // We could not determine the real type of this object,
      // let's assume it is the one given by the caller.
      clActual = const_cast<TClass*>(cl);
      actualStart = obj;
   }

   Build(motherDir, clActual->GetName(), -1);

   fBufferRef = new TBufferFile(TBuffer::kWrite, bufsize);
   fBufferRef->SetParent(GetFile());
   fCycle     = fMotherDir->AppendKey(this);

   Streamer(*fBufferRef);         //write key itself
   fKeylen    = fBufferRef->Length();

   Int_t lbuf, nout, noutot, bufmax, nzip;

   fBufferRef->MapObject(actualStart,clActual);         //register obj in map in case of self reference
   clActual->Streamer((void*)actualStart, *fBufferRef); //write object
   lbuf       = fBufferRef->Length();
   fObjlen    = lbuf - fKeylen;

   Int_t cxlevel = GetFile() ? GetFile()->GetCompressionLevel() : 0;
   Int_t cxAlgorithm = GetFile() ? GetFile()->GetCompressionAlgorithm() : 0;
   if (cxlevel > 0 && fObjlen > 256) {
      Int_t nbuffers = 1 + (fObjlen - 1)/kMAXZIPBUF;
      Int_t buflen = TMath::Max(512,fKeylen + fObjlen + 9*nbuffers + 28); //add 28 bytes in case object is placed in a deleted gap
      fBuffer = new char[buflen];
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      char *bufcur = &fBuffer[fKeylen];
      noutot = 0;
      nzip   = 0;
      for (Int_t i = 0; i < nbuffers; ++i) {
         if (i == nbuffers - 1) bufmax = fObjlen - nzip;
         else               bufmax = kMAXZIPBUF;
         R__zipMultipleAlgorithm(cxlevel, &bufmax, objbuf, &bufmax, bufcur, &nout, cxAlgorithm);
         if (nout == 0 || nout >= fObjlen) { //this happens when the buffer cannot be compressed
            fBuffer = fBufferRef->Buffer();
            Create(fObjlen);
            fBufferRef->SetBufferOffset(0);
            Streamer(*fBufferRef);         //write key itself again
            return;
         }
         bufcur += nout;
         noutot += nout;
         objbuf += kMAXZIPBUF;
         nzip   += kMAXZIPBUF;
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
void TKey::Build(TDirectory* motherDir, const char* classname, Long64_t filepos)
{
   // method used in all TKey constructor to initialize basic data fields
   // filepos is used to calculate correct version number of key
   // if filepos==-1, end of file position is used

   fMotherDir = motherDir;

   fPidOffset  = 0;
   fNbytes     = 0;
   fBuffer     = 0;
   fKeylen     = 0;
   fObjlen     = 0;
   fBufferRef  = 0;
   fCycle      = 0;
   fSeekPdir   = 0;
   fSeekKey    = 0;
   fLeft       = 0;

   fClassName = classname;
   //the following test required for forward and backward compatibility
   if (fClassName == "TDirectoryFile") SetBit(kIsDirectoryFile);

   fVersion = TKey::Class_Version();

   if ((filepos==-1) && GetFile()) filepos = GetFile()->GetEND();
   if (filepos > TFile::kStartBigFile) fVersion += 1000;

   if (fTitle.Length() > kTitleMax) fTitle.Resize(kTitleMax);
}

//______________________________________________________________________________
void TKey::Browse(TBrowser *b)
{
   // Read object from disk and call its Browse() method.
   // If object with same name already exist in memory delete it (like
   // TDirectoryFile::Get() is doing), except when the key references a
   // folder in which case we don't want to re-read the folder object
   // since it might contain new objects not yet saved.

   if (fMotherDir==0) return;

   TClass *objcl = TClass::GetClass(GetClassName());

   void* obj = fMotherDir->GetList()->FindObject(GetName());
   if (obj && objcl->IsTObject()) {
      TObject *tobj = (TObject*)obj;
      if (!tobj->IsFolder()) {
         if (tobj->InheritsFrom(TCollection::Class()))
            tobj->Delete();   // delete also collection elements
         delete tobj;
         obj = 0;
      }
   }

   if (!obj)
      obj = ReadObj();

   if (b && obj) {
      objcl->Browse(obj,b);
      b->SetRefreshFlag(kTRUE);
   }
}

//______________________________________________________________________________
void TKey::Create(Int_t nbytes, TFile* externFile)
{
   // Create a TKey object of specified size
   // if externFile!=0, key will be allocated in specified file,
   // otherwise file of mother directory will be used

   keyAbsNumber++; SetUniqueID(keyAbsNumber);

   TFile *f = externFile;
   if (!f) f = GetFile();
   if (!f) {
      Error("Create","Cannot create key without file");
      return;
   }

   Int_t nsize      = nbytes + fKeylen;
   TList *lfree     = f->GetListOfFree();
   TFree *f1        = (TFree*)lfree->First();
//*-*-------------------find free segment
//*-*                    =================
   TFree *bestfree  = f1->GetBestFree(lfree,nsize);
   if (bestfree == 0) {
      Error("Create","Cannot allocate %d bytes for ID = %s Title = %s",
            nsize,GetName(),GetTitle());
      return;
   }
   fDatime.Set();
   fSeekKey  = bestfree->GetFirst();
//*-*----------------- Case Add at the end of the file
   if (fSeekKey >= f->GetEND()) {
      f->SetEND(fSeekKey+nsize);
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
      lfree->Remove(bestfree);
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

   fSeekPdir = externFile ? externFile->GetSeekDir() : fMotherDir->GetSeekDir();
}

//______________________________________________________________________________
TKey::~TKey()
{
   // TKey default destructor.

   //   delete [] fBuffer; fBuffer = 0;
   //   delete fBufferRef; fBufferRef = 0;

   DeleteBuffer();
}

//______________________________________________________________________________
void TKey::Delete(Option_t *option)
{
   // Delete an object from the file.
   // Note: the key is not deleted. You still have to call "delete key".
   // This is different from the behaviour of TObject::Delete()!

   if (option && option[0] == 'v') printf("Deleting key: %s at address %lld, nbytes = %d\n",GetName(),fSeekKey,fNbytes);
   Long64_t first = fSeekKey;
   Long64_t last  = fSeekKey + fNbytes -1;
   if (GetFile()) GetFile()->MakeFree(first, last);  // release space used by this key
   fMotherDir->GetListOfKeys()->Remove(this);
}

//______________________________________________________________________________
void TKey::DeleteBuffer()
{
   // Delete key buffer(s).

   if (fBufferRef) {
      delete fBufferRef;
      fBufferRef = 0;
   } else {
      // We only need to delete fBuffer if fBufferRef is zero because
      // if fBufferRef exists, we delegate ownership of fBuffer to fBufferRef.
      if (fBuffer) {
         delete [] fBuffer;
      }
   }
   fBuffer = 0;
}

//______________________________________________________________________________
Short_t TKey::GetCycle() const
{
   // Return cycle number associated to this key.

   return ((fCycle >0) ? fCycle : -fCycle);
}

//______________________________________________________________________________
TFile *TKey::GetFile() const
{
   // Returns file to which key belong

   return fMotherDir!=0 ? fMotherDir->GetFile() : gFile;
}

//______________________________________________________________________________
Short_t TKey::GetKeep() const
{
   // Returns the "KEEP" status.

   return ((fCycle >0) ? 0 : 1);
}

//______________________________________________________________________________
void TKey::FillBuffer(char *&buffer)
{
   // Encode key header into output buffer.

   tobuf(buffer, fNbytes);
   Version_t version = fVersion;
   tobuf(buffer, version);

   tobuf(buffer, fObjlen);
   fDatime.FillBuffer(buffer);
   tobuf(buffer, fKeylen);
   tobuf(buffer, fCycle);
   if (fVersion > 1000) {
      tobuf(buffer, fSeekKey);

      // We currently store in the 16 highest bit of fSeekPdir the value of
      // fPidOffset.  This offset is used when a key (or basket) is transfered from one
      // file to the other.  In this case the TRef and TObject might have stored a
      // pid index (to retrieve TProcessIDs) which refered to their order on the original
      // file, the fPidOffset is to be added to those values to correctly find the
      // TProcessID.  This fPidOffset needs to be increment if the key/basket is copied
      // and need to be zero for new key/basket.
      Long64_t pdir = (((Long64_t)fPidOffset)<<kPidOffsetShift) | (kPidOffsetMask & fSeekPdir);
      tobuf(buffer, pdir);
   } else {
      tobuf(buffer, (Int_t)fSeekKey);
      tobuf(buffer, (Int_t)fSeekPdir);
   }
   if (TestBit(kIsDirectoryFile)) {
      // We want to record "TDirectory" instead of TDirectoryFile so that the file can be read by ancient version of ROOT.
      gTDirectoryString().FillBuffer(buffer);
   } else {
      fClassName.FillBuffer(buffer);
   }
   fName.FillBuffer(buffer);
   fTitle.FillBuffer(buffer);
}

//______________________________________________________________________________
ULong_t TKey::Hash() const
{
   // This Hash function should redefine the default from TNamed.

   return TNamed::Hash();
}

//______________________________________________________________________________
void TKey::IncrementPidOffset(UShort_t offset)
{
   // Increment fPidOffset by 'offset'.
   // This offset is used when a key (or basket) is transfered from one file to
   // the other.  In this case the TRef and TObject might have stored a pid
   // index (to retrieve TProcessIDs) which refered to their order on the
   // original file, the fPidOffset is to be added to those values to correctly
   // find the TProcessID.  This fPidOffset needs to be increment if the
   // key/basket is copied and need to be zero for new key/basket.

   fPidOffset += offset;
   if (fPidOffset) {
      // We currently store fPidOffset in the 16 highest bit of fSeekPdir, which
      // need to be store as a 64 bit integer.  So we require this key to be
      // a 'large file' key.
      if (fVersion<1000) fVersion += 1000;
   }
}

//______________________________________________________________________________
Bool_t TKey::IsFolder() const
{
   // Check if object referenced by the key is a folder.

   Bool_t ret = kFALSE;

   TClass *classPtr = TClass::GetClass((const char *) fClassName);
   if (classPtr && classPtr->GetState() > TClass::kEmulated && classPtr->IsTObject()) {
      TObject *obj = (TObject *) classPtr->New(TClass::kDummyNew);
      if (obj) {
         ret = obj->IsFolder();
         delete obj;
      }
   }

   return ret;
}

//______________________________________________________________________________
void TKey::Keep()
{
   // Set the "KEEP" status.
   // When the KEEP flag is set to 1 the object cannot be purged.

   if (fCycle >0)  fCycle = -fCycle;
}

//______________________________________________________________________________
void TKey::ls(Option_t *) const
{
   // List Key contents.

   TROOT::IndentLevel();
   std::cout <<"KEY: "<<fClassName<<"\t"<<GetName()<<";"<<GetCycle()<<"\t"<<GetTitle()<<std::endl;
}

//______________________________________________________________________________
void TKey::Print(Option_t *) const
{
   // Print key contents.

   printf("TKey Name = %s, Title = %s, Cycle = %d\n",GetName(),GetTitle(),GetCycle());
}

//______________________________________________________________________________
TObject *TKey::ReadObj()
{
   // To read a TObject* from the file.
   //
   //  The object associated to this key is read from the file into memory
   //  Once the key structure is read (via Streamer) the class identifier
   //  of the object is known.
   //  Using the class identifier we find the TClass object for this class.
   //  A TClass object contains a full description (i.e. dictionary) of the
   //  associated class. In particular the TClass object can create a new
   //  object of the class type it describes. This new object now calls its
   //  Streamer function to rebuilt itself.
   //
   //  Use TKey::ReadObjectAny to read any object non-derived from TObject
   //
   //  Note:
   //  A C style cast can only be used in the case where the final class
   //  of this object derives from TObject as a first inheritance, otherwise
   //  one must use a dynamic_cast.
   //
   //  Example1: simplified case:
   //      class MyClass : public TObject, public AnotherClass
   //   then on return, one get away with using:
   //    MyClass *obj = (MyClass*)key->ReadObj();
   //
   //  Example2: Usual case (recommended unless performance is critical)
   //    MyClass *obj = dynamic_cast<MyClass*>(key->ReadObj());
   //  which support also the more complex inheritance like:
   //    class MyClass : public AnotherClass, public TObject
   //
   //  Of course, dynamic_cast<> can also be used in the example 1.

   TClass *cl = TClass::GetClass(fClassName.Data());
   if (!cl) {
      Error("ReadObj", "Unknown class %s", fClassName.Data());
      return 0;
   }
   if (!cl->IsTObject()) {
      // in principle user should call TKey::ReadObjectAny!
      return (TObject*)ReadObjectAny(0);
   }

   fBufferRef = new TBufferFile(TBuffer::kRead, fObjlen+fKeylen);
   if (!fBufferRef) {
      Error("ReadObj", "Cannot allocate buffer: fObjlen = %d", fObjlen);
      return 0;
   }
   if (GetFile()==0) return 0;
   fBufferRef->SetParent(GetFile());
   fBufferRef->SetPidOffset(fPidOffset);

   if (fObjlen > fNbytes-fKeylen) {
      fBuffer = new char[fNbytes];
      if( !ReadFile() )                    //Read object structure from file
      {
        delete fBufferRef;
        delete [] fBuffer;
        fBufferRef = 0;
        fBuffer = 0;
        return 0;
      }
      memcpy(fBufferRef->Buffer(),fBuffer,fKeylen);
   } else {
      fBuffer = fBufferRef->Buffer();
      if( !ReadFile() ) {                   //Read object structure from file
         delete fBufferRef;
         fBufferRef = 0;
         fBuffer = 0;
         return 0;
      }
   }

   // get version of key
   fBufferRef->SetBufferOffset(sizeof(fNbytes));
   Version_t kvers = fBufferRef->ReadVersion();

   fBufferRef->SetBufferOffset(fKeylen);
   TObject *tobj = 0;
   // Create an instance of this class

   char *pobj = (char*)cl->New();
   if (!pobj) {
      Error("ReadObj", "Cannot create new object of class %s", fClassName.Data());
      return 0;
   }
   Int_t baseOffset = cl->GetBaseClassOffset(TObject::Class());
   if (baseOffset==-1) {
      // cl does not inherit from TObject.
      // Since this is not possible yet, the only reason we could reach this code
      // is because something is screw up in the ROOT code.
      Fatal("ReadObj","Incorrect detection of the inheritance from TObject for class %s.\n",
            fClassName.Data());
   }
   tobj = (TObject*)(pobj+baseOffset);
   if (kvers > 1)
      fBufferRef->MapObject(pobj,cl);  //register obj in map to handle self reference

   if (fObjlen > fNbytes-fKeylen) {
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      UChar_t *bufcur = (UChar_t *)&fBuffer[fKeylen];
      Int_t nin, nout = 0, nbuf;
      Int_t noutot = 0;
      while (1) {
         Int_t hc = R__unzip_header(&nin, bufcur, &nbuf);
         if (hc!=0) break;
         R__unzip(&nin, bufcur, &nbuf, (unsigned char*) objbuf, &nout);
         if (!nout) break;
         noutot += nout;
         if (noutot >= fObjlen) break;
         bufcur += nin;
         objbuf += nout;
      }
      if (nout) {
         tobj->Streamer(*fBufferRef); //does not work with example 2 above
         delete [] fBuffer;
      } else {
         delete [] fBuffer;
         // Even-though we have a TObject, if the class is emulated the virtual
         // table may not be 'right', so let's go via the TClass.
         cl->Destructor(pobj);
         pobj = 0;
         tobj = 0;
         goto CLEAR;
      }
   } else {
      tobj->Streamer(*fBufferRef);
   }

   if (gROOT->GetForceStyle()) tobj->UseCurrentStyle();

   if (cl->InheritsFrom(TDirectoryFile::Class())) {
      TDirectory *dir = static_cast<TDirectoryFile*>(tobj);
      dir->SetName(GetName());
      dir->SetTitle(GetTitle());
      dir->SetMother(fMotherDir);
      fMotherDir->Append(dir);
   }

   // Append the object to the directory if requested:
   {
      ROOT::DirAutoAdd_t addfunc = cl->GetDirectoryAutoAdd();
      if (addfunc) {
         addfunc(pobj, fMotherDir);
      }
   }

CLEAR:
   delete fBufferRef;
   fBufferRef = 0;
   fBuffer    = 0;

   return tobj;
}

//______________________________________________________________________________
TObject *TKey::ReadObjWithBuffer(char *bufferRead)
{
   // To read a TObject* from bufferRead.
   // This function is identical to TKey::ReadObj, but it reads directly
   // from bufferRead instead of reading from a file.
   //  The object associated to this key is read from the buffer into memory
   //  Using the class identifier we find the TClass object for this class.
   //  A TClass object contains a full description (i.e. dictionary) of the
   //  associated class. In particular the TClass object can create a new
   //  object of the class type it describes. This new object now calls its
   //  Streamer function to rebuilt itself.
   //
   //  NOTE :
   //  This function is called only internally by ROOT classes.
   //  Although being public it is not supposed to be used outside ROOT.
   //  If used, you must make sure that the bufferRead is large enough to
   //  accomodate the object being read.


   TClass *cl = TClass::GetClass(fClassName.Data());
   if (!cl) {
      Error("ReadObjWithBuffer", "Unknown class %s", fClassName.Data());
      return 0;
   }
   if (!cl->IsTObject()) {
      // in principle user should call TKey::ReadObjectAny!
      return (TObject*)ReadObjectAny(0);
   }

   fBufferRef = new TBufferFile(TBuffer::kRead, fObjlen+fKeylen);
   if (!fBufferRef) {
      Error("ReadObjWithBuffer", "Cannot allocate buffer: fObjlen = %d", fObjlen);
      return 0;
   }
   if (GetFile()==0) return 0;
   fBufferRef->SetParent(GetFile());
   fBufferRef->SetPidOffset(fPidOffset);

   if (fObjlen > fNbytes-fKeylen) {
      fBuffer = bufferRead;
      memcpy(fBufferRef->Buffer(),fBuffer,fKeylen);
   } else {
      fBuffer = fBufferRef->Buffer();
      ReadFile();                    //Read object structure from file
   }

   // get version of key
   fBufferRef->SetBufferOffset(sizeof(fNbytes));
   Version_t kvers = fBufferRef->ReadVersion();

   fBufferRef->SetBufferOffset(fKeylen);
   TObject *tobj = 0;
   // Create an instance of this class

   char *pobj = (char*)cl->New();
   if (!pobj) {
      Error("ReadObjWithBuffer", "Cannot create new object of class %s", fClassName.Data());
      return 0;
   }
   Int_t baseOffset = cl->GetBaseClassOffset(TObject::Class());
   if (baseOffset==-1) {
      // cl does not inherit from TObject.
      // Since this is not possible yet, the only reason we could reach this code
      // is because something is screw up in the ROOT code.
      Fatal("ReadObjWithBuffer","Incorrect detection of the inheritance from TObject for class %s.\n",
            fClassName.Data());
   }
   tobj = (TObject*)(pobj+baseOffset);

   if (kvers > 1)
      fBufferRef->MapObject(pobj,cl);  //register obj in map to handle self reference

   if (fObjlen > fNbytes-fKeylen) {
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      UChar_t *bufcur = (UChar_t *)&fBuffer[fKeylen];
      Int_t nin, nout = 0, nbuf;
      Int_t noutot = 0;
      while (1) {
         Int_t hc = R__unzip_header(&nin, bufcur, &nbuf);
         if (hc!=0) break;
         R__unzip(&nin, bufcur, &nbuf, (unsigned char*) objbuf, &nout);
         if (!nout) break;
         noutot += nout;
         if (noutot >= fObjlen) break;
         bufcur += nin;
         objbuf += nout;
      }
      if (nout) {
         tobj->Streamer(*fBufferRef); //does not work with example 2 above
      } else {
         // Even-though we have a TObject, if the class is emulated the virtual
         // table may not be 'right', so let's go via the TClass.
         cl->Destructor(pobj);
         pobj = 0;
         tobj = 0;
         goto CLEAR;
      }
   } else {
      tobj->Streamer(*fBufferRef);
   }

   if (gROOT->GetForceStyle()) tobj->UseCurrentStyle();

   if (cl->InheritsFrom(TDirectoryFile::Class())) {
      TDirectory *dir = static_cast<TDirectoryFile*>(tobj);
      dir->SetName(GetName());
      dir->SetTitle(GetTitle());
      dir->SetMother(fMotherDir);
      fMotherDir->Append(dir);
   }

   // Append the object to the directory if requested:
   {
      ROOT::DirAutoAdd_t addfunc = cl->GetDirectoryAutoAdd();
      if (addfunc) {
         addfunc(pobj, fMotherDir);
      }
   }

CLEAR:
   delete fBufferRef;
   fBufferRef = 0;
   fBuffer    = 0;

   return tobj;
}

//______________________________________________________________________________
void *TKey::ReadObjectAny(const TClass* expectedClass)
{
   //  To read an object (non deriving from TObject) from the file.
   //
   //  If expectedClass is not null, we checked that that actual class of
   //  the object stored is suitable to be stored in a pointer pointing
   //  to an object of class 'expectedClass'.  We also adjust the value
   //  of the returned address so that it is suitable to be cast (C-Style)
   //  a  a pointer pointing to an object of class 'expectedClass'.
   //
   //  So for example if the class Bottom inherits from Top and the object
   //  stored is of type Bottom you can safely do:
   //
   //     TClass *TopClass = TClass::GetClass("Top");
   //     Top *ptr = (Top*) key->ReadObjectAny( TopClass );
   //     if (ptr==0) printError("the object stored in the key is not of the expected type\n");
   //
   //  The object associated to this key is read from the file into memory
   //  Once the key structure is read (via Streamer) the class identifier
   //  of the object is known.
   //  Using the class identifier we find the TClass object for this class.
   //  A TClass object contains a full description (i.e. dictionary) of the
   //  associated class. In particular the TClass object can create a new
   //  object of the class type it describes. This new object now calls its
   //  Streamer function to rebuilt itself.

   fBufferRef = new TBufferFile(TBuffer::kRead, fObjlen+fKeylen);
   if (!fBufferRef) {
      Error("ReadObj", "Cannot allocate buffer: fObjlen = %d", fObjlen);
      return 0;
   }
   if (GetFile()==0) return 0;
   fBufferRef->SetParent(GetFile());
   fBufferRef->SetPidOffset(fPidOffset);

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
   TClass *cl = TClass::GetClass(fClassName.Data());
   TClass *clOnfile = 0;
   if (!cl) {
      Error("ReadObjectAny", "Unknown class %s", fClassName.Data());
      return 0;
   }
   Int_t baseOffset = 0;
   if (expectedClass) {
       // baseOffset will be -1 if cl does not inherit from expectedClass
      baseOffset = cl->GetBaseClassOffset(expectedClass);
      if (baseOffset == -1) {
         // The 2 classes are unrelated, maybe there is a converter between the 2.

         if (!expectedClass->GetSchemaRules() ||
             !expectedClass->GetSchemaRules()->HasRuleWithSourceClass(cl->GetName()))
         {
            // There is no converter
            return 0;
         }
         baseOffset = 0; // For now we do not support requesting from a class that is the base of one of the class for which there is transformation to ....
         clOnfile = cl;
         cl = const_cast<TClass*>(expectedClass);
         Info("ReadObjectAny","Using Converter StreamerInfo from %s to %s",clOnfile->GetName(),expectedClass->GetName());
      }
      if (cl->GetState() > TClass::kEmulated && expectedClass->GetState() <= TClass::kEmulated) {
         //we cannot mix a compiled class with an emulated class in the inheritance
         Warning("ReadObjectAny",
                 "Trying to read an emulated class (%s) to store in a compiled pointer (%s)",
                 cl->GetName(),expectedClass->GetName());
      }
   }
   // Create an instance of this class

   void *pobj = cl->New();
   if (!pobj) {
      Error("ReadObjectAny", "Cannot create new object of class %s", fClassName.Data());
      return 0;
   }

   if (kvers > 1)
      fBufferRef->MapObject(pobj,cl);  //register obj in map to handle self reference

   if (fObjlen > fNbytes-fKeylen) {
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      UChar_t *bufcur = (UChar_t *)&fBuffer[fKeylen];
      Int_t nin, nout = 0, nbuf;
      Int_t noutot = 0;
      while (1) {
         Int_t hc = R__unzip_header(&nin, bufcur, &nbuf);
         if (hc!=0) break;
         R__unzip(&nin, bufcur, &nbuf, (unsigned char*) objbuf, &nout);
         if (!nout) break;
         noutot += nout;
         if (noutot >= fObjlen) break;
         bufcur += nin;
         objbuf += nout;
      }
      if (nout) {
         cl->Streamer((void*)pobj, *fBufferRef, clOnfile);    //read object
         delete [] fBuffer;
      } else {
         delete [] fBuffer;
         cl->Destructor(pobj);
         pobj = 0;
         goto CLEAR;
      }
   } else {
      cl->Streamer((void*)pobj, *fBufferRef, clOnfile);    //read object
   }

   if (cl->IsTObject()) {
      baseOffset = cl->GetBaseClassOffset(TObject::Class());
      if (baseOffset==-1) {
         Fatal("ReadObj","Incorrect detection of the inheritance from TObject for class %s.\n",
               fClassName.Data());
      }
      TObject *tobj = (TObject*)( ((char*)pobj) +baseOffset);

      // See similar adjustments in ReadObj
      if (gROOT->GetForceStyle()) tobj->UseCurrentStyle();

      if (cl->InheritsFrom(TDirectoryFile::Class())) {
         TDirectory *dir = static_cast<TDirectoryFile*>(tobj);
         dir->SetName(GetName());
         dir->SetTitle(GetTitle());
         dir->SetMother(fMotherDir);
         fMotherDir->Append(dir);
      }
   }

   {
      // Append the object to the directory if requested:
      ROOT::DirAutoAdd_t addfunc = cl->GetDirectoryAutoAdd();
      if (addfunc) {
         addfunc(pobj, fMotherDir);
      }
   }

   CLEAR:
   delete fBufferRef;
   fBufferRef = 0;
   fBuffer    = 0;

   return ( ((char*)pobj) + baseOffset );
}

//______________________________________________________________________________
Int_t TKey::Read(TObject *obj)
{
   // To read an object from the file.
   // The object associated to this key is read from the file into memory.
   // Before invoking this function, obj has been created via the
   // default constructor.

   if (!obj || (GetFile()==0)) return 0;

   fBufferRef = new TBufferFile(TBuffer::kRead, fObjlen+fKeylen);
   fBufferRef->SetParent(GetFile());
   fBufferRef->SetPidOffset(fPidOffset);

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
   if (fObjlen > fNbytes-fKeylen) {
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      UChar_t *bufcur = (UChar_t *)&fBuffer[fKeylen];
      Int_t nin, nout = 0, nbuf;
      Int_t noutot = 0;
      while (1) {
         Int_t hc = R__unzip_header(&nin, bufcur, &nbuf);
         if (hc!=0) break;
         R__unzip(&nin, bufcur, &nbuf, (unsigned char*) objbuf, &nout);
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

   // Append the object to the directory if requested:
   {
      ROOT::DirAutoAdd_t addfunc = obj->IsA()->GetDirectoryAutoAdd();
      if (addfunc) {
         addfunc(obj, fMotherDir);
      }
   }

   delete fBufferRef;
   fBufferRef = 0;
   fBuffer    = 0;
   return fNbytes;
}

//______________________________________________________________________________
void TKey::ReadBuffer(char *&buffer)
{
   // Decode input buffer.
   // In some situation will add key to gDirectory ???

   ReadKeyBuffer(buffer);

   if (!gROOT->ReadingObject() && gDirectory) {
      if (fSeekPdir != gDirectory->GetSeekDir()) gDirectory->AppendKey(this);
   }
}

//______________________________________________________________________________
void TKey::ReadKeyBuffer(char *&buffer)
{
   // Decode input buffer.

   frombuf(buffer, &fNbytes);
   Version_t version;
   frombuf(buffer,&version);
   fVersion = (Int_t)version;
   frombuf(buffer, &fObjlen);
   fDatime.ReadBuffer(buffer);
   frombuf(buffer, &fKeylen);
   frombuf(buffer, &fCycle);
   if (fVersion > 1000) {
      frombuf(buffer, &fSeekKey);

      // We currently store in the 16 highest bit of fSeekPdir the value of
      // fPidOffset.  This offset is used when a key (or basket) is transfered from one
      // file to the other.  In this case the TRef and TObject might have stored a
      // pid index (to retrieve TProcessIDs) which refered to their order on the original
      // file, the fPidOffset is to be added to those values to correctly find the
      // TProcessID.  This fPidOffset needs to be increment if the key/basket is copied
      // and need to be zero for new key/basket.
      Long64_t pdir;
      frombuf(buffer, &pdir);
      fPidOffset = pdir >> kPidOffsetShift;
      fSeekPdir = pdir & kPidOffsetMask;
   } else {
      Int_t seekkey,seekdir;
      frombuf(buffer, &seekkey); fSeekKey = (Long64_t)seekkey;
      frombuf(buffer, &seekdir); fSeekPdir= (Long64_t)seekdir;
   }
   fClassName.ReadBuffer(buffer);
   //the following test required for forward and backward compatibility
   if (fClassName == "TDirectory") {
      fClassName = "TDirectoryFile";
      SetBit(kIsDirectoryFile);
   }

   fName.ReadBuffer(buffer);
   fTitle.ReadBuffer(buffer);
}

//______________________________________________________________________________
Bool_t TKey::ReadFile()
{
   // Read the key structure from the file

   TFile* f = GetFile();
   if (f==0) return kFALSE;

   Int_t nsize = fNbytes;
   f->Seek(fSeekKey);
#if 0
   for (Int_t i = 0; i < nsize; i += kMAXFILEBUFFER) {
      int nb = kMAXFILEBUFFER;
      if (i+nb > nsize) nb = nsize - i;
      f->ReadBuffer(fBuffer+i,nb);
   }
#else
   if( f->ReadBuffer(fBuffer,nsize) )
   {
      Error("ReadFile", "Failed to read data.");
      return kFALSE;
   }
#endif
   if (gDebug) {
      std::cout << "TKey Reading "<<nsize<< " bytes at address "<<fSeekKey<<std::endl;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TKey::SetParent(const TObject *parent)
{
   // Set parent in key buffer.

   if (fBufferRef) fBufferRef->SetParent((TObject*)parent);
}

//______________________________________________________________________________
void TKey::Reset()
{
   // Reset the key as it had not been 'filled' yet.

   fPidOffset  = 0;
   fNbytes     = 0;
   fBuffer     = 0;
   fObjlen     = 0;
   fCycle      = 0;
   fSeekPdir   = 0;
   fSeekKey    = 0;
   fLeft       = 0;
   fDatime     = (UInt_t)0;

   // fBufferRef and fKeylen intentionally not reset/changed

   keyAbsNumber++; SetUniqueID(keyAbsNumber);
}

//______________________________________________________________________________
Int_t TKey::Sizeof() const
{
   // Return the size in bytes of the key header structure.
   // Int_t nbytes = sizeof fNbytes;      4
   //             += sizeof(Version_t);   2
   //             += sizeof fObjlen;      4
   //             += sizeof fKeylen;      2
   //             += sizeof fCycle;       2
   //             += sizeof fSeekKey;     4 or 8
   //             += sizeof fSeekPdir;    4 or 8
   //              =                     22

   Int_t nbytes = 22; if (fVersion > 1000) nbytes += 8;
   nbytes      += fDatime.Sizeof();
   if (TestBit(kIsDirectoryFile)) {
      nbytes   += 11; // strlen("TDirectory")+1
   } else {
      nbytes   += fClassName.Sizeof();
   }
   nbytes      += fName.Sizeof();
   nbytes      += fTitle.Sizeof();
   return nbytes;
}

//_______________________________________________________________________
void TKey::Streamer(TBuffer &b)
{
   // Stream a class object.

   Version_t version;
   if (b.IsReading()) {
      b >> fNbytes;
      b >> version; fVersion = (Int_t)version;
      b >> fObjlen;
      fDatime.Streamer(b);
      b >> fKeylen;
      b >> fCycle;
      if (fVersion > 1000) {
         b >> fSeekKey;

         // We currently store in the 16 highest bit of fSeekPdir the value of
         // fPidOffset.  This offset is used when a key (or basket) is transfered from one
         // file to the other.  In this case the TRef and TObject might have stored a
         // pid index (to retrieve TProcessIDs) which refered to their order on the original
         // file, the fPidOffset is to be added to those values to correctly find the
         // TProcessID.  This fPidOffset needs to be increment if the key/basket is copied
         // and need to be zero for new key/basket.
         Long64_t pdir;
         b >> pdir;
         fPidOffset = pdir >> kPidOffsetShift;
         fSeekPdir = pdir & kPidOffsetMask;
      } else {
         Int_t seekkey, seekdir;
         b >> seekkey; fSeekKey = (Long64_t)seekkey;
         b >> seekdir; fSeekPdir= (Long64_t)seekdir;
      }
      fClassName.Streamer(b);
      //the following test required for forward and backward compatibility
      if (fClassName == "TDirectory") {
         fClassName = "TDirectoryFile";
         SetBit(kIsDirectoryFile);
      }
      fName.Streamer(b);
      fTitle.Streamer(b);
      if (fKeylen < 0) {
         Error("Streamer","The value of fKeylen is incorrect (%d) ; trying to recover by setting it to zero",fKeylen);
         MakeZombie();
         fKeylen = 0;
      }
      if (fObjlen < 0) {
         Error("Streamer","The value of fObjlen is incorrect (%d) ; trying to recover by setting it to zero",fObjlen);
         MakeZombie();
         fObjlen = 0;
      }
      if (fNbytes < 0) {
         Error("Streamer","The value of fNbytes is incorrect (%d) ; trying to recover by setting it to zero",fNbytes);
         MakeZombie();
         fNbytes = 0;
      }

   } else {
      b << fNbytes;
      version = (Version_t)fVersion;
      b << version;
      b << fObjlen;
      if (fDatime.Get() == 0) fDatime.Set();
      fDatime.Streamer(b);
      b << fKeylen;
      b << fCycle;
      if (fVersion > 1000) {
         b << fSeekKey;

         // We currently store in the 16 highest bit of fSeekPdir the value of
         // fPidOffset.  This offset is used when a key (or basket) is transfered from one
         // file to the other.  In this case the TRef and TObject might have stored a
         // pid index (to retrieve TProcessIDs) which refered to their order on the original
         // file, the fPidOffset is to be added to those values to correctly find the
         // TProcessID.  This fPidOffset needs to be increment if the key/basket is copied
         // and need to be zero for new key/basket.
         Long64_t pdir = (((Long64_t)fPidOffset)<<kPidOffsetShift) | (kPidOffsetMask & fSeekPdir);
         b << pdir;
      } else {
         b << (Int_t)fSeekKey;
         b << (Int_t)fSeekPdir;
      }
      if (TestBit(kIsDirectoryFile)) {
         // We want to record "TDirectory" instead of TDirectoryFile so that the file can be read by ancient version of ROOT.
         gTDirectoryString().Streamer(b);
      } else {
         fClassName.Streamer(b);
      }
      fName.Streamer(b);
      fTitle.Streamer(b);
   }
}

//______________________________________________________________________________
Int_t TKey::WriteFile(Int_t cycle, TFile* f)
{
   // Write the encoded object supported by this key.
   // The function returns the number of bytes committed to the file.
   // If a write error occurs, the number of bytes returned is -1.

   if (!f) f = GetFile();
   if (!f) return -1;

   Int_t nsize  = fNbytes;
   char *buffer = fBuffer;
   if (cycle) {
      fCycle = cycle;
      FillBuffer(buffer);
      buffer = fBuffer;
   }

   if (fLeft > 0) nsize += sizeof(Int_t);
   f->Seek(fSeekKey);
#if 0
   for (Int_t i=0;i<nsize;i+=kMAXFILEBUFFER) {
      Int_t nb = kMAXFILEBUFFER;
      if (i+nb > nsize) nb = nsize - i;
      f->WriteBuffer(buffer,nb);
      buffer += nb;
   }
#else
   Bool_t result = f->WriteBuffer(buffer,nsize);
#endif
   //f->Flush(); Flushing takes too much time.
   //            Let user flush the file when they want.
   if (gDebug) {
      std::cout <<"   TKey Writing "<<nsize<< " bytes at address "<<fSeekKey
           <<" for ID= " <<GetName()<<" Title= "<<GetTitle()<<std::endl;
   }

   DeleteBuffer();
   return result==kTRUE ? -1 : nsize;
}

//______________________________________________________________________________
Int_t TKey::WriteFileKeepBuffer(TFile *f)
{
   // Write the encoded object supported by this key.
   // The function returns the number of bytes committed to the file.
   // If a write error occurs, the number of bytes returned is -1.

   if (!f) f = GetFile();
   if (!f) return -1;

   Int_t nsize  = fNbytes;
   char *buffer = fBuffer;

   if (fLeft > 0) nsize += sizeof(Int_t);
   f->Seek(fSeekKey);
#if 0
   for (Int_t i=0;i<nsize;i+=kMAXFILEBUFFER) {
      Int_t nb = kMAXFILEBUFFER;
      if (i+nb > nsize) nb = nsize - i;
      f->WriteBuffer(buffer,nb);
      buffer += nb;
   }
#else
   Bool_t result = f->WriteBuffer(buffer,nsize);
#endif
   //f->Flush(); Flushing takes too much time.
   //            Let user flush the file when they want.
   if (gDebug) {
      std::cout <<"   TKey Writing "<<nsize<< " bytes at address "<<fSeekKey
      <<" for ID= " <<GetName()<<" Title= "<<GetTitle()<<std::endl;
   }

   return result==kTRUE ? -1 : nsize;
}

//______________________________________________________________________________
const char *TKey::GetIconName() const
{
   // Title can keep 32x32 xpm thumbnail/icon of the parent object.

   return (!fTitle.IsNull() && fTitle.BeginsWith("/* ") ?  fTitle.Data() : 0);
}

//______________________________________________________________________________
const char *TKey::GetTitle() const
{
   // Returns title (title can contain 32x32 xpm thumbnail/icon).

   if (!fTitle.IsNull() && fTitle.BeginsWith("/* ")) { // title contains xpm thumbnail
      static TString ret;
      int start = fTitle.Index("/*") + 3;
      int stop = fTitle.Index("*/") - 1;
      ret = fTitle(start, stop - start);
      return ret.Data();
   }
   return fTitle.Data();
}
