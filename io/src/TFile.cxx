// @(#)root/base:$Name:  $:$Id: TFile.cxx,v 1.57 2002/03/26 14:09:07 brun Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#ifndef WIN32
#   include <unistd.h>
#else
#   define ssize_t int
#   include <io.h>
#   include <sys/types.h>
#endif

#include "Riostream.h"
#include "Strlen.h"
#include "TFile.h"
#include "TWebFile.h"
#include "TNetFile.h"
#include "TROOT.h"
#include "TFree.h"
#include "TKey.h"
#include "TDatime.h"
#include "TSystem.h"
#include "TError.h"
#include "Bytes.h"
#include "TInterpreter.h"
#include "TStreamerInfo.h"
#include "TArrayC.h"
#include "TClassTable.h"
#include "TProcessID.h"
#include "TPluginManager.h"


TFile *gFile;                 //Pointer to current file

const Int_t  TFile::kBegin = 64;
const Char_t TFile::kUnits = 4;

Double_t TFile::fgBytesRead  = 0;
Double_t TFile::fgBytesWrite = 0;


ClassImp(TFile)

//*-*x17 macros/layout_file
//______________________________________________________________________________
TFile::TFile() : TDirectory()
{
//*-*-*-*-*-*-*-*-*-*-*-*File default Constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================
   fD          = -1;
   fFree       = 0;
   fWritten    = 0;
   fSumBuffer  = 0;
   fSum2Buffer = 0;
   fClassIndex = 0;
   fCache      = 0;
   fProcessIDs = 0;

   if (gDebug)
      cerr << "TFile default ctor" <<endl;
}

//1_____________________________________________________________________________
TFile::TFile(const char *fname1, Option_t *option, const char *ftitle, Int_t compress)
           :TDirectory()
{
//*-*-*-*-*-*-*-*-*-*-*-*File Constructor*-*--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================
//  If Option = NEW or CREATE   create a new file and open it for writing,
//                              if the file already exists the file is
//                              not opened.
//            = RECREATE        create a new file, if the file already
//                              exists it will be overwritten.
//            = UPDATE          open an existing file for writing.
//                              if no file exists, it is created.
//            = READ            open an existing file for reading.
//  If the constructor failed in any way IsZombie() will return true.
//  Use IsOpen() to check if the file is (still) open.
//
//  The parameter name is used to identify the file in the current job
//    name may be different in a job writing the file and another job
//    reading/writing the file.
//  When the file is created, the name of the file seen from the file system
//    is fname. It is recommended to specify fname as "file.root"
//    The suffix root will be used by object browsers to automatically
//    identify ROOT files.
//  The title of the file (ftitle) will be shown by the ROOT browsers.
//
//  A ROOT file (like a Unix file system) may contain objects and
//    directories. There are no restrictions for the number of levels
//    of directories.
//
//  A ROOT file is designed such that one can write in the file in pure
//    sequential mode (case of BATCH jobs). In this case, the file may be
//    read sequentially again without using the file index written
//    at the end of the file. In case of a job crash, all the information
//    on the file is therefore protected.
//  A ROOT file can be used interactively. In this case, one has the
//    possibility to delete existing objects and add new ones.
//  When an object is deleted from the file, the freed space is added
//  into the FREE linked list (fFree). The FREE list consists of a chain
//  of consecutive free segments on the file. At the same time, the first
//  4 bytes of the freed record on the file are overwritten by GAPSIZE
//  where GAPSIZE = -(Number of bytes occupied by the record).
//
//  compress = 0 objects written to this file will not be compressed.
//  compress = 1 minimal compression level but fast.
//  ....
//  compress = 9 maximal compression level but slow.
//
//  Note that the compression level may be changed at any time.
//  The new compression level will only apply to newly written objects.
//  The function TFile::Map shows the compression factor
//  for each object written to this file.
//  The function TFile::GetCompressionFactor returns the global
//  compression factor for this file.
//
//  A ROOT file is a suite of consecutive data records with the following
//    format (see also the TKey class);
// TKey ---------------------
//      byte 1->4  Nbytes    = Length of compressed object (in bytes)
//           5->6  Version   = TKey version identifier
//           7->10 ObjLen    = Length of uncompressed object
//          11->14 Datime    = Date and time when object was written to file
//          15->16 KeyLen    = Length of the key structure (in bytes)
//          17->18 Cycle     = Cycle of key
//          19->22 SeekKey   = Pointer to record itself (consistency check)
//          23->26 SeekPdir  = Pointer to directory header
//          27->27 lname     = Number of bytes in the class name
//          28->.. ClassName = Object Class Name
//          ..->.. lname     = Number of bytes in the object name
//          ..->.. Name      = lName bytes with the name of the object
//          ..->.. lTitle    = Number of bytes in the object title
//          ..->.. Title     = Title of the object
//          -----> DATA      = Data bytes associated to the object
//
//  The first data record starts at byte fBEGIN (currently set to kBegin)
//  Bytes 1->kBegin contain the file description:
//       byte  1->4  "root"      = Root file identifier
//             5->8  fVersion    = File format version
//             9->12 fBEGIN      = Pointer to first data record
//            13->16 fEND        = Pointer to first free word at the EOF
//            17->20 fSeekFree   = Pointer to FREE data record
//            21->24 fNbytesFree = Number of bytes in FREE data record
//            25->28 nfree       = Number of free data records
//            29->32 fNbytesName = Number of bytes in TNamed at creation time
//            33->33 fUnits      = Number of bytes for file pointers
//            34->37 fCompress   = Zip compression level
//Begin_Html
/*
<img src="gif/file_layout.gif">
*/
//End_Html
//
//  The structure of a directory is shown in TDirectory::TDirectory
//
//

   if (!gROOT)
      ::Fatal("TFile::TFile", "ROOT system not initialized");

   gDirectory = 0;
   SetName(fname1);
   SetTitle(ftitle);
   TDirectory::Build();

   fD          = -1;
   fFile       = this;
   fFree       = 0;
   fVersion    = gROOT->GetVersionInt();  //ROOT version in integer format
   fUnits      = kUnits;
   fOption     = option;
   fCompress   = compress;
   fWritten    = 0;
   fSumBuffer  = 0;
   fSum2Buffer = 0;
   fBytesRead  = 0;
   fBytesWrite = 0;
   fClassIndex = 0;
   fSeekInfo   = 0;
   fNbytesInfo = 0;
   fCache      = 0;
   fProcessIDs = 0;

   if (!fOption.CompareTo("NET", TString::kIgnoreCase))
      return;

   if (!fOption.CompareTo("WEB", TString::kIgnoreCase)) {
      fOption   = "READ";
      fWritable = kFALSE;
      return;
   }

   Bool_t create = kFALSE;
   if (!fOption.CompareTo("NEW", TString::kIgnoreCase) ||
       !fOption.CompareTo("CREATE", TString::kIgnoreCase))
       create = kTRUE;
   Bool_t recreate = fOption.CompareTo("RECREATE", TString::kIgnoreCase)
                    ? kFALSE : kTRUE;
   Bool_t update   = fOption.CompareTo("UPDATE", TString::kIgnoreCase)
                    ? kFALSE : kTRUE;
   Bool_t read     = fOption.CompareTo("READ", TString::kIgnoreCase)
                    ? kFALSE : kTRUE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   const char *fname;
   if ((fname = gSystem->ExpandPathName(fname1))) {
      SetName(fname);
      delete [] (char*)fname;
      fname = GetName();
   } else {
      Error("TFile", "error expanding path %s", fname1);
      goto zombie;
   }

   if (recreate) {
      if (!gSystem->AccessPathName(fname, kFileExists))
         gSystem->Unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   if (create && !gSystem->AccessPathName(fname, kFileExists)) {
      Error("TFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && gSystem->AccessPathName(fname, kWritePermission)) {
         Error("TFile", "no write permission, could not open file %s", fname);
         goto zombie;
      }
   }
   if (read) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         Error("TFile", "file %s does not exist", fname);
         goto zombie;
      }
      if (gSystem->AccessPathName(fname, kReadPermission)) {
         Error("TFile", "no read permission, could not open file %s", fname);
         goto zombie;
      }
   }

//*-*--------------Connect to file system stream
   if (create || update) {
#ifndef WIN32
      fD = SysOpen(fname, O_RDWR | O_CREAT, 0644);
#else
      fD = SysOpen(fname, O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TFile", "file %s can not be opened", fname);
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
#ifndef WIN32
      fD = SysOpen(fname, O_RDONLY, 0644);
#else
      fD = SysOpen(fname, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TFile", "file %s can not be opened for reading", fname);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   Init(create);

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
TFile::TFile(const TFile &file)
{
   ((TFile&)file).Copy(*this);
}

//______________________________________________________________________________
TFile::~TFile()
{
//*-*-*-*-*-*-*-*-*-*-*-*File destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ===============

   Close();

   delete fProcessIDs; fProcessIDs = 0;

   SafeDelete(fFree);
   SafeDelete(fCache);

   gROOT->GetListOfFiles()->Remove(this);

   if (gDebug)
      cerr <<"TFile dtor called for " << GetName() << endl;
}

//______________________________________________________________________________
void TFile::Init(Bool_t create)
{
   // Initialize a TFile object.

   Int_t max_file_size = 2000000000;  // should rather check disk quota
   Int_t nfree;

   // make newly opened file the current file and directory
   cd();

//*-*---------------NEW file
   if (create) {
      fFree        = new TList;
      fBEGIN       = kBegin;    //First used word in file following the file header
      fEND         = fBEGIN;    //Pointer to end of file
      new TFree(fFree, fBEGIN, max_file_size);  //Create new free list

//*-* Write Directory info
      Int_t namelen= TNamed::Sizeof();
      Int_t nbytes = namelen + TDirectory::Sizeof();
      TKey *key    = new TKey(fName,fTitle,IsA(),nbytes);
      fNbytesName  = key->GetKeylen() + namelen;
      fSeekDir     = key->GetSeekKey();
      fSeekFree    = 0;
      fNbytesFree  = 0;
      WriteHeader();
      char *buffer = key->GetBuffer();
      TNamed::FillBuffer(buffer);
      TDirectory::FillBuffer(buffer);
      key->WriteFile();
      delete key;
   }
//*-*----------------UPDATE
   else {
      char *header = new char[kBegin];
      Seek(0);
      ReadBuffer(header,kBegin);

      // make sure this is a root file
      if (strncmp(header, "root", 4)) {
         Error("TFile", "%s not a ROOT file", fName.Data());
         delete [] header;
         goto zombie;
      }

      char *buffer = header + 4;    // skip the "root" file identifier
      frombuf(buffer, &fVersion);
      frombuf(buffer, &fBEGIN);
      frombuf(buffer, &fEND);
      frombuf(buffer, &fSeekFree);
      frombuf(buffer, &fNbytesFree);
      frombuf(buffer, &nfree);
      frombuf(buffer, &fNbytesName);
      frombuf(buffer, &fUnits );
      frombuf(buffer, &fCompress);
      frombuf(buffer, &fSeekInfo);
      frombuf(buffer, &fNbytesInfo);
      fSeekDir = fBEGIN;
      delete [] header;
//*-*-------------Read Free segments structure if file is writable
      if (fWritable) {
        fFree = new TList;
        if (fSeekFree > fBEGIN) {
           ReadFree();
        } else {
           Warning("TFile","file %s probably not closed, cannot read free segments",GetName());
        }
      }
//*-*-------------Read directory info
      Int_t nbytes = fNbytesName + TDirectory::Sizeof();
      header       = new char[nbytes];
      buffer       = header;
      Seek(fBEGIN);
      ReadBuffer(buffer,nbytes);
      buffer = header+fNbytesName;
      Version_t versiondir;
      frombuf(buffer,&versiondir);
      fDatimeC.ReadBuffer(buffer);
      fDatimeM.ReadBuffer(buffer);
      frombuf(buffer, &fNbytesKeys);
      frombuf(buffer, &fNbytesName);
      frombuf(buffer, &fSeekDir);
      frombuf(buffer, &fSeekParent);
      frombuf(buffer, &fSeekKeys);
//*-*---------read TKey::FillBuffer info
      Int_t nk = sizeof(Int_t) +sizeof(Version_t) +2*sizeof(Int_t)+2*sizeof(Short_t)
                +2*sizeof(Seek_t);
      buffer = header+nk;
      TString cname;
      cname.ReadBuffer(buffer);
      cname.ReadBuffer(buffer); // fName.ReadBuffer(buffer); file may have been renamed
      fTitle.ReadBuffer(buffer);
      delete [] header;
      if (fNbytesName < 10 || fNbytesName > 1000) {
         Error("Init","Cannot read directory info");
         goto zombie;
      }
//*-* -------------Check if file is truncated
      Long_t size;
      if ((size = GetSize()) == -1) {
         Error("TFile", "cannot stat the file %s", GetName());
         goto zombie;
      }
//*-* -------------Read keys of the top directory

      if (fSeekKeys > fBEGIN && fEND <= size) {
         TDirectory::ReadKeys();
         gDirectory = this;
         if (!GetNkeys()) Recover();
      } else {
         if (fEND > size) {
            Error("TFile","file %s is truncated at %d bytes: should be %d, trying to recover",GetName(),size,fEND);
         } else {
            Warning("TFile","file %s probably not closed, trying to recover",GetName());
         }
         Int_t nrecov = Recover();
         if (nrecov) {
            Warning("Recover", "successfully recovered %d keys", nrecov);
         } else {
            Warning("Recover", "no keys recovered: file is a Zombie");
            goto zombie;
         }
      }
   }
   gROOT->GetListOfFiles()->Add(this);

   // Create StreamerInfo index
   {
      Int_t lenIndex = gROOT->GetListOfStreamerInfo()->GetSize()+1;
      if (lenIndex < 5000) lenIndex = 5000;
      fClassIndex = new TArrayC(lenIndex);
      if (fSeekFree > fBEGIN) ReadStreamerInfo();
   }

   fProcessIDs = new TObjArray(10);
   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
void TFile::Close(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*Close a file*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============

   if (!IsOpen()) return;

   if (IsWritable()) {
      WriteStreamerInfo();
   }

   delete fClassIndex;
   fClassIndex = 0;

   TCollection::StartGarbageCollection();

   TDirectory *cursav = gDirectory;
   cd();

   if (gFile == this) {
      cursav = 0;
   }

   // Delete all supported directories structures from memory
   TDirectory::Close();
   cd();      // Close() sets gFile = 0

   if (IsWritable()) {
      TFree *f1 = (TFree*)fFree->First();
      if (f1) {
         WriteFree();       //*-*- Write free segments linked list
         WriteHeader();     //*-*- Now write file header
      }
      if (fCache) fCache->Flush();
   }

   // Delete free segments from free list (but don't delete list header)
   if (fFree) {
      fFree->Delete();
   }

   if (IsOpen()) {
      SysClose(fD);
      fD = -1;
   }

   fWritable = kFALSE;

   if (cursav)
      cursav->cd();
   else {
      gFile      = 0;
      gDirectory = gROOT;
   }

   //delete the TProcessIDs
   TIter next(fProcessIDs);
   TProcessID *pid;
   while ((pid = (TProcessID*)next())) {
      if (!pid->DecrementCount()) {
         if (pid != TProcessID::GetSessionProcessID()) delete pid;
      }
  }

   gROOT->GetListOfFiles()->Remove(this);

   TCollection::EmptyGarbageCollection();
}

//______________________________________________________________________________
void TFile::Delete(const char *namecycle)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Delete object namecycle*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =======================
//  namecycle identifies an object in the top directory of the file
//   namecycle has the format name;cycle
//   name  = * means all
//   cycle = * means all cycles (memory and keys)
//   cycle = "" or cycle = 9999 ==> apply to a memory object
//   When name=* use T* to delete subdirectories also
//
//   examples:
//     foo   : delete object named foo in memory
//     foo;1 : delete cycle 1 of foo on file
//     foo;* : delete all cycles of foo on disk and also from memory
//     *;2   : delete all objects on file having the cycle 2
//     *;*   : delete all objects from memory and file
//    T*;*   : delete all objects from memory and file and all subdirectories
//

   if (gDebug)
      Printf("TFile Deleting Name=%s",namecycle);

   TDirectory::Delete(namecycle);
}
//______________________________________________________________________________
void TFile::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Fill Graphics Structure and Paint-*-*-*-*-*-*-*-*-*-*
//*-*                    =================================
// Loop on all objects (memory or file) and all subdirectories
//

   GetList()->ForEach(TObject,Draw)(option);
}

//______________________________________________________________________________
void TFile::Flush()
{
   // Synchornize a file's in-core and on-disk states.

   if (IsOpen() && fWritable) {
      if (SysSync(fD) < 0)
         SysError("Flush", "error flushing file %s", GetName());
   }
}

//______________________________________________________________________________
void TFile::FillBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*-*Encode file output buffer*-*-*-*-*-*-*
//*-*                    =========================
// The file output buffer contains only the FREE data record
//

   Version_t version = TFile::Class_Version();
   tobuf(buffer, version);
}

//______________________________________________________________________________
Int_t TFile::GetBestBuffer() const
{
//*-*-*-*-*-*-*-*Return the best buffer size of objects on this file*-*-*-*-*-*
//*-*            ===================================================
//
//  The best buffer size is estimated based on the current mean value
//  and standard deviation of all objects written so far to this file.
//  Returns mean value + one standard deviation.
//

   if (!fWritten) return TBuffer::kInitialSize;
   Double_t mean = fSumBuffer/fWritten;
   Double_t rms2 = TMath::Abs(fSum2Buffer/fSumBuffer -mean*mean);
   return (Int_t)(mean + TMath::Sqrt(rms2));
}

//______________________________________________________________________________
Float_t TFile::GetCompressionFactor()
{
//*-*-*-*-*-*-*-*-*-*Return the file compression factor*-*-*-*-*-*-*-*-*-*
//*-*                =================================
//
//  Add total number of compressed/uncompressed bytes for each key.
//  return ratio of the two.
//
   Short_t  keylen;
   UInt_t   datime;
   Int_t    nbytes, objlen, nwh = 64;
   char    *header = new char[kBegin];
   char    *buffer;
   Seek_t   idcur = fBEGIN;
   Float_t comp,uncomp;
   comp = uncomp = fBEGIN;

   while (idcur < fEND-100) {
      Seek(idcur);
      ReadBuffer(header, nwh);
      buffer=header;
      frombuf(buffer, &nbytes);
      if (nbytes < 0) {
         idcur -= nbytes;
         Seek(idcur);
         continue;
      }
      if (nbytes == 0) break; //this may happen when the file is corrupted
      Version_t versionkey;
      frombuf(buffer, &versionkey);
      frombuf(buffer, &objlen);
      frombuf(buffer, &datime);
      frombuf(buffer, &keylen);
      if (!objlen) objlen = nbytes-keylen;
      comp   += nbytes;
      uncomp += keylen + objlen;
      idcur  += nbytes;
   }
   delete [] header;
   return uncomp/comp;
}

//______________________________________________________________________________
Int_t TFile::GetErrno() const
{
   // Method returning errno. Is overriden in TRFIOFile.

   return TSystem::GetErrno();
}

//______________________________________________________________________________
void TFile::ResetErrno() const
{
   // Method resetting the errno. Is overridden in TRFIOFile.

   TSystem::ResetErrno();
}

//______________________________________________________________________________
Int_t TFile::GetRecordHeader(char *buf, Seek_t first, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen)
{
//*-*-*-*-*-*-*-*-*Read the logical record header starting at position first
//*-*              =========================================================
// maxbytes bytes are read into buf
// the function reads nread bytes where nread is the minimum of maxbytes
// and the number of bytes before the end of file.
// the function returns nread.
// In output arguments:
//    nbytes : number of bytes in record
//             if negative, this is a deleted record
//             if 0, cannot read record, wrong value of argument first
//    objlen : uncompressed object size
//    keylen : length of logical record header
// Note that the arguments objlen and keylen are returned only if maxbytes >=16

   if (first < fBEGIN) return 0;
   if (first > fEND)   return 0;
   Seek(first);
   Int_t nread = maxbytes;
   if (first+maxbytes > fEND) nread = fEND-maxbytes;
   if (nread < 4) {
      Warning("GetRecordHeader","parameter maxbytes=%d must be >= 4",nread);
      return nread;
   }
   ReadBuffer(buf,nread);
   Version_t versionkey;
   Short_t  klen;
   UInt_t   datime;
   Int_t    nb,olen;
   char *buffer = buf;
   frombuf(buffer,&nb);
   nbytes = nb;
   if (nb < 0) return nread;
//   const Int_t headerSize = Int_t(sizeof(nb) +sizeof(versionkey) +sizeof(olen) +sizeof(datime) +sizeof(klen));
   const Int_t headerSize = 16;
   if (nread < headerSize) return nread;
   frombuf(buffer, &versionkey);
   frombuf(buffer, &olen);
   frombuf(buffer, &datime);
   frombuf(buffer, &klen);
   if (!olen) olen = nbytes-klen;
   objlen = olen;
   keylen = klen;
   return nread;
}

//______________________________________________________________________________
Seek_t TFile::GetSize() const
{
   // Returns the current file size. Returns -1 in case the file could not
   // be stat'ed.

   Long_t id, size, flags, modtime;
#if defined(R__AIX) || (defined(R__HPUX) && !defined(R__ACC))
   if (((TFile*)this)->SysStat(fD, &id, &size, &flags, &modtime)) {
#else
   if (const_cast<TFile*>(this)->SysStat(fD, &id, &size, &flags, &modtime)) {
#endif
      Error("GetSize", "cannot stat the file %s", GetName());
      return -1;
   }

   return size;
}

//______________________________________________________________________________
TList *TFile::GetStreamerInfoList() 
{
// Read the list of TStreamerInfo objects written to this file.
// The function returns a TList. It is the user'responsability
// to delete the list created by this function.
//
// Using the list, one can access additional information,eg:
//   TFile f("myfile.root");
//   TList *list = f.GetStreamerInfoList();
//   TStreamerInfo *info = (TStreamerInfo*)list->FindObject("MyClass");
//   Int_t classversionid = info->GetClassVersion();
//   delete list;

   TList *list = 0;
   if (fSeekInfo) {
      TKey *key = new TKey();
      char *buffer = new char[fNbytesInfo+1];
      char *buf    = buffer;
      Seek(fSeekInfo);
      ReadBuffer(buf,fNbytesInfo);
      key->ReadBuffer(buf);
      list = (TList*)key->ReadObj();
      list->SetOwner();
      delete [] buffer;
      delete key;
   } else {
      list = (TList*)Get("StreamerInfo"); //for versions 2.26 (never released)
   }

   if (list == 0) {
      printf("Cannot find the StreamerInfo record in this file\n");
      return 0;
   }

   return list;
}

//______________________________________________________________________________
void TFile::ls(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*List File contents*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==================
//  Indentation is used to identify the file tree
//  Subdirectories are listed first
//  then objects in memory
//  then objects on the file
//

   TROOT::IndentLevel();
   cout <<ClassName()<<"**\t\t"<<GetName()<<"\t"<<GetTitle()<<endl;
   TROOT::IncreaseDirLevel();
   TDirectory::ls(option);
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
Bool_t TFile::IsOpen() const
{
   // Returns kTRUE in case file is open and kFALSE if file is not open.

   return fD == -1 ? kFALSE : kTRUE;
}

//______________________________________________________________________________
void TFile::MakeFree(Seek_t first, Seek_t last)
{
//*-*-*-*-*-*-*-*-*-*-*-*Mark unused bytes on the file*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//  The list of free segments is in the fFree linked list
//  When an object is deleted from the file, the freed space is added
//  into the FREE linked list (fFree). The FREE list consists of a chain
//  of consecutive free segments on the file. At the same time, the first
//  4 bytes of the freed record on the file are overwritten by GAPSIZE
//  where GAPSIZE = -(Number of bytes occupied by the record).
//

   TFree *f1      = (TFree*)fFree->First();
   if (!f1) return;
   TFree *newfree = f1->AddFree(fFree,first,last);
   if(!newfree) return;
   Seek_t nfirst  = newfree->GetFirst();
   Seek_t nlast   = newfree->GetLast();
   Int_t nbytes   = Int_t (nfirst - nlast -1);
   Int_t nb       = sizeof(Int_t);
   char * buffer  = new char[nb];
   char * psave   = buffer;
   tobuf(buffer, nbytes);
   if (nlast == fEND-1) fEND = nfirst;
   Seek(nfirst);
   WriteBuffer(psave, nb);
   Flush();
   delete [] psave;
}


//______________________________________________________________________________
void TFile::Map()
{
//*-*-*-*-*-*-*-*-*-*List the contents of a file sequentially*-*-*-*-*-*
//*-*                ========================================
//
//  For each logical record found, it prints
//     Date/Time  Record_Adress Logical_Record_Length  ClassName  CompressionFactor
//
//  Example of output
//  20010404/150437  At:64        N=150       TFile
//  20010404/150440  At:214       N=28326     TBasket        CX =  1.13
//  20010404/150440  At:28540     N=29616     TBasket        CX =  1.08
//  20010404/150440  At:58156     N=29640     TBasket        CX =  1.08
//  20010404/150440  At:87796     N=29076     TBasket        CX =  1.10
//  20010404/150440  At:116872    N=10151     TBasket        CX =  3.15
//  20010404/150441  At:127023    N=28341     TBasket        CX =  1.13
//  20010404/150441  At:155364    N=29594     TBasket        CX =  1.08
//  20010404/150441  At:184958    N=29616     TBasket        CX =  1.08
//  20010404/150441  At:214574    N=29075     TBasket        CX =  1.10
//  20010404/150441  At:243649    N=9583      TBasket        CX =  3.34
//  20010404/150442  At:253232    N=28324     TBasket        CX =  1.13
//  20010404/150442  At:281556    N=29641     TBasket        CX =  1.08
//  20010404/150442  At:311197    N=29633     TBasket        CX =  1.08
//  20010404/150442  At:340830    N=29091     TBasket        CX =  1.10
//  20010404/150442  At:369921    N=10341     TBasket        CX =  3.09
//  20010404/150442  At:380262    N=509       TH1F           CX =  1.93
//  20010404/150442  At:380771    N=1769      TH2F           CX =  4.32
//  20010404/150442  At:382540    N=1849      TProfile       CX =  1.65
//  20010404/150442  At:384389    N=18434     TNtuple        CX =  4.51
//  20010404/150442  At:402823    N=307       KeysList
//  20010404/150443  At:403130    N=4548      StreamerInfo   CX =  3.65
//  20010404/150443  At:407678    N=86        FreeSegments
//  20010404/150443  At:407764    N=1         END
//
   Short_t  keylen,cycle;
   UInt_t   datime;
   Int_t    nbytes,date,time,objlen,nwheader;
   Seek_t   seekkey,seekpdir;
   char    *buffer;
   char     nwhc;
   Seek_t   idcur = fBEGIN;

   nwheader = 64;
   Int_t nread = nwheader;

   char header[kBegin];
   char classname[512];

   while (idcur < fEND) {
      Seek(idcur);
      if (idcur+nread >= fEND) nread = fEND-idcur-1;
      ReadBuffer(header, nread);
      buffer=header;
      frombuf(buffer, &nbytes);
      if (!nbytes) {
         Printf("Address = %d\tNbytes = %d\t=====E R R O R=======", idcur, nbytes);
         break;
      }
      if (nbytes < 0) {
         Printf("Address = %d\tNbytes = %d\t=====G A P===========", idcur, nbytes);
         idcur -= nbytes;
         Seek(idcur);
         continue;
      }
      Version_t versionkey;
      frombuf(buffer, &versionkey);
      frombuf(buffer, &objlen);
      frombuf(buffer, &datime);
      frombuf(buffer, &keylen);
      frombuf(buffer, &cycle);
      frombuf(buffer, &seekkey);
      frombuf(buffer, &seekpdir);
      frombuf(buffer, &nwhc);
      int i;
      for (i = 0;i < nwhc; i++) frombuf(buffer, &classname[i]);
      classname[nwhc] = '\0';
      if (idcur == fSeekFree) strcpy(classname,"FreeSegments");
      if (idcur == fSeekInfo) strcpy(classname,"StreamerInfo");
      if (idcur == fSeekKeys) strcpy(classname,"KeysList");
      TDatime::GetDateTime(datime, date, time);
      if (objlen != nbytes-keylen) {
         Float_t cx = Float_t(objlen+keylen)/Float_t(nbytes);
         Printf("%d/%06d  At:%-8d  N=%-8d  %-14s CX = %5.2f",date,time,idcur,nbytes,classname,cx);
      } else {
         Printf("%d/%06d  At:%-8d  N=%-8d  %-14s",date,time,idcur,nbytes,classname);
      }
      idcur += nbytes;
   }
   Printf("%d/%06d  At:%-8d  N=%-8d  %-14s",date,time,idcur,1,"END");
}

//______________________________________________________________________________
void TFile::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Paint all objects in the file*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//

   GetList()->ForEach(TObject,Paint)(option);
}

//______________________________________________________________________________
void TFile::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print all objects in the file*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//

   Printf("TFile: name=%s, title=%s, option=%s", GetName(), GetTitle(), GetOption());
   GetList()->ForEach(TObject,Print)(option);
}

//______________________________________________________________________________
Bool_t TFile::ReadBuffer(char *buf, Int_t len)
{
   // Read a buffer from the file. This is the basic low level read operation.
   // Returns kTRUE in case of failure.

   if (IsOpen()) {
      ssize_t siz;
      while ((siz = SysRead(fD, buf, len)) < 0 && GetErrno() == EINTR)
         ResetErrno();
      if (siz < 0) {
         SysError("ReadBuffer", "error reading from file %s", GetName());
         return kTRUE;
      }
      if (siz != len) {
         Error("ReadBuffer", "error reading all requested bytes from file %s, got %d of %d",
               GetName(), siz, len);
         return kTRUE;
      }
      fBytesRead  += siz;
      fgBytesRead += siz;

      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TFile::ReadFree()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Read the FREE linked list*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =========================
//  Every file has a linked list (fFree) of free segments
//  This linked list has been written on the file via WriteFree
//  as a single data record
//

   TKey *headerfree  = new TKey(fSeekFree,fNbytesFree);
   headerfree->ReadFile();
   char *buffer = headerfree->GetBuffer();
   headerfree->ReadBuffer(buffer);
   buffer =  headerfree->GetBuffer();
   while(1) {
      TFree *afree = new TFree();
      afree->ReadBuffer(buffer);
      fFree->Add(afree);
      if (afree->GetLast() > fEND) break;
   }
   delete headerfree;
}

//______________________________________________________________________________
Int_t TFile::Recover()
{
//*-*-*-*-*-*-*-*-*Attempt to recover file if not correctly closed*-*-*-*-*
//*-*              ===============================================
//
//  The function returns the number of keys that have been recovered.
//  If no keys can be recovered, the file will be declared Zombie by
//  the calling function.

   Short_t  keylen,cycle;
   UInt_t   datime;
   Int_t    nbytes,date,time,objlen,nwheader;
   Seek_t   seekkey,seekpdir;
   char    *header = new char[kBegin];
   char    *buffer, *bufread;
   char     nwhc;
   Seek_t   idcur = fBEGIN;

   Long_t size;
   if ((size = GetSize()) == -1) {
      Error("Recover", "cannot stat the file %s", GetName());
      return 0;
   }

   fEND = Seek_t(size);

   if (fWritable && !fFree) fFree  = new TList;

   TKey *key;
   Int_t nrecov = 0;
   nwheader = 64;
   Int_t nread = nwheader;

   while (idcur < fEND) {
      Seek(idcur);
      if (idcur+nread >= fEND) nread = fEND-idcur-1;
      ReadBuffer(header, nread);
      buffer  = header;
      bufread = header;
      frombuf(buffer, &nbytes);
      if (!nbytes) {
         Printf("Address = %d\tNbytes = %d\t=====E R R O R=======", idcur, nbytes);
         break;
      }
      if (nbytes < 0) {
         idcur -= nbytes;
         if (fWritable) new TFree(fFree,idcur,idcur-nbytes-1);
         Seek(idcur);
         continue;
      }
      Version_t versionkey;
      frombuf(buffer, &versionkey);
      frombuf(buffer, &objlen);
      frombuf(buffer, &datime);
      frombuf(buffer, &keylen);
      frombuf(buffer, &cycle);
      frombuf(buffer, &seekkey);
      frombuf(buffer, &seekpdir);
      frombuf(buffer, &nwhc);
      char *classname = new char[nwhc+1];
      int i;
      for (i = 0;i < nwhc; i++) frombuf(buffer, &classname[i]);
      classname[nwhc] = '\0';
      TDatime::GetDateTime(datime, date, time);
      if (seekpdir == fSeekDir && strcmp(classname,"TBasket")) {
         key = new TKey();
         key->ReadBuffer(bufread);
         AppendKey(key);
         nrecov++;
         Printf("Recovering key: %s:%s at address:%d",key->GetClassName(),key->GetName(),idcur);
      }
      delete [] classname;
      idcur += nbytes;
   }
   if (fWritable) {
      new TFree(fFree,fEND,2000000000);
      if (nrecov) Write();
   }
   delete [] header;
   return nrecov;
}

//______________________________________________________________________________
void TFile::Seek(Seek_t offset, ERelativeTo pos)
{
   // Seek to a specific position in the file. Pos it either kBeg, kCur or kEnd.

   int whence = 0;
   switch (pos) {
   case kBeg:
      whence = SEEK_SET;
      break;
   case kCur:
      whence = SEEK_CUR;
      break;
   case kEnd:
      whence = SEEK_END;
      break;
   }
   if (SysSeek(fD, offset, whence) < 0)
      SysError("Seek", "cannot seek to position %d in file %s", offset, GetName());
}

//______________________________________________________________________________
void TFile::SetCompressionLevel(Int_t level)
{
//*-*-*-*-*-*-*-*-*-*Set level of compression for this file*-*-*-*-*-*-*-*
//*-*                ======================================
//
//  level = 0 objects written to this file will not be compressed.
//  level = 1 minimal compression level but fast.
//  ....
//  level = 9 maximal compression level but slow.
//
//  Note that the compression level may be changed at any time.
//  The new compression level will only apply to newly written objects.
//  The function TFile::Map shows the compression factor
//  for each object written to this file.
//  The function TFile::GetCompressionFactor returns the global
//  compression factor for this file.
//

   if (level < 0) level = 0;
   if (level > 9) level = 9;
   fCompress = level;
}

//______________________________________________________________________________
Int_t TFile::Sizeof() const
{
//*-*-*-*-*-*-*Return the size in bytes of the file header-*-*-*-*-*-*-*-*
//*-*          ===========================================

   return 0;
}

//_______________________________________________________________________
void TFile::Streamer(TBuffer &b)
{
   // Stream a TFile object.

   if (b.IsReading()) {
      b.ReadVersion();  //Version_t v = b.ReadVersion();
   } else {
      b.WriteVersion(TFile::IsA());
   }
}

//_______________________________________________________________________
void TFile::SumBuffer(Int_t bufsize)
{
//*-*-*-*-*Increment statistics for buffer sizes of objects in this file*-*-*
//*-*      =============================================================

   fWritten++;
   fSumBuffer  += bufsize;
   fSum2Buffer += bufsize*bufsize;
}

//_______________________________________________________________________
void TFile::UseCache(Int_t maxCacheSize, Int_t pageSize)
{
   // Activate caching. Use maxCacheSize to specify the maximum cache size
   // in MB's (default is 10 MB) and pageSize to specify the page size
   // (default is 512 KB). To turn off the cache use maxCacheSize=0.
   // Not needed for normal disk files since the operating system will
   // do proper caching (via the "buffer cache"). Use it for TNetFile,
   // TWebFile and TRFIOFile.

   if (IsA() == TFile::Class())
      return;

   if (maxCacheSize == 0) {
      if (fCache) {
         if (IsWritable())
            fCache->Flush();
         delete fCache;
         fCache = 0;
      }
      return;
   }

   if (fCache) {
      // if pageSize is changed, we need to delete the cache and recreate it
      if (pageSize != fCache->GetPageSize()) {
         if (IsWritable())
            fCache->Flush();
         delete fCache;
      } else if (maxCacheSize != fCache->GetMaxCacheSize()) {
         fCache->Resize(maxCacheSize);
         return;
      }
   }
   fCache = new TCache(maxCacheSize, this, pageSize);
}

//______________________________________________________________________________
Int_t TFile::Write(const char *, Int_t opt, Int_t bufsiz)
{
//*-*-*-*-*-*-*-*-*-*Write memory objects to this file*-*-*-*-*-*-*-*-*-*
//*-*                =================================
//  Loop on all objects in memory (including subdirectories).
//  A new key is created in the KEYS linked list for each object.
//  The list of keys is then saved on the file (via WriteKeys)
//  as a single data record.
//  For values of opt see TObject::Write().
//  The directory header info is rewritten on the directory header record.
//  The linked list of FREE segments is written.
//  The file header is written (bytes 1->fBEGIN).
//

   if (!IsWritable()) {
      Warning("Write", "file not opened in write mode");
      return 0;
   }

   TDirectory *cursav = gDirectory;
   cd();

   if (gDebug) {
      if (!GetTitle() || strlen(GetTitle()) == 0)
         Printf("TFile Writing Name=%s", GetName());
      else
         Printf("TFile Writing Name=%s Title=%s", GetName(), GetTitle());
   }

   Int_t nbytes = TDirectory::Write(0, opt, bufsiz); // Write directory tree
   WriteStreamerInfo();
   WriteFree();                       // Write free segments linked list
   WriteHeader();                     // Now write file header

   cursav->cd();
   return nbytes;
}

//______________________________________________________________________________
Bool_t TFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write a buffer to the file. This is the basic low level write operation.
   // Returns kTRUE in case of failure.

   if (IsOpen() && fWritable) {
      ssize_t siz;
      gSystem->IgnoreInterrupt();
      while ((siz = SysWrite(fD, buf, len)) < 0 && GetErrno() == EINTR)
         ResetErrno();
      gSystem->IgnoreInterrupt(kFALSE);
      if (siz < 0) {
         SysError("WriteBuffer", "error writing to file %s", GetName());
         return kTRUE;
      }
      if (siz != len) {
         Error("WriteBuffer", "error writing all requested bytes to file %s, wrote %d of %d",
               GetName(), siz, len);
         return kTRUE;
      }
      fBytesWrite  += siz;
      fgBytesWrite += siz;

      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TFile::WriteFree()
{
//*-*-*-*-*-*-*-*-*-*-*-*Write FREE linked list on the file *-*-*-*-*-*-*-*
//*-*                    ==================================
//  The linked list of FREE segments (fFree) is written as a single data
//  record
//

//*-* Delete old record if it exists
   if (fSeekFree != 0){
      MakeFree(fSeekFree, fSeekFree + fNbytesFree -1);
   }

   Int_t nbytes = 0;
   TFree *afree;
   TIter next (fFree);
   while ((afree = (TFree*) next())) {
      nbytes += afree->Sizeof();
   }
   if (!nbytes) return;
   TKey *key    = new TKey(fName,fTitle,IsA(),nbytes);
   if (key->GetSeekKey() == 0) {
      delete key;
      return;
   }
   char *buffer = key->GetBuffer();

   next.Reset();
   while ((afree = (TFree*) next())) {
      afree->FillBuffer(buffer);
   }
   fNbytesFree = key->GetNbytes();
   fSeekFree   = key->GetSeekKey();
   key->WriteFile();
   delete key;
}

//______________________________________________________________________________
void TFile::WriteHeader()
{
//*-*-*-*-*-*-*-*-*-*-*-*Write File Header*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =================
//
   TFree *lastfree = (TFree*)fFree->Last();
   if (lastfree) fEND  = lastfree->GetFirst();
   const char *root = "root";
   char *psave  = new char[kBegin];
   char *buffer = psave;
   Int_t nfree  = fFree->GetSize();
   memcpy(buffer, root, 4); buffer += 4;
   tobuf(buffer, fVersion);
   tobuf(buffer, fBEGIN);
   tobuf(buffer, fEND);
   tobuf(buffer, fSeekFree);
   tobuf(buffer, fNbytesFree);
   tobuf(buffer, nfree);
   tobuf(buffer, fNbytesName);
   tobuf(buffer, fUnits);
   tobuf(buffer, fCompress);
   tobuf(buffer, fSeekInfo);
   tobuf(buffer, fNbytesInfo);
   Int_t nbytes  = buffer - psave;
   Seek(0);
   WriteBuffer(psave, nbytes);
   Flush();
   delete [] psave;
}

//______________________________________________________________________________
void TFile::MakeProject(const char *dirname, const char *classes, Option_t *option)
{
// Generate code in directory dirname for all classes specified in argument classes
// If classes = "*" (default), the function generates an include file for each
// class in the StreamerInfo list for which a TClass object does not exist.
// One can restrict the list of classes to be generated by using expressions like:
//   classes = "Ali*" generate code only for classes starting with Ali
//   classes = "myClass" generate code for class MyClass only.
//
// if option = "new" (default) a new directory dirname is created.
//                   If dirname already exist, an error message is printed
//                   and the function returns.
// if option = "recreate", then;
//                   if dirname does not exist, it is created (like in "new")
//                   if dirname already exist, all existing files in dirname
//                   are deleted before creating the new files.
// if option = "update", then new classes are added to the existing directory.
//                   Existing classes with the same name are replaced by the
//                   new definition. If the directory dirname doest not exist,
//                   same effect as "new".
// if, in addition to one of the 3 above options, the option "+" is specified,
// the function will generate:
//   - a script called MAKE to build the shared lib
//   - a LinkDef.h file
//   - rootcint will be run to generate a dirnameProjectDict.cxx file
//   - dirnameProjectDict.cxx will be compiled with the current options in compiledata.h
//   - a shared lib dirname.so will be created.
// if the option "++" is specified, the generated shared lib is dynamically
// linked with the current executable module.
// example:
//  file.MakeProject("demo","*","recreate++");
//  - creates a new directory demo unless it already exist
//  - clear the previous directory content
//  - generate the xxx.h files for all classes xxx found in this file
//    and not yet known to the CINT dictionary.
//  - creates the build script MAKE
//  - creates a LinkDef.h file
//  - runs rootcint generating demoProjectDict.cxx
//  - compiles demoProjectDict.cxx into demoProjectDict.o
//  - generates a shared lib demo.so
//  - dynamically links the shared lib demo.so to the executable
//  If only the option "+" had been specified, one can still link the
//  shared lib to the current executable module with:
//     gSystem->load("demo/demo.so");
//

   TString opt = option;
   opt.ToLower();
   void *dir = gSystem->OpenDirectory(dirname);
   char *path = new char[4000];

   if (opt.Contains("update")) {
      // check that directory exist, if not create it
      if (dir == 0) {
         gSystem->mkdir(dirname);
      }

   } else if (opt.Contains("recreate")) {
      // check that directory exist, if not create it
      if (dir == 0) {
         gSystem->mkdir(dirname);
      }
      // clear directory
      while (dir) {
         const char *afile = gSystem->GetDirEntry(dir);
         if (afile == 0) break;
         if (strcmp(afile,".") == 0) continue;
         if (strcmp(afile,"..") == 0) continue;
         sprintf(path,"%s/%s",dirname,afile);
         gSystem->Unlink(path);
      }

   } else {
      // new is assumed
      // if directory already exist, print error message and return
      if (dir) {
         Error("MakeProject","Cannot create directory:%s, already existing",dirname);
         delete [] path;
         return;
      }
      gSystem->mkdir(dirname);
   }

   // we are now ready to generate the classes
   // loop on all TStreamerInfo
   TList *list = 0;
   if (fSeekInfo) {
      TKey *key = new TKey();
      char *buffer = new char[fNbytesInfo+1];
      char *buf    = buffer;
      Seek(fSeekInfo);
      ReadBuffer(buf,fNbytesInfo);
      key->ReadBuffer(buf);
      list = (TList*)key->ReadObj();
      delete [] buffer;
      delete key;
   } else {
      list = (TList*)Get("StreamerInfo"); //for versions 2.26 (never released)
   }
   if (list == 0) {
      Error("MakeProject","File has no StreamerInfo");
      delete [] path;
      return;
   }

   // loop on all TStreamerInfo classes
   TStreamerInfo *info;
   TIter next(list);
   Int_t ngener = 0;
   while ((info = (TStreamerInfo*)next())) {
      ngener += info->GenerateHeaderFile(dirname);
   }
   list->Delete();
   delete list;
   printf("MakeProject has generated %d classes in %s\n",ngener,dirname);

   // generate the shared lib
   if (!opt.Contains("+")) { delete [] path; return;}

   // create the MAKE file by looping on all *.h files
   // delete MAKE if it already exists
#ifdef WIN32
   sprintf(path,"%s/make.cmd",dirname);
#else
   sprintf(path,"%s/MAKE",dirname);
#endif
   FILE *fpMAKE = fopen(path,"w");
   if (!fpMAKE) {
      printf("Cannot open file:%s\n",path);
      delete [] path;
      return;
   }

   // add rootcint statement generating ProjectDict.cxx
   fprintf(fpMAKE,"rootcint -f %sProjectDict.cxx -c %s ",dirname,gSystem->GetIncludePath());

   // create the LinkDef.h file by looping on all *.h files
   // delete LinkDef.h if it already exists
   sprintf(path,"%s/LinkDef.h",dirname);
   FILE *fp = fopen(path,"w");
   if (!fp) {
      printf("Cannot open path file:%s\n",path);
      delete [] path;
      return;
   }
   fprintf(fp,"#ifdef __CINT__\n");
   fprintf(fp,"#pragma link off all globals;\n");
   fprintf(fp,"#pragma link off all classes;\n");
   fprintf(fp,"#pragma link off all functions;\n");
   fprintf(fp,"\n");
   dir = gSystem->OpenDirectory(dirname);
   while (dir) {
      const char *afile = gSystem->GetDirEntry(dir);
      if (afile == 0) break;
      if(strcmp(afile,"LinkDef.h") == 0) continue;
      if(strstr(afile,"ProjectDict.h") != 0) continue;
      strcpy(path,afile);
      char *h = strstr(path,".h");
      if (!h) continue;
      *h = 0;
      fprintf(fp,"#pragma link C++ class %s+;\n",path);
      fprintf(fpMAKE,"%s ",afile);
  }
   fprintf(fp,"#endif\n");
   fclose(fp);
   fprintf(fpMAKE,"LinkDef.h \n");

   // add compilation line
   TString sdirname(dirname);

   TString cmd = gSystem->GetMakeSharedLib();
   cmd.ReplaceAll("$SourceFiles",sdirname+"ProjectDict.cxx");
   cmd.ReplaceAll("$ObjectFiles",sdirname+"ProjectDict."+gSystem->GetObjExt());
   cmd.ReplaceAll("$IncludePath",TString(gSystem->GetIncludePath()) + " -I" + dirname);
   cmd.ReplaceAll("$SharedLib",sdirname+"."+gSystem->GetSoExt());
   cmd.ReplaceAll("$LinkedLibs",gSystem->GetLibraries("","SDL"));
   cmd.ReplaceAll("$LibName",sdirname);
   cmd.ReplaceAll("$BuildDir",".");

   fprintf(fpMAKE,"%s\n",cmd.Data());

   fclose(fpMAKE);
   printf("%s/MAKE file has been generated\n",dirname);

   // now execute the generated script compiling and generating the shared lib
   strcpy(path,gSystem->WorkingDirectory());
   gSystem->ChangeDirectory(dirname);
#ifndef WIN32
   gSystem->Exec("chmod +x MAKE");
#else
   // not really needed for Windows but it would work both both Unix and NT
   chmod("make.cmd",00700);
#endif
   int res = !gSystem->Exec("MAKE");
   gSystem->ChangeDirectory(path);
   sprintf(path,"%s/%s.%s",dirname,dirname,gSystem->GetSoExt());
   if (res) printf("Shared lib %s has been generated\n",path);

   //dynamically link the generated shared lib
   if (opt.Contains("++")) {
      res = !gSystem->Load(path);
      if (res) printf("Shared lib %s has been dynamically linked\n",path);
   }
   delete [] path;
}

//______________________________________________________________________________
void TFile::ReadStreamerInfo()
{
// Read the list of StreamerInfo from this file
// The key with name holding the list of TStreamerInfo objects is read.
// The corresponding TClass objects are updated.

   TList *list = 0;
   if (fSeekInfo > 0 && fSeekInfo < fEND) {
      TKey *key = new TKey();
      char *buffer = new char[fNbytesInfo+1];
      char *buf    = buffer;
      Seek(fSeekInfo);
      ReadBuffer(buf,fNbytesInfo);
      key->ReadBuffer(buf);
      list = (TList*)key->ReadObj();
      delete [] buffer;
      delete key;
   } else {
      list = (TList*)Get("StreamerInfo"); //for versions 2.26 (never released)
   }

   if (list == 0) return;
   if (gDebug > 0) printf("Calling ReadStreamerInfo for file: %s\n",GetName());
//list->Dump();
   // loop on all TStreamerInfo classes
   TStreamerInfo *info;
   TIter next(list);
   while ((info = (TStreamerInfo*)next())) {
      if (info->IsA() != TStreamerInfo::Class()) {
         Warning("ReadStreamerInfo"," not a TStreamerInfo object");
         continue;
      }
      info->BuildCheck();
      Int_t uid = info->GetNumber();
      if (uid >= 0) fClassIndex->fArray[uid] = 1;
      if (gDebug > 0) printf(" -class: %s version: %d info read at slot %d\n",info->GetName(), info->GetClassVersion(),uid);
   }
   fClassIndex->fArray[0] = 0;
   list->Clear();  //this will delete all TStreamerInfo objects with kCanDelete bit set
   delete list;
}

//______________________________________________________________________________
void TFile::ShowStreamerInfo()
{
// Show the StreamerInfo of all classes written to this file.

   TList *list = GetStreamerInfoList();

   if (!list) return;
   
   list->ls();
   delete list;
}

//______________________________________________________________________________
void TFile::WriteStreamerInfo()
{
//  Write the list of TStreamerInfo as a single object in this file
//  The class Streamer description for all classes written to this file
//  is saved.
//  see class TStreamerInfo

   //if (!gFile) return;
   if (!fWritable) return;
   if (!fClassIndex) return;
   //no need to update the index if no new classes added to the file
   if (fClassIndex->fArray[0] == 0) return;
   if (gDebug > 0) printf("Calling WriteStreamerInfo for file: %s\n",GetName());

   // build a temporary list with the marked files
   TIter next(gROOT->GetListOfStreamerInfo());
   TStreamerInfo *info;
   TList list;

   while ((info = (TStreamerInfo*)next())) {
      Int_t uid = info->GetNumber();
      if (fClassIndex->fArray[uid]) list.Add(info);
      if (gDebug > 0) printf(" -class: %s info number %d saved\n",info->GetName(),uid);
   }
   if (list.GetSize() == 0) return;
   fClassIndex->fArray[0] = 0;

   // always write with compression on
   Int_t compress = fCompress;
   fCompress = 1;
   TFile * fileSave = gFile;
   TDirectory *dirSave = gDirectory;
   gFile = this;
   gDirectory = this;

   //free previous StreamerInfo record
   if (fSeekInfo) MakeFree(fSeekInfo,fSeekInfo+fNbytesInfo-1);
   //Create new key
   TKey key(&list,"StreamerInfo",GetBestBuffer());
   fKeys->Remove(&key);
   fSeekInfo   = key.GetSeekKey();
   fNbytesInfo = key.GetNbytes();
   SumBuffer(key.GetObjlen());
   key.WriteFile(0);

   gFile = fileSave;
   gDirectory = dirSave;
   fCompress = compress;
}

//______________________________________________________________________________
TFile *TFile::Open(const char *name, Option_t *option, const char *ftitle,
                   Int_t compress, Int_t netopt)
{
   // Static member function allowing the creation/opening of either a
   // TFile, TNetFile, TWebFile or any TFile derived class for which an
   // plugin library handler has been registered with the plugin manager
   // (for the plugin manager see the TPluginManager class). The returned
   // type of TFile depends on the file name. If the file starts with
   // "root:", "roots:" or "rootk:" a TNetFile object will be returned,
   // with "http:" a TWebFile, with "file:" a local TFile, etc. (see the
   // list of TFile plugin handlers for regular axpressions that will be
   // checked) and as last a local file will be tried.
   // Before opening a file via TNetFile a check is made to see if the URL
   // specifies a local file. If that is the case the file will be opened
   // via a normal TFile. To force the opening of a local file via a
   // TNetFile use either TNetFile directly or specify as host "localhost".
   // For the meaning of the options and other arguments see the constructors
   // of the individual file classes. In case of error returns 0.

   TPluginHandler *h;
   TFile *f = 0;

   if (!strncmp(name, "root:", 5) || !strncmp(name, "roots:", 6) ||
       !strncmp(name, "rootk:", 6)) {
      TUrl url(name);
      TInetAddress a(gSystem->GetHostByName(url.GetHost()));
      TInetAddress b(gSystem->GetHostByName(gSystem->HostName()));
      if (strcmp(a.GetHostName(), b.GetHostName()))
         f = new TNetFile(name, option, ftitle, compress, netopt);
      else {
         const char *fname = url.GetFile();
         if (fname[1] == '/' || fname[1] == '~' || fname[1] == '$')
            f = new TFile(&fname[1], option, ftitle, compress);
         else
            f = new TFile(Form("%s%s", gSystem->HomeDirectory(), fname),
                          option, ftitle, compress);
      }
   } else if (!strncmp(name, "http:", 5))
      f = new TWebFile(name);
   else if (!strncmp(name, "file:", 5))
      f = new TFile(name+5, option, ftitle, compress);
   else if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name))) {
      if (h->LoadPlugin() == -1)
         return 0;
      f = (TFile*) gROOT->ProcessLineFast(Form("new %s(\"%s\",\"%s\",\"%s\",%d)",
                   h->GetClass(), name, option, ftitle, compress));
   } else
      f = new TFile(name, option, ftitle, compress);

   return f;
}

//______________________________________________________________________________
Int_t TFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // Interface to system open. All arguments like in POSIX open().

   return ::open(pathname, flags, mode);
}

//______________________________________________________________________________
Int_t TFile::SysClose(Int_t fd)
{
   // Interface to system close. All arguments like in POSIX close().

   return ::close(fd);
}

//______________________________________________________________________________
Int_t TFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   // Interface to system read. All arguments like in POSIX read().

   return ::read(fd, buf, len);
}

//______________________________________________________________________________
Int_t TFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   // Interface to system write. All arguments like in POSIX write().

   return ::write(fd, buf, len);
}

//______________________________________________________________________________
Seek_t TFile::SysSeek(Int_t fd, Seek_t offset, Int_t whence)
{
   // Interface to system lseek. All arguments like in POSIX lseek()
   // except that the offset and return value are of a type which will
   // be able to handle 64 bit file systems in the future.

   return ::lseek(fd, offset, whence);
}

//______________________________________________________________________________
Int_t TFile::SysStat(Int_t, Long_t *id, Long_t *size, Long_t *flags,
                     Long_t *modtime)
{
   // Return file stat information. The interface and return value is
   // identical to TSystem::GetPathInfo(). The function returns 0 in
   // case of success and 1 if the file could not be stat'ed.

   return gSystem->GetPathInfo(GetName(), id, size, flags, modtime);
}

//______________________________________________________________________________
Int_t TFile::SysSync(Int_t fd)
{
   // Interface to system fsync. All arguments like in POSIX fsync().

#ifndef WIN32
   return ::fsync(fd);
#else
   return 0;
#endif
}

//______________________________________________________________________________
Double_t TFile::GetFileBytesRead() { return fgBytesRead; }

//______________________________________________________________________________
Double_t TFile::GetFileBytesWritten() { return fgBytesWrite; }

//______________________________________________________________________________
void TFile::SetFileBytesRead(Double_t bytes){ fgBytesRead = bytes; }

//______________________________________________________________________________
void TFile::SetFileBytesWritten(Double_t bytes){ fgBytesWrite = bytes; }
