// @(#)root/base:$Name:  $:$Id: TFile.cxx,v 1.153 2006/04/18 14:23:20 rdm Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfig.h"

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

#include "Bytes.h"
#include "Riostream.h"
#include "Strlen.h"
#include "TArrayC.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TDatime.h"
#include "TError.h"
#include "TFile.h"
#include "TFree.h"
#include "TInterpreter.h"
#include "TKey.h"
#include "TNetFile.h"
#include "TPluginManager.h"
#include "TProcessUUID.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TStreamerInfo.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TVirtualPerfStats.h"
#include "TWebFile.h"
#include "TArchiveFile.h"
#include "TEnv.h"
#include "TVirtualMutex.h"


TFile *gFile;                 //Pointer to current file


Long64_t TFile::fgBytesRead  = 0;
Long64_t TFile::fgBytesWrite = 0;
TList *TFile::fgAsyncOpenRequests = 0;

const Int_t kBEGIN = 100;

ClassImp(TFile)

//*-*x17 macros/layout_file

//______________________________________________________________________________
TFile::TFile() : TDirectory(), fInfoCache(0)
{
   // File default Constructor.

   fD             = -1;
   fFree          = 0;
   fWritten       = 0;
   fSumBuffer     = 0;
   fSum2Buffer    = 0;
   fClassIndex    = 0;
   fCache         = 0;
   fProcessIDs    = 0;
   fNProcessIDs   = 0;
   fOffset        = 0;
   fArchive       = 0;
   fArchiveOffset = 0;
   fIsRootFile    = kTRUE;
   fIsArchive     = kFALSE;
   fInitDone      = kFALSE;
   fAsyncHandle   = 0;
   fAsyncOpenStatus   = kAOSNotAsync;

   if (gDebug)
      Info("TFile", "default ctor");
}

//1_____________________________________________________________________________
TFile::TFile(const char *fname1, Option_t *option, const char *ftitle, Int_t compress)
           : TDirectory(), fUrl(fname1,kTRUE), fInfoCache(0)
{
   // Opens or creates a local ROOT file whose name is fname1. It is
   // recommended to specify fname1 as "<file>.root". The suffix ".root"
   // will be used by object browsers to automatically identify the file as
   // a ROOT file. If the constructor fails in any way IsZombie() will
   // return true. Use IsOpen() to check if the file is (still) open.
   //
   // To open non-local files use the static TFile::Open() method, that
   // will take care of opening the files using the correct remote file
   // access plugin.
   //
   // If option = NEW or CREATE   create a new file and open it for writing,
   //                             if the file already exists the file is
   //                             not opened.
   //           = RECREATE        create a new file, if the file already
   //                             exists it will be overwritten.
   //           = UPDATE          open an existing file for writing.
   //                             if no file exists, it is created.
   //           = READ            open an existing file for reading.
   //           = NET             used by derived remote file access
   //                             classes, not a user callable option
   //           = WEB             used by derived remote http access
   //                             class, not a user callable option
   //
   // The file can be specified as a URL of the form:
   //    file:///user/rdm/bla.root or file:/user/rdm/bla.root
   //
   // The file can also be a member of an archive, in which case it is
   // specified as:
   //    multi.zip#file.root or multi.zip#0
   // which will open file.root which is a member of the file multi.zip
   // archive or member 1 from the archive. For more on archive file
   // support see the TArchiveFile class.
   //
   // TFile and its remote access plugins can also be used to open any
   // file, i.e. also non ROOT files, using:
   //    file.tar?filetype=raw
   // This is convenient because the many remote file access plugins allow
   // easy access to/from the many different mass storage systems.
   //
   // The title of the file (ftitle) will be shown by the ROOT browsers.
   //
   // A ROOT file (like a Unix file system) may contain objects and
   // directories. There are no restrictions for the number of levels
   // of directories.
   //
   // A ROOT file is designed such that one can write in the file in pure
   // sequential mode (case of BATCH jobs). In this case, the file may be
   // read sequentially again without using the file index written
   // at the end of the file. In case of a job crash, all the information
   // on the file is therefore protected.
   //
   // A ROOT file can be used interactively. In this case, one has the
   // possibility to delete existing objects and add new ones.
   // When an object is deleted from the file, the freed space is added
   // into the FREE linked list (fFree). The FREE list consists of a chain
   // of consecutive free segments on the file. At the same time, the first
   // 4 bytes of the freed record on the file are overwritten by GAPSIZE
   // where GAPSIZE = -(Number of bytes occupied by the record).
   //
   // Option compress is used to specify the compression level:
   //  compress = 0 objects written to this file will not be compressed.
   //  compress = 1 minimal compression level but fast.
   //  ....
   //  compress = 9 maximal compression level but slow.
   //
   // Note that the compression level may be changed at any time.
   // The new compression level will only apply to newly written objects.
   // The function TFile::Map() shows the compression factor
   // for each object written to this file.
   // The function TFile::GetCompressionFactor returns the global
   // compression factor for this file.
   //
   // In case the file does not exist or is not a valid ROOT file,
   // it is made a Zombie. One can detect this situation with a code like:
   //    TFile f("file.root");
   //    if (f.IsZombie()) {
   //       cout << "Error opening file" << endl;
   //       exit(-1);
   //    }
   //
   // A ROOT file is a suite of consecutive data records (TKey's) with
   // the following format (see also the TKey class). If the key is
   // located past the 32 bit file limit (> 2 GB) then some fields will
   // be 8 instead of 4 bytes:
   //    1->4            Nbytes    = Length of compressed object (in bytes)
   //    5->6            Version   = TKey version identifier
   //    7->10           ObjLen    = Length of uncompressed object
   //    11->14          Datime    = Date and time when object was written to file
   //    15->16          KeyLen    = Length of the key structure (in bytes)
   //    17->18          Cycle     = Cycle of key
   //    19->22 [19->26] SeekKey   = Pointer to record itself (consistency check)
   //    23->26 [27->34] SeekPdir  = Pointer to directory header
   //    27->27 [35->35] lname     = Number of bytes in the class name
   //    28->.. [36->..] ClassName = Object Class Name
   //    ..->..          lname     = Number of bytes in the object name
   //    ..->..          Name      = lName bytes with the name of the object
   //    ..->..          lTitle    = Number of bytes in the object title
   //    ..->..          Title     = Title of the object
   //    ----->          DATA      = Data bytes associated to the object
   //
   // The first data record starts at byte fBEGIN (currently set to kBEGIN).
   // Bytes 1->kBEGIN contain the file description, when fVersion >= 1000000
   // it is a large file (> 2 GB) and the offsets will be 8 bytes long and
   // fUnits will be set to 8:
   //    1->4            "root"      = Root file identifier
   //    5->8            fVersion    = File format version
   //    9->12           fBEGIN      = Pointer to first data record
   //    13->16 [13->20] fEND        = Pointer to first free word at the EOF
   //    17->20 [21->28] fSeekFree   = Pointer to FREE data record
   //    21->24 [29->32] fNbytesFree = Number of bytes in FREE data record
   //    25->28 [33->36] nfree       = Number of free data records
   //    29->32 [37->40] fNbytesName = Number of bytes in TNamed at creation time
   //    33->33 [41->41] fUnits      = Number of bytes for file pointers
   //    34->37 [42->45] fCompress   = Zip compression level
   //    38->41 [46->53] fSeekInfo   = Pointer to TStreamerInfo record
   //    42->45 [54->57] fNbytesInfo = Number of bytes in TStreamerInfo record
   //    46->63 [58->75] fUUID       = Universal Unique ID
//Begin_Html
/*
<img src="gif/file_layout.gif">
*/
//End_Html
   //
   // The structure of a directory is shown in TDirectory::TDirectory

   if (!gROOT)
      ::Fatal("TFile::TFile", "ROOT system not initialized");

   // store original name and title
   SetName(fname1);
   SetTitle(ftitle);

   // accept also URL like "file:..." syntax
   fname1 = fUrl.GetFile();

   // if option contains filetype=raw then go into raw file mode
   fIsRootFile = kTRUE;
   if (strstr(fUrl.GetOptions(), "filetype=raw"))
      fIsRootFile = kFALSE;

   // Init initialization control flag
   fInitDone   = kFALSE;

   // We are opening synchronously
   fAsyncHandle = 0;
   fAsyncOpenStatus = kAOSNotAsync;

   TDirectory::Build(this, 0);

   fD          = -1;
   fFree       = 0;
   fVersion    = gROOT->GetVersionInt();  //ROOT version in integer format
   fUnits      = 4;
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
   fNProcessIDs= 0;
   fOffset     = 0;

   fOption.ToUpper();

   fArchiveOffset = 0;
   fIsArchive     = kFALSE;
   fArchive = TArchiveFile::Open(fUrl.GetUrl(), this);
   if (fArchive) {
      fname1 = fArchive->GetArchiveName();
      // if no archive member is specified then this TFile is just used
      // to read the archive contents
      if (!strlen(fArchive->GetMemberName()))
         fIsArchive = kTRUE;
   }

   if (fOption == "NET")
      return;

   if (fOption == "WEB") {
      fOption   = "READ";
      fWritable = kFALSE;
      return;
   }

   if (fOption == "NEW")
      fOption = "CREATE";

   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   Bool_t devnull = kFALSE;

   if (!fname1 || !strlen(fname1)) {
      Error("TFile", "file name is not specified");
      goto zombie;
   }

   // support dumping to /dev/null on UNIX
   if (!strcmp(fname1, "/dev/null") &&
       !gSystem->AccessPathName(fname1, kWritePermission)) {
      devnull  = kTRUE;
      create   = kTRUE;
      recreate = kFALSE;
      update   = kFALSE;
      read     = kFALSE;
      fOption  = "CREATE";
      SetBit(kDevNull);
   }

   const char *fname;
   if ((fname = gSystem->ExpandPathName(fname1))) {
      SetName(fname);
      delete [] (char*)fname;
      fRealName = GetName();
      fname = fRealName.Data();
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
   if (create && !devnull && !gSystem->AccessPathName(fname, kFileExists)) {
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

   // Connect to file system stream
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
TFile::TFile(const TFile &file) : TDirectory(), fInfoCache(0)
{
   // Copy constructor.
   ((TFile&)file).Copy(*this);
}

//______________________________________________________________________________
TFile::~TFile()
{
   // File destructor.

   Close();

   SafeDelete(fProcessIDs);
   SafeDelete(fFree);
   SafeDelete(fCache);
   SafeDelete(fArchive);
   SafeDelete(fInfoCache);
   SafeDelete(fAsyncHandle);

   R__LOCKGUARD2(gROOTMutex);
   gROOT->GetListOfFiles()->Remove(this);
   gROOT->GetUUIDs()->RemoveUUID(GetUniqueID());

   if (gDebug)
      Info("~TFile", "dtor called for %s [%d]", GetName(),this);
}

//______________________________________________________________________________
void TFile::Init(Bool_t create)
{
   // Initialize a TFile object.
   // TFile implementations providing asynchronous open functionality need to
   // override this method to run the appropriate checks before calling this
   // standard initialization part. See TXNetFile::Init for an example.

   if (fInitDone)
      // Already called once
      return;
   fInitDone = kTRUE;

   if (!fIsRootFile) {
      gDirectory = gROOT;
      return;
   }

   if (fArchive) {
      if (fOption != "READ") {
         Error("Init", "archive %s can only be opened in read mode", GetName());
         delete fArchive;
         fArchive = 0;
         fIsArchive = kFALSE;
         goto zombie;
      }

      fArchive->OpenArchive();

      if (fIsArchive) return;

      TString full = fRealName != "" ? fRealName.Data() : GetName();
      full += "#"; full += fArchive->GetMemberName();
      SetName(full);

      if (fArchive->SetCurrentMember() != -1)
         fArchiveOffset = fArchive->GetMemberFilePosition();
      else {
         Error("Init", "member %s not found in archive %s",
               fArchive->GetMemberName(), fArchive->GetArchiveName());
         delete fArchive;
         fArchive = 0;
         fIsArchive = kFALSE;
         goto zombie;
      }
   }

   Int_t nfree;
   fBEGIN = (Long64_t)kBEGIN;    //First used word in file following the file header

   // make newly opened file the current file and directory
   cd();

//*-*---------------NEW file
   if (create) {
      fFree        = new TList;
      fEND         = fBEGIN;    //Pointer to end of file
      new TFree(fFree, fBEGIN, Long64_t(kStartBigFile));  //Create new free list

//*-* Write Directory info
      Int_t namelen= TNamed::Sizeof();
      Int_t nbytes = namelen + TDirectory::Sizeof();
      TKey *key    = new TKey(fName, fTitle, IsA(), nbytes, this);
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
      char *header = new char[kBEGIN];
      Seek(0);
      ReadBuffer(header, kBEGIN);

      // make sure this is a ROOT file
      if (strncmp(header, "root", 4)) {
         Error("Init", "%s not a ROOT file", GetName());
         delete [] header;
         goto zombie;
      }

      char *buffer = header + 4;    // skip the "root" file identifier
      frombuf(buffer, &fVersion);
      Int_t headerLength;
      frombuf(buffer, &headerLength);
      fBEGIN = (Long64_t)headerLength;
      if (fVersion < 1000000) { //small file
         Int_t send,sfree,sinfo;
         frombuf(buffer, &send);         fEND     = (Long64_t)send;
         frombuf(buffer, &sfree);        fSeekFree= (Long64_t)sfree;
         frombuf(buffer, &fNbytesFree);
         frombuf(buffer, &nfree);
         frombuf(buffer, &fNbytesName);
         frombuf(buffer, &fUnits );
         frombuf(buffer, &fCompress);
         frombuf(buffer, &sinfo);        fSeekInfo = (Long64_t)sinfo;
         frombuf(buffer, &fNbytesInfo);
      } else { // new format to support large files
         frombuf(buffer, &fEND);
         frombuf(buffer, &fSeekFree);
         frombuf(buffer, &fNbytesFree);
         frombuf(buffer, &nfree);
         frombuf(buffer, &fNbytesName);
         frombuf(buffer, &fUnits );
         frombuf(buffer, &fCompress);
         frombuf(buffer, &fSeekInfo);
         frombuf(buffer, &fNbytesInfo);
      }
      fSeekDir = fBEGIN;
      delete [] header;
//*-*-------------Read Free segments structure if file is writable
      if (fWritable) {
         fFree = new TList;
         if (fSeekFree > fBEGIN) {
            ReadFree();
         } else {
            Warning("Init","file %s probably not closed, cannot read free segments",GetName());
         }
      }
//*-*-------------Read directory info
      Int_t nbytes = fNbytesName + TDirectory::Sizeof();
      header       = new char[nbytes];
      buffer       = header;
      Seek(fBEGIN);
      ReadBuffer(buffer,nbytes);
      buffer = header+fNbytesName;
      Version_t version,versiondir;
      frombuf(buffer,&version); versiondir = version%1000;
      fDatimeC.ReadBuffer(buffer);
      fDatimeM.ReadBuffer(buffer);
      frombuf(buffer, &fNbytesKeys);
      frombuf(buffer, &fNbytesName);
      Int_t nk = sizeof(Int_t) +sizeof(Version_t) +2*sizeof(Int_t)+2*sizeof(Short_t)
                +2*sizeof(Int_t);
      if (version > 1000) {
         frombuf(buffer, &fSeekDir);
         frombuf(buffer, &fSeekParent);
         frombuf(buffer, &fSeekKeys);
      } else {
         Int_t sdir,sparent,skeys;
         frombuf(buffer, &sdir);    fSeekDir    = (Long64_t)sdir;
         frombuf(buffer, &sparent); fSeekParent = (Long64_t)sparent;
         frombuf(buffer, &skeys);   fSeekKeys   = (Long64_t)skeys;
      }
      if (versiondir > 1) fUUID.ReadBuffer(buffer);

//*-*---------read TKey::FillBuffer info
      buffer = header+nk;
      TString cname;
      cname.ReadBuffer(buffer);
      cname.ReadBuffer(buffer); // fName.ReadBuffer(buffer); file may have been renamed
      fTitle.ReadBuffer(buffer);
      delete [] header;
      if (fNbytesName < 10 || fNbytesName > 2000) {
         Error("Init","cannot read directory info of file %s", GetName());
         goto zombie;
      }
//*-* -------------Check if file is truncated
      Long64_t size;
      if ((size = GetSize()) == -1) {
         Error("Init", "cannot stat the file %s", GetName());
         goto zombie;
      }
//*-* -------------Read keys of the top directory

      if (fSeekKeys > fBEGIN && fEND <= size) {
         //normal case. Recover only if file has no keys
         TDirectory::ReadKeys();
         gDirectory = this;
         if (!GetNkeys()) Recover();
      } else if ((fBEGIN+nbytes == fEND) && (fEND == size)) {
         //the file might be open by another process and nothing written to the file yet
         Warning("Init","file %s has no keys", GetName());
         gDirectory = this;
      } else {
         //something had been written to the file. Trailer is missing, must recover
         if (fEND > size) {
            Error("Init","file %s is truncated at %lld bytes: should be %lld, trying to recover",
                  GetName(), size, fEND);
         } else {
            Warning("Init","file %s probably not closed, trying to recover",
                    GetName());
         }
         Int_t nrecov = Recover();
         if (nrecov) {
            Warning("Init", "successfully recovered %d keys", nrecov);
         } else {
            Warning("Init", "no keys recovered, file has been made a Zombie");
            goto zombie;
         }
      }
   }

   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfFiles()->Add(this);
      gROOT->GetUUIDs()->AddUUID(fUUID,this);

   // Create StreamerInfo index
      Int_t lenIndex = gROOT->GetListOfStreamerInfo()->GetSize()+1;
      if (lenIndex < 5000) lenIndex = 5000;
      fClassIndex = new TArrayC(lenIndex);
      if (fSeekInfo > fBEGIN) ReadStreamerInfo();
   }

   // Count number of TProcessIDs in this file
   {
      TIter next(fKeys);
      TKey *key;
      while ((key = (TKey*)next())) {
         if (!strcmp(key->GetClassName(),"TProcessID")) fNProcessIDs++;
      }
      fProcessIDs = new TObjArray(fNProcessIDs+1);
      return;
   }

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
void TFile::Close(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Close a file*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============
// if option == "R", all TProcessIDs referenced by this file are deleted.
// Calling TFile::Close("R") might be necessary in case one reads a long list
// of files having TRef, writing some of the referenced objects or TRef
// to a new file. If the TRef or referenced objects of the file being closed
// will not be referenced again, it is possible to minimize the size
// of the TProcessID data structures in memory by forcing a delete of
// the unused TProcessID.

   TString opt = option;
   opt.ToLower();

   if (!IsOpen()) return;

   if (fIsArchive || !fIsRootFile) {
      SysClose(fD);
      fD = -1;
      return;
   }

   if (IsWritable()) {
      WriteStreamerInfo();
   }

   delete fClassIndex;
   fClassIndex = 0;

   TCollection::StartGarbageCollection();

   TDirectory *cursav = gDirectory;
   cd();

   if (cursav == this || (cursav && cursav->GetFile() == this)) {
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
   TList pidDeleted;
   TIter next(fProcessIDs);
   TProcessID *pid;
   while ((pid = (TProcessID*)next())) {
      if (!pid->DecrementCount()) {
         if (pid != TProcessID::GetSessionProcessID()) pidDeleted.Add(pid);
      } else if(opt.Contains("r")) {
         pid->Clear();
      }
   }
   pidDeleted.Delete();

   R__LOCKGUARD2(gROOTMutex);
   gROOT->GetListOfFiles()->Remove(this);

   TCollection::EmptyGarbageCollection();
}

//____________________________________________________________________________________
TKey* TFile::CreateKey(TDirectory* mother, const TObject* obj, const char* name, Int_t bufsize)
{
   // Creates key for object and converts data to buffer.
   return new TKey(obj, name, bufsize, mother);
}

//____________________________________________________________________________________
TKey* TFile::CreateKey(TDirectory* mother, const void* obj, const TClass* cl, const char* name, Int_t bufsize)
{
   // Creates key for object and converts data to buffer.
   return new TKey(obj, cl, name, bufsize, mother);
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
      Info("Delete", "deleting name = %s", namecycle);

   TDirectory::Delete(namecycle);
}

//______________________________________________________________________________
void TFile::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Fill Graphics Structure and Paint-*-*-*-*-*-*-*-*-*-*
//*-*                    =================================
// Loop on all objects (memory or file) and all subdirectories
//

   GetList()->R__FOR_EACH(TObject,Draw)(option);
}

//______________________________________________________________________________
void TFile::DrawMap(const char *keys, Option_t *option)
{
// Draw map of objects in this file

   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TFileDrawMap"))) {
      if (h->LoadPlugin() == -1)
         return;
      h->ExecPlugin(3, this, keys, option);
   }
}

//______________________________________________________________________________
void TFile::Flush()
{
   // Synchronize a file's in-core and on-disk states.

   if (IsOpen() && fWritable) {
      if (SysSync(fD) < 0) {
         // Write the system error only once for this file
         SetBit(kWriteError); SetWritable(kFALSE);
         SysError("Flush", "error flushing file %s", GetName());
      }
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
   char    *header = new char[fBEGIN];
   char    *buffer;
   Long64_t   idcur = fBEGIN;
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
Int_t TFile::GetRecordHeader(char *buf, Long64_t first, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen)
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
      Warning("GetRecordHeader","%s: parameter maxbytes = %d must be >= 4",
              GetName(), nread);
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
Long64_t TFile::GetSize() const
{
   // Returns the current file size. Returns -1 in case the file could not
   // be stat'ed.

   Long64_t size;

   if (fArchive && fArchive->GetMember()) {
      size = fArchive->GetMember()->GetDecompressedSize();
   } else {
      Long_t id, flags, modtime;
      if (const_cast<TFile*>(this)->SysStat(fD, &id, &size, &flags, &modtime)) {
         Error("GetSize", "cannot stat the file %s", GetName());
         return -1;
      }
   }
   return size;
}

//______________________________________________________________________________
const TList *TFile::GetStreamerInfoCache()
{
   // Returns the cached list of StreamerInfos used in this file.
   return fInfoCache ?  fInfoCache : (fInfoCache=GetStreamerInfoList());
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
      TDirectory::TContext ctx(gDirectory,this); // gFile and gDirectory used in ReadObj

      TKey *key = new TKey(this);
      char *buffer = new char[fNbytesInfo+1];
      char *buf    = buffer;
      Seek(fSeekInfo);
      ReadBuffer(buf,fNbytesInfo);
      key->ReadKeyBuffer(buf);
      list = (TList*)key->ReadObj();
      if (list) list->SetOwner();
      delete [] buffer;
      delete key;
   } else {
      list = (TList*)Get("StreamerInfo"); //for versions 2.26 (never released)
   }

   if (list == 0) {
      Info("GetStreamerInfoList", "cannot find the StreamerInfo record in file %s",
           GetName());
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
void TFile::MakeFree(Long64_t first, Long64_t last)
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
   Long64_t nfirst = newfree->GetFirst();
   Long64_t nlast  = newfree->GetLast();
   Long64_t nbytesl= nlast-nfirst+1;
   if (nbytesl > 2000000000) nbytesl = 2000000000;
   Int_t nbytes    = -Int_t (nbytesl);
   Int_t nb        = sizeof(Int_t);
   char * buffer   = new char[nb];
   char * psave    = buffer;
   tobuf(buffer, nbytes);
   if (last == fEND-1) fEND = nfirst;
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
   Long64_t seekkey,seekpdir;
   char    *buffer;
   char     nwhc;
   Long64_t idcur = fBEGIN;

   nwheader = 64;
   Int_t nread = nwheader;

   char header[kBEGIN];
   char classname[512];

   while (idcur < fEND) {
      Seek(idcur);
      if (idcur+nread >= fEND) nread = fEND-idcur-1;
      ReadBuffer(header, nread);
      buffer=header;
      frombuf(buffer, &nbytes);
      if (!nbytes) {
         Printf("Address = %lld\tNbytes = %d\t=====E R R O R=======", idcur, nbytes);
         break;
      }
      if (nbytes < 0) {
         Printf("Address = %lld\tNbytes = %d\t=====G A P===========", idcur, nbytes);
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
      if (versionkey > 1000) {
         frombuf(buffer, &seekkey);
         frombuf(buffer, &seekpdir);
      } else {
         Int_t skey,sdir;
         frombuf(buffer, &skey);  seekkey  = (Long64_t)skey;
         frombuf(buffer, &sdir);  seekpdir = (Long64_t)sdir;
      }
      frombuf(buffer, &nwhc);
      int i;
      for (i = 0;i < nwhc; i++) frombuf(buffer, &classname[i]);
      classname[(int)nwhc] = '\0'; //cast to avoid warning with gcc3.4
      if (idcur == fSeekFree) strcpy(classname,"FreeSegments");
      if (idcur == fSeekInfo) strcpy(classname,"StreamerInfo");
      if (idcur == fSeekKeys) strcpy(classname,"KeysList");
      TDatime::GetDateTime(datime, date, time);
      if (objlen != nbytes-keylen) {
         Float_t cx = Float_t(objlen+keylen)/Float_t(nbytes);
         //Printf("%d/%06d  At:%-8d  N=%-8d  %-14s CX = %5.2f",date,time,idcur,nbytes,classname,cx);
         Printf("%d/%06d  At:%lld  N=%-8d  %-14s CX = %5.2f",date,time,idcur,nbytes,classname,cx);
      } else {
         //Printf("%d/%06d  At:%-8d  N=%-8d  %-14s",date,time,idcur,nbytes,classname);
         Printf("%d/%06d  At:%lld  N=%-8d  %-14s",date,time,idcur,nbytes,classname);
      }
      idcur += nbytes;
   }
   //Printf("%d/%06d  At:%-8d  N=%-8d  %-14s",date,time,idcur,1,"END");
   Printf("%d/%06d  At:%lld  N=%-8d  %-14s",date,time,idcur,1,"END");
}

//______________________________________________________________________________
void TFile::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Paint all objects in the file*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//

   GetList()->R__FOR_EACH(TObject,Paint)(option);
}

//______________________________________________________________________________
void TFile::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print all objects in the file*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//

   Printf("TFile: name=%s, title=%s, option=%s", GetName(), GetTitle(), GetOption());
   GetList()->R__FOR_EACH(TObject,Print)(option);
}

//______________________________________________________________________________
Bool_t TFile::ReadBuffer(char *buf, Int_t len)
{
   // Read a buffer from the file. This is the basic low level read operation.
   // Returns kTRUE in case of failure.

   if (IsOpen()) {
      ssize_t siz;

      Double_t start = 0;
      if (gPerfStats != 0) start = TTimeStamp();


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

      if (gPerfStats != 0) {
         gPerfStats->FileReadEvent(this, len, double(TTimeStamp())-start);
      }
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TFile::ReadBufferViaCache(char *buf, Int_t len)
{
   // Read buffer via cache. Returns 0 if cache is not active, 1 in case
   // read via cache was successful, 2 in case read via cache failed.

   if (!fCache) return 0;

   Int_t st;
   Long64_t off = GetRelOffset();
   if ((st = fCache->ReadBuffer(off, buf, len)) < 0) {
      Error("ReadBuffer", "error reading from cache");
      return 2;
   }
   if (st > 0) {
      // fOffset might have been changed via TCache::ReadBuffer(), reset it
      Seek(off + len);
      return 1;
   }
   return 0;
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

   TKey *headerfree = new TKey(fSeekFree, fNbytesFree, this);
   headerfree->ReadFile();
   char *buffer = headerfree->GetBuffer();
   headerfree->ReadKeyBuffer(buffer);
   buffer = headerfree->GetBuffer();
   while (1) {
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
   Long64_t seekkey,seekpdir;
   char     header[1024];
   char    *buffer, *bufread;
   char     nwhc;
   Long64_t idcur = fBEGIN;

   Long64_t size;
   if ((size = GetSize()) == -1) {
      Error("Recover", "cannot stat the file %s", GetName());
      return 0;
   }

   fEND = Long64_t(size);

   if (fWritable && !fFree) fFree  = new TList;

   TKey *key;
   Int_t nrecov = 0;
   nwheader = 1024;
   Int_t nread = nwheader;

   while (idcur < fEND) {
      Seek(idcur);
      if (idcur+nread >= fEND) nread = fEND-idcur-1;
      ReadBuffer(header, nread);
      buffer  = header;
      bufread = header;
      frombuf(buffer, &nbytes);
      if (!nbytes) {
         Printf("Address = %lld\tNbytes = %d\t=====E R R O R=======", idcur, nbytes);
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
      if (versionkey > 1000) {
         frombuf(buffer, &seekkey);
         frombuf(buffer, &seekpdir);
      } else {
         Int_t skey,sdir;
         frombuf(buffer, &skey);  seekkey  = (Long64_t)skey;
         frombuf(buffer, &sdir);  seekpdir = (Long64_t)sdir;
      }
      frombuf(buffer, &nwhc);
      char *classname = 0;
      if (nwhc <= 0 || nwhc > 100) break;
      classname = new char[nwhc+1];
      int i;
      for (i = 0;i < nwhc; i++) frombuf(buffer, &classname[i]);
      classname[nwhc] = '\0';
      TDatime::GetDateTime(datime, date, time);
      if (seekpdir == fSeekDir && strcmp(classname,"TFile") && strcmp(classname,"TBasket")) {
         key = new TKey(this);
         key->ReadKeyBuffer(bufread);
         if (!strcmp(key->GetName(),"StreamerInfo")) {
            fSeekInfo = seekkey;
            SafeDelete(fInfoCache);
            fNbytesInfo = nbytes;
         } else {
            AppendKey(key);
            nrecov++;
            SetBit(kRecovered);
            Info("Recover", "%s, recovered key %s:%s at address %lld",GetName(),key->GetClassName(),key->GetName(),idcur);
         }
      }
      delete [] classname;
      idcur += nbytes;
   }
   if (fWritable) {
      Long64_t max_file_size = Long64_t(kStartBigFile);
      if (max_file_size < fEND) max_file_size = fEND+1000000000;
      new TFree(fFree,fEND,max_file_size);
      if (nrecov) Write();
   }
   return nrecov;
}

//______________________________________________________________________________
Int_t TFile::ReOpen(Option_t *mode)
{
   // Reopen a file with a different access mode, like from READ to
   // UPDATE or from NEW, CREATE, RECREATE, UPDATE to READ. Thus the
   // mode argument can be either "READ" or "UPDATE". The method returns
   // 0 in case the mode was successfully modified, 1 in case the mode
   // did not change (was already as requested or wrong input arguments)
   // and -1 in case of failure, in which case the file cannot be used
   // anymore. The current directory (gFile) is changed to this file.

   cd();

   TString opt = mode;
   opt.ToUpper();

   if (opt != "READ" && opt != "UPDATE") {
      Error("ReOpen", "mode must be either READ or UPDATE, not %s", opt.Data());
      return 1;
   }

   if (opt == fOption || (opt == "UPDATE" && fOption == "CREATE"))
      return 1;

   if (opt == "READ") {
      // switch to READ mode

      // flush data still in the pipeline and close the file
      if (IsOpen() && IsWritable()) {
         WriteStreamerInfo();

         // save directory key list and header
         Save();

         TFree *f1 = (TFree*)fFree->First();
         if (f1) {
            WriteFree();       // write free segments linked list
            WriteHeader();     // now write file header
         }
         if (fCache) fCache->Flush();

         // delete free segments from free list
         if (fFree) {
            fFree->Delete();
            SafeDelete(fFree);
         }

         SysClose(fD);
         fD = -1;

         SetWritable(kFALSE);
      }

      // open in READ mode
      fOption = opt;    // set fOption before SysOpen() for TNetFile
#ifndef WIN32
      fD = SysOpen(fRealName, O_RDONLY, 0644);
#else
      fD = SysOpen(fRealName, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("ReOpen", "file %s can not be opened in read mode", GetName());
         return -1;
      }
      SetWritable(kFALSE);

   } else {
      // switch to UPDATE mode

      // close readonly file
      if (IsOpen()) {
         SysClose(fD);
         fD = -1;
      }

      // open in UPDATE mode
      fOption = opt;    // set fOption before SysOpen() for TNetFile
#ifndef WIN32
      fD = SysOpen(fRealName, O_RDWR | O_CREAT, 0644);
#else
      fD = SysOpen(fRealName, O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("ReOpen", "file %s can not be opened in update mode", GetName());
         return -1;
      }
      SetWritable(kTRUE);

      fFree = new TList;
      if (fSeekFree > fBEGIN)
         ReadFree();
      else
         Warning("ReOpen","file %s probably not closed, cannot read free segments", GetName());
   }

   return 0;
}

//______________________________________________________________________________
void TFile::Seek(Long64_t offset, ERelativeTo pos)
{
   // Seek to a specific position in the file. Pos it either kBeg, kCur or kEnd.

   int whence = 0;
   switch (pos) {
      case kBeg:
         whence = SEEK_SET;
         offset += fArchiveOffset;
         break;
      case kCur:
         whence = SEEK_CUR;
         break;
      case kEnd:
         whence = SEEK_END;
         // this option is not used currently in the ROOT code
         if (fArchiveOffset)
            Error("Seek", "seeking from end in archive is not (yet) supported");
         break;
   }
   if (Long64_t retpos = SysSeek(fD, offset, whence) < 0)
      SysError("Seek", "cannot seek to position %lld in file %s, retpos=%lld",
               offset, GetName(), retpos);
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
   // Return the size in bytes of the file header.

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
   // TWebFile, TRFIOFile, TDCacheFile, etc.

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
      if (!TestBit(kWriteError)) {
         // Do not print the warning if we already had a SysError.
         Warning("Write", "file %s not opened in write mode", GetName());
      }
      return 0;
   }

   TDirectory *cursav = gDirectory;
   cd();

   if (gDebug) {
      if (!GetTitle() || strlen(GetTitle()) == 0)
         Info("Write", "writing name = %s", GetName());
      else
         Info("Write", "writing name = s title = %s", GetName(), GetTitle());
   }

   Int_t nbytes = TDirectory::Write(0, opt, bufsiz); // Write directory tree
   WriteStreamerInfo();
   WriteFree();                       // Write free segments linked list
   WriteHeader();                     // Now write file header

   cursav->cd();
   return nbytes;
}

//______________________________________________________________________________
Int_t TFile::Write(const char *n, Int_t opt, Int_t bufsize) const
{
   // One can not save a const TDirectory object.

   Error("Write const","A const TFile object should not be saved. We try to proceed anyway.");
   return const_cast<TFile*>(this)->Write(n, opt, bufsize);
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
         // Write the system error only once for this file
         SetBit(kWriteError); SetWritable(kFALSE);
         SysError("WriteBuffer", "error writing to file %s (%d)", GetName(), siz);
         return kTRUE;
      }
      if (siz != len) {
         SetBit(kWriteError);
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
Int_t TFile::WriteBufferViaCache(const char *buf, Int_t len)
{
   // Write buffer via cache. Returns 0 if cache is not active, 1 in case
   // write via cache was successful, 2 in case write via cache failed.

   if (!fCache) return 0;

   Int_t st;
   Long64_t off = GetRelOffset();
   if ((st = fCache->WriteBuffer(off, buf, len)) < 0) {
      SetBit(kWriteError);
      Error("WriteBuffer", "error writing to cache");
      return 2;
   }
   if (st > 0) {
      // fOffset might have been changed via TCache::WriteBuffer(), reset it
      Seek(off + len);
      return 1;
   }
   return 0;
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
   TKey *key    = new TKey(fName,fTitle,IsA(),nbytes,this);
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
   SafeDelete(fInfoCache);
   TFree *lastfree = (TFree*)fFree->Last();
   if (lastfree) fEND  = lastfree->GetFirst();
   const char *root = "root";
   char *psave  = new char[fBEGIN];
   char *buffer = psave;
   Int_t nfree  = fFree->GetSize();
   memcpy(buffer, root, 4); buffer += 4;
   Int_t version = fVersion;
   if (fEND > kStartBigFile) {version += 1000000; fUnits = 8;}
   tobuf(buffer, version);
   tobuf(buffer, (Int_t)fBEGIN);
   if (version < 1000000) {
      tobuf(buffer, (Int_t)fEND);
      tobuf(buffer, (Int_t)fSeekFree);
      tobuf(buffer, fNbytesFree);
      tobuf(buffer, nfree);
      tobuf(buffer, fNbytesName);
      tobuf(buffer, fUnits);
      tobuf(buffer, fCompress);
      tobuf(buffer, (Int_t)fSeekInfo);
      tobuf(buffer, fNbytesInfo);
   } else {
      tobuf(buffer, fEND);
      tobuf(buffer, fSeekFree);
      tobuf(buffer, fNbytesFree);
      tobuf(buffer, nfree);
      tobuf(buffer, fNbytesName);
      tobuf(buffer, fUnits);
      tobuf(buffer, fCompress);
      tobuf(buffer, fSeekInfo);
      tobuf(buffer, fNbytesInfo);
   }
   fUUID.FillBuffer(buffer);
   Int_t nbytes  = buffer - psave;
   Seek(0);
   WriteBuffer(psave, nbytes);
   Flush();
   delete [] psave;
}

//______________________________________________________________________________
void TFile::MakeProject(const char *dirname, const char * /*classes*/,
                        Option_t *option)
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
         Error("MakeProject","cannot create directory %s, already existing",dirname);
         delete [] path;
         return;
      }
      gSystem->mkdir(dirname);
   }

   // we are now ready to generate the classes
   // loop on all TStreamerInfo
   TList *list = 0;
   if (fSeekInfo) {
      TKey *key = new TKey(this);
      char *buffer = new char[fNbytesInfo+1];
      char *buf    = buffer;
      Seek(fSeekInfo);
      ReadBuffer(buf,fNbytesInfo);
      key->ReadKeyBuffer(buf);
      list = (TList*)key->ReadObj();
      delete [] buffer;
      delete key;
   } else {
      list = (TList*)Get("StreamerInfo"); //for versions 2.26 (never released)
   }
   if (list == 0) {
      Error("MakeProject","file %s has no StreamerInfo", GetName());
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
#ifdef R__WINGCC
   FILE *fpMAKE = fopen(path,"wb");
#else
   FILE *fpMAKE = fopen(path,"w");
#endif
   if (!fpMAKE) {
      Error("MakeProject", "cannot open file %s", path);
      delete [] path;
      return;
   }

   // add rootcint statement generating ProjectDict.cxx
   fprintf(fpMAKE,"rootcint -f %sProjectDict.cxx -c %s ",dirname,gSystem->GetIncludePath());

   // create the LinkDef.h file by looping on all *.h files
   // delete LinkDef.h if it already exists
   sprintf(path,"%s/LinkDef.h",dirname);
#ifdef R__WINGCC
   FILE *fp = fopen(path,"wb");
#else
   FILE *fp = fopen(path,"w");
#endif
   if (!fp) {
      Error("MakeProject", "cannot open path file %s", path);
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

   TList *list = GetStreamerInfoList();
   if (!list) {
      MakeZombie();
      return;
   }

   list->SetOwner(kFALSE);

   if (gDebug > 0) Info("ReadStreamerInfo", "called for file %s",GetName());

   // loop on all TStreamerInfo classes
   TStreamerInfo *info;
   TIter next(list);
   while ((info = (TStreamerInfo*)next())) {
      if (info->IsA() != TStreamerInfo::Class()) {
         Warning("ReadStreamerInfo","%s: not a TStreamerInfo object", GetName());
         continue;
      }
      info->BuildCheck();
      Int_t uid = info->GetNumber();
      Int_t asize = fClassIndex->GetSize();
      if (uid >= asize && uid <100000) fClassIndex->Set(2*asize);
      if (uid >= 0 && uid < fClassIndex->GetSize()) fClassIndex->fArray[uid] = 1;
      else {
         printf("ReadStreamerInfo, class:%s, illegal uid=%d\n",info->GetName(),uid);
      }
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
   if (gDebug > 0) Info("WriteStreamerInfo", "called for file %s",GetName());

   SafeDelete(fInfoCache);

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
   fClassIndex->fArray[0] = 2; //to prevent adding classes in TStreamerInfo::TagFile

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
   TKey key(&list,"StreamerInfo",GetBestBuffer(), this);
   fKeys->Remove(&key);
   fSeekInfo   = key.GetSeekKey();
   fNbytesInfo = key.GetNbytes();
   SumBuffer(key.GetObjlen());
   key.WriteFile(0);

   fClassIndex->fArray[0] = 0;
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
   // list of TFile plugin handlers in $ROOTSYS/etc/system.rootrc for regular
   // expressions that will be checked) and as last a local file will be tried.
   // Before opening a file via TNetFile a check is made to see if the URL
   // specifies a local file. If that is the case the file will be opened
   // via a normal TFile. To force the opening of a local file via a
   // TNetFile use either TNetFile directly or specify as host "localhost".
   // The netopt argument is only used by TNetFile. For the meaning of the
   // options and other arguments see the constructors of the individual
   // file classes. In case of error returns 0.

   TPluginHandler *h;
   TFile *f = 0;

   // change names from e.g. /castor/cern.ch/alice/file.root to
   // castor:/castor/cern.ch/alice/file.root as recognized by the plugin manager
   TUrl urlname(name, kTRUE);
   name = urlname.GetUrl();

   // Check first if a pending async open request matches this one
   if (fgAsyncOpenRequests && (fgAsyncOpenRequests->GetSize() > 0)) {
      TIter nxr(fgAsyncOpenRequests);
      TFileOpenHandle *fh = 0;
      while ((fh = (TFileOpenHandle *)nxr()))
         if (fh->Matches(name))
            return TFile::Open(fh);
   }

   // Resolve the file type; this also adjusts names
   EFileType type = GetType(name, option);

   if (type == kLocal) {

      // Local files
      f = new TFile(urlname.GetFile(), option, ftitle, compress);

   } else if (type == kNet) {

      // Network files
      if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
           h->LoadPlugin() == 0)
         f = (TFile*) h->ExecPlugin(5, name, option, ftitle, compress, netopt);
      else
         f = new TNetFile(name, option, ftitle, compress, netopt);

   } else if (type == kWeb) {

      // Web files
      if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
          h->LoadPlugin() == 0)
         f = (TFile*) h->ExecPlugin(1, name);
      else
         f = new TWebFile(name);

   } else if (type == kFile) {

      // 'file:' protocol
      if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
          h->LoadPlugin() == 0)
         f = (TFile*) h->ExecPlugin(4, name+5, option, ftitle, compress);
      else
         f = new TFile(name, option, ftitle, compress);

   } else {

      // no recognized specification: try the plugin manager
      if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name))) {
         if (h->LoadPlugin() == -1)
            return 0;
         TClass *cl = gROOT->GetClass(h->GetClass());
         if (cl && cl->InheritsFrom("TNetFile"))
            f = (TFile*) h->ExecPlugin(5, name, option, ftitle, compress, netopt);
         else
            f = (TFile*) h->ExecPlugin(4, name, option, ftitle, compress);
      } else
         // just try to open it locally
         f = new TFile(name, option, ftitle, compress);
   }

   if (f && f->IsZombie()) {
      delete f;
      f = 0;
   }

   return f;
}

//______________________________________________________________________________
TFileOpenHandle *TFile::AsyncOpen(const char *name, Option_t *option,
                                  const char *ftitle, Int_t compress,
                                  Int_t netopt)
{
   // Static member function to submit an open request. The request will be
   // processed asynchronously. See TFile::Open(const char *, ...) for an
   // explanation of the arguments. A handler is returned which is to be passed
   // to TFile::Open(TFileOpenHandle *) to get the real TFile instance once
   // the file is open.
   // This call never blocks and it is provided to allow parallel submission
   // of file opening operations expected to take a long time.
   // TFile::Open(TFileOpenHandle *) may block if the file is not yet ready.
   // The sequence
   //    TFile::Open(TFile::AsyncOpen(const char *, ...))
   // is equivalent to
   //    TFile::Open(const char *, ...) .
   // To be effective, the underlying TFile implementation must be able to
   // support asynchronous open functionality. Currently, only TXNetFile
   // supports it. If the functionality is not implemented, this call acts
   // transparently by returning an handle with the arguments for the
   // standard synchronous open run by TFile::Open(TFileOpenHandle *).
   // The retuned handle will be adopted by TFile after opening completion
   // in TFile::Open(TFileOpenHandle *); if opening is not finalized the
   // handle must be deleted by the caller.

   TFileOpenHandle *fh = 0;
   TPluginHandler *h;
   TFile *f = 0;
   Bool_t notfound = kTRUE;

   // change names from e.g. /castor/cern.ch/alice/file.root to
   // castor:/castor/cern.ch/alice/file.root as recognized by the plugin manager
   TUrl urlname(name, kTRUE);
   name = urlname.GetUrl();

   // Resolve the file type; this also adjusts names
   EFileType type = GetType(name, option);

   // Here we send the asynchronous request if the functionality is implemented
   if (type == kNet) {
      // Network files
      if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
           !strcmp(h->GetClass(),"TXNetFile") && h->LoadPlugin() == 0) {
         f = (TFile*) h->ExecPlugin(6, name, option, ftitle, compress, netopt, kTRUE);
         notfound = kFALSE;
      }
   }

   // Make sure that no error occured
   if (notfound) {
      SafeDelete(f);
      // Save the arguments in the handler, so that a standard open can be
      // attempted later on
      fh = new TFileOpenHandle(name, option, ftitle, compress, netopt);
   } else if (f) {
      // Fill the opaque handler to be use to attach the file later on
      fh = new TFileOpenHandle(f);
   }

   // Record this request
   if (fh) {
      // Create the lst, if not done already
      if (!fgAsyncOpenRequests)
         fgAsyncOpenRequests = new TList;
      fgAsyncOpenRequests->Add(fh);
   }

   // We are done
   return fh;
}

//______________________________________________________________________________
TFile *TFile::Open(TFileOpenHandle *fh)
{
   // Waits for the completion of an asynchronous open request.
   // Returns the associated TFile, transferring ownership of the
   // handle to the TFile instance.

   TFile *f = 0;

   // Note that the request may have failed
   if (fh && fgAsyncOpenRequests) {
      // Was asynchronous open functionality implemented?
      if ((f = fh->GetFile()) && !(f->IsZombie())) {
         // Yes: wait for the completion of the open phase, if needed
         Bool_t cr = (!strcmp(f->GetOption(),"CREATE") ||
                      !strcmp(f->GetOption(),"RECREATE") ||
                      !strcmp(f->GetOption(),"NEW")) ? kTRUE : kFALSE;
         f->Init(cr);
      } else {
         // No: process a standard open
         f = TFile::Open(fh->GetName(), fh->GetOpt(), fh->GetTitle(),
                         fh->GetCompress(), fh->GetNetOpt());
      }

      // Adopt the handle instance in the TFile instance so that it gets
      // automatically cleaned up
      f->fAsyncHandle = fh;

      // Remove it from the pending list
      fgAsyncOpenRequests->Remove(fh);
   }

   // We are done
   return f;
}

//______________________________________________________________________________
Int_t TFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // Interface to system open. All arguments like in POSIX open().

#if defined(R__WINGCC)
   // ALWAYS use binary mode - even cygwin text should be in unix format
   // although this is posix default it has to be set explicitly
   return ::open(pathname, flags | O_BINARY, mode);
#elif defined(R__SEEK64)
   return ::open64(pathname, flags, mode);
#else
   return ::open(pathname, flags, mode);
#endif
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
Long64_t TFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   // Interface to system lseek. All arguments like in POSIX lseek()
   // except that the offset and return value are of a type which are
   // able to handle 64 bit file systems.

#if defined (R__SEEK64)
   return ::lseek64(fd, offset, whence);
#elif defined(WIN32)
   return ::_lseeki64(fd, offset, whence);
#else
   return ::lseek(fd, offset, whence);
#endif
}

//______________________________________________________________________________
Int_t TFile::SysStat(Int_t, Long_t *id, Long64_t *size, Long_t *flags,
                     Long_t *modtime)
{
   // Return file stat information. The interface and return value is
   // identical to TSystem::GetPathInfo(). The function returns 0 in
   // case of success and 1 if the file could not be stat'ed.

   return gSystem->GetPathInfo(fRealName, id, size, flags, modtime);
}

//______________________________________________________________________________
Int_t TFile::SysSync(Int_t fd)
{
   // Interface to system fsync. All arguments like in POSIX fsync().

   if (TestBit(kDevNull)) return 0;

#ifndef WIN32
   return ::fsync(fd);
#else
   return ::_commit(fd);
#endif
}

//______________________________________________________________________________
Long64_t TFile::GetFileBytesRead() { return fgBytesRead; }

//______________________________________________________________________________
Long64_t TFile::GetFileBytesWritten() { return fgBytesWrite; }

//______________________________________________________________________________
void TFile::SetFileBytesRead(Long64_t bytes){ fgBytesRead = bytes; }

//______________________________________________________________________________
void TFile::SetFileBytesWritten(Long64_t bytes){ fgBytesWrite = bytes; }

//______________________________________________________________________________
Bool_t TFile::Matches(const char *url)
{
   // Return kTRUE if 'url' matches the coordinates of this file.
   // The check is implementation dependent and may need to be overload
   // by each TFile implememtation relying on this check.
   // The default implementation checks teh file name only.

   // Check the full URL, including port and FQDN.
   TUrl u(url);
   TInetAddress a = gSystem->GetHostByName(u.GetHost());
   TString fqdn = a.GetHostName();
   if (fqdn == "UnNamedHost")
      fqdn = a.GetHostAddress();

   // Check
   if (!strcmp(u.GetFile(),fUrl.GetFile())) {
      // Check ports
      if (u.GetPort() == fUrl.GetPort()) {
         TInetAddress aref = gSystem->GetHostByName(fUrl.GetHost());
         TString fqdnref = aref.GetHostName();
         if (fqdnref == "UnNamedHost")
            fqdnref = aref.GetHostAddress();
         if (fqdn == fqdnref)
            // Ok, coordinates match
            return kTRUE;
      }
   }

   // Default is not matching
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TFileOpenHandle::Matches(const char *url)
{
   // Return kTRUE if this async request matches the open request
   // specified by 'url'

   if (fFile) {
      return fFile->Matches(url);
   } else if (fName.Length() > 0){
      // Deep check of URLs
      TUrl u(url);
      TUrl uref(fName);
      if (!strcmp(u.GetFile(),uref.GetFile())) {
         // Check ports
         if (u.GetPort() == uref.GetPort()) {
            // Check also the host name
            TInetAddress a = gSystem->GetHostByName(u.GetHost());
            TInetAddress aref = gSystem->GetHostByName(uref.GetHost());
            TString fqdn = a.GetHostName();
            if (fqdn == "UnNamedHost")
               fqdn = a.GetHostAddress();
            TString fqdnref = aref.GetHostName();
            if (fqdnref == "UnNamedHost")
               fqdnref = aref.GetHostAddress();
            if (fqdn == fqdnref)
               // Ok, coordinates match
               return kTRUE;
         }
      }
   }

   // Default is not matching
   return kFALSE;
}

//______________________________________________________________________________
TFile::EFileType TFile::GetType(const char *name, Option_t *option)
{
   // Resolve the file type as a function of the protocol field in 'name'

   EFileType type = kDefault;

   TRegexp re("^root.*:");
   TString sname = name;
   if (sname.Index(re) != kNPOS) {
      //
      // Should be a network file ...
      type = kNet;
      // ... but make sure that is not local or that a remote-like connection
      // is forced. Treat it as local if:
      //    i)  the url points to the localhost, the file will be opened in
      //        readonly mode and the current user has read access;
      //    ii) the specified user is equal to the current user then open local
      //        TFile.
      const char *lfname = 0;
      Bool_t localFile = kFALSE;
      TUrl url(name);
      Bool_t forceRemote = gEnv->GetValue("TFile.ForceRemote",0);
      if (!forceRemote) {
         TInetAddress a(gSystem->GetHostByName(url.GetHost()));
         TInetAddress b(gSystem->GetHostByName(gSystem->HostName()));
         if (!strcmp(a.GetHostName(), b.GetHostName())) {
            Bool_t read = kFALSE;
            TString opt = option;
            opt.ToUpper();
            if (opt == "" || opt == "READ") read = kTRUE;
            const char *fname = url.GetFile();
            if (fname[1] == '/' || fname[1] == '~' || fname[1] == '$')
               lfname = &fname[1];
            else
               lfname = Form("%s%s", gSystem->HomeDirectory(), fname);
            if (read) {
               char *fn;
               if ((fn = gSystem->ExpandPathName(lfname))) {
                  if (gSystem->AccessPathName(fn, kReadPermission))
                     read = kFALSE;
                  delete [] fn;
               }
            }
            Bool_t sameUser = kFALSE;
            UserGroup_t *u = gSystem->GetUserInfo();
            if (u && !strcmp(u->fUser, url.GetUser()))
               sameUser = kTRUE;
            delete u;
            if (read || sameUser)
               localFile = kTRUE;
         }
      }
      //
      // Adjust the type according to findings
      type = (localFile) ? kLocal : type;
   } else if (!strncmp(name, "http:", 5)) {
      //
      // Web file
      type = kWeb;
   } else if (!strncmp(name, "file:", 5)) {
      //
      // 'file' protocol
      type = kFile;
   }

   // We are done
   return type;
}

//______________________________________________________________________________
TFile::EAsyncOpenStatus TFile::GetAsyncOpenStatus(const char* name)
{
   // Get status of the async open request related to 'name'.

   // Check the list of pending async opem requests
   if (fgAsyncOpenRequests && (fgAsyncOpenRequests->GetSize() > 0)) {
      TIter nxr(fgAsyncOpenRequests);
      TFileOpenHandle *fh = 0;
      while ((fh = (TFileOpenHandle *)nxr()))
         if (fh->Matches(name))
            return TFile::GetAsyncOpenStatus(fh);
   }

   // Check also the list of files open
   TSeqCollection *of = gROOT->GetListOfFiles();
   if (of && (of->GetSize() > 0)) {
      TIter nxf(of);
      TFile *f = 0;
      while ((f = (TFile *)nxf()))
         if (f->Matches(name))
            return f->GetAsyncOpenStatus();
   }

   // Default is synchronous mode
   return kAOSNotAsync;
}

//______________________________________________________________________________
TFile::EAsyncOpenStatus TFile::GetAsyncOpenStatus(TFileOpenHandle *handle)
{
   // Get status of the async open request related to 'handle'.

   if (handle && handle->fFile)
      if (!handle->fFile->IsZombie())
         return handle->fFile->GetAsyncOpenStatus();
      else
         return TFile::kAOSFailure;

   // Default is synchronous mode
   return TFile::kAOSNotAsync;
}

//______________________________________________________________________________
const TUrl *TFile::GetEndpointUrl(const char* name)
{
   // Get final URL for file being opened asynchronously.
   // Returns 0 is the information is not yet available.

   // Check the list of pending async opem requests
   if (fgAsyncOpenRequests && (fgAsyncOpenRequests->GetSize() > 0)) {
      TIter nxr(fgAsyncOpenRequests);
      TFileOpenHandle *fh = 0;
      while ((fh = (TFileOpenHandle *)nxr()))
         if (fh->Matches(name))
            if (fh->fFile)
               return fh->fFile->GetEndpointUrl();
   }

   // Check also the list of files open
   TSeqCollection *of = gROOT->GetListOfFiles();
   if (of && (of->GetSize() > 0)) {
      TIter nxf(of);
      TFile *f = 0;
      while ((f = (TFile *)nxf()))
         if (f->Matches(name))
            return f->GetEndpointUrl();
   }

   // Information not yet available
   return (const TUrl *)0;
}
