
// @(#)root/io:$Id$
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
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
//    34->37 [42->45] fCompress   = Compression level and algorithm
//    38->41 [46->53] fSeekInfo   = Pointer to TStreamerInfo record
//    42->45 [54->57] fNbytesInfo = Number of bytes in TStreamerInfo record
//    46->63 [58->75] fUUID       = Universal Unique ID
//Begin_Html
/*
<img src="gif/file_layout.gif">
*/
//End_Html
//
// The structure of a directory is shown in TDirectoryFile::TDirectoryFile
//
////////////////////////////////////////////////////////////////////////////////

#include "RConfig.h"

#ifdef R__LINUX
// for posix_fadvise
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#endif
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
#include "Compression.h"
#include "Riostream.h"
#include "RConfigure.h"
#include "Strlen.h"
#include "TArrayC.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassTable.h"
#include "TDatime.h"
#include "TError.h"
#include "TFile.h"
#include "TFileCacheRead.h"
#include "TFileCacheWrite.h"
#include "TFree.h"
#include "TInterpreter.h"
#include "TKey.h"
#include "TMakeProject.h"
#include "TPluginManager.h"
#include "TProcessUUID.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TVirtualPerfStats.h"
#include "TArchiveFile.h"
#include "TEnv.h"
#include "TVirtualMonitoring.h"
#include "TVirtualMutex.h"
#include "TMathBase.h"
#include "TObjString.h"
#include "TStopwatch.h"
#include "compiledata.h"
#include <cmath>
#include <set>
#include "TSchemaRule.h"
#include "TSchemaRuleSet.h"
#include "TThreadSlots.h"

Long64_t TFile::fgBytesRead  = 0;
Long64_t TFile::fgBytesWrite = 0;
Long64_t TFile::fgFileCounter = 0;
Int_t    TFile::fgReadaheadSize = 256000;
Int_t    TFile::fgReadCalls = 0;
Bool_t   TFile::fgReadInfo = kTRUE;
TList   *TFile::fgAsyncOpenRequests = 0;
TString  TFile::fgCacheFileDir;
Bool_t   TFile::fgCacheFileForce = kFALSE;
Bool_t   TFile::fgCacheFileDisconnected = kTRUE;
UInt_t   TFile::fgOpenTimeout = TFile::kEternalTimeout;
Bool_t   TFile::fgOnlyStaged = 0;

const Int_t kBEGIN = 100;

ClassImp(TFile)

//*-*x17 macros/layout_file

//______________________________________________________________________________
TFile::TFile() : TDirectoryFile(), fInfoCache(0)
{
   // File default Constructor.

   fD               = -1;
   fFree            = 0;
   fWritten         = 0;
   fSumBuffer       = 0;
   fSum2Buffer      = 0;
   fClassIndex      = 0;
   fCompress        = 0;
   fProcessIDs      = 0;
   fNProcessIDs     = 0;
   fOffset          = 0;
   fArchive         = 0;
   fCacheRead       = 0;
   fCacheWrite      = 0;
   fArchiveOffset   = 0;
   fReadCalls       = 0;
   fInfoCache       = 0;
   fOpenPhases      = 0;
   fNoAnchorInName  = kFALSE;
   fIsRootFile      = kTRUE;
   fIsArchive       = kFALSE;
   fInitDone        = kFALSE;
   fMustFlush       = kTRUE;
   fAsyncHandle     = 0;
   fAsyncOpenStatus = kAOSNotAsync;
   SetBit(kBinaryFile, kTRUE);

   fBEGIN          = 0;
   fEND            = 0;
   fBytesRead      = 0;
   fBytesReadExtra = 0;
   fBytesWrite     = 0;
   fCompress       = 0;
   fNbytesFree     = 0;
   fNbytesInfo     = 0;
   fSeekFree       = 0;
   fSeekInfo       = 0;
   fUnits          = 0;
   fVersion        = 0;

   if (gDebug)
      Info("TFile", "default ctor");
}

//_____________________________________________________________________________
TFile::TFile(const char *fname1, Option_t *option, const char *ftitle, Int_t compress)
           : TDirectoryFile(), fUrl(fname1,kTRUE), fInfoCache(0), fOpenPhases(0)
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
   //           = READ            open an existing file for reading (default).
   //           = NET             used by derived remote file access
   //                             classes, not a user callable option
   //           = WEB             used by derived remote http access
   //                             class, not a user callable option
   // If option = "" (default), READ is assumed.
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
   // Option compress is used to specify the compression level and algorithm:
   //  compress = 100 * algorithm + level
   //  level = 0, objects written to this file will not be compressed.
   //  level = 1, minimal compression level but fast.
   //  ....
   //  level = 9, maximal compression level but slower and might use more memory.
   // (For the currently supported algorithms, the maximum level is 9)
   // If compress is negative it indicates the compression level is not set yet.
   //
   // The enumeration ROOT::ECompressionAlgorithm associates each
   // algorithm with a number. There is a utility function to help
   // to set the value of compress. For example,
   //   ROOT::CompressionSettings(ROOT::kLZMA, 1)
   // will build an integer which will set the compression to use
   // the LZMA algorithm and compression level 1.  These are defined
   // in the header file Compression.h.
   //
   // Note that the compression settings may be changed at any time.
   // The new compression settings will only apply to branches created
   // or attached after the setting is changed and other objects written
   // after the setting is changed.
   //
   // In case the file does not exist or is not a valid ROOT file,
   // it is made a Zombie. One can detect this situation with a code like:
   //    TFile f("file.root");
   //    if (f.IsZombie()) {
   //       cout << "Error opening file" << endl;
   //       exit(-1);
   //    }
   //
   //  When opening the file, the system checks the validity of this directory.
   //  If something wrong is detected, an automatic Recovery is performed. In
   //  this case, the file is scanned sequentially reading all logical blocks
   //  and attempting to rebuild a correct directory (see TFile::Recover).
   //  One can disable the automatic recovery procedure when reading one
   //  or more files by setting the environment variable "TFile::Recover 0"
   //  in the system.rootrc file.
   //

   if (!gROOT)
      ::Fatal("TFile::TFile", "ROOT system not initialized");

   // store name without the options as name and title
   TString sfname1 = fname1;
   fNoAnchorInName = kFALSE;
   if (sfname1.Index("?") != kNPOS) {
      TString s = sfname1(0, sfname1.Index("?"));
      SetName(s);
      fNoAnchorInName = kTRUE;
   } else
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
   fMustFlush  = kTRUE;

   // We are opening synchronously
   fAsyncHandle = 0;
   fAsyncOpenStatus = kAOSNotAsync;

   TDirectoryFile::Build(this, 0);

   fD            = -1;
   fFree         = 0;
   fVersion      = gROOT->GetVersionInt();  //ROOT version in integer format
   fUnits        = 4;
   fOption       = option;
   fCompress     = compress;
   fWritten      = 0;
   fSumBuffer    = 0;
   fSum2Buffer   = 0;
   fBytesRead    = 0;
   fBytesReadExtra = 0;
   fBytesWrite   = 0;
   fClassIndex   = 0;
   fSeekInfo     = 0;
   fNbytesInfo   = 0;
   fProcessIDs   = 0;
   fNProcessIDs  = 0;
   fOffset       = 0;
   fCacheRead    = 0;
   fCacheWrite   = 0;
   fReadCalls    = 0;
   SetBit(kBinaryFile, kTRUE);

   fOption.ToUpper();

   fArchiveOffset = 0;
   fIsArchive     = kFALSE;
   fArchive       = 0;
   if (fIsRootFile) {
      fArchive = TArchiveFile::Open(fUrl.GetUrl(), this);
      if (fArchive) {
        fname1 = fArchive->GetArchiveName();
         // if no archive member is specified then this TFile is just used
         // to read the archive contents
         if (!strlen(fArchive->GetMemberName()))
            fIsArchive = kTRUE;
      }
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
      delete [] fname;
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
   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfClosedObjects()->Add(this);
   }
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
TFile::TFile(const TFile &) : TDirectoryFile(), fInfoCache(0)
{
   // TFile objects can not be copied.

   MayNotUse("TFile::TFile(const TFile &)");
}

//______________________________________________________________________________
TFile::~TFile()
{
   // File destructor.

   Close();

   SafeDelete(fProcessIDs);
   SafeDelete(fFree);
   SafeDelete(fArchive);
   SafeDelete(fInfoCache);
   SafeDelete(fOpenPhases);
   SafeDelete(fAsyncHandle);
   SafeDelete(fCacheRead);
   SafeDelete(fCacheWrite);

   R__LOCKGUARD2(gROOTMutex);
   gROOT->GetListOfClosedObjects()->Remove(this);
   gROOT->GetUUIDs()->RemoveUUID(GetUniqueID());

   if (IsOnHeap()) {
      // Delete object from CINT symbol table so it can not be used anymore.
      // CINT object are always on the heap.
      gInterpreter->ResetGlobalVar(this);
   }

   if (gDebug)
      Info("~TFile", "dtor called for %s [%lx]", GetName(),(Long_t)this);
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

      // Make sure the anchor is in the name
      if (!fNoAnchorInName)
         if (!strchr(GetName(),'#'))
            SetName(Form("%s#%s", GetName(), fArchive->GetMemberName()));

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

   if (create) {
      //*-*---------------NEW file
      fFree        = new TList;
      fEND         = fBEGIN;    //Pointer to end of file
      new TFree(fFree, fBEGIN, Long64_t(kStartBigFile));  //Create new free list

      //*-* Write Directory info
      Int_t namelen= TNamed::Sizeof();
      Int_t nbytes = namelen + TDirectoryFile::Sizeof();
      TKey *key    = new TKey(fName, fTitle, IsA(), nbytes, this);
      fNbytesName  = key->GetKeylen() + namelen;
      fSeekDir     = key->GetSeekKey();
      fSeekFree    = 0;
      fNbytesFree  = 0;
      WriteHeader();
      char *buffer = key->GetBuffer();
      TNamed::FillBuffer(buffer);
      TDirectoryFile::FillBuffer(buffer);
      key->WriteFile();
      delete key;
   } else {
      //*-*----------------UPDATE
      //char *header = new char[kBEGIN];
      char *header = new char[kBEGIN+200];
      Seek(0);
      //ReadBuffer(header, kBEGIN);
      ReadBuffer(header, kBEGIN+200);

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
      // buffer_keyloc is the start of the key record.
      char *buffer_keyloc = 0;

      Int_t nbytes = fNbytesName + TDirectoryFile::Sizeof();
      if (nbytes+fBEGIN > kBEGIN+200) {
         delete [] header;
         header       = new char[nbytes];
         buffer       = header;
         Seek(fBEGIN);
         ReadBuffer(buffer,nbytes);
         buffer = header+fNbytesName;
         buffer_keyloc = header;
      } else {
         buffer = header+fBEGIN+fNbytesName;
         buffer_keyloc = header+fBEGIN;
      }
      Version_t version,versiondir;
      frombuf(buffer,&version); versiondir = version%1000;
      fDatimeC.ReadBuffer(buffer);
      fDatimeM.ReadBuffer(buffer);
      frombuf(buffer, &fNbytesKeys);
      frombuf(buffer, &fNbytesName);
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
      buffer_keyloc += sizeof(Int_t); // Skip NBytes;
      Version_t keyversion;
      frombuf(buffer_keyloc, &keyversion);
      // Skip ObjLen, DateTime, KeyLen, Cycle, SeekKey, SeekPdir
      if (keyversion > 1000) {
         // Large files
         buffer_keyloc += 2*sizeof(Int_t)+2*sizeof(Short_t)+2*sizeof(Long64_t);
      } else {
         buffer_keyloc += 2*sizeof(Int_t)+2*sizeof(Short_t)+2*sizeof(Int_t);
      }
      TString cname;
      cname.ReadBuffer(buffer_keyloc);
      cname.ReadBuffer(buffer_keyloc); // fName.ReadBuffer(buffer); file may have been renamed
      fTitle.ReadBuffer(buffer_keyloc);
      delete [] header;
      if (fNbytesName < 10 || fNbytesName > 10000) {
         Error("Init","cannot read directory info of file %s", GetName());
         goto zombie;
      }

      //*-* -------------Check if file is truncated
      Long64_t size;
      if ((size = GetSize()) == -1) {
         Error("Init", "cannot stat the file %s", GetName());
         goto zombie;
      }

      //*-* -------------Check if, in case of inconsistencies, we are requested to
      //*-* -------------attempt recovering the file
      Bool_t tryrecover = (gEnv->GetValue("TFile.Recover", 1) == 1) ? kTRUE : kFALSE;

      //*-* -------------Read keys of the top directory
      if (fSeekKeys > fBEGIN && fEND <= size) {
         //normal case. Recover only if file has no keys
         TDirectoryFile::ReadKeys(kFALSE);
         gDirectory = this;
         if (!GetNkeys()) {
            if (tryrecover) {
               Recover();
            } else {
               Error("Init", "file %s has no keys", GetName());
               goto zombie;
            }
         }
      } else if ((fBEGIN+nbytes == fEND) && (fEND == size)) {
         //the file might be open by another process and nothing written to the file yet
         Warning("Init","file %s has no keys", GetName());
         gDirectory = this;
      } else {
         //something had been written to the file. Trailer is missing, must recover
         if (fEND > size) {
            if (tryrecover) {
               Error("Init","file %s is truncated at %lld bytes: should be %lld, "
                     "trying to recover", GetName(), size, fEND);
            } else {
               Error("Init","file %s is truncated at %lld bytes: should be %lld",
                     GetName(), size, fEND);
               goto zombie;
            }
         } else {
            if (tryrecover) {
               Warning("Init","file %s probably not closed, "
                       "trying to recover", GetName());
            } else {
               Warning("Init","file %s probably not closed", GetName());
               goto zombie;
            }
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
   }

   // Create StreamerInfo index
   {
      Int_t lenIndex = gROOT->GetListOfStreamerInfo()->GetSize()+1;
      if (lenIndex < 5000) lenIndex = 5000;
      fClassIndex = new TArrayC(lenIndex);
      if (fgReadInfo) {
         if (fSeekInfo > fBEGIN) {
            ReadStreamerInfo();
         } else if (fVersion != gROOT->GetVersionInt() && fVersion > 30000) {
            Warning("Init","no StreamerInfo found in %s therefore preventing schema evolution when reading this file.",GetName());
         }
      }
   }

   // Count number of TProcessIDs in this file
   {
      TIter next(fKeys);
      TKey *key;
      while ((key = (TKey*)next())) {
         if (!strcmp(key->GetClassName(),"TProcessID")) fNProcessIDs++;
      }
      fProcessIDs = new TObjArray(fNProcessIDs+1);
   }
   return;

zombie:
   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfClosedObjects()->Add(this);
   }
   // error in file opening occured, make this object a zombie
   fWritable = kFALSE;
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
void TFile::Close(Option_t *option)
{
   // Close a file.
   // If option == "R", all TProcessIDs referenced by this file are deleted.
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
      FlushWriteCache();
      SysClose(fD);
      fD = -1;

      if (gMonitoringWriter)
         gMonitoringWriter->SendFileCloseEvent(this);

      return;
   }

   if (IsWritable()) {
      WriteStreamerInfo();
   }

   delete fClassIndex;
   fClassIndex = 0;

   // Delete all supported directories structures from memory
   // If gDirectory points to this object or any of the nested
   // TDirectoryFile, TDirectoryFile::Close will induce the proper cd.
   TDirectoryFile::Close();

   if (IsWritable()) {
      TFree *f1 = (TFree*)fFree->First();
      if (f1) {
         WriteFree();       //*-*- Write free segments linked list
         WriteHeader();     //*-*- Now write file header
      }
   }

   FlushWriteCache();

   if (gMonitoringWriter)
      gMonitoringWriter->SendFileCloseEvent(this);

   // Delete free segments from free list (but don't delete list header)
   if (fFree) {
      fFree->Delete();
   }

   if (IsOpen()) {
      SysClose(fD);
      fD = -1;
   }

   fWritable = kFALSE;

   // delete the TProcessIDs
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

   if (!IsZombie()) {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfFiles()->Remove(this);
      gROOT->GetListOfBrowsers()->RecursiveRemove(this);
      gROOT->GetListOfClosedObjects()->Add(this);
   } else {
      // If we are a zombie, we are already in the list of closed objects.
   }
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

//____________________________________________________________________________________
TFile *&TFile::CurrentFile()
{
   // Return the current ROOT file if any.
   // Note that if 'cd' has been called on a TDirectory that does not belong to a file,
   // gFile will be unchanged and still points to the file of the previous current
   // directory that was a file.
   
   static TFile *currentFile = 0;
   if (!gThreadTsd)
      return currentFile;
   else
      return *(TFile**)(*gThreadTsd)(&currentFile,ROOT::kFileThreadSlot);   
}

//______________________________________________________________________________
void TFile::Delete(const char *namecycle)
{
   // Delete object namecycle.
   // Namecycle identifies an object in the top directory of the file
   //   namecycle has the format name;cycle
   //   name  = * means all
   //   cycle = * means all cycles (memory and keys)
   //   cycle = "" or cycle = 9999 ==> apply to a memory object
   //   When name=* use T* to delete subdirectories also
   //
   // Examples:
   //     foo   : delete object named foo in memory
   //     foo;1 : delete cycle 1 of foo on file
   //     foo;* : delete all cycles of foo on disk and also from memory
   //     *;2   : delete all objects on file having the cycle 2
   //     *;*   : delete all objects from memory and file
   //    T*;*   : delete all objects from memory and file and all subdirectories

   if (gDebug)
      Info("Delete", "deleting name = %s", namecycle);

   TDirectoryFile::Delete(namecycle);
}

//______________________________________________________________________________
void TFile::Draw(Option_t *option)
{
   // Fill Graphics Structure and Paint.
   // Loop on all objects (memory or file) and all subdirectories.

   GetList()->R__FOR_EACH(TObject,Draw)(option);
}

//______________________________________________________________________________
void TFile::DrawMap(const char *keys, Option_t *option)
{
   // Draw map of objects in this file.

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
      FlushWriteCache();
      if (SysSync(fD) < 0) {
         // Write the system error only once for this file
         SetBit(kWriteError); SetWritable(kFALSE);
         SysError("Flush", "error flushing file %s", GetName());
      }
   }
}

//______________________________________________________________________________
Bool_t TFile::FlushWriteCache()
{
   // Flush the write cache if active.
   // Return kTRUE in case of error

   if (fCacheWrite && IsOpen() && fWritable)
      return fCacheWrite->Flush();
   return kFALSE;
}

//______________________________________________________________________________
void TFile::FillBuffer(char *&buffer)
{
   // Encode file output buffer.
   // The file output buffer contains only the FREE data record.

   Version_t version = TFile::Class_Version();
   tobuf(buffer, version);
}

//______________________________________________________________________________
Int_t TFile::GetBestBuffer() const
{
   // Return the best buffer size of objects on this file.
   // The best buffer size is estimated based on the current mean value
   // and standard deviation of all objects written so far to this file.
   // Returns mean value + one standard deviation.

   if (!fWritten) return TBuffer::kInitialSize;
   Double_t mean = fSumBuffer/fWritten;
   Double_t rms2 = TMath::Abs(fSum2Buffer/fSumBuffer -mean*mean);
   return (Int_t)(mean + sqrt(rms2));
}

//______________________________________________________________________________
Float_t TFile::GetCompressionFactor()
{
   // Return the file compression factor.
   // Add total number of compressed/uncompressed bytes for each key.
   // return ratio of the two.

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
TFileCacheRead *TFile::GetCacheRead() const
{
   // Return a pointer to the current read cache.

   return fCacheRead;
}

//______________________________________________________________________________
TFileCacheWrite *TFile::GetCacheWrite() const
{
   // Return a pointer to the current write cache.

   return fCacheWrite;
}

//______________________________________________________________________________
Int_t TFile::GetRecordHeader(char *buf, Long64_t first, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen)
{
   // Read the logical record header starting at position first.
   // Maxbytes bytes are read into buf the function reads nread bytes
   // where nread is the minimum of maxbytes and the number of bytes
   // before the end of file. The function returns nread.
   // In output arguments:
   //    nbytes : number of bytes in record
   //             if negative, this is a deleted record
   //             if 0, cannot read record, wrong value of argument first
   //    objlen : uncompressed object size
   //    keylen : length of logical record header
   // Note that the arguments objlen and keylen are returned only
   // if maxbytes >=16

   nbytes = 0;
   objlen = 0;
   keylen = 0;
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
      list = (TList*)key->ReadObjWithBuffer(buffer);
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
   // List File contents.
   // Indentation is used to identify the file tree.
   // Subdirectories are listed first, then objects in memory,
   // then objects on the file.

   TROOT::IndentLevel();
   cout <<ClassName()<<"**\t\t"<<GetName()<<"\t"<<GetTitle()<<endl;
   TROOT::IncreaseDirLevel();
   TDirectoryFile::ls(option);
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
   // Mark unused bytes on the file.
   // The list of free segments is in the fFree linked list.
   // When an object is deleted from the file, the freed space is added
   // into the FREE linked list (fFree). The FREE list consists of a chain
   // of consecutive free segments on the file. At the same time, the first
   // 4 bytes of the freed record on the file are overwritten by GAPSIZE
   // where GAPSIZE = -(Number of bytes occupied by the record).

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
   // List the contents of a file sequentially.
   // For each logical record found, it prints:
   //  Date/Time  Record_Adress Logical_Record_Length  ClassName  CompressionFactor
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
         date = 0; time = 0;
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
      for (int i = 0;i < nwhc; i++) frombuf(buffer, &classname[i]);
      classname[(int)nwhc] = '\0'; //cast to avoid warning with gcc3.4
      if (idcur == fSeekFree) strlcpy(classname,"FreeSegments",512);
      if (idcur == fSeekInfo) strlcpy(classname,"StreamerInfo",512);
      if (idcur == fSeekKeys) strlcpy(classname,"KeysList",512);
      TDatime::GetDateTime(datime, date, time);
      if (objlen != nbytes-keylen) {
         Float_t cx = Float_t(objlen+keylen)/Float_t(nbytes);
         Printf("%d/%06d  At:%lld  N=%-8d  %-14s CX = %5.2f",date,time,idcur,nbytes,classname,cx);
      } else {
         Printf("%d/%06d  At:%lld  N=%-8d  %-14s",date,time,idcur,nbytes,classname);
      }
      idcur += nbytes;
   }
   Printf("%d/%06d  At:%lld  N=%-8d  %-14s",date,time,idcur,1,"END");
}

//______________________________________________________________________________
void TFile::Paint(Option_t *option)
{
   // Paint all objects in the file.

   GetList()->R__FOR_EACH(TObject,Paint)(option);
}

//______________________________________________________________________________
void TFile::Print(Option_t *option) const
{
   // Print all objects in the file.

   Printf("TFile: name=%s, title=%s, option=%s", GetName(), GetTitle(), GetOption());
   GetList()->R__FOR_EACH(TObject,Print)(option);
}

//______________________________________________________________________________
Bool_t TFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read a buffer from the file at the offset 'pos' in the file.
   // Returns kTRUE in case of failure.
   // Compared to ReadBuffer(char*, Int_t), this routine does _not_
   // change the cursor on the physical file representation (fD)
   // if the data is in this TFile's cache.

   if (IsOpen()) {

      SetOffset(pos);

      Int_t st;
      Double_t start = 0;
      if (gPerfStats != 0) start = TTimeStamp();

      if ((st = ReadBufferViaCache(buf, len))) {
         if (st == 2)
            return kTRUE;
         return kFALSE;
      }

      Seek(pos);
      ssize_t siz;

      while ((siz = SysRead(fD, buf, len)) < 0 && GetErrno() == EINTR)
         ResetErrno();

      if (siz < 0) {
         SysError("ReadBuffer", "error reading from file %s", GetName());
         return kTRUE;
      }
      if (siz != len) {
         Error("ReadBuffer", "error reading all requested bytes from file %s, got %ld of %d",
               GetName(), (Long_t)siz, len);
         return kTRUE;
      }
      fBytesRead  += siz;
      fgBytesRead += siz;
      fReadCalls++;
      fgReadCalls++;

      if (gMonitoringWriter)
         gMonitoringWriter->SendFileReadProgress(this);
      if (gPerfStats != 0) {
         gPerfStats->FileReadEvent(this, len, start);
      }
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFile::ReadBuffer(char *buf, Int_t len)
{
   // Read a buffer from the file. This is the basic low level read operation.
   // Returns kTRUE in case of failure.

   if (IsOpen()) {

      Int_t st;
      if ((st = ReadBufferViaCache(buf, len))) {
         if (st == 2)
            return kTRUE;
         return kFALSE;
      }

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
         Error("ReadBuffer", "error reading all requested bytes from file %s, got %ld of %d",
               GetName(), (Long_t)siz, len);
         return kTRUE;
      }
      fBytesRead  += siz;
      fgBytesRead += siz;
      fReadCalls++;
      fgReadCalls++;

      if (gMonitoringWriter)
         gMonitoringWriter->SendFileReadProgress(this);
      if (gPerfStats != 0) {
         gPerfStats->FileReadEvent(this, len, start);
      }
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFile::ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
   // Read the nbuf blocks described in arrays pos and len,
   // where pos[i] is the seek position of block i of length len[i].
   // Note that for nbuf=1, this call is equivalent to TFile::ReafBuffer.
   // This function is overloaded by TNetFile, TWebFile, etc.
   // Returns kTRUE in case of failure.

   // called with buf=0, from TFileCacheRead to pass list of readahead buffers
   if (!buf) {
      for (Int_t j = 0; j < nbuf; j++) {
         if (ReadBufferAsync(pos[j], len[j])) {
             return kTRUE;
         }
      }
      return kFALSE;
   }

   Int_t k = 0;
   Bool_t result = kTRUE;
   TFileCacheRead *old = fCacheRead;
   fCacheRead = 0;
   Long64_t curbegin = pos[0];
   Long64_t cur;
   char *buf2 = 0;
   Int_t i = 0, n = 0;
   while (i < nbuf) {
      cur = pos[i]+len[i];
      Bool_t bigRead = kTRUE;
      if (cur -curbegin < fgReadaheadSize) {n++; i++; bigRead = kFALSE;}
      if (bigRead || (i>=nbuf)) {
         if (n == 0) {
            //if the block to read is about the same size as the read-ahead buffer
            //we read the block directly
            Seek(pos[i]);
            result = ReadBuffer(&buf[k], len[i]);
            if (result) break;
            k += len[i];
            i++;
         } else {
            //otherwise we read all blocks that fit in the read-ahead buffer
            Seek(curbegin);
            if (buf2 == 0) buf2 = new char[fgReadaheadSize];
            //we read ahead
            Long64_t nahead = pos[i-1]+len[i-1]-curbegin;
            result = ReadBuffer(buf2, nahead);
            if (result) break;
            //now copy from the read-ahead buffer to the cache
            Int_t kold = k;
            for (Int_t j=0;j<n;j++) {
               memcpy(&buf[k],&buf2[pos[i-n+j]-curbegin],len[i-n+j]);
               k += len[i-n+j];
            }
            Int_t nok = k-kold;
            Long64_t extra = nahead-nok;
            fBytesReadExtra += extra;
            fBytesRead      -= extra;
            fgBytesRead     -= extra;
            n = 0;
         }
         curbegin = pos[i];
      }
   }
   if (buf2) delete [] buf2;
   fCacheRead = old;
   return result;
}

//______________________________________________________________________________
Int_t TFile::ReadBufferViaCache(char *buf, Int_t len)
{
   // Read buffer via cache. Returns 0 if the requested block is
   // not in the cache, 1 in case read via cache was successful,
   // 2 in case read via cache failed.

   Long64_t off = GetRelOffset();
   if (fCacheRead) {
      Int_t st = fCacheRead->ReadBuffer(buf, off, len);
      if (st < 0)
         return 2;  // failure reading
      else if (st == 1) {
         // fOffset might have been changed via TFileCacheRead::ReadBuffer(), reset it
         SetOffset(off + len);
         return 1;
      }
      // fOffset might have been changed via TFileCacheRead::ReadBuffer(), reset it
      Seek(off);
   } else {
      // if write cache is active check if data still in write cache
      if (fWritable && fCacheWrite) {
         if (fCacheWrite->ReadBuffer(buf, off, len) == 0) {
            SetOffset(off + len);
            return 1;
         }
         // fOffset might have been changed via TFileCacheWrite::ReadBuffer(), reset it
         SetOffset(off);
      }
   }

   return 0;
}

//______________________________________________________________________________
void TFile::ReadFree()
{
   // Read the FREE linked list.
   // Every file has a linked list (fFree) of free segments.
   // This linked list has been written on the file via WriteFree
   // as a single data record.

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
TProcessID  *TFile::ReadProcessID(UShort_t pidf)
{
   //The TProcessID with number pidf is read from this file.
   //If the object is not already entered in the gROOT list, it is added.

   TProcessID *pid = 0;
   TObjArray *pids = GetListOfProcessIDs();
   if (pidf < pids->GetSize()) pid = (TProcessID *)pids->UncheckedAt(pidf);
   if (pid) {
      pid->CheckInit();
      return pid;
   }

   //check if fProcessIDs[uid] is set in file
   //if not set, read the process uid from file
   char pidname[32];
   snprintf(pidname,32,"ProcessID%d",pidf);
   pid = (TProcessID *)Get(pidname);
   if (gDebug > 0) {
      printf("ReadProcessID, name=%s, file=%s, pid=%lx\n",pidname,GetName(),(Long_t)pid);
   }
   if (!pid) {
      //file->Error("ReadProcessID","Cannot find %s in file %s",pidname,file->GetName());
      return pid;
   }
      //check that a similar pid is not already registered in fgPIDs
   TObjArray *pidslist = TProcessID::GetPIDs();
   TIter next(pidslist);
   TProcessID *p;
   while ((p = (TProcessID*)next())) {
      if (!strcmp(p->GetTitle(),pid->GetTitle())) {
         delete pid;
         pids->AddAtAndExpand(p,pidf);
         p->IncrementCount();
         return p;
      }
   }
   pids->AddAtAndExpand(pid,pidf);
   pid->IncrementCount();
   pidslist->Add(pid);
   Int_t ind = pidslist->IndexOf(pid);
   pid->SetUniqueID((UInt_t)ind);
   return pid;
}


//______________________________________________________________________________
Int_t TFile::Recover()
{
   // Attempt to recover file if not correctly closed.
   // The function returns the number of keys that have been recovered.
   // If no keys can be recovered, the file will be declared Zombie by
   // the calling function. This function is automatically called when
   // opening a file.
   //
   // If the file is open in read only mode, the file is not modified.
   // If open in update mode and the function finds something to recover,
   //  a new directory header is written to the file. When opening the file gain
   //  no message from Recover will be reported.
   // If keys have been recovered, the file is usable and you can safely
   // read the corresponding objects.
   // If the file is not usable (a zombie), you can test for this case
   // with code like:
   //   TFile f("myfile.root");
   //   if (f.IsZombie()) {file is unusable)
   // If the file has been recovered, the bit kRecovered is set in the TFile object in memory.
   // You can test if the file has been recovered with
   //   if (f.TestBit(TFile::kRecovered)) {.. the file has been recovered}
   //
   // When writing TTrees to a file, it is important to save the Tree header
   // at regular intervals (see TTree::AutoSave). If a file containing a Tree
   // is recovered, the last Tree header written to the file will be used.
   // In this case all the entries in all the branches written before writing
   // the header are valid entries.
   //
   // One can disable the automatic recovery procedure by setting
   // TFile.Recover 0
   // in the system.rootrc file.

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
      int i, nwhci = nwhc;
      for (i = 0;i < nwhc; i++) frombuf(buffer, &classname[i]);
      classname[nwhci] = '\0';
      TDatime::GetDateTime(datime, date, time);
      TClass *tclass = TClass::GetClass(classname);
      if (seekpdir == fSeekDir && tclass && !tclass->InheritsFrom(TFile::Class())
                               && strcmp(classname,"TBasket")) {
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
      TFree *last = (TFree*)fFree->Last();
      if (last) {
         last->AddFree(fFree,fEND,max_file_size);
      } else {
         new TFree(fFree,fEND,max_file_size);
      }
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

         FlushWriteCache();

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
void TFile::SetOffset(Long64_t offset, ERelativeTo pos)
{
   // Set position from where to start reading.

   switch (pos) {
      case kBeg:
         fOffset = offset + fArchiveOffset;
         break;
      case kCur:
         fOffset += offset;
         break;
      case kEnd:
         // this option is not used currently in the ROOT code
         if (fArchiveOffset)
            Error("SetOffset", "seeking from end in archive is not (yet) supported");
         fOffset = fEND + offset;  // is fEND really EOF or logical EOF?
         break;
   }
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
   Long64_t retpos;
   if ((retpos = SysSeek(fD, offset, whence)) < 0)
      SysError("Seek", "cannot seek to position %lld in file %s, retpos=%lld",
               offset, GetName(), retpos);

   // used by TFileCacheRead::ReadBuffer()
   fOffset = retpos;
}

//______________________________________________________________________________
void TFile::SetCompressionAlgorithm(Int_t algorithm)
{
   // See comments for function SetCompressionSettings
   if (algorithm < 0 || algorithm >= ROOT::kUndefinedCompressionAlgorithm) algorithm = 0;
   if (fCompress < 0) {
      // if the level is not defined yet use 1 as a default
      fCompress = 100 * algorithm + 1;
   } else {
      int level = fCompress % 100;
      fCompress = 100 * algorithm + level;
   }
}

//______________________________________________________________________________
void TFile::SetCompressionLevel(Int_t level)
{
   // See comments for function SetCompressionSettings
   if (level < 0) level = 0;
   if (level > 99) level = 99;
   if (fCompress < 0) {
      // if the algorithm is not defined yet use 0 as a default
      fCompress = level;
   } else {
      int algorithm = fCompress / 100;
      if (algorithm >= ROOT::kUndefinedCompressionAlgorithm) algorithm = 0;
      fCompress = 100 * algorithm + level;
   }
}

//______________________________________________________________________________
void TFile::SetCompressionSettings(Int_t settings)
{
   // Used to specify the compression level and algorithm:
   //  settings = 100 * algorithm + level
   //
   //  level = 0, objects written to this file will not be compressed.
   //  level = 1, minimal compression level but fast.
   //  ....
   //  level = 9, maximal compression level but slower and might use more memory.
   // (For the currently supported algorithms, the maximum level is 9)
   // If compress is negative it indicates the compression level is not set yet.
   //
   // The enumeration ROOT::ECompressionAlgorithm associates each
   // algorithm with a number. There is a utility function to help
   // to set the value of the argument. For example,
   //   ROOT::CompressionSettings(ROOT::kLZMA, 1)
   // will build an integer which will set the compression to use
   // the LZMA algorithm and compression level 1.  These are defined
   // in the header file Compression.h.
   //
   // Note that the compression settings may be changed at any time.
   // The new compression settings will only apply to branches created
   // or attached after the setting is changed and other objects written
   // after the setting is changed.
   fCompress = settings;
}

//______________________________________________________________________________
void TFile::SetCacheRead(TFileCacheRead *cache)
{
   // Set a pointer to the read cache.
   // NOTE:  This relinquish ownership of the previous cache, so if you do not
   // already have a pointer to the previous cache (and there was a previous
   // cache), you ought to retrieve (and delete it if needed) using:
   //    TFileCacheRead *older = myfile->GetCacheRead();

   fCacheRead = cache;
}

//______________________________________________________________________________
void TFile::SetCacheWrite(TFileCacheWrite *cache)
{
   // Set a pointer to the write cache.
   // If file is null the existing write cache is deleted.

   if (!cache && fCacheWrite) delete fCacheWrite;
   fCacheWrite = cache;
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
   // Increment statistics for buffer sizes of objects in this file.

   fWritten++;
   fSumBuffer  += bufsize;
   fSum2Buffer += bufsize*bufsize;
}

//_______________________________________________________________________
void TFile::UseCache(Int_t /*maxCacheSize*/, Int_t /*pageSize*/)
{
   // Dummy function kept for backward compatibility.
   // The read  cache is now managed by TFileCacheRead
   // The write cache is now managed by TFileCacheWrite
   // Both caches are created automatically by the system.

   Obsolete("UseCache", "v5-30-00", "v5-32-00");
}

//______________________________________________________________________________
Int_t TFile::Write(const char *, Int_t opt, Int_t bufsiz)
{
   // Write memory objects to this file.
   // Loop on all objects in memory (including subdirectories).
   // A new key is created in the KEYS linked list for each object.
   // The list of keys is then saved on the file (via WriteKeys)
   // as a single data record.
   // For values of opt see TObject::Write().
   // The directory header info is rewritten on the directory header record.
   // The linked list of FREE segments is written.
   // The file header is written (bytes 1->fBEGIN).

   if (!IsWritable()) {
      if (!TestBit(kWriteError)) {
         // Do not print the warning if we already had a SysError.
         Warning("Write", "file %s not opened in write mode", GetName());
      }
      return 0;
   }

   if (gDebug) {
      if (!GetTitle() || strlen(GetTitle()) == 0)
         Info("Write", "writing name = %s", GetName());
      else
         Info("Write", "writing name = %s title = %s", GetName(), GetTitle());
   }

   fMustFlush = kFALSE;
   Int_t nbytes = TDirectoryFile::Write(0, opt, bufsiz); // Write directory tree
   WriteStreamerInfo();
   WriteFree();                       // Write free segments linked list
   WriteHeader();                     // Now write file header
   fMustFlush = kTRUE;

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

      Int_t st;
      if ((st = WriteBufferViaCache(buf, len))) {
         if (st == 2)
            return kTRUE;
         return kFALSE;
      }

      ssize_t siz;
      gSystem->IgnoreInterrupt();
      while ((siz = SysWrite(fD, buf, len)) < 0 && GetErrno() == EINTR)
         ResetErrno();
      gSystem->IgnoreInterrupt(kFALSE);
      if (siz < 0) {
         // Write the system error only once for this file
         SetBit(kWriteError); SetWritable(kFALSE);
         SysError("WriteBuffer", "error writing to file %s (%ld)", GetName(), (Long_t)siz);
         return kTRUE;
      }
      if (siz != len) {
         SetBit(kWriteError);
         Error("WriteBuffer", "error writing all requested bytes to file %s, wrote %ld of %d",
               GetName(), (Long_t)siz, len);
         return kTRUE;
      }
      fBytesWrite  += siz;
      fgBytesWrite += siz;

      if (gMonitoringWriter)
         gMonitoringWriter->SendFileWriteProgress(this);

      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TFile::WriteBufferViaCache(const char *buf, Int_t len)
{
   // Write buffer via cache. Returns 0 if cache is not active, 1 in case
   // write via cache was successful, 2 in case write via cache failed.

   if (!fCacheWrite) return 0;

   Int_t st;
   Long64_t off = GetRelOffset();
   if ((st = fCacheWrite->WriteBuffer(buf, off, len)) < 0) {
      SetBit(kWriteError);
      Error("WriteBuffer", "error writing to cache");
      return 2;
   }
   if (st > 0) {
      // fOffset might have been changed via TFileCacheWrite::WriteBuffer(), reset it
      Seek(off + len);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void TFile::WriteFree()
{
   // Write FREE linked list on the file.
   // The linked list of FREE segments (fFree) is written as a single data
   // record.

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
   char *start = buffer;

   next.Reset();
   while ((afree = (TFree*) next())) {
      afree->FillBuffer(buffer);
   }
   if ( (buffer-start)!=nbytes ) {
      // Most likely one of the 'free' segment was used to store this
      // TKey, so we had one less TFree to store than we planned.
      memset(buffer,0,nbytes-(buffer-start));
   }
   fNbytesFree = key->GetNbytes();
   fSeekFree   = key->GetSeekKey();
   key->WriteFile();
   delete key;
}

//______________________________________________________________________________
void TFile::WriteHeader()
{
   // Write File Header.

   SafeDelete(fInfoCache);
   TFree *lastfree = (TFree*)fFree->Last();
   if (lastfree) fEND  = lastfree->GetFirst();
   const char *root = "root";
   char *psave  = new char[fBEGIN];
   char *buffer = psave;
   Int_t nfree  = fFree->GetSize();
   memcpy(buffer, root, 4); buffer += 4;
   Int_t version = fVersion;
   if (version <1000000 && fEND > kStartBigFile) {version += 1000000; fUnits = 8;}
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
   // Generate code in directory dirname for all classes specified in
   // argument classes If classes = "*" (default and currently the
   // only supported value), the function generates an include file
   // for each class in the StreamerInfo list for which a TClass
   // object does not exist.
   //
   // The code generated includes:
   //    dirnameProjectHeaders.h  // contains one #include statement per generated header file
   //    dirnameProjectSource.cxx // contains all the constructors and destructors implementation.
   // and one header per class that is not nested inside another class.
   // The header file name is the fully qualified name of the class after all the special characters
   // "<>,:" are replaced by underscored.  For example for pair<edm::Vertex,int> the file name is
   // pair_edm__Vertex_int_.h
   //
   // In the generated classes, map, multimap when the first template parameter is a class
   // are replaced by a vector of pair. set and multiset when the tempalte parameter
   // is a class are replaced by a vector. This is required since we do not have the
   // code needed to order and/or compare the object of the classes.
   //
   // If option = "new" (default) a new directory dirname is created.
   //                   If dirname already exist, an error message is printed
   //                   and the function returns.
   // If option = "recreate", then;
   //                   if dirname does not exist, it is created (like in "new")
   //                   if dirname already exist, all existing files in dirname
   //                   are deleted before creating the new files.
   // If option = "update", then new classes are added to the existing directory.
   //                   Existing classes with the same name are replaced by the
   //                   new definition. If the directory dirname doest not exist,
   //                   same effect as "new".
   // If option = "genreflex", then use genreflex rather than rootcint to generate
   //                   the dictionary.
   // If option = "par", create a PAR file with the minimal set of code needed to read the content
   //                   of the ROOT file. The name of the PAR file is basename(dirname), with extension
   //                   '.par' enforced; the PAR file will be created at dirname(dirname) .
   // If, in addition to one of the 3 above options, the option "+" is specified,
   // the function will generate:
   //   - a script called MAKEP to build the shared lib
   //   - a dirnameLinkDef.h file
   //   - rootcint will be run to generate a dirnameProjectDict.cxx file
   //   - dirnameProjectDict.cxx will be compiled with the current options in compiledata.h
   //   - a shared lib dirname.so will be created.
   // If the option "++" is specified, the generated shared lib is dynamically
   // linked with the current executable module.
   // If the option "+" and "nocompile" are specified, the utility files are generated
   // as in the option "+" but they are not executed.
   // Example:
   //  file.MakeProject("demo","*","recreate++");
   //  - creates a new directory demo unless it already exist
   //  - clear the previous directory content
   //  - generate the xxx.h files for all classes xxx found in this file
   //    and not yet known to the CINT dictionary.
   //  - creates the build script MAKEP
   //  - creates a LinkDef.h file
   //  - runs rootcint generating demoProjectDict.cxx
   //  - compiles demoProjectDict.cxx into demoProjectDict.o
   //  - generates a shared lib demo.so
   //  - dynamically links the shared lib demo.so to the executable
   //  If only the option "+" had been specified, one can still link the
   //  shared lib to the current executable module with:
   //     gSystem->load("demo/demo.so");
   //
   // The following feature is not yet enabled:
   // One can restrict the list of classes to be generated by using expressions like:
   //   classes = "Ali*" generate code only for classes starting with Ali
   //   classes = "myClass" generate code for class MyClass only.
   //

   TString opt = option;
   opt.ToLower();
   Bool_t makepar = kFALSE;
   TString parname, pardir;
   if (opt.Contains("par")) {
      // Create a PAR file
      parname = gSystem->BaseName(dirname);
      if (parname.EndsWith(".par")) parname.ReplaceAll(".par","");
      pardir = gSystem->DirName(dirname);
      // Cleanup or prepare the dirs
      TString path, filepath;
      void *dir = gSystem->OpenDirectory(pardir);
      if (dir) {
         path.Form("%s/%s", pardir.Data(), parname.Data());
         void *dirp = gSystem->OpenDirectory(path);
         if (dirp) {
            path += "/PROOF-INF";
            void *dirinf = gSystem->OpenDirectory(path);
            const char *afile = 0;
            if (dirinf) {
               while ((afile = gSystem->GetDirEntry(dirinf))) {
                  if (strcmp(afile,".") == 0) continue;
                  if (strcmp(afile,"..") == 0) continue;
                  filepath.Form("%s/%s", path.Data(), afile);
                  if (gSystem->Unlink(filepath))
                     Warning("MakeProject", "1: problems unlinking '%s' ('%s', '%s')", filepath.Data(), path.Data(), afile);
               }
               gSystem->FreeDirectory(dirinf);
            }
            gSystem->Unlink(path);
            path.Form("%s/%s", pardir.Data(), parname.Data());
            while ((afile = gSystem->GetDirEntry(dirp))) {
               if (strcmp(afile,".") == 0) continue;
               if (strcmp(afile,"..") == 0) continue;
               filepath.Form("%s/%s", path.Data(), afile);
               if (gSystem->Unlink(filepath))
                  Warning("MakeProject", "2: problems unlinking '%s' ('%s', '%s')", filepath.Data(), path.Data(), afile);
            }
            gSystem->FreeDirectory(dirp);
            if (gSystem->Unlink(path))
               Warning("MakeProject", "problems unlinking '%s'", path.Data());
         }
      }
      // Make sure that the releavnt dirs exists: this is mandatory, so we fail if unsuccessful
      path.Form("%s/%s/PROOF-INF", pardir.Data(), parname.Data());
      if (gSystem->mkdir(path, kTRUE)) {
         Error("MakeProject", "problems creating '%s'", path.Data());
         return;
      }
      makepar = kTRUE;

   } else {
      void *dir = gSystem->OpenDirectory(dirname);
      TString dirpath;

      if (opt.Contains("update")) {
         // check that directory exist, if not create it
         if (dir == 0) {
            gSystem->mkdir(dirname);
         }

      } else if (opt.Contains("recreate")) {
         // check that directory exist, if not create it
         if (dir == 0) {
            if (gSystem->mkdir(dirname) < 0) {
               Error("MakeProject","cannot create directory '%s'",dirname);
               return;               
            }
         }
         // clear directory
         while (dir) {
            const char *afile = gSystem->GetDirEntry(dir);
            if (afile == 0) break;
            if (strcmp(afile,".") == 0) continue;
            if (strcmp(afile,"..") == 0) continue;
            dirpath.Form("%s/%s",dirname,afile);
            gSystem->Unlink(dirpath);
         }

      } else {
         // new is assumed
         // if directory already exist, print error message and return
         if (dir) {
            Error("MakeProject","cannot create directory %s, already existing",dirname);
            gSystem->FreeDirectory(dir);
            return;
         }
         if (gSystem->mkdir(dirname) < 0) {
            Error("MakeProject","cannot create directory '%s'",dirname);
            return;               
         }
      }
      if (dir) {
         gSystem->FreeDirectory(dir);
      }
   }
   Bool_t genreflex = opt.Contains("genreflex");

   // we are now ready to generate the classes
   // loop on all TStreamerInfo
   TList *filelist = (TList*)GetStreamerInfoCache()->Clone();
   if (filelist == 0) {
      Error("MakeProject","file %s has no StreamerInfo", GetName());
      return;
   }

   TString clean_dirname(dirname);
   if (makepar) clean_dirname.Form("%s/%s", pardir.Data(), parname.Data());
   if (clean_dirname[clean_dirname.Length()-1]=='/') {
      clean_dirname.Remove(clean_dirname.Length()-1);
   } else if (clean_dirname[clean_dirname.Length()-1]=='\\') {
      clean_dirname.Remove(clean_dirname.Length()-1);
      if (clean_dirname[clean_dirname.Length()-1]=='\\') {
         clean_dirname.Remove(clean_dirname.Length()-1);
      }
   }
   TString subdirname( gSystem->BaseName(clean_dirname) );
   if (makepar) subdirname = parname;
   if (subdirname == "") {
      Error("MakeProject","Directory name must not be empty.");
      return;
   }

   // Start the source file
   TString spath; spath.Form("%s/%sProjectSource.cxx",clean_dirname.Data(),subdirname.Data());
   FILE *sfp = fopen(spath.Data(),"w");
   if (sfp ==0) {
      Error("MakeProject","Unable to create the source file %s.",spath.Data());
      return;
   }
   fprintf(sfp, "#include \"%sProjectHeaders.h\"\n\n",subdirname.Data() );
   if (!genreflex) fprintf(sfp, "#include \"%sLinkDef.h\"\n\n",subdirname.Data() );
   fprintf(sfp, "#include \"%sProjectDict.cxx\"\n\n",subdirname.Data() );
   fprintf(sfp, "struct DeleteObjectFunctor {\n");
   fprintf(sfp, "   template <typename T>\n");
   fprintf(sfp, "   void operator()(const T *ptr) const {\n");
   fprintf(sfp, "      delete ptr;\n");
   fprintf(sfp, "   }\n");
   fprintf(sfp, "   template <typename T, typename Q>\n");
   fprintf(sfp, "   void operator()(const std::pair<T,Q> &) const {\n");
   fprintf(sfp, "      // Do nothing\n");
   fprintf(sfp, "   }\n");
   fprintf(sfp, "   template <typename T, typename Q>\n");
   fprintf(sfp, "   void operator()(const std::pair<T,Q*> &ptr) const {\n");
   fprintf(sfp, "      delete ptr.second;\n");
   fprintf(sfp, "   }\n");
   fprintf(sfp, "   template <typename T, typename Q>\n");
   fprintf(sfp, "   void operator()(const std::pair<T*,Q> &ptr) const {\n");
   fprintf(sfp, "      delete ptr.first;\n");
   fprintf(sfp, "   }\n");
   fprintf(sfp, "   template <typename T, typename Q>\n");
   fprintf(sfp, "   void operator()(const std::pair<T*,Q*> &ptr) const {\n");
   fprintf(sfp, "      delete ptr.first;\n");
   fprintf(sfp, "      delete ptr.second;\n");
   fprintf(sfp, "   }\n");
   fprintf(sfp, "};\n\n");
   fclose( sfp );

   // loop on all TStreamerInfo classes to check for empty classes
   // and enums listed either as data member or template parameters,
   // and filter out 'duplicates' classes/streamerInfos.
   TStreamerInfo *info;
   TIter flnext(filelist);
   TList extrainfos;
   TList *list = new TList();
   while ((info = (TStreamerInfo*)flnext())) {
      if (info->IsA() != TStreamerInfo::Class()) {
         continue;
      }
      if (strstr(info->GetName(),"@@")) {
         // Skip schema evolution support streamerInfo
         continue;
      }
      TClass *cl = TClass::GetClass(info->GetName());
      if (cl) {
         if (cl->GetClassInfo()) continue; // skip known classes
      }
      // Find and use the proper rules for the TStreamerInfos.
      TMakeProject::GenerateMissingStreamerInfos( &extrainfos, info->GetName() );
      TIter enext( info->GetElements() );
      TStreamerElement *el;
      const ROOT::TSchemaMatch* rules = 0;
      if (cl && cl->GetSchemaRules()) {
         rules = cl->GetSchemaRules()->FindRules(cl->GetName(), info->GetClassVersion());
      }
      while( (el=(TStreamerElement*)enext()) ) {
         if (rules) {
            for(Int_t art = 0; art < rules->GetEntries(); ++art) {
               ROOT::TSchemaRule *rule = (ROOT::TSchemaRule*)rules->At(art);
               if( rule->IsRenameRule() || rule->IsAliasRule() )
                  continue;
               // Check whether this is an 'attribute' rule.
               if ( rule->HasTarget( el->GetName()) && rule->GetAttributes()[0] != 0 ) {
                  TString attr( rule->GetAttributes() );
                  attr.ToLower();
                  if (attr.Contains("owner")) {
                     if (attr.Contains("notowner")) {
                        el->SetBit(TStreamerElement::kDoNotDelete);
                     } else {
                        el->ResetBit(TStreamerElement::kDoNotDelete);
                     }
                  }
               }
            }
         }
         TMakeProject::GenerateMissingStreamerInfos(&extrainfos, el);
      }
      delete rules;
      TVirtualStreamerInfo *alternate = (TVirtualStreamerInfo*)list->FindObject(info->GetName());
      if (alternate) {
         if ((info->GetClass() && info->GetClassVersion() == info->GetClass()->GetClassVersion())
             || (info->GetClassVersion() > alternate->GetClassVersion()) ) {
            list->AddAfter(alternate, info);
            list->Remove(alternate);
         } // otherwise ingnore this info as not being the official one.
      } else {
         list->Add(info);
      }
   }
   // Now transfer the new StreamerInfo onto the main list and
   // to the owning list.
   TIter nextextra(&extrainfos);
   while ((info = (TStreamerInfo*)nextextra())) {
      list->Add(info);
      filelist->Add(info);
   }

   // loop on all TStreamerInfo classes
   TIter next(list);
   Int_t ngener = 0;
   while ((info = (TStreamerInfo*)next())) {
      if (info->IsA() != TStreamerInfo::Class()) {
         continue;
      }
      if (info->GetClassVersion()==-4) continue; // Skip outer level namespace
      TIter subnext(list);
      TStreamerInfo *subinfo;
      TList subClasses;
      Int_t len = strlen(info->GetName());
      while ((subinfo = (TStreamerInfo*)subnext())) {
         if (subinfo->IsA() != TStreamerInfo::Class()) {
            continue;
         }
         if (strncmp(info->GetName(),subinfo->GetName(),len)==0) {
            // The 'sub' StreamerInfo start with the main StreamerInfo name,
            // it subinfo is likely to be a nested class.
            const Int_t sublen = strlen(subinfo->GetName());
            if ( (sublen > len) && subinfo->GetName()[len+1]==':'
               && !subClasses.FindObject(subinfo->GetName()) /* We need to insure uniqueness */)
            {
               subClasses.Add(subinfo);
            }
         }
      }
      ngener += info->GenerateHeaderFile(clean_dirname.Data(),&subClasses,&extrainfos);
      subClasses.Clear("nodelete");
   }
   TString path;
   path.Form("%s/%sProjectHeaders.h",clean_dirname.Data(),subdirname.Data());
   FILE *allfp = fopen(path,"a");
   if (!allfp) {
      Error("MakeProject","Cannot open output file:%s\n",path.Data());
   } else {
      fprintf(allfp,"#include \"%sProjectInstances.h\"\n", subdirname.Data());
      fclose(allfp);
   }

   printf("MakeProject has generated %d classes in %s\n",ngener,clean_dirname.Data());

   // generate the shared lib
   if (!opt.Contains("+") && !makepar) {
      delete list;
      filelist->Delete();
      delete filelist;
      return;
   }

   // Makefiles files
   FILE *fpMAKE = 0;
   if (!makepar) {
      // Create the MAKEP file by looping on all *.h files
      // delete MAKEP if it already exists
#ifdef WIN32
      path.Form("%s/makep.cmd",clean_dirname.Data());
#else
      path.Form("%s/MAKEP",clean_dirname.Data());
#endif
#ifdef R__WINGCC
      fpMAKE = fopen(path,"wb");
#else
      fpMAKE = fopen(path,"w");
#endif
      if (!fpMAKE) {
         Error("MakeProject", "cannot open file %s", path.Data());
         delete list;
         filelist->Delete();
         delete filelist;
         return;
      }
   }

   // Add rootcint/genreflex statement generating ProjectDict.cxx
   FILE *ifp = 0;
   path.Form("%s/%sProjectInstances.h",clean_dirname.Data(),subdirname.Data());
#ifdef R__WINGCC
   ifp = fopen(path,"wb");
#else
   ifp = fopen(path,"w");
#endif
   if (!ifp) {
      Error("MakeProject", "cannot open path file %s", path.Data());
      delete list;
      filelist->Delete();
      delete filelist;
      fclose(fpMAKE);
      return;
   }

   if (!makepar) {
      if (genreflex) {
         fprintf(fpMAKE,"genreflex %sProjectHeaders.h -o %sProjectDict.cxx --comments --iocomments %s ",subdirname.Data(),subdirname.Data(),gSystem->GetIncludePath());
         path.Form("%s/%sSelection.xml",clean_dirname.Data(),subdirname.Data());
      } else {
         fprintf(fpMAKE,"rootcint -f %sProjectDict.cxx -c %s ",subdirname.Data(),gSystem->GetIncludePath());
         path.Form("%s/%sLinkDef.h",clean_dirname.Data(),subdirname.Data());
      }
   } else {
      path.Form("%s/%sLinkDef.h",clean_dirname.Data(),subdirname.Data());
   }
      
   // Create the LinkDef.h or xml selection file by looping on all *.h files
   // replace any existing file.
#ifdef R__WINGCC
   FILE *fp = fopen(path,"wb");
#else
   FILE *fp = fopen(path,"w");
#endif
   if (!fp) {
      Error("MakeProject", "cannot open path file %s", path.Data());
      delete list;
      filelist->Delete();
      delete filelist;
      fclose(fpMAKE);
      fclose(ifp);
      return;
   }
   if (genreflex) {
      fprintf(fp,"<lcgdict>\n");
      fprintf(fp,"\n");
   } else {
      fprintf(fp,"#ifdef __CINT__\n");
      fprintf(fp,"\n");
   }

   TString tmp;
   TString instances;
   TString selections;
   next.Reset();
   while ((info = (TStreamerInfo*)next())) {
      if (info->IsA() != TStreamerInfo::Class()) {
         continue;
      }
      TClass *cl = TClass::GetClass(info->GetName());
      if (cl) {
         if (cl->GetClassInfo()) continue; // skip known classes
         const ROOT::TSchemaMatch* rules = 0;
         if (cl->GetSchemaRules()) {
            rules = cl->GetSchemaRules()->FindRules(cl->GetName(), info->GetClassVersion());
            TString strrule;
            if (rules) {
               for(Int_t art = 0; art < rules->GetEntries(); ++art) {
                  ROOT::TSchemaRule *rule = (ROOT::TSchemaRule*)rules->At(art);
                  strrule.Clear();
                  if (genreflex) {
                     rule->AsString(strrule,"x");
                     strrule.Append("\n");
                     if ( selections.Index(strrule) == kNPOS ) {
                        selections.Append(strrule);
                     }
                  } else {
                     rule->AsString(strrule);
                     if (strncmp(strrule.Data(),"type=",5)==0) {
                        strrule.Remove(0,5);
                     }
                     fprintf(fp,"#pragma %s;\n",strrule.Data());
                  }
               }
            }
            delete rules;
         }

      }
      if (TClassEdit::IsSTLCont(info->GetName())) {
         std::vector<std::string> inside;
         int nestedLoc;
         TClassEdit::GetSplit( info->GetName(), inside, nestedLoc, TClassEdit::kLong64 );
         Int_t stlkind =  TClassEdit::STLKind(inside[0].c_str());
         TClass *key = TClass::GetClass(inside[1].c_str());
         if (key) {
            TString what;
            switch ( stlkind )  {
            case TClassEdit::kMap:
            case TClassEdit::kMultiMap:
               if (TClass::GetClass(inside[1].c_str())) {
                  what = "std::pair<";
                  what += TMakeProject::UpdateAssociativeToVector( inside[1].c_str() );
                  what += ",";
                  what += TMakeProject::UpdateAssociativeToVector( inside[2].c_str() );
                  if (what[what.Length()-1]=='>') {
                     what += " >";
                  } else {
                     what += ">";
                  }
                  if (genreflex) {
                     tmp.Form("<class name=\"%s\" />\n",what.Data());
                     if ( selections.Index(tmp) == kNPOS ) {
                        selections.Append(tmp);
                     }
                     tmp.Form("template class %s;\n",what.Data());
                     if ( instances.Index(tmp) == kNPOS ) {
                        instances.Append(tmp);
                     }
                  } else {
                     what.ReplaceAll("std::","");
                     TClass *paircl = TClass::GetClass(what.Data());
                     if (paircl == 0 || paircl->GetClassInfo() == 0) {
                        fprintf(fp,"#pragma link C++ class %s+;\n",what.Data());
                     }
                  }
                  break;
               }
            default:
               if (strncmp(key->GetName(),"pair<",strlen("pair<"))==0) {
                  if (genreflex) {
                     tmp.Form("<class name=\"%s\" />\n",key->GetName());
                     if ( selections.Index(tmp) == kNPOS ) {
                        selections.Append(tmp);
                     }
                     tmp.Form("template class %s;\n",key->GetName());
                     if ( instances.Index(tmp) == kNPOS ) {
                        instances.Append(tmp);
                     }
                  } else {
                     what.ReplaceAll("std::","");
                     fprintf(fp,"#pragma link C++ class %s+;\n",key->GetName());
                  }
               }
               break;
            }
         }
         continue;
      }
      {
         TString what(TMakeProject::UpdateAssociativeToVector(info->GetName()).Data());
         if (genreflex) {
            tmp.Form("<class name=\"%s\" />\n",what.Data());
            if ( selections.Index(tmp) == kNPOS ) {
               selections.Append(tmp);
            }
            if (what[what.Length()-1] == '>') {
               tmp.Form("template class %s;\n",what.Data());
               if ( instances.Index(tmp) == kNPOS ) {
                  instances.Append(tmp);
               }
            }
         } else {
            what.ReplaceAll("std::","");
            fprintf(fp,"#pragma link C++ class %s+;\n",what.Data());
         }
      }
      if (genreflex) {
         // Also request the dictionary for the STL container used as members ...
         TIter eliter( info->GetElements() );
         TStreamerElement *element;
         while( (element = (TStreamerElement*)eliter() ) ) {
            if (element->GetClass() && !element->GetClass()->IsLoaded() && element->GetClass()->GetCollectionProxy()) {
               TString what( TMakeProject::UpdateAssociativeToVector(element->GetClass()->GetName()) );
               tmp.Form("<class name=\"%s\" />\n",what.Data());
               if ( selections.Index(tmp) == kNPOS ) {
                  selections.Append(tmp);
               }
               tmp.Form("template class %s;\n",what.Data());
               if ( instances.Index(tmp) == kNPOS ) {
                  instances.Append(tmp);
               }
            }
         }
      }
   }
   if (genreflex) {
      fprintf(ifp,"#ifndef PROJECT_INSTANCES_H\n");
      fprintf(ifp,"#define PROJECT_INSTANCES_H\n");
      fprintf(ifp,"%s",instances.Data());
      fprintf(ifp,"#endif\n");
      fprintf(fp,"%s",selections.Data());
      fprintf(fp,"</lcgdict>\n");
   } else {
      fprintf(fp,"#endif\n");
   }
   fclose(fp);
   fclose(ifp);

   if (!makepar) {
      // add compilation line
      TString sdirname(subdirname);
      
      TString cmd = gSystem->GetMakeSharedLib();
      TString sources = TString::Format("%sProjectSource.cxx ", sdirname.Data());
      cmd.ReplaceAll("$SourceFiles",sources.Data());
      TString object = TString::Format("%sProjectSource.", sdirname.Data());
      object.Append( gSystem->GetObjExt() );
      cmd.ReplaceAll("$ObjectFiles", object.Data());
      cmd.ReplaceAll("$IncludePath",TString(gSystem->GetIncludePath()) + " -I" + clean_dirname.Data());
      cmd.ReplaceAll("$SharedLib",sdirname+"."+gSystem->GetSoExt());
      cmd.ReplaceAll("$LinkedLibs",gSystem->GetLibraries("","SDL"));
      cmd.ReplaceAll("$LibName",sdirname);
      cmd.ReplaceAll("$BuildDir",".");
      TString sOpt;
      TString rootbuild = ROOTBUILD;
      if (rootbuild.Index("debug",0,TString::kIgnoreCase)==kNPOS) {
         sOpt = gSystem->GetFlagsOpt();
      } else {
         sOpt = gSystem->GetFlagsDebug();
      }
      cmd.ReplaceAll("$Opt", sOpt);

      if (genreflex) {
         fprintf(fpMAKE,"-s %sSelection.xml \n",subdirname.Data());
      } else {
         fprintf(fpMAKE,"%sProjectHeaders.h ",subdirname.Data());
         fprintf(fpMAKE,"%sLinkDef.h \n",subdirname.Data());
      }

      fprintf(fpMAKE,"%s\n",cmd.Data());

      printf("%s/MAKEP file has been generated\n", clean_dirname.Data());

      fclose(fpMAKE);

   } else {

      // Create the Makefile
      TString filemake = TString::Format("%s/Makefile", clean_dirname.Data());
      if (MakeProjectParMake(parname, filemake.Data()) != 0) {
         Error("MakeProject", "problems creating PAR make file '%s'", filemake.Data());
         delete list;
         filelist->Delete();
         delete filelist;
         return;
      }
      // Get Makefile.arch
#ifdef ROOTETCDIR
      TString mkarchsrc = TString::Format("%s/Makefile.arch", ROOTETCDIR);
#else
      TString mkarchsrc("$(ROOTSYS)/etc/Makefile.arch");
#endif
      if (gSystem->ExpandPathName(mkarchsrc))
         Warning("MakeProject", "problems expanding '%s'", mkarchsrc.Data());
      TString mkarchdst = TString::Format("%s/Makefile.arch", clean_dirname.Data());
      if (gSystem->CopyFile(mkarchsrc.Data(), mkarchdst.Data(), kTRUE) != 0) {
         Error("MakeProject", "problems retrieving '%s' to '%s'", mkarchsrc.Data(), mkarchdst.Data());
         delete list;
         filelist->Delete();
         delete filelist;
         return;
      }
      // Create the Makefile
      TString proofinf = TString::Format("%s/PROOF-INF", clean_dirname.Data());
      if (MakeProjectParProofInf(parname, proofinf.Data()) != 0) {
         Error("MakeProject", "problems creating BUILD.sh and/or SETUP.C under '%s'", proofinf.Data());
         delete list;
         filelist->Delete();
         delete filelist;
         return;
      }

      // Make sure BUILD.sh is executable and create SETUP.C
      TString cmod = TString::Format("chmod +x %s/PROOF-INF/BUILD.sh", clean_dirname.Data());
#ifndef WIN32
      gSystem->Exec(cmod.Data());
#else
      // not really needed for Windows but it would work both both Unix and NT
      chmod(cmod.Data(), 00700);
#endif
      Printf("Files Makefile, Makefile.arch, PROOF-INF/BUILD.sh and"
             " PROOF-INF/SETUP.C have been generated under '%s'", clean_dirname.Data());

      // Generate the PAR file, if not Windows
#ifndef WIN32
      TString curdir = gSystem->WorkingDirectory();
      if (gSystem->ChangeDirectory(pardir)) {
         TString cmd = TString::Format("tar czvf %s.par %s", parname.Data(), parname.Data());
         gSystem->Exec(cmd.Data());
         if (gSystem->ChangeDirectory(curdir)) {
            Info("MakeProject", "PAR file %s.par generated", clean_dirname.Data());
         } else {
            Warning("MakeProject", "problems changing directory back to '%s'", curdir.Data());
         }
      } else {
         Error("MakeProject", "problems changing directory to '%s' - skipping PAR file generation", pardir.Data());
      }
#else
      Warning("MakeProject", "on Windows systems the PAR file cannot be generated out of the package directory!");
#endif
   }

   
   if (!makepar && !opt.Contains("nocompilation")) {
      // now execute the generated script compiling and generating the shared lib
      path = gSystem->WorkingDirectory();
      gSystem->ChangeDirectory(clean_dirname.Data());
#ifndef WIN32
      gSystem->Exec("chmod +x MAKEP");
      int res = !gSystem->Exec("./MAKEP");
#else
      // not really needed for Windows but it would work both both Unix and NT
      chmod("makep.cmd",00700);
      int res = !gSystem->Exec("MAKEP");
#endif
      gSystem->ChangeDirectory(path);
      path.Form("%s/%s.%s",clean_dirname.Data(),subdirname.Data(),gSystem->GetSoExt());
      if (res) printf("Shared lib %s has been generated\n",path.Data());

      //dynamically link the generated shared lib
      if (opt.Contains("++")) {
         res = !gSystem->Load(path);
         if (res) printf("Shared lib %s has been dynamically linked\n",path.Data());
      }
   }

   extrainfos.Clear("nodelete");
   // filelist->Clear("nodetele");
   delete list;
   filelist->Delete();
   delete filelist;
}

//______________________________________________________________________________
Int_t TFile::MakeProjectParMake(const char *pack, const char *filemake)
{
   // Create makefile at 'filemake' for PAR package 'pack'.
   // Called by MakeProject when option 'par' is given.
   // Return 0 on success, -1 on error.

   // Output file path must be defined
   if (!filemake || (filemake && strlen(filemake) <= 0)) {
      Error("MakeProjectParMake", "path for output file undefined!");
      return -1;
   }

   // Package name must be defined
   if (!pack || (pack && strlen(pack) <= 0)) {
      Error("MakeProjectParMake", "package name undefined!");
      return -1;
   }

#ifdef R__WINGCC
   FILE *fmk = fopen(filemake, "wb");
#else
   FILE *fmk = fopen(filemake, "w");
#endif
   if (!fmk) {
      Error("MakeProjectParMake", "cannot create file '%s' (errno: %d)", filemake, TSystem::GetErrno());
      return -1;
   }

   // Fill the file now
   fprintf(fmk, "# Makefile for the ROOT test programs.\n");
   fprintf(fmk, "# This Makefile shows how to compile and link applications\n");
   fprintf(fmk, "# using the ROOT libraries on all supported platforms.\n");
   fprintf(fmk, "#\n");
   fprintf(fmk, "# Copyright (c) 2000 Rene Brun and Fons Rademakers\n");
   fprintf(fmk, "#\n");
   fprintf(fmk, "# Author: this makefile has been automatically generated via TFile::MakeProject\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "include Makefile.arch\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "#------------------------------------------------------------------------------\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "PACKO        = %sProjectSource.$(ObjSuf)\n", pack);
   fprintf(fmk, "PACKS        = %sProjectSource.$(SrcSuf) %sProjectDict.$(SrcSuf)\n", pack, pack);
   fprintf(fmk, "PACKSO       = lib%s.$(DllSuf)\n", pack);
   fprintf(fmk, "\n");
   fprintf(fmk, "ifeq ($(PLATFORM),win32)\n");
   fprintf(fmk, "PACKLIB      = lib%s.lib\n", pack);
   fprintf(fmk, "else\n");
   fprintf(fmk, "PACKLIB      = $(PACKSO)\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "OBJS          = $(PACKO)\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "PROGRAMS      =\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "#------------------------------------------------------------------------------\n");
   fprintf(fmk, "\n");
   fprintf(fmk, ".SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "all:            $(PACKLIB)\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "$(PACKSO):     $(PACKO)\n");
   fprintf(fmk, "ifeq ($(ARCH),aix)\n");
   fprintf(fmk, "\t\t/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^\n");
   fprintf(fmk, "else\n");
   fprintf(fmk, "ifeq ($(ARCH),aix5)\n");
   fprintf(fmk, "\t\t/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^\n");
   fprintf(fmk, "else\n");
   fprintf(fmk, "ifeq ($(PLATFORM),macosx)\n");
   fprintf(fmk, "# We need to make both the .dylib and the .so\n");
   fprintf(fmk, "\t\t$(LD) $(SOFLAGS)$@ $(LDFLAGS) $^ $(OutPutOpt) $@ $(LIBS)\n");
   fprintf(fmk, "ifneq ($(subst $(MACOSX_MINOR),,1234),1234)\n");
   fprintf(fmk, "ifeq ($(MACOSX_MINOR),4)\n");
   fprintf(fmk, "\t\tln -sf $@ $(subst .$(DllSuf),.so,$@)\n");
   fprintf(fmk, "else\n");
   fprintf(fmk, "\t\t$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \\\n");
   fprintf(fmk, "\t\t   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "else\n");
   fprintf(fmk, "ifeq ($(PLATFORM),win32)\n");
   fprintf(fmk, "\t\tbindexplib $* $^ > $*.def\n");
   fprintf(fmk, "\t\tlib -nologo -MACHINE:IX86 $^ -def:$*.def \\\n");
   fprintf(fmk, "\t\t   $(OutPutOpt)$(PACKLIB)\n");
   fprintf(fmk, "\t\t$(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \\\n");
   fprintf(fmk, "\t\t   $(OutPutOpt)$@\n");
   fprintf(fmk, "else\n");
   fprintf(fmk, "\t\t$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(LIBS) $(EXPLLINKLIBS)\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "endif\n");
   fprintf(fmk, "\t\t@echo \"$@ done\"\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "clean:\n");
   fprintf(fmk, "\t\t@rm -f $(OBJS) core\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "distclean:      clean\n");
   fprintf(fmk, "\t\t@rm -f $(PROGRAMS) $(PACKSO) $(PACKLIB) *Dict.* *.def *.exp \\\n");
   fprintf(fmk, "\t\t   *.so *.lib *.dll *.d *.log .def so_locations\n");
   fprintf(fmk, "\t\t@rm -rf cxx_repository\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "# Dependencies\n");
   fprintf(fmk, "\n");
   fprintf(fmk, "%sProjectSource.$(ObjSuf): %sProjectHeaders.h %sLinkDef.h %sProjectDict.$(SrcSuf)\n", pack, pack, pack, pack);
   fprintf(fmk, "\n");
   fprintf(fmk, "%sProjectDict.$(SrcSuf): %sProjectHeaders.h %sLinkDef.h\n", pack, pack, pack);
   fprintf(fmk, "\t\t@echo \"Generating dictionary $@...\"\n");
   fprintf(fmk, "\t\t@rootcint -f $@ -c $^\n");
   fprintf(fmk, "\n");
   fprintf(fmk, ".$(SrcSuf).$(ObjSuf):\n");
   fprintf(fmk, "\t\t$(CXX) $(CXXFLAGS) -c $<\n");
   fprintf(fmk, "\n");

   // Close the file
   fclose(fmk);

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TFile::MakeProjectParProofInf(const char *pack, const char *proofinf)
{
   // Create BUILD.sh and SETUP.C under 'proofinf' for PAR package 'pack'.
   // Called by MakeProject when option 'par' is given.
   // Return 0 on success, -1 on error.

   // Output directory path must be defined ...
   if (!proofinf || (proofinf && strlen(proofinf) <= 0)) {
      Error("MakeProjectParProofInf", "directory path undefined!");
      return -1;
   }

   // ... and exist and be a directory
   Int_t rcst = 0;
   FileStat_t st;
   if ((rcst = gSystem->GetPathInfo(proofinf, st)) != 0 || !R_ISDIR(st.fMode)) {
      Error("MakeProjectParProofInf", "path '%s' %s", proofinf,
                       ((rcst == 0) ? "is not a directory" : "does not exist"));
      return -1;
   }

   // Package name must be defined
   if (!pack || (pack && strlen(pack) <= 0)) {
      Error("MakeProjectParProofInf", "package name undefined!");
      return -1;
   }

   TString path;

   // The BUILD.sh first
   path.Form("%s/BUILD.sh", proofinf);
#ifdef R__WINGCC
   FILE *f = fopen(path.Data(), "wb");
#else
   FILE *f = fopen(path.Data(), "w");
#endif
   if (!f) {
      Error("MakeProjectParProofInf", "cannot create file '%s' (errno: %d)",
                                      path.Data(), TSystem::GetErrno());
      return -1;
   }

   fprintf(f, "#! /bin/sh\n");
   fprintf(f, "# Build libEvent library.\n");
   fprintf(f, "\n");
   fprintf(f, "#\n");
   fprintf(f, "# The environment variables ROOTPROOFLITE and ROOTPROOFCLIENT can be used to\n");
   fprintf(f, "# adapt the script to the calling environment\n");
   fprintf(f, "#\n");
   fprintf(f, "# if test ! \"x$ROOTPROOFLITE\" = \"x\"; then\n");
   fprintf(f, "#    echo \"event-BUILD: PROOF-Lite node (session has $ROOTPROOFLITE workers)\"\n");
   fprintf(f, "# elif test ! \"x$ROOTPROOFCLIENT\" = \"x\"; then\n");
   fprintf(f, "#    echo \"event-BUILD: PROOF client\"\n");
   fprintf(f, "# else\n");
   fprintf(f, "#    echo \"event-BUILD: standard PROOF node\"\n");
   fprintf(f, "# fi\n");
   fprintf(f, "\n");
   fprintf(f, "if [ \"\" = \"clean\" ]; then\n");
   fprintf(f, "   make distclean\n");
   fprintf(f, "   exit 0\n");
   fprintf(f, "fi\n");
   fprintf(f, "\n");
   fprintf(f, "make\n");
   fprintf(f, "rc=$?\n");
   fprintf(f, "echo \"rc=$?\"\n");
   fprintf(f, "if [ $? != \"0\" ] ; then\n");
   fprintf(f, "   exit 1\n");
   fprintf(f, "fi\n");
   fprintf(f, "exit 0\n");

   // Close the file
   fclose(f);

   // Then SETUP.C
   path.Form("%s/SETUP.C", proofinf);
#ifdef R__WINGCC
   f = fopen(path.Data(), "wb");
#else
   f = fopen(path.Data(), "w");
#endif
   if (!f) {
      Error("MakeProjectParProofInf", "cannot create file '%s' (errno: %d)",
                                      path.Data(), TSystem::GetErrno());
      return -1;
   }

   fprintf(f, "Int_t SETUP()\n");
   fprintf(f, "{\n");
   fprintf(f, "\n");
   fprintf(f, "//\n");
   fprintf(f, "// The environment variables ROOTPROOFLITE and ROOTPROOFCLIENT can be used to\n");
   fprintf(f, "// adapt the macro to the calling environment\n");
   fprintf(f, "//\n");
   fprintf(f, "//   if (gSystem->Getenv(\"ROOTPROOFLITE\")) {\n");
   fprintf(f, "//      Printf(\"event-SETUP: PROOF-Lite node (session has %%s workers)\",\n");
   fprintf(f, "//                                   gSystem->Getenv(\"ROOTPROOFLITE\"));\n");
   fprintf(f, "//   } else if (gSystem->Getenv(\"ROOTPROOFCLIENT\")) {\n");
   fprintf(f, "//      Printf(\"event-SETUP: PROOF client\");\n");
   fprintf(f, "//   } else {\n");
   fprintf(f, "//      Printf(\"event-SETUP: standard PROOF node\");\n");
   fprintf(f, "//   }\n");
   fprintf(f, "\n");
   fprintf(f, "   if (gSystem->Load(\"lib%s\") == -1)\n", pack);
   fprintf(f, "      return -1;\n");
   fprintf(f, "   return 0;\n");
   fprintf(f, "}\n");
   fprintf(f, "\n");

   // Close the file
   fclose(f);

   // Done
   return 0;
}

//______________________________________________________________________________
void TFile::ReadStreamerInfo()
{
   // Read the list of StreamerInfo from this file.
   // The key with name holding the list of TStreamerInfo objects is read.
   // The corresponding TClass objects are updated.
   // Note that this function is not called if the static member fgReadInfo is falsse.
   //  (see TFile::SetReadStreamerInfo)

   TList *list = GetStreamerInfoList();
   if (!list) {
      MakeZombie();
      return;
   }

   list->SetOwner(kFALSE);

   if (gDebug > 0) Info("ReadStreamerInfo", "called for file %s",GetName());

   // loop on all TStreamerInfo classes
   TStreamerInfo *info;
   for (int mode=0;mode<2; ++mode) {
      // In order for the collection proxy to be initialized properly, we need
      // to setup the TStreamerInfo for non-stl class before the stl classes.
      TObjLink *lnk = list->FirstLink();
      while (lnk) {
         info = (TStreamerInfo*)lnk->GetObject();
         if (info == 0) {
            lnk = lnk->Next();
            continue;
         }
         if (info->IsA() != TStreamerInfo::Class()) {
            if (mode==1) {
               TObject *obj = (TObject*)info;
               if (strcmp(obj->GetName(),"listOfRules")==0) {
                  TList *listOfRules = (TList*)obj;
                  TObjLink *rulelnk = listOfRules->FirstLink();
                  while (rulelnk) {
                     TObjString *rule = (TObjString*)rulelnk->GetObject();
                     TClass::AddRule( rule->String().Data() );
                     rulelnk = rulelnk->Next();
                  }
               } else {
                  Warning("ReadStreamerInfo","%s has a %s in the list of TStreamerInfo.", GetName(), info->IsA()->GetName());
               }
               info->SetBit(kCanDelete);
            }
            lnk = lnk->Next();
            continue;
         }
         // This is a quick way (instead of parsing the name) to see if this is
         // the description of an STL container.
         if (info->GetElements()==0) {
            Warning("ReadStreamerInfo","The StreamerInfo for %s does not have a list of elements.",info->GetName());
            lnk = lnk->Next();
            continue;
         }
         TObject *element = info->GetElements()->UncheckedAt(0);
         Bool_t isstl = element && strcmp("This",element->GetName())==0;

         if ( (!isstl && mode ==0) || (isstl && mode ==1) ) {
               // Skip the STL container the first time around
               // Skip the regular classes the second time around;
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
         lnk = lnk->Next();
      }
   }
   fClassIndex->fArray[0] = 0;
   list->Clear();  //this will delete all TStreamerInfo objects with kCanDelete bit set
   delete list;
}

//______________________________________________________________________________
void TFile::SetReadStreamerInfo(Bool_t readinfo)
{
   // static function to set fgReadInfo.
   // If fgReadInfo is true (default) TFile::ReadStreamerInfo is called
   //  when opening the file.
   // It may be interesting to set fgReadInfo to false to speedup the file
   // opening time or in case libraries containing classes referenced
   // by the file have not yet been loaded.
   // if fgReadInfo is false, one can still read the StreamerInfo with
   //    myfile.ReadStreamerInfo();

   fgReadInfo = readinfo;
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
UShort_t TFile::WriteProcessID(TProcessID *pidd)
{
   // Check if the ProcessID pidd is already in the file,
   // if not, add it and return the index  number in the local file list.

   TProcessID *pid = pidd;
   if (!pid) pid = TProcessID::GetPID();
   TObjArray *pids = GetListOfProcessIDs();
   Int_t npids = GetNProcessIDs();
   for (Int_t i=0;i<npids;i++) {
      if (pids->At(i) == pid) return (UShort_t)i;
   }

   this->SetBit(TFile::kHasReferences);
   pids->AddAtAndExpand(pid,npids);
   pid->IncrementCount();
   char name[32];
   snprintf(name,32,"ProcessID%d",npids);
   this->WriteTObject(pid,name);
   this->IncrementProcessIDs();
   if (gDebug > 0) {
      Info("WriteProcessID", "name=%s, file=%s", name, GetName());
   }
   return (UShort_t)npids;
}


//______________________________________________________________________________
void TFile::WriteStreamerInfo()
{
   // Write the list of TStreamerInfo as a single object in this file
   // The class Streamer description for all classes written to this file
   // is saved. See class TStreamerInfo.

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
   TList listOfRules;
   listOfRules.SetOwner(kTRUE);
   listOfRules.SetName("listOfRules");
   std::set<TClass*> classSet;


   while ((info = (TStreamerInfo*)next())) {
      Int_t uid = info->GetNumber();
      if (fClassIndex->fArray[uid]) {
         list.Add(info);
         if (gDebug > 0) printf(" -class: %s info number %d saved\n",info->GetName(),uid);

         // Add the IO customization rules to the list to be saved for the underlying
         // class but make sure to add them only once.
         TClass *clinfo = info->GetClass();
         if (clinfo && clinfo->GetSchemaRules()) {
            if ( classSet.find( clinfo ) == classSet.end() ) {
               if (gDebug > 0) printf(" -class: %s stored the I/O customization rules\n",info->GetName());

               TObjArrayIter it( clinfo->GetSchemaRules()->GetRules() );
               ROOT::TSchemaRule *rule;
               while( (rule = (ROOT::TSchemaRule*)it.Next()) ) {
                  TObjString *obj = new TObjString();
                  rule->AsString(obj->String());
                  listOfRules.Add(obj);
               }
               classSet.insert(clinfo);
            }
         }
      }
   }

   // Write the StreamerInfo list even if it is empty.
   fClassIndex->fArray[0] = 2; //to prevent adding classes in TStreamerInfo::TagFile

   if (listOfRules.GetEntries()) {
      // Only add the list of rules if we have something to say.
      list.Add(&listOfRules);
   }

   // always write with compression on
   Int_t compress = fCompress;
   fCompress = 1;

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
   fCompress = compress;

   list.RemoveLast(); // remove the listOfRules.
}

//______________________________________________________________________________
TFile *TFile::OpenFromCache(const char *name, Option_t *, const char *ftitle,
                   Int_t compress, Int_t netopt)
{
   // Static member function allowing to open a file for reading through the file
   // cache. The file will be downloaded to the cache and opened from there.
   // If the download fails, it will be opened remotely.
   // The file will be downloaded to the directory specified by SetCacheFileDir().

   TFile *f = 0;

   if (fgCacheFileDir == "") {
      ::Warning("TFile::OpenFromCache",
                "you want to read through a cache, but you have no valid cache "
                "directory set - reading remotely");
      ::Info("TFile::OpenFromCache", "set cache directory using TFile::SetCacheFileDir()");
   } else {
      TUrl fileurl(name);
      TUrl tagurl;

      if ((!strcmp(fileurl.GetProtocol(), "file"))) {
         // it makes no sense to read local files through a file cache
         if (!fgCacheFileForce)
            ::Warning("TFile::OpenFromCache",
                      "you want to read through a cache, but you are reading "
                      "local files - CACHEREAD disabled");
      } else {
         // this is a remote file and worthwhile to be put into the local cache
         // now create cachepath to put it
         TString cachefilepath;
         TString cachefilepathbasedir;
         cachefilepath = fgCacheFileDir;
         cachefilepath += fileurl.GetFile();
         cachefilepathbasedir = gSystem->DirName(cachefilepath);
         if ((gSystem->mkdir(cachefilepathbasedir, kTRUE) < 0) &&
               (gSystem->AccessPathName(cachefilepathbasedir, kFileExists))) {
            ::Warning("TFile::OpenFromCache","you want to read through a cache, but I "
                      "cannot create the directory %s - CACHEREAD disabled",
                      cachefilepathbasedir.Data());
         } else {
            // check if this should be a zip file
            if (strlen(fileurl.GetAnchor())) {
               // remove the anchor and change the target name
               cachefilepath += "__";
               cachefilepath += fileurl.GetAnchor();
               fileurl.SetAnchor("");
            }
            if (strstr(name,"zip=")) {
               // filter out this option and change the target cache name
               TString urloptions = fileurl.GetOptions();
               TString newoptions;
               TObjArray *objOptions = urloptions.Tokenize("&");
               Int_t optioncount = 0;
               TString zipname;
               for (Int_t n = 0; n < objOptions->GetEntries(); n++) {
                  TString loption = ((TObjString*)objOptions->At(n))->GetName();
                  TObjArray *objTags = loption.Tokenize("=");
                  if (objTags->GetEntries() == 2) {
                     TString key   = ((TObjString*)objTags->At(0))->GetName();
                     TString value = ((TObjString*)objTags->At(1))->GetName();
                     if (key.CompareTo("zip", TString::kIgnoreCase)) {
                        if (optioncount!=0) {
                           newoptions += "&";
                        }
                        newoptions += key;
                        newoptions += "=";
                        newoptions += value;
                        ++optioncount;
                     } else {
                        zipname = value;
                     }
                  }
                  delete objTags;
               }
               delete objOptions;
               fileurl.SetOptions(newoptions.Data());
               cachefilepath += "__";
               cachefilepath += zipname;
               fileurl.SetAnchor("");
            }

            Bool_t need2copy = kFALSE;

            // check if file is in the cache
            Long_t id;
            Long64_t size;
            Long_t flags;
            Long_t modtime;
            if (!gSystem->GetPathInfo(cachefilepath, &id, &size, &flags, &modtime)) {
               // file is in the cache
               if (!fgCacheFileDisconnected) {
                  char cacheblock[256];
                  char remotblock[256];
                  // check the remote file for it's size and compare some magic bytes
                  TString cfurl;
                  cfurl = cachefilepath;
                  cfurl += "?filetype=raw";
                  TUrl rurl(name);
                  TString ropt = rurl.GetOptions();
                  ropt += "&filetype=raw";
                  rurl.SetOptions(ropt);

                  Bool_t forcedcache = fgCacheFileForce;
                  fgCacheFileForce = kFALSE;

                  TFile *cachefile = TFile::Open(cfurl, "READ");
                  TFile *remotfile = TFile::Open(rurl.GetUrl(), "READ");

                  fgCacheFileForce = forcedcache;

                  if (!cachefile) {
                     need2copy = kTRUE;
                     ::Error("TFile::OpenFromCache",
                             "cannot open the cache file to check cache consistency");
                     return 0;
                  }

                  if (!remotfile) {
                     ::Error("TFile::OpenFromCache",
                             "cannot open the remote file to check cache consistency");
                     return 0;
                  }

                  cachefile->Seek(0);
                  remotfile->Seek(0);

                  if ((!cachefile->ReadBuffer(cacheblock,256)) &&
                      (!remotfile->ReadBuffer(remotblock,256))) {
                     if (memcmp(cacheblock, remotblock, 256)) {
                        ::Warning("TFile::OpenFromCache", "the header of the cache file "
                                  "differs from the remote file - forcing an update");
                        need2copy = kTRUE;
                     }
                  } else {
                     ::Warning("TFile::OpenFromCache", "the header of the cache and/or "
                               "remote file are not readable - forcing an update");
                     need2copy = kTRUE;
                  }

                  delete remotfile;
                  delete cachefile;
               }
            } else {
               need2copy = kTRUE;
            }

            // try to fetch the file (disable now the forced caching)
            Bool_t forcedcache = fgCacheFileForce;
            fgCacheFileForce = kFALSE;
            if (need2copy && !TFile::Cp(name, cachefilepath)) {
               ::Warning("TFile::OpenFromCache", "you want to read through a cache, but I "
                         "cannot make a cache copy of %s - CACHEREAD disabled",
                         cachefilepathbasedir.Data());
               fgCacheFileForce = forcedcache;
               if (fgOpenTimeout != 0)
                  return 0;
            } else {
               fgCacheFileForce = forcedcache;
               ::Info("TFile::OpenFromCache", "using local cache copy of %s [%s]",
                       name, cachefilepath.Data());
               // finally we have the file and can open it locally
               fileurl.SetProtocol("file");
               fileurl.SetFile(cachefilepath);

               tagurl = fileurl;
               TString tagfile;
               tagfile = cachefilepath;
               tagfile += ".ROOT.cachefile";
               tagurl.SetFile(tagfile);
               // we symlink this file as a ROOT cached file
               gSystem->Symlink(cachefilepath, tagfile);
               return TFile::Open(fileurl.GetUrl(), "READ", ftitle, compress, netopt);
            }
         }
      }
   }

   // Failed
   return f;
}

//______________________________________________________________________________
TFile *TFile::Open(const char *url, Option_t *options, const char *ftitle,
                   Int_t compress, Int_t netopt)
{
   // Static member function allowing the creation/opening of either a
   // TFile, TNetFile, TWebFile or any TFile derived class for which an
   // plugin library handler has been registered with the plugin manager
   // (for the plugin manager see the TPluginManager class). The returned
   // type of TFile depends on the file name specified by 'url'.
   // If 'url' is a '|'-separated list of file URLs, the 'URLs' are tried
   // sequentially in the specified order until a successful open.
   // If the file starts with "root:", "roots:" or "rootk:" a TNetFile object
   // will be returned, with "http:" a TWebFile, with "file:" a local TFile,
   // etc. (see the list of TFile plugin handlers in $ROOTSYS/etc/system.rootrc
   // for regular expressions that will be checked) and as last a local file will
   // be tried.
   // Before opening a file via TNetFile a check is made to see if the URL
   // specifies a local file. If that is the case the file will be opened
   // via a normal TFile. To force the opening of a local file via a
   // TNetFile use either TNetFile directly or specify as host "localhost".
   // The netopt argument is only used by TNetFile. For the meaning of the
   // options and other arguments see the constructors of the individual
   // file classes. In case of error returns 0.
   //
   // For TFile implementations supporting asynchronous file open, see
   // TFile::AsyncOpen(...), it is possible to request a timeout with the
   // option:
   //  TIMEOUT=<secs>   the timeout must be specified in seconds and
   //                   it will be internally checked with granularity of
   //                   one millisec.
   //
   // For remote files there is the option:
   //  CACHEREAD     opens an existing file for reading through the file cache.
   //                The file will be downloaded to the cache and opened from there.
   //                If the download fails, it will be opened remotely.
   //                The file will be downloaded to the directory specified by
   //                SetCacheFileDir().

   TPluginHandler *h;
   TFile *f = 0;
   EFileType type = kFile;

   // Check input
   if (!url || strlen(url) <= 0) {
      ::Error("TFile::Open", "no url specified");
      return f;
   }

   TString expandedUrl(url);
   gSystem->ExpandPathName(expandedUrl);

   // If a timeout has been specified extract the value and try to apply it (it requires
   // support for asynchronous open, though; the following is completely transparent if
   // such support if not available for the required protocol)
   TString opts(options);
   Int_t ito = opts.Index("TIMEOUT=");
   if (ito != kNPOS) {
      TString sto = opts(ito + strlen("TIMEOUT="), opts.Length());
      while (!(sto.IsDigit()) && !(sto.IsNull())) { sto.Remove(sto.Length()-1,1); }
      if (!(sto.IsNull())) {
         // Timeout in millisecs
         Int_t toms = sto.Atoi() * 1000;
         if (gDebug > 0) ::Info("TFile::Open", "timeout of %d millisec requested", toms);
         // Remove from the options field
         sto.Insert(0, "TIMEOUT=");
         opts.ReplaceAll(sto, "");
         // Asynchrounous open
         TFileOpenHandle *fh = TFile::AsyncOpen(expandedUrl, opts, ftitle, compress, netopt);
         // Check the result in steps of 1 millisec
         TFile::EAsyncOpenStatus aos = TFile::kAOSNotAsync;
         aos = TFile::GetAsyncOpenStatus(fh);
         Int_t xtms = toms;
         while (aos != TFile::kAOSNotAsync && aos == TFile::kAOSInProgress && xtms > 0) {
            gSystem->Sleep(1);
            xtms -= 1;
            aos = TFile::GetAsyncOpenStatus(fh);
         }
         if (aos == TFile::kAOSNotAsync || aos == TFile::kAOSSuccess) {
            // Do open the file now
            f = TFile::Open(fh);
            if (gDebug > 0) {
               if (aos == TFile::kAOSSuccess)
                  ::Info("TFile::Open", "waited %d millisec for asynchronous open", toms - xtms);
               else
                  ::Info("TFile::Open", "timeout option not supported (requires asynchronous"
                                        " open support)");
            }
         } else {
            if (xtms <= 0)
               ::Error("TFile::Open", "timeout expired while opening '%s'", expandedUrl.Data());
            // Cleanup the request
            SafeDelete(fh);
         }
         // Done
         return f;
      } else {
         ::Warning("TFile::Open", "incomplete 'TIMEOUT=' option specification - ignored");
         opts.ReplaceAll("TIMEOUT=", "");
      }
   }

   // We will use this from now on
   const char *option = opts;

   // Many URLs? Redirect output and print errors in case of global failure
   TString namelist(expandedUrl);
   Ssiz_t ip = namelist.Index("|");
   Bool_t rediroutput = (ip != kNPOS &&
                         ip != namelist.Length()-1 && gDebug <= 0) ? kTRUE : kFALSE;
   RedirectHandle_t rh;
   if (rediroutput) {
      TString outf = ".TFileOpen_";
      FILE *fout = gSystem->TempFileName(outf);
      if (fout) {
         fclose(fout);
         gSystem->RedirectOutput(outf, "w", &rh);
      }
   }

   // Try sequentially all names in 'names'
   TString name, n;
   Ssiz_t from = 0;
   while (namelist.Tokenize(n, from, "|") && !f) {

      // check if we read through a file cache
      if (!strcasecmp(option, "CACHEREAD") ||
         ((!strcasecmp(option,"READ") || !strlen(option)) && fgCacheFileForce)) {
         // Try opening the file from the cache
         if ((f = TFile::OpenFromCache(n, option, ftitle, compress, netopt)))
            return f;
      }

      IncrementFileCounter();

      // change names from e.g. /castor/cern.ch/alice/file.root to
      // castor:/castor/cern.ch/alice/file.root as recognized by the plugin manager
      TUrl urlname(n, kTRUE);
      name = urlname.GetUrl();
      // Check first if a pending async open request matches this one
      if (fgAsyncOpenRequests && (fgAsyncOpenRequests->GetSize() > 0)) {
         TIter nxr(fgAsyncOpenRequests);
         TFileOpenHandle *fh = 0;
         while ((fh = (TFileOpenHandle *)nxr()))
            if (fh->Matches(name))
               return TFile::Open(fh);
      }

      TString urlOptions(urlname.GetOptions());
      if (urlOptions.BeginsWith("pmerge") || urlOptions.Contains("&pmerge") || urlOptions.Contains(" pmerge")) {
         type = kMerge;

         // Pass the full name including the url options:
         f = (TFile*) gROOT->ProcessLineFast(TString::Format("new TParallelMergingFile(\"%s\",\"%s\",\"%s\",%d)",n.Data(),option,ftitle,compress));
        
      } else {
         // Resolve the file type; this also adjusts names
         TString lfname = gEnv->GetValue("Path.Localroot", "");
         type = GetType(name, option, &lfname);
         
         if (type == kLocal) {
            
            // Local files
            if (lfname.IsNull()) {
               urlname.SetHost("");
               urlname.SetProtocol("file");
               lfname = urlname.GetUrl();
            }
            f = new TFile(lfname.Data(), option, ftitle, compress);
            
         } else if (type == kNet) {
            
            // Network files
            if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name))) {
               if (h->LoadPlugin() == -1)
                  return 0;
               f = (TFile*) h->ExecPlugin(5, name.Data(), option, ftitle, compress, netopt);
            }
            
         } else if (type == kWeb) {
            
            // Web files
            if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name))) {
               if (h->LoadPlugin() == -1)
                  return 0;
               f = (TFile*) h->ExecPlugin(2, name.Data(), option);
            }
            
         } else if (type == kFile) {
            
            // 'file:' protocol
            if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
                h->LoadPlugin() == 0) {
               name.ReplaceAll("file:", "");
               f = (TFile*) h->ExecPlugin(4, name.Data(), option, ftitle, compress);
            } else
               f = new TFile(name.Data(), option, ftitle, compress);
            
         } else {
            
            // no recognized specification: try the plugin manager
            if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name.Data()))) {
               if (h->LoadPlugin() == -1)
                  return 0;
               TClass *cl = TClass::GetClass(h->GetClass());
               if (cl && cl->InheritsFrom("TNetFile"))
                  f = (TFile*) h->ExecPlugin(5, name.Data(), option, ftitle, compress, netopt);
               else
                  f = (TFile*) h->ExecPlugin(4, name.Data(), option, ftitle, compress);
            } else {
               // Just try to open it locally but via TFile::Open, so that we pick-up the correct
               // plug-in in the case file name contains information about a special backend (e.g.
               // "srm://srm.cern.ch//castor/cern.ch/grid/..." should be considered a castor file
               // /castor/cern.ch/grid/...").
               f = TFile::Open(urlname.GetFileAndOptions(), option, ftitle, compress);
            }
         }
      }
      
      if (f && f->IsZombie()) {
         delete f;
         f = 0;
      }
   }

   if (rediroutput) {
      // Restore output to stdout
      gSystem->RedirectOutput(0, "", &rh);
      // If we failed print error messages
      if (!f)
         gSystem->ShowOutput(&rh);
      // Remove the file
      gSystem->Unlink(rh.fFile);
   }

   // if the file is writable, non local, and not opened in raw mode
   // we create a default write cache of 512 KBytes
   if (type != kLocal && type != kFile &&
       f && f->IsWritable() && !f->IsRaw()) {
      new TFileCacheWrite(f, 1);
   }

   return f;
}

//______________________________________________________________________________
TFileOpenHandle *TFile::AsyncOpen(const char *url, Option_t *option,
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

   // Check input
   if (!url || strlen(url) <= 0) {
      ::Error("TFile::AsyncOpen", "no url specified");
      return fh;
   }

   // Many URLs? Redirect output and print errors in case of global failure
   TString namelist(url);
   gSystem->ExpandPathName(namelist);
   Ssiz_t ip = namelist.Index("|");
   Bool_t rediroutput = (ip != kNPOS &&
                         ip != namelist.Length()-1 && gDebug <= 0) ? kTRUE : kFALSE;
   RedirectHandle_t rh;
   if (rediroutput) {
      TString outf = ".TFileAsyncOpen_";
      FILE *fout = gSystem->TempFileName(outf);
      if (fout) {
         fclose(fout);
         gSystem->RedirectOutput(outf, "w", &rh);
      }
   }

   // Try sequentially all names in 'names'
   TString name, n;
   Ssiz_t from = 0;
   while (namelist.Tokenize(n, from, "|") && !f) {

      // change names from e.g. /castor/cern.ch/alice/file.root to
      // castor:/castor/cern.ch/alice/file.root as recognized by the plugin manager
      TUrl urlname(n, kTRUE);
      name = urlname.GetUrl();

      // Resolve the file type; this also adjusts names
      EFileType type = GetType(name, option);

      // Here we send the asynchronous request if the functionality is implemented
      if (type == kNet) {
         // Network files
         if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
            !strcmp(h->GetClass(),"TXNetFile") && h->LoadPlugin() == 0) {
            f = (TFile*) h->ExecPlugin(6, name.Data(), option, ftitle, compress, netopt, kTRUE);
            notfound = kFALSE;
         }
      }
      if ((h = gROOT->GetPluginManager()->FindHandler("TFile", name)) &&
         !strcmp(h->GetClass(),"TAlienFile") && h->LoadPlugin() == 0) {
         f = (TFile*) h->ExecPlugin(5, name.Data(), option, ftitle, compress, kTRUE);
         notfound = kFALSE;
      }

   }

   if (rediroutput) {
      // Restore output to stdout
      gSystem->RedirectOutput(0, "", &rh);
      // If we failed print error messages
      if (!notfound && !f)
         gSystem->ShowOutput(&rh);
      // Remove the file
      gSystem->Unlink(rh.fFile);
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
      // Remove it from the pending list: we need to do it at this level to avoid
      // recursive calls in the standard TFile::Open
      fgAsyncOpenRequests->Remove(fh);
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

   if (fd < 0) return 0;
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
Long64_t TFile::GetBytesWritten() const
{
   // Return the total number of bytes written so far to the file.

   return fCacheWrite ? fCacheWrite->GetBytesInCache() + fBytesWrite : fBytesWrite;
}

//______________________________________________________________________________
Long64_t TFile::GetFileBytesRead()
{
   // Static function returning the total number of bytes read from all files.

   return fgBytesRead;
}

//______________________________________________________________________________
Long64_t TFile::GetFileBytesWritten()
{
   // Static function returning the total number of bytes written to all files.
   // Does not take into account what might still be in the write caches.

   return fgBytesWrite;
}

//______________________________________________________________________________
Int_t TFile::GetFileReadCalls()
{
   // Static function returning the total number of read calls from all files.

   return fgReadCalls;
}

//______________________________________________________________________________
Int_t TFile::GetReadaheadSize()
{
   // Static function returning the readahead buffer size.

   return fgReadaheadSize;
}

//______________________________________________________________________________
void TFile::SetReadaheadSize(Int_t bytes) { fgReadaheadSize = bytes; }

//______________________________________________________________________________
void TFile::SetFileBytesRead(Long64_t bytes) { fgBytesRead = bytes; }

//______________________________________________________________________________
void TFile::SetFileBytesWritten(Long64_t bytes) { fgBytesWrite = bytes; }

//______________________________________________________________________________
void TFile::SetFileReadCalls(Int_t readcalls) { fgReadCalls = readcalls; }

//______________________________________________________________________________
Long64_t TFile::GetFileCounter() { return fgFileCounter; }

//______________________________________________________________________________
void TFile::IncrementFileCounter() { fgFileCounter++; }

//______________________________________________________________________________
Bool_t TFile::SetCacheFileDir(const char *cachedir, Bool_t operatedisconnected,
                              Bool_t forcecacheread )
{
   // Sets the directory where to locally stage/cache remote files.
   // If the directory is not writable by us return kFALSE.

   TString cached = cachedir;
   if (!cached.EndsWith("/"))
      cached += "/";

   if (gSystem->AccessPathName(cached, kFileExists)) {
      // try to create it
      gSystem->mkdir(cached, kTRUE);
      if (gSystem->AccessPathName(cached, kFileExists)) {
         ::Error("TFile::SetCacheFileDir", "no suffcient permissions on cache directory %s or cannot create it", cachedir);
         fgCacheFileDir = "";
         return kFALSE;
      }
      gSystem->Chmod(cached, 0700);
   }
   if (gSystem->AccessPathName(cached, kWritePermission))
      gSystem->Chmod(cached, 0700);
   fgCacheFileDir          = cached;
   fgCacheFileDisconnected = operatedisconnected;
   fgCacheFileForce        = forcecacheread;
   return kTRUE;
}

//______________________________________________________________________________
const char *TFile::GetCacheFileDir()
{
   // Get the directory where to locally stage/cache remote files.

   return fgCacheFileDir;
}

//______________________________________________________________________________
Bool_t TFile::ShrinkCacheFileDir(Long64_t shrinksize, Long_t cleanupinterval)
{
   // We try to shrink the cache to the desired size.
   // With the clenupinterval you can specify the minimum amount of time after
   // the previous cleanup before the cleanup operation is repeated in
   // the cache directory

   if (fgCacheFileDir == "") {
      return kFALSE;
   }

   // check the last clean-up in the cache
   Long_t id;
   Long64_t size;
   Long_t flags;
   Long_t modtime;

   TString cachetagfile = fgCacheFileDir;
   cachetagfile += ".tag.ROOT.cache";
   if (!gSystem->GetPathInfo(cachetagfile, &id, &size, &flags, &modtime)) {
      // check the time passed since last cache cleanup
      Long_t lastcleanuptime = ((Long_t)time(0) - modtime);
      if (lastcleanuptime < cleanupinterval) {
         ::Info("TFile::ShrinkCacheFileDir", "clean-up is skipped - last cleanup %lu seconds ago - you requested %lu", lastcleanuptime, cleanupinterval);
         return kTRUE;
      }
   }

   // (re-)create the cache tag file
   cachetagfile += "?filetype=raw";
   TFile *tagfile = 0;

   if (!(tagfile = TFile::Open(cachetagfile, "RECREATE"))) {
      ::Error("TFile::ShrinkCacheFileDir", "cannot create the cache tag file %s", cachetagfile.Data());
      return kFALSE;
   }

   // the shortest garbage collector in the world - one long line of PERL - unlinks files only,
   // if there is a symbolic link with '.ROOT.cachefile' for safety ;-)

   TString cmd;
#if defined(R__WIN32)
   cmd = "echo <TFile::ShrinkCacheFileDir>: cleanup to be implemented";
#elif defined(R__MACOSX)
   cmd.Format("perl -e 'my $cachepath = \"%s\"; my $cachesize = %lld;my $findcommand=\"find $cachepath -type f -exec stat -f \\\"\\%%a::\\%%N::\\%%z\\\" \\{\\} \\\\\\;\";my $totalsize=0;open FIND, \"$findcommand | sort -k 1 |\";while (<FIND>) { my ($accesstime, $filename, $filesize) = split \"::\",$_; $totalsize += $filesize;if ($totalsize > $cachesize) {if ( ( -e \"${filename}.ROOT.cachefile\" ) && ( -e \"${filename}\" ) ) {unlink \"$filename.ROOT.cachefile\";unlink \"$filename\";}}}close FIND;' ", fgCacheFileDir.Data(),shrinksize);
#else
   cmd.Format("perl -e 'my $cachepath = \"%s\"; my $cachesize = %lld;my $findcommand=\"find $cachepath -type f -exec stat -c \\\"\\%%x::\\%%n::\\%%s\\\" \\{\\} \\\\\\;\";my $totalsize=0;open FIND, \"$findcommand | sort -k 1 |\";while (<FIND>) { my ($accesstime, $filename, $filesize) = split \"::\",$_; $totalsize += $filesize;if ($totalsize > $cachesize) {if ( ( -e \"${filename}.ROOT.cachefile\" ) && ( -e \"${filename}\" ) ) {unlink \"$filename.ROOT.cachefile\";unlink \"$filename\";}}}close FIND;' ", fgCacheFileDir.Data(),shrinksize);
#endif

   tagfile->WriteBuffer(cmd, 4096);
   delete tagfile;

   if ((gSystem->Exec(cmd)) != 0) {
      ::Error("TFile::ShrinkCacheFileDir", "error executing clean-up script");
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
UInt_t TFile::SetOpenTimeout(UInt_t timeout)
{
   // Sets open timeout time (in ms). Returns previous timeout value.

   UInt_t to = fgOpenTimeout;
   fgOpenTimeout = timeout;
   return to;
}

//______________________________________________________________________________
UInt_t TFile::GetOpenTimeout()
{
   // Returns open timeout (in ms).

   return fgOpenTimeout;
}

//______________________________________________________________________________
Bool_t TFile::SetOnlyStaged(Bool_t onlystaged)
{
   // Sets only staged flag. Returns previous value of flag.
   // When true we check before opening the file if it is staged, if not,
   // the open fails.

   Bool_t f = fgOnlyStaged;
   fgOnlyStaged = onlystaged;
   return f;
}

//______________________________________________________________________________
Bool_t TFile::GetOnlyStaged()
{
   // Returns staged only flag.

   return fgOnlyStaged;
}

//______________________________________________________________________________
Bool_t TFile::Matches(const char *url)
{
   // Return kTRUE if 'url' matches the coordinates of this file.
   // The check is implementation dependent and may need to be overload
   // by each TFile implememtation relying on this check.
   // The default implementation checks the file name only.

   // Check the full URL, including port and FQDN.
   TUrl u(url);

   // Check
   if (!strcmp(u.GetFile(), fUrl.GetFile())) {
      // Check ports
      if (u.GetPort() == fUrl.GetPort()) {
         if (!strcmp(u.GetHostFQDN(), fUrl.GetHostFQDN())) {
            // Ok, coordinates match
            return kTRUE;
         }
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
      if (!strcmp(u.GetFile(), uref.GetFile())) {
         // Check ports
         if (u.GetPort() == uref.GetPort()) {
            // Check also the host name
            if (!strcmp(u.GetHostFQDN(), uref.GetHostFQDN())) {
               // Ok, coordinates match
               return kTRUE;
            }
         }
      }
   }

   // Default is not matching
   return kFALSE;
}

//______________________________________________________________________________
TFile::EFileType TFile::GetType(const char *name, Option_t *option, TString *prefix)
{
   // Resolve the file type as a function of the protocol field in 'name'
   // If defined, the string 'prefix' is added when testing the locality of
   // a 'name' with network-like structure (i.e. root://host//path); if the file
   // is local, on return 'prefix' will contain the actual local path of the file.

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
      Bool_t localFile = kFALSE;
      TUrl url(name);
      //
      // Check whether we should try to optimize for local files
      Bool_t forceRemote = gEnv->GetValue("Path.ForceRemote", 0);
      forceRemote = (forceRemote) ? kTRUE : gEnv->GetValue("TFile.ForceRemote", 0);
      TString opts = url.GetOptions();
      if (opts.Contains("remote=1"))
         forceRemote = kTRUE;
      else if (opts.Contains("remote=0"))
         forceRemote = kFALSE;
      if (!forceRemote) {
         // Generic locality test
         localFile = gSystem->IsPathLocal(name);
         if (localFile) {
            // Local path including the prefix
            const char *fname = url.GetFileAndOptions();
            TString lfname;
            if (fname[0] == '/') {
               if (prefix)
                  lfname = Form("%s%s", prefix->Data(), fname);
               else
                  lfname = fname;
            } else if (fname[0] == '~' || fname[0] == '$') {
               lfname = fname;
            } else {
               lfname = Form("%s/%s", gSystem->HomeDirectory(), fname);
            }
            // If option "READ" test existence and access
            TString opt = option;
            Bool_t read = (opt.IsNull() ||
                          !opt.CompareTo("READ", TString::kIgnoreCase)) ? kTRUE : kFALSE;
            if (read) {
               char *fn;
               if ((fn = gSystem->ExpandPathName(TUrl(lfname).GetFile()))) {
                  if (gSystem->AccessPathName(fn, kReadPermission))
                     localFile = kFALSE;
                  delete [] fn;
               }
            }
            // Return full local path if requested (and if the case)
            if (localFile && prefix)
               *prefix = lfname;
         }
      }
      //
      // Adjust the type according to findings
      type = (localFile) ? kLocal : type;
   } else if (!strncmp(name, "http:", 5) ||
              !strncmp(name, "as3:", 4) ||
              !strncmp(name, "gs:", 3)) {
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

   if (handle && handle->fFile) {
      if (!handle->fFile->IsZombie())
         return handle->fFile->GetAsyncOpenStatus();
      else
         return TFile::kAOSFailure;
   }

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

//______________________________________________________________________________
void TFile::CpProgress(Long64_t bytesread, Long64_t size, TStopwatch &watch)
{
   // Print file copy progress.

   fprintf(stderr, "[TFile::Cp] Total %.02f MB\t|", (Double_t)size/1048576);

   for (int l = 0; l < 20; l++) {
      if (size > 0) {
         if (l < 20*bytesread/size)
            fprintf(stderr, "=");
         else if (l == 20*bytesread/size)
            fprintf(stderr, ">");
         else if (l > 20*bytesread/size)
            fprintf(stderr, ".");
      } else
         fprintf(stderr, "=");
   }
   // Allow to update the GUI while uploading files
   gSystem->ProcessEvents();
   watch.Stop();
   Double_t lCopy_time = watch.RealTime();
   fprintf(stderr, "| %.02f %% [%.01f MB/s]\r",
           100.0*(size?(bytesread/size):1), bytesread/lCopy_time/1048576.);
   watch.Continue();
}

//______________________________________________________________________________
Bool_t TFile::Cp(const char *dst, Bool_t progressbar, UInt_t buffersize)
{
   // Allows to copy this file to the dst URL. Returns kTRUE in case of success,
   // kFALSE otherwise.

   Bool_t rmdestiferror = kFALSE;
   TStopwatch watch;
   Bool_t success = kFALSE;

   TUrl dURL(dst, kTRUE);

   TString oopt = "RECREATE";
   TString ourl = dURL.GetUrl();

   // Files will be open in RAW mode
   TString raw = "filetype=raw";

   // Set optimization options for the destination file
   TString opt = dURL.GetOptions();
   if (opt != "") opt += "&";
   opt += raw;
   dURL.SetOptions(opt);

   char *copybuffer = 0;

   TFile *sfile = this;
   TFile *dfile = 0;

   // "RECREATE" does not work always well with XROOTD
   // namely when some pieces of the path are missing;
   // we force "NEW" in such a case
   if (TFile::GetType(ourl, "") == TFile::kNet) {
      if (gSystem->AccessPathName(ourl)) {
         oopt = "NEW";
         // Force creation of the missing parts of the path
         opt += "&mkpath=1";
         dURL.SetOptions(opt);
      }
   }

   // Open destination file
   if (!(dfile = TFile::Open(dURL.GetUrl(), oopt))) {
      ::Error("TFile::Cp", "cannot open destination file %s", dst);
      goto copyout;
   }

   // Probably we created a new file
   // We have to remove it in case of errors
   rmdestiferror = kTRUE;

   sfile->Seek(0);
   dfile->Seek(0);

   copybuffer = new char[buffersize];
   if (!copybuffer) {
      ::Error("TFile::Cp", "cannot allocate the copy buffer");
      goto copyout;
   }

   Bool_t   readop;
   Bool_t   writeop;
   Long64_t read;
   Long64_t written;
   Long64_t totalread;
   Long64_t filesize;
   Long64_t b00;
   filesize  = sfile->GetSize();
   totalread = 0;
   watch.Start();

   b00 = sfile->GetBytesRead();

   do {
      if (progressbar) CpProgress(totalread, filesize,watch);

      Long64_t b1 = sfile->GetBytesRead() - b00;

      Long64_t readsize;
      if (filesize - b1 > (Long64_t)buffersize) {
         readsize = buffersize;
      } else {
         readsize = filesize - b1;
      }

      if (readsize == 0) break;

      Long64_t b0 = sfile->GetBytesRead();
      sfile->Seek(totalread,TFile::kBeg);
      readop = sfile->ReadBuffer(copybuffer, (Int_t)readsize);
      read   = sfile->GetBytesRead() - b0;
      if ((read <= 0) || readop) {
         ::Error("TFile::Cp", "cannot read from source file %s. readsize=%lld read=%lld readop=%d",
                              sfile->GetName(), readsize, read, readop);
         goto copyout;
      }

      Long64_t w0 = dfile->GetBytesWritten();
      writeop = dfile->WriteBuffer(copybuffer, (Int_t)read);
      written = dfile->GetBytesWritten() - w0;
      if ((written != read) || writeop) {
         ::Error("TFile::Cp", "cannot write %lld bytes to destination file %s", read, dst);
         goto copyout;
      }
      totalread += read;
   } while (read == (Long64_t)buffersize);

   if (progressbar) {
      CpProgress(totalread, filesize,watch);
      fprintf(stderr, "\n");
   }

   success = kTRUE;

copyout:
   if (dfile) dfile->Close();

   if (dfile) delete dfile;
   if (copybuffer) delete[] copybuffer;

   if (rmdestiferror && (success != kTRUE))
      gSystem->Unlink(dst);

   watch.Stop();
   watch.Reset();

   return success;
}

//______________________________________________________________________________
Bool_t TFile::Cp(const char *src, const char *dst, Bool_t progressbar,
                 UInt_t buffersize)
{
   // Allows to copy file from src to dst URL. Returns kTRUE in case of success,
   // kFALSE otherwise.
   
   TUrl sURL(src, kTRUE);

   // Files will be open in RAW mode
   TString raw = "filetype=raw";
   
   // Set optimization options for the source file
   TString opt = sURL.GetOptions();
   if (opt != "") opt += "&";
   opt += raw;
   // Netx-related options:
   //    cachesz = 4*buffersize     -> 4 buffers as peak mem usage
   //    readaheadsz = 2*buffersize -> Keep at max 4*buffersize bytes outstanding when reading
   //    rmpolicy = 1               -> Remove from the cache the blk with the least offset
   opt += Form("&cachesz=%d&readaheadsz=%d&rmpolicy=1", 4*buffersize, 2*buffersize);
   sURL.SetOptions(opt);

   TFile *sfile = 0;

   Bool_t success = kFALSE;

   // Open source file
   if (!(sfile = TFile::Open(sURL.GetUrl(), "READ"))) {
      ::Error("TFile::Cp", "cannot open source file %s", src);
   } else {
      success = sfile->Cp(dst, progressbar, buffersize);
   }

   if (sfile) sfile->Close();
   if (sfile) delete sfile;
   
   return success;
}

//______________________________________________________________________________
//The next statement is not active anymore on Linux.
//Using posix_fadvise introduces a performance penalty (10 %) on optimized files
//and in addition it destroys the information of TTreePerfStats
#if defined(R__neverLINUX) && !defined(R__WINGCC)
Bool_t TFile::ReadBufferAsync(Long64_t offset, Int_t len)
{
   // Read specified byte range asynchronously. Actually we tell the kernel
   // which blocks we are going to read so it can start loading these blocks
   // in the buffer cache.

   // Shortcut to avoid having to implement dummy ReadBufferAsync() in all
   // I/O plugins. Override ReadBufferAsync() in plugins if async is supported.
   if (IsA() != TFile::Class())
      return kTRUE;

   int advice = POSIX_FADV_WILLNEED;
   if (len == 0) {
      // according POSIX spec if len is zero, all data following offset
      // is specified. Nevertheless ROOT uses zero to probe readahead
      // capadilites.
      advice = POSIX_FADV_NORMAL;
   }
   Double_t start = 0;
   if (gPerfStats != 0) start = TTimeStamp();
#if defined(R__SEEK64)
   Int_t result = posix_fadvise64(fD, offset, len, advice);
#else
   Int_t result = posix_fadvise(fD, offset, len, advice);
#endif
   if (gPerfStats != 0) {
      gPerfStats->FileReadEvent(this, len, start);
   }
   return (result != 0);
}
#else
Bool_t TFile::ReadBufferAsync(Long64_t, Int_t)
{
   // Not supported yet on non Linux systems.

   return kTRUE;
}
#endif

//______________________________________________________________________________
Int_t TFile::GetBytesToPrefetch() const
{
   // Max number of bytes to prefetch. By default this is 75% of the
   // read cache size. But specific TFile implementations may need to change it

   TFileCacheRead *cr = 0;
   if ((cr = GetCacheRead())) {
      Int_t bytes = cr->GetBufferSize() / 4 * 3;
      return ((bytes < 0) ? 0 : bytes);
   }
   return 0;
}
