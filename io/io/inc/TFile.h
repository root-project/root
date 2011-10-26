// @(#)root/io:$Id$
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFile
#define ROOT_TFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFile                                                                //
//                                                                      //
// ROOT file.                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDirectoryFile
#include "TDirectoryFile.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TFree;
class TArrayC;
class TArchiveFile;
class TFileOpenHandle;
class TFileCacheRead;
class TFileCacheWrite;
class TProcessID;
class TStopwatch;
class TFilePrefetch;

class TFile : public TDirectoryFile {
  friend class TDirectoryFile;
  friend class TFilePrefetch;

public:
   // Asynchronous open request status
   enum EAsyncOpenStatus { kAOSNotAsync = -1,  kAOSFailure = 0,
                           kAOSInProgress = 1, kAOSSuccess = 2 };
   // Open timeout constants
   enum EOpenTimeOut { kInstantTimeout = 0, kEternalTimeout = 999999999 };

protected:
   Double_t         fSumBuffer;      //Sum of buffer sizes of objects written so far
   Double_t         fSum2Buffer;     //Sum of squares of buffer sizes of objects written so far
   Long64_t         fBytesWrite;     //Number of bytes written to this file
   Long64_t         fBytesRead;      //Number of bytes read from this file
   Long64_t         fBytesReadExtra; //Number of extra bytes (overhead) read by the readahead buffer
   Long64_t         fBEGIN;          //First used byte in file
   Long64_t         fEND;            //Last used byte in file
   Long64_t         fSeekFree;       //Location on disk of free segments structure
   Long64_t         fSeekInfo;       //Location on disk of StreamerInfo record
   Int_t            fD;              //File descriptor
   Int_t            fVersion;        //File format version
   Int_t            fCompress;       //Compression level and algorithm
   Int_t            fNbytesFree;     //Number of bytes for free segments structure
   Int_t            fNbytesInfo;     //Number of bytes for StreamerInfo record
   Int_t            fWritten;        //Number of objects written so far
   Int_t            fNProcessIDs;    //Number of TProcessID written to this file
   Int_t            fReadCalls;      //Number of read calls ( not counting the cache calls )
   TString          fRealName;       //Effective real file name (not original url)
   TString          fOption;         //File options
   Char_t           fUnits;          //Number of bytes for file pointers
   TList           *fFree;           //Free segments linked list table
   TArrayC         *fClassIndex;     //!Index of TStreamerInfo classes written to this file
   TObjArray       *fProcessIDs;     //!Array of pointers to TProcessIDs
   Long64_t         fOffset;         //!Seek offset cache
   TArchiveFile    *fArchive;        //!Archive file from which we read this file
   TFileCacheRead  *fCacheRead;      //!Pointer to the read cache (if any)
   TFileCacheWrite *fCacheWrite;     //!Pointer to the write cache (if any)
   Long64_t         fArchiveOffset;  //!Offset at which file starts in archive
   Bool_t           fIsArchive;      //!True if this is a pure archive file
   Bool_t           fNoAnchorInName; //!True if we don't want to force the anchor to be appended to the file name
   Bool_t           fIsRootFile;     //!True is this is a ROOT file, raw file otherwise
   Bool_t           fInitDone;       //!True if the file has been initialized
   Bool_t           fMustFlush;      //!True if the file buffers must be flushed
   TFileOpenHandle *fAsyncHandle;    //!For proper automatic cleanup
   EAsyncOpenStatus fAsyncOpenStatus; //!Status of an asynchronous open request
   TUrl             fUrl;            //!URL of file

   TList           *fInfoCache;      //!Cached list of the streamer infos in this file
   TList           *fOpenPhases;     //!Time info about open phases

   static TList    *fgAsyncOpenRequests; //List of handles for pending open requests

   static TString   fgCacheFileDir;          //Directory where to locally stage files
   static Bool_t    fgCacheFileDisconnected; //Indicates, we trust in the files in the cache dir without stat on the cached file
   static Bool_t    fgCacheFileForce;        //Indicates, to force all READ to CACHEREAD
   static UInt_t    fgOpenTimeout;           //Timeout for open operations in ms  - 0 corresponds to blocking i/o
   static Bool_t    fgOnlyStaged ;           //Before the file is opened, it is checked, that the file is staged, if not, the open fails
   static Long64_t  fgBytesWrite;            //Number of bytes written by all TFile objects
   static Long64_t  fgBytesRead;             //Number of bytes read by all TFile objects
   static Long64_t  fgFileCounter;           //Counter for all opened files
   static Int_t     fgReadCalls;             //Number of bytes read from all TFile objects
   static Int_t     fgReadaheadSize;         //Readahead buffer size
   static Bool_t    fgReadInfo;              //if true (default) ReadStreamerInfo is called when opening a file

   virtual EAsyncOpenStatus GetAsyncOpenStatus() { return fAsyncOpenStatus; }
   virtual void  Init(Bool_t create);
   Bool_t        FlushWriteCache();
   Int_t         ReadBufferViaCache(char *buf, Int_t len);
   Int_t         WriteBufferViaCache(const char *buf, Int_t len);

   // Creating projects
   Int_t         MakeProjectParMake(const char *packname, const char *filename);
   Int_t         MakeProjectParProofInf(const char *packname, const char *proofinfdir);

   // Interface to basic system I/O routines
   virtual Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   virtual Int_t    SysClose(Int_t fd);
   virtual Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   virtual Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   virtual Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   virtual Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   virtual Int_t    SysSync(Int_t fd);

   // Interface for text-based TDirectory I/O
   virtual Long64_t DirCreateEntry(TDirectory*) { return 0; }
   virtual Int_t    DirReadKeys(TDirectory*) { return 0; }
   virtual void     DirWriteKeys(TDirectory*) {}
   virtual void     DirWriteHeader(TDirectory*) {}

private:
   TFile(const TFile &);            //Files cannot be copied
   void operator=(const TFile &);

   static void   CpProgress(Long64_t bytesread, Long64_t size, TStopwatch &watch);
   static TFile *OpenFromCache(const char *name, Option_t * = "",
                               const char *ftitle = "", Int_t compress = 1,
                               Int_t netopt = 0);

public:
   // TFile status bits
   enum EStatusBits {
      kRecovered     = BIT(10),
      kHasReferences = BIT(11),
      kDevNull       = BIT(12),
      kWriteError    = BIT(14), // BIT(13) is taken up by TObject
      kBinaryFile    = BIT(15),
      kRedirected    = BIT(16)
   };
   enum ERelativeTo { kBeg = 0, kCur = 1, kEnd = 2 };
   enum { kStartBigFile  = 2000000000 };
   // File type
   enum EFileType { kDefault = 0, kLocal = 1, kNet = 2, kWeb = 3, kFile = 4, kMerge = 5};

   TFile();
   TFile(const char *fname, Option_t *option="", const char *ftitle="", Int_t compress=1);
   virtual ~TFile();
   virtual void        Close(Option_t *option=""); // *MENU*
   virtual void        Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual Bool_t      Cp(const char *dst, Bool_t progressbar = kTRUE,UInt_t buffersize = 1000000);
   virtual TKey*       CreateKey(TDirectory* mother, const TObject* obj, const char* name, Int_t bufsize);
   virtual TKey*       CreateKey(TDirectory* mother, const void* obj, const TClass* cl,
                                 const char* name, Int_t bufsize);
   static TFile      *&CurrentFile(); // Return the current file for this thread.
   virtual void        Delete(const char *namecycle="");
   virtual void        Draw(Option_t *option="");
   virtual void        DrawMap(const char *keys="*",Option_t *option=""); // *MENU*
   virtual void        FillBuffer(char *&buffer);
   virtual void        Flush();
   TArchiveFile       *GetArchive() const { return fArchive; }
   Int_t               GetBestBuffer() const;
   virtual Int_t       GetBytesToPrefetch() const;
   TFileCacheRead     *GetCacheRead() const;
   TFileCacheWrite    *GetCacheWrite() const;
   TArrayC            *GetClassIndex() const { return fClassIndex; }
   Int_t               GetCompressionAlgorithm() const;
   Int_t               GetCompressionLevel() const;
   Int_t               GetCompressionSettings() const;
   Float_t             GetCompressionFactor();
   virtual Long64_t    GetEND() const { return fEND; }
   virtual Int_t       GetErrno() const;
   virtual void        ResetErrno() const;
   Int_t               GetFd() const { return fD; }
   virtual const TUrl *GetEndpointUrl() const { return &fUrl; }
   TObjArray          *GetListOfProcessIDs() const {return fProcessIDs;}
   TList              *GetListOfFree() const { return fFree; }
   virtual Int_t       GetNfree() const { return fFree->GetSize(); }
   virtual Int_t       GetNProcessIDs() const { return fNProcessIDs; }
   Option_t           *GetOption() const { return fOption.Data(); }
   virtual Long64_t    GetBytesRead() const { return fBytesRead; }
   virtual Long64_t    GetBytesReadExtra() const { return fBytesReadExtra; }
   virtual Long64_t    GetBytesWritten() const;
   virtual Int_t       GetReadCalls() const { return fReadCalls; }
   Int_t               GetVersion() const { return fVersion; }
   Int_t               GetRecordHeader(char *buf, Long64_t first, Int_t maxbytes,
                                       Int_t &nbytes, Int_t &objlen, Int_t &keylen);
   virtual Int_t       GetNbytesInfo() const {return fNbytesInfo;}
   virtual Int_t       GetNbytesFree() const {return fNbytesFree;}
   Long64_t            GetRelOffset() const { return fOffset - fArchiveOffset; }
   virtual Long64_t    GetSeekFree() const {return fSeekFree;}
   virtual Long64_t    GetSeekInfo() const {return fSeekInfo;}
   virtual Long64_t    GetSize() const;
   virtual TList      *GetStreamerInfoList();
   const   TList      *GetStreamerInfoCache();
   virtual void        IncrementProcessIDs() { fNProcessIDs++; }
   virtual Bool_t      IsArchive() const { return fIsArchive; }
           Bool_t      IsBinary() const { return TestBit(kBinaryFile); }
           Bool_t      IsRaw() const { return !fIsRootFile; }
   virtual Bool_t      IsOpen() const;
   virtual void        ls(Option_t *option="") const;
   virtual void        MakeFree(Long64_t first, Long64_t last);
   virtual void        MakeProject(const char *dirname, const char *classes="*",
                                   Option_t *option="new"); // *MENU*
   virtual void        Map(); // *MENU*
   virtual Bool_t      Matches(const char *name);
   virtual Bool_t      MustFlush() const {return fMustFlush;}
   virtual void        Paint(Option_t *option="");
   virtual void        Print(Option_t *option="") const;
   virtual Bool_t      ReadBufferAsync(Long64_t offs, Int_t len);
   virtual Bool_t      ReadBuffer(char *buf, Int_t len);
   virtual Bool_t      ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual Bool_t      ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);
   virtual void        ReadFree();
   virtual TProcessID *ReadProcessID(UShort_t pidf);
   virtual void        ReadStreamerInfo();
   virtual Int_t       Recover();
   virtual Int_t       ReOpen(Option_t *mode);
   virtual void        Seek(Long64_t offset, ERelativeTo pos = kBeg);
   virtual void        SetCacheRead(TFileCacheRead *cache);
   virtual void        SetCacheWrite(TFileCacheWrite *cache);
   virtual void        SetCompressionAlgorithm(Int_t algorithm=0);
   virtual void        SetCompressionLevel(Int_t level=1);
   virtual void        SetCompressionSettings(Int_t settings=1);
   virtual void        SetEND(Long64_t last) { fEND = last; }
   virtual void        SetOffset(Long64_t offset, ERelativeTo pos = kBeg);
   virtual void        SetOption(Option_t *option=">") { fOption = option; }
   virtual void        SetReadCalls(Int_t readcalls = 0) { fReadCalls = readcalls; }
   virtual void        ShowStreamerInfo();
   virtual Int_t       Sizeof() const;
   void                SumBuffer(Int_t bufsize);
   virtual void        UseCache(Int_t maxCacheSize = 10, Int_t pageSize = 0);
   virtual Bool_t      WriteBuffer(const char *buf, Int_t len);
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0);
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0) const;
   virtual void        WriteFree();
   virtual void        WriteHeader();
   virtual UShort_t    WriteProcessID(TProcessID *pid);
   virtual void        WriteStreamerInfo();

   static TFileOpenHandle
                      *AsyncOpen(const char *name, Option_t *option = "",
                                 const char *ftitle = "", Int_t compress = 1,
                                 Int_t netopt = 0);
   static TFile       *Open(const char *name, Option_t *option = "",
                            const char *ftitle = "", Int_t compress = 1,
                            Int_t netopt = 0);
   static TFile       *Open(TFileOpenHandle *handle);

   static EFileType    GetType(const char *name, Option_t *option = "", TString *prefix = 0);

   static EAsyncOpenStatus GetAsyncOpenStatus(const char *name);
   static EAsyncOpenStatus GetAsyncOpenStatus(TFileOpenHandle *handle);
   static const TUrl  *GetEndpointUrl(const char *name);

   static Long64_t     GetFileBytesRead();
   static Long64_t     GetFileBytesWritten();
   static Int_t        GetFileReadCalls();
   static Int_t        GetReadaheadSize();

   static void         SetFileBytesRead(Long64_t bytes = 0);
   static void         SetFileBytesWritten(Long64_t bytes = 0);
   static void         SetFileReadCalls(Int_t readcalls = 0);
   static void         SetReadaheadSize(Int_t bufsize = 256000);
   static void         SetReadStreamerInfo(Bool_t readinfo=kTRUE);

   static Long64_t     GetFileCounter();
   static void         IncrementFileCounter();

   static Bool_t       SetCacheFileDir(const char *cacheDir, Bool_t operateDisconnected = kTRUE,
                                       Bool_t forceCacheread = kFALSE);
   static const char  *GetCacheFileDir();
   static Bool_t       ShrinkCacheFileDir(Long64_t shrinkSize, Long_t cleanupInteval = 0);
   static Bool_t       Cp(const char *src, const char *dst, Bool_t progressbar = kTRUE,
                          UInt_t buffersize = 1000000);

   static UInt_t       SetOpenTimeout(UInt_t timeout);  // in ms
   static UInt_t       GetOpenTimeout(); // in ms
   static Bool_t       SetOnlyStaged(Bool_t onlystaged);
   static Bool_t       GetOnlyStaged();

   ClassDef(TFile,8)  //ROOT file
};

#ifndef __CINT__
#define gFile (TFile::CurrentFile())

#elif defined(__MAKECINT__)
// To properly handle the use of gFile in header files (in static declarations)
R__EXTERN TFile   *gFile;
#endif

//
// Class holding info about the file being opened
//
class TFileOpenHandle : public TNamed {

friend class TFile;
friend class TAlienFile;

private:
   TString  fOpt;        // Options
   Int_t    fCompress;   // Compression level and algorithm
   Int_t    fNetOpt;     // Network options
   TFile   *fFile;       // TFile instance of the file being opened

   TFileOpenHandle(TFile *f) : TNamed("",""), fOpt(""), fCompress(1),
                               fNetOpt(0), fFile(f) { }
   TFileOpenHandle(const char *n, const char *o, const char *t, Int_t cmp,
                   Int_t no) : TNamed(n,t), fOpt(o), fCompress(cmp),
                               fNetOpt(no), fFile(0) { }
   TFileOpenHandle(const TFileOpenHandle&);
   TFileOpenHandle& operator=(const TFileOpenHandle&);

   TFile      *GetFile() const { return fFile; }

public:
   ~TFileOpenHandle() { }

   Bool_t      Matches(const char *name);

   const char *GetOpt() const { return fOpt; }
   Int_t       GetCompress() const { return fCompress; }
   Int_t       GetNetOpt() const { return fNetOpt; }
};

//______________________________________________________________________________
inline Int_t TFile::GetCompressionAlgorithm() const
{
   return (fCompress < 0) ? -1 : fCompress / 100;
}

//______________________________________________________________________________
inline Int_t TFile::GetCompressionLevel() const
{
   return (fCompress < 0) ? -1 : fCompress % 100;
}

//______________________________________________________________________________
inline Int_t TFile::GetCompressionSettings() const
{
   return (fCompress < 0) ? -1 : fCompress;
}

#endif
