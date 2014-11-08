// @(#)root/io:$Id$
// Author: Fons Rademakers   08/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifdef WIN32
#pragma optimize("",off)
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMapFile                                                             //
//                                                                      //
// This class implements a shared memory region mapped to a file.       //
// Objects can be placed into this shared memory area using the Add()   //
// member function. To actually place a copy of the object is shared    //
// memory call Update() also whenever the mapped object(s) change(s)    //
// call Update() to put a fresh copy in the shared memory. This extra   //
// step is necessary since it is not possible to share objects with     //
// virtual pointers between processes (the vtbl ptr points to the       //
// originators unique address space and can not be used by the          //
// consumer process(es)). Consumer processes can map the memory region  //
// from this file and access the objects stored in it via the Get()     //
// method (which returns a copy of the object stored in the shared      //
// memory with correct vtbl ptr set). Only objects of classes with a    //
// Streamer() member function defined can be shared.                    //
//                                                                      //
// I know the current implementation is not ideal (you need to copy to  //
// and from the shared memory file) but the main problem is with the    //
// class' virtual_table pointer. This pointer points to a table unique  //
// for every process. Therefore, different options are:                 //
//   1) One could allocate an object directly in shared memory in the   //
//      producer, but the consumer still has to copy the object from    //
//      shared memory into a local object which has the correct vtbl    //
//      pointer for that process (copy ctor's can be used for creating  //
//      the local copy).                                                //
//   2) Another possibility is to only allow objects without virtual    //
//      functions in shared memory (like simple C structs), or to       //
//      forbid (how?) the consumer from calling any virtual functions   //
//      of the objects in shared memory.                                //
//   3) A last option is to copy the object internals to shared memory  //
//      and copy them again from there. This is what is done in the     //
//      TMapFile (using the object Streamer() to make a deep copy).     //
// Option 1) saves one copy, but requires solid copy ctor's (along the  //
// full inheritance chain) to rebuild the object in the consumer. Most  //
// classes don't provide these copy ctor's, especially not when objects //
// contain collections, etc. 2) is too limiting or dangerous (calling   //
// accidentally a virtual function will segv). So since we have a       //
// robust Streamer mechanism I opted for 3).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifdef WIN32
#  include <windows.h>
#  include <process.h>
#  ifdef GetObject
#    undef GetObject
#  endif
#  define HAVE_SEMOP

# ifdef CreateSemaphore
#   undef CreateSemaphore
# endif

# ifdef AcquireSemaphore
#   undef AcquireSemaphore;
# endif

# ifdef ReleaseSemaphore
#   undef ReleaseSemaphore
# endif

# ifdef DeleteSemaphore
#   undef DeleteSemaphore
# endif

#else
#  define INVALID_HANDLE_VALUE -1
#endif

#include <fcntl.h>
#include <errno.h>

#include "TMapFile.h"
#include "TKeyMapFile.h"
#include "TDirectoryFile.h"
#include "TBrowser.h"
#include "TString.h"
#include "TSystem.h"
#include "TClass.h"
#include "TBufferFile.h"
#include "TVirtualMutex.h"
#include <cmath>

#if defined(R__UNIX) && !defined(R__MACOSX) && !defined(R__WINGCC)
#define HAVE_SEMOP
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#if defined(R__HPUX) || \
    defined (R__SOLARIS) || defined(R__AIX) || defined(R__HIUX) || \
    __GLIBC_MINOR__ > 0
union semun {
   int val;                      // value for SETVAL
   struct semid_ds *buf;         // buffer for IPC_STAT & IPC_SET
   ushort *array;                // array for GETALL & SETALL
};
#endif
#if defined(R__LINUX) || defined(R__LYNXOS) || defined(R__HURD)
#  define       SEM_A   0200     // alter permission
#  define       SEM_R   0400     // read permission
#endif
#endif


Long_t TMapFile::fgMapAddress = 0;
void  *TMapFile::fgMmallocDesc = 0;

//void *gMmallocDesc = 0; //is initialized in TClass.cxx



//______________________________________________________________________________
TMapRec::TMapRec(const char *name, const TObject *obj, Int_t size, void *buf)
{
   /// Constructor.
   fName      = StrDup(name);
   fClassName = 0;
   fObject    = (TObject*)obj;
   fBuffer    = buf;
   fBufSize   = size;
   fNext      = 0;
}

//______________________________________________________________________________
TMapRec::~TMapRec()
{
   // Destructor.
   delete [] fName;
   delete [] fClassName;
}

//______________________________________________________________________________
TObject *TMapRec::GetObject() const
{
   // This method returns a pointer to the original object. NOTE: this pointer
   // is only valid in the process that produces the shared memory file. In a
   // consumer process this pointer is illegal! Be careful.

   return fObject;
}




ClassImp(TMapFile)

//______________________________________________________________________________
TMapFile::TMapFile()
{
   // Default ctor. Does not much except setting some basic values.

   fFd          = -1;
   fVersion     = 0;
   fName        = 0;
   fTitle       = 0;
   fOption      = 0;
   fMmallocDesc = 0;
   fBaseAddr    = 0;
   fSize        = 0;
   fFirst       = 0;
   fLast        = 0;
   fOffset      = 0;
   fDirectory   = 0;
   fBrowseList  = 0;
   fWritable    = kFALSE;
   fSemaphore   = -1;
   fhSemaphore  = 0;
   fGetting     = 0;
   fWritten     = 0;
   fSumBuffer   = 0;
   fSum2Buffer  = 0;
}

//______________________________________________________________________________
TMapFile::TMapFile(const char *name, const char *title, Option_t *option,
                   Int_t size, TMapFile *&newMapFile)
{
   // Create a memory mapped file. This opens a file (to which the
   // memory will be mapped) and attaches a memory region to it.
   // Option can be either: "NEW", "CREATE", "RECREATE", "UPDATE" or
   // "READ" (see TFile). The default open mode is "READ". The size
   // argument specifies the maximum size of shared memory file in bytes.
   // This protected ctor is called via the static Create() method.

#ifndef WIN32
   fFd          = -1;
   fSemaphore   = -1;
   fhSemaphore  = 0;
#else
   fFd          = (Int_t) INVALID_HANDLE_VALUE;
   fSemaphore   = (Int_t) INVALID_HANDLE_VALUE;
#endif
   fMmallocDesc = 0;
   fSize        = size;
   fFirst       = 0;
   fOffset      = 0;
   fVersion     = gROOT->GetVersionInt();
   fTitle       = StrDup(title);
   fOption      = StrDup(option);
   fDirectory   = 0;
   fBrowseList  = 0;
   fGetting     = 0;
   fWritten     = 0;
   fSumBuffer   = 0;
   fSum2Buffer  = 0;

   char  *cleanup = 0;
   Bool_t create  = kFALSE;
   Bool_t recreate, update, read;

   {
      TString opt = option;

      if (!opt.CompareTo("NEW", TString::kIgnoreCase) ||
          !opt.CompareTo("CREATE", TString::kIgnoreCase))
         create = kTRUE;
      recreate = opt.CompareTo("RECREATE", TString::kIgnoreCase)
                 ? kFALSE : kTRUE;
      update   = opt.CompareTo("UPDATE", TString::kIgnoreCase)
                 ? kFALSE : kTRUE;
      read     = opt.CompareTo("READ", TString::kIgnoreCase)
                 ? kFALSE : kTRUE;
      if (!create && !recreate && !update && !read) {
         read    = kTRUE;
         delete [] fOption;
         fOption = StrDup("READ");
      }
   }

   const char *fname;
   if ((fname = gSystem->ExpandPathName(name))) {
      fName = StrDup(fname);
      delete [] (char*)fname;
      fname = fName;
   } else {
      Error("TMapFile", "error expanding path %s", fname);
      goto zombie;
   }

   if (recreate) {
      if (!gSystem->AccessPathName(fname, kFileExists))
         gSystem->Unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      delete [] fOption;
      fOption  = StrDup("CREATE");
   }
   if (create && !gSystem->AccessPathName(fname, kFileExists)) {
      Error("TMapFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && gSystem->AccessPathName(fname, kWritePermission)) {
         Error("TMapFile", "no write permission, could not open file %s", fname);
         goto zombie;
      }
   }
   if (read) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         Error("TMapFile", "file %s does not exist", fname);
         goto zombie;
      }
      if (gSystem->AccessPathName(fname, kReadPermission)) {
         Error("TMapFile", "no read permission, could not open file %s", fname);
         goto zombie;
      }
   }

   // Open file to which memory will be mapped
   if (create || update) {
#ifndef WIN32
      fFd = open(fname, O_RDWR | O_CREAT, 0644);
#else
      fFd = (Int_t) CreateFile(fname,                    // pointer to name of the file
                     GENERIC_WRITE | GENERIC_READ,       // access (read-write) mode
                     FILE_SHARE_WRITE | FILE_SHARE_READ, // share mode
                     NULL,                               // pointer to security attributes
                     OPEN_ALWAYS,                        // how to create
                     FILE_ATTRIBUTE_TEMPORARY,           // file attributes
                     (HANDLE) NULL);                     // handle to file with attributes to copy
#endif
      if (fFd == (Int_t)INVALID_HANDLE_VALUE) {
         SysError("TMapFile", "file %s can not be opened", fname);
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
#ifndef WIN32
      fFd = open(fname, O_RDONLY);
#else
      fFd = (Int_t) CreateFile(fname,                    // pointer to name of the file
                     GENERIC_READ,                       // access (read-write) mode
                     FILE_SHARE_WRITE | FILE_SHARE_READ, // share mode
                     NULL,                               // pointer to security attributes
                     OPEN_EXISTING,                      // how to create
                     FILE_ATTRIBUTE_TEMPORARY,           // file attributes
                     (HANDLE) NULL);                     // handle to file with attributes to copy
#endif
      if (fFd == (Int_t)INVALID_HANDLE_VALUE) {
         SysError("TMapFile", "file %s can not be opened for reading", fname);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   // Attach memory region to file.
   void *mapto;
   TMapFile *mapfil;

   if (((mapto = MapToAddress()) == (void *)-1) ||
#ifndef WIN32
       ((fMmallocDesc = mmalloc_attach(fFd, mapto, fSize)) == 0)) {
#else
       ((fMmallocDesc = mmalloc_attach((HANDLE) fFd, mapto, fSize)) == 0)) {
#endif

      if (mapto == (void *)-1) {
         Error("TMapFile", "no memory mapped file capability available\n"
                           "Use rootn.exe or link application against \"-lNew\"");
      } else {
         if (fMmallocDesc == 0 && fWritable)
            Error("TMapFile", "mapped file not in mmalloc format or\n"
                              "already open in RW mode by another process");
         if (fMmallocDesc == 0 && !fWritable)
            Error("TMapFile", "mapped file not in mmalloc format");
      }
#ifndef WIN32
      close(fFd);
#else
      CloseHandle((HANDLE) fFd);
#endif
      fFd = -1;
      if (create)
         gSystem->Unlink(fname);
      goto zombie;

   } else if ((mapfil = (TMapFile *) mmalloc_getkey(fMmallocDesc, 0)) != 0) {

      // File contains mmalloc heap. If we are in write mode and mapped
      // file already connected in write mode switch to read-only mode.
      // Check if ROOT versions are compatible.
      // If so update mapped version of TMapFile to reflect current
      // situation (only if not opened in READ mode).
      if (mapfil->fVersion != fVersion) {
         Error("TMapFile", "map file %s (%d) incompatible with current ROOT version (%d)",
               fname, mapfil->fVersion, fVersion);
         mmalloc_detach(fMmallocDesc);
#ifndef WIN32
         close(fFd);
#else
         CloseHandle((HANDLE) fFd);
#endif
         fFd = -1;
         fMmallocDesc = 0;
         goto zombie;
      }

      if (mapfil->fWritable && fWritable) {
         Warning("TMapFile", "map file already open in write mode, opening in read-only mode");
         fWritable = kFALSE;
      }

      fBaseAddr = mapfil->fBaseAddr;
      fSize     = mapfil->fSize;

      if (fWritable) {
         // create new TMapFile object in mapped heap to get correct vtbl ptr
         CreateSemaphore();
         gMmallocDesc = fMmallocDesc;
         TMapFile *mf = new TMapFile(*mapfil);
         mf->fFd        = fFd;
         mf->fWritable  = kTRUE;
         cleanup        = mf->fOption;
         mf->fOption    = StrDup(fOption);
         mf->fSemaphore = fSemaphore;
#ifdef WIN32
         mf->CreateSemaphore(fSemaphore);
#endif
         mmalloc_setkey(fMmallocDesc, 0, mf);
         gMmallocDesc = 0;
         mapfil = mf;
      } else {
         gMmallocDesc = 0;    // make sure we are in sbrk heap
         fOffset      = ((struct mdesc *) fMmallocDesc)->offset;
         TMapFile *mf = new TMapFile(*mapfil, fOffset);
         delete [] mf->fOption;
         mf->fFd          = fFd;
         mf->fOption      = StrDup("READ");
         mf->fMmallocDesc = fMmallocDesc;
         mf->fWritable    = kFALSE;
         mapfil = mf;
      }

      // store shadow mapfile (it contains the real fFd in case map
      // is not writable)
      fVersion  = -1;   // make this the shadow map file
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfMappedFiles()->AddLast(this);

   } else {

      // New file. If the file is writable create a new copy of the
      // TMapFile which will now be allocated on the memory mapped heap.
      if (!fWritable) {
         Error("TMapFile", "map file is not writable");
         mmalloc_detach(fMmallocDesc);
#ifndef WIN32
         close(fFd);
#else
         CloseHandle((HANDLE) fFd);
#endif
         fFd = -1;
         fMmallocDesc = 0;
         goto zombie;
      }

      fBaseAddr = (ULong_t)((struct mdesc *) fMmallocDesc)->base;

      CreateSemaphore();

      gMmallocDesc = fMmallocDesc;

      mapfil = new TMapFile(*this);
      mmalloc_setkey(fMmallocDesc, 0, mapfil);

      gMmallocDesc = 0;

      // store shadow mapfile
      fVersion  = -1;   // make this the shadow map file
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfMappedFiles()->AddLast(this);

   }

   mapfil->InitDirectory();
   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfMappedFiles()->AddFirst(mapfil);
   }

   if (cleanup) delete [] cleanup;

   newMapFile = mapfil;

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   newMapFile   = this;
   gMmallocDesc = 0;
}

//______________________________________________________________________________
TMapFile::TMapFile(const TMapFile &f, Long_t offset) : TObject(f)
{
   // Private copy ctor. Used by the the ctor to create a new version
   // of TMapFile in the memory mapped heap. It's main purpose is to
   // correctly create the string data members.

   fFd          = f.fFd;
   fVersion     = f.fVersion;
   fName        = StrDup((char *)((Long_t)f.fName + offset));
   fTitle       = StrDup((char *)((Long_t)f.fTitle + offset));
   fOption      = StrDup((char *)((Long_t)f.fOption + offset));
   fMmallocDesc = f.fMmallocDesc;
   fBaseAddr    = f.fBaseAddr;
   fSize        = f.fSize;
   fFirst       = f.fFirst;
   fLast        = f.fLast;
   fWritable    = f.fWritable;
   fSemaphore   = f.fSemaphore;
   fOffset      = offset;
   fDirectory   = 0;
   fBrowseList  = 0;
   fGetting     = 0;
   fWritten     = f.fWritten;
   fSumBuffer   = f.fSumBuffer;
   fSum2Buffer  = f.fSum2Buffer;
#ifdef WIN32
   CreateSemaphore(fSemaphore);
#else
   fhSemaphore = f.fhSemaphore;
#endif
}

//______________________________________________________________________________
TMapFile::~TMapFile()
{
   // TMapFiles may not be deleted, since we want to keep the complete
   // TMapFile object in the mapped file for later re-use. To enforce this
   // the delete operator has been made private. Use Close() to properly
   // terminate a TMapFile (also done via the TROOT dtor).

   if (fDirectory == gDirectory) gDirectory = gROOT;
   delete fDirectory; fDirectory = 0;
   if (fBrowseList) fBrowseList->Delete();
   delete fBrowseList; fBrowseList = 0;

   // if shadow map file we are done here
   if (fVersion == -1)
      return;

   // Writable mapfile is allocated in mapped memory. This object should
   // not be deleted by ::operator delete(), because it is needed if we
   // want to connect later to the file again.
   if (fWritable)
      TObject::SetDtorOnly(this);

   Close("dtor");

   fgMmallocDesc = fMmallocDesc;
}

//______________________________________________________________________________
void TMapFile::InitDirectory()
{
   // Create the directory associated to this mapfile

   gDirectory = 0;
   fDirectory = new TDirectoryFile();
   fDirectory->SetName(GetName());
   fDirectory->SetTitle(GetTitle());
   fDirectory->Build();
   fDirectory->SetMother(this);
   gDirectory = fDirectory;
}

//______________________________________________________________________________
void TMapFile::Add(const TObject *obj, const char *name)
{
   // Add an object to the list of objects to be stored in shared memory.
   // To place the object actually into shared memory call Update().

   if (!fWritable || !fMmallocDesc) return;

   Bool_t lock = fGetting != obj ? kTRUE : kFALSE;

   if (lock)
      AcquireSemaphore();

   gMmallocDesc = fMmallocDesc;

   const char *n;
   if (name && *name)
      n = name;
   else
      n = obj->GetName();

   if (Remove(n, kFALSE)) {
      //Warning("Add", "replaced object with same name %s", n);
   }

   TMapRec *mr = new TMapRec(n, obj, 0, 0);
   if (!fFirst) {
      fFirst = mr;
      fLast  = mr;
   } else {
      fLast->fNext = mr;
      fLast        = mr;
   }

   gMmallocDesc = 0;

   if (lock)
      ReleaseSemaphore();
}

//______________________________________________________________________________
void TMapFile::Update(TObject *obj)
{
   // Update an object (or all objects, if obj == 0) in shared memory.

   if (!fWritable || !fMmallocDesc) return;

   AcquireSemaphore();

   gMmallocDesc = fMmallocDesc;

   Bool_t all = (obj == 0) ? kTRUE : kFALSE;

   TMapRec *mr = fFirst;
   while (mr) {
      if (all || mr->fObject == obj) {
         TBufferFile *b;
         if (!mr->fBufSize) {
            b = new TBufferFile(TBuffer::kWrite, GetBestBuffer());
            mr->fClassName = StrDup(mr->fObject->ClassName());
         } else
            b = new TBufferFile(TBuffer::kWrite, mr->fBufSize, mr->fBuffer);
         b->MapObject(mr->fObject);  //register obj in map to handle self reference
         mr->fObject->Streamer(*b);
         mr->fBufSize = b->BufferSize();
         mr->fBuffer  = b->Buffer();
         SumBuffer(b->Length());
         b->DetachBuffer();
         delete b;
      }
      mr = mr->fNext;
   }

   gMmallocDesc = 0;

   ReleaseSemaphore();
}

//______________________________________________________________________________
TObject *TMapFile::Remove(TObject *obj, Bool_t lock)
{
   // Remove object from shared memory. Returns pointer to removed
   // object if successful, 0 otherwise.

   if (!fWritable || !fMmallocDesc) return 0;

   if (lock)
      AcquireSemaphore();

   TObject *retObj = 0;
   TMapRec *prev = 0, *mr = fFirst;
   while (mr) {
      if (mr->fObject == obj) {
         if (mr == fFirst) {
            fFirst = mr->fNext;
            if (mr == fLast)
               fLast = 0;
         } else {
            prev->fNext = mr->fNext;
            if (mr == fLast)
               fLast = prev;
         }
         retObj = obj;
         delete mr;
         break;
      }
      prev = mr;
      mr   = mr->fNext;
   }

   if (lock)
      ReleaseSemaphore();

   return retObj;
}

//______________________________________________________________________________
TObject *TMapFile::Remove(const char *name, Bool_t lock)
{
   // Remove object by name from shared memory. Returns pointer to removed
   // object if successful, 0 otherwise.

   if (!fWritable || !fMmallocDesc) return 0;

   if (lock)
      AcquireSemaphore();

   TObject *retObj = 0;
   TMapRec *prev = 0, *mr = fFirst;
   while (mr) {
      if (!strcmp(mr->fName, name)) {
         if (mr == fFirst) {
            fFirst = mr->fNext;
            if (mr == fLast)
               fLast = 0;
         } else {
            prev->fNext = mr->fNext;
            if (mr == fLast)
               fLast = prev;
         }
         retObj = mr->fObject;
         delete mr;
         break;
      }
      prev = mr;
      mr   = mr->fNext;
   }

   if (lock)
      ReleaseSemaphore();

   return retObj;
}

//______________________________________________________________________________
void TMapFile::RemoveAll()
{
   // Remove all objects from shared memory.

   if (!fWritable || !fMmallocDesc) return;

   AcquireSemaphore();

   TMapRec *mr = fFirst;
   while (mr) {
      TMapRec *t = mr;
      mr = mr->fNext;
      delete t;
   }
   fFirst = fLast = 0;

   ReleaseSemaphore();
}

//______________________________________________________________________________
TObject *TMapFile::Get(const char *name, TObject *delObj)
{
   // Return pointer to object retrieved from shared memory. The object must
   // be deleted after use. If delObj is a pointer to a previously allocated
   // object it will be deleted. Returns 0 in case object with the given
   // name does not exist.

   if (!fMmallocDesc) return 0;

   AcquireSemaphore();

   delete delObj;

   TObject *obj = 0;
   TMapRec *mr = GetFirst();
   while (OrgAddress(mr)) {
      if (!strcmp(mr->GetName(fOffset), name)) {
         if (!mr->fBufSize) goto release;
         TClass *cl = TClass::GetClass(mr->GetClassName(fOffset));
         if (!cl) {
            Error("Get", "unknown class %s", mr->GetClassName(fOffset));
            goto release;
         }

         obj = (TObject *)cl->New();
         if (!obj) {
            Error("Get", "cannot create new object of class %s", mr->GetClassName(fOffset));
            goto release;
         }

         fGetting = obj;
         TBufferFile *b = new TBufferFile(TBuffer::kRead, mr->fBufSize, mr->GetBuffer(fOffset));
         b->MapObject(obj);  //register obj in map to handle self reference
         obj->Streamer(*b);
         b->DetachBuffer();
         delete b;
         fGetting = 0;
         goto release;
      }
      mr = mr->GetNext(fOffset);
   }

release:
   ReleaseSemaphore();

   return obj;
}

//______________________________________________________________________________
#ifndef WIN32
void TMapFile::CreateSemaphore(int)
#else
void TMapFile::CreateSemaphore(int pid)
#endif
{
   // Create semaphore used for synchronizing access to shared memory.

#ifdef HAVE_SEMOP
#ifndef WIN32
   // create semaphore to synchronize access (should use read/write lock)
   fSemaphore = semget(IPC_PRIVATE, 1, SEM_R|SEM_A|(SEM_R>>3)|(SEM_A>>3)|
                                       (SEM_R>>6)|(SEM_A>>6));

   // set semaphore to 1
   if (fSemaphore != -1) {
      union semun set;
      set.val = 1;
      semctl(fSemaphore, 0, SETVAL, set);
   }
#else
   char buffer[] ="ROOT_Semaphore_xxxxxxxx";
   int lbuf = strlen(buffer);
   if (!pid) fSemaphore = getpid();
   fhSemaphore = (ULong_t)CreateMutex(NULL,FALSE,itoa(fSemaphore,&buffer[lbuf-8],16));
   if (fhSemaphore == 0) fSemaphore = (Int_t)INVALID_HANDLE_VALUE;
#endif
#endif
}

//______________________________________________________________________________
void TMapFile::DeleteSemaphore()
{
   // Delete the semaphore.

#ifdef HAVE_SEMOP
   // remove semaphore
#ifndef WIN32
   if (fSemaphore != -1) {
      int semid  = fSemaphore;
      fSemaphore = -1;
      union semun set;
      set.val = 0;
      semctl(semid, 0, IPC_RMID, set);
   }
#else
   if (fSemaphore != (Int_t)INVALID_HANDLE_VALUE) {
      CloseHandle((HANDLE)fhSemaphore);
      fhSemaphore = 0;
      fSemaphore  = (Int_t)INVALID_HANDLE_VALUE;
   }
#endif
#endif
}

//______________________________________________________________________________
Int_t TMapFile::AcquireSemaphore()
{
   // Acquire semaphore. Returns 0 if OK, -1 on error.

#ifdef HAVE_SEMOP
#ifndef WIN32
   if (fSemaphore != -1) {
      struct sembuf buf = { 0, -1, SEM_UNDO };
      int intr = 0;
again:
      if (semop(fSemaphore, &buf, 1) == -1) {
#if defined(R__FBSD) || defined(R__OBSD)
         if (TSystem::GetErrno() == EINVAL)
#else
         if (TSystem::GetErrno() == EIDRM)
#endif
            fSemaphore = -1;
#if !defined(R__FBSD)
         if (TSystem::GetErrno() == EINTR) {
            if (intr > 2)
               return -1;
            TSystem::ResetErrno();
            intr++;
            goto again;
         }
#endif
      }
   }
#else
   // Enter Critical section to "write" lock
   if (fSemaphore != (Int_t)INVALID_HANDLE_VALUE)
      WaitForSingleObject((HANDLE)fhSemaphore,INFINITE);
#endif
#endif

   // file might have grown, update mapping on reader to new size
   if (!fWritable && fMmallocDesc) {
      if (mmalloc_update_mapping(fMmallocDesc) == -1)
         Error("AcquireSemaphore", "cannot update mapping");
   }

   return 0;
}

//______________________________________________________________________________
Int_t TMapFile::ReleaseSemaphore()
{
   // Release semaphore. Returns 0 if OK, -1 on error.

#ifdef HAVE_SEMOP
#ifndef WIN32
   if (fSemaphore != -1) {
      struct sembuf buf = { 0, 1, SEM_UNDO };
      if (semop(fSemaphore, &buf, 1) == -1) {
#if defined(R__FBSD) || defined(R__OBSD)
         if (TSystem::GetErrno() == EINVAL)
#else
         if (TSystem::GetErrno() == EIDRM)
#endif
            fSemaphore = -1;
      }
   }
#else
   if (fSemaphore != (Int_t)INVALID_HANDLE_VALUE)
      ReleaseMutex((HANDLE)fhSemaphore);
#endif
#endif
   return 0;
}

//______________________________________________________________________________
void TMapFile::Close(Option_t *option)
{
   // Close a mapped file. First detach mapped memory then close file.
   // No member functions of a TMapFile that was opened in write mode
   // may be called after Close() (this includes, of course, "delete" which
   // would call the dtors). The option="dtor" is only used when called
   // via the ~TMapFile.

   if (!fMmallocDesc) return;

   TMapFile *shadow = FindShadowMapFile();
   if (!shadow) {
      Error("Close", "shadow map == 0, should never happen!");
      return;
   }

   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfMappedFiles()->Remove(shadow);
      gROOT->GetListOfMappedFiles()->Remove(this);
   }

   if (shadow->fWritable) {
      fWritable = kFALSE;
      DeleteSemaphore();
   }

   if (fMmallocDesc) {
      if (strcmp(option, "dtor"))
         mmalloc_detach(fMmallocDesc);

      // If writable cannot access fMmallocDesc anymore since
      // it points to the just unmapped memory region. Any further
      // access to this TMapFile will cause a crash.
      if (!shadow->fWritable)
         fMmallocDesc = 0;
   }

   if (shadow->fFd != -1)
#ifndef WIN32
      close(shadow->fFd);
#else
      CloseHandle((HANDLE)shadow->fFd);
#endif

   delete shadow;
}

//______________________________________________________________________________
TMapFile *TMapFile::FindShadowMapFile()
{
   // Returns shadow map file.

   R__LOCKGUARD2(gROOTMutex);
   TObjLink *lnk = ((TList *)gROOT->GetListOfMappedFiles())->LastLink();
   while (lnk) {
      TMapFile *mf = (TMapFile*)lnk->GetObject();
      if (mf->fVersion == -1 && fBaseAddr == mf->fBaseAddr && fSize == mf->fSize)
         return mf;
      lnk = lnk->Prev();
   }
   return 0;
}

//______________________________________________________________________________
void TMapFile::Print(Option_t *) const
{
   // Print some info about the mapped file.

   Printf("Memory mapped file:   %s", fName);
   Printf("Title:                %s", fTitle);
   if (fMmallocDesc) {
      Printf("Option:               %s", fOption);
      ULong_t size = (ULong_t)((struct mdesc *)fMmallocDesc)->top - fBaseAddr;
      Printf("Mapped Memory region: 0x%lx - 0x%lx (%.2f MB)", fBaseAddr, fBaseAddr + size,
             (float)size/1048576);
      Printf("Current breakval:     0x%lx", (ULong_t)GetBreakval());
   } else
      Printf("Option:               file closed");
}

//______________________________________________________________________________
Bool_t TMapFile::IsFolder() const
{
   // Returns kTRUE in case object is a folder (i.e. contains browsable lists).

   if (fMmallocDesc && fVersion > 0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TMapFile::Browse(TBrowser *b)
{
   // Browse contents of TMapFile.

   if (b && fMmallocDesc) {

      AcquireSemaphore();

      TMapRec *mr = GetFirst();
      TKeyMapFile *keymap;
      if (!fBrowseList) fBrowseList = new TList();
      while (OrgAddress(mr)) {
         keymap = (TKeyMapFile*)fBrowseList->FindObject(mr->GetName(fOffset));
         if (!keymap) {
            keymap = new TKeyMapFile(mr->GetName(fOffset),mr->GetClassName(fOffset),this);
            fBrowseList->Add(keymap);
         }
         b->Add(keymap, keymap->GetName());
         mr = mr->GetNext(fOffset);
      }

      ReleaseSemaphore();

   }
}

//______________________________________________________________________________
Bool_t TMapFile::cd(const char *path)
{
   // Cd to associated directory,

   if (fDirectory)
      return fDirectory->cd(path);
   return kFALSE;
}

//______________________________________________________________________________
void TMapFile::ls(Option_t *) const
{
   // List contents of TMapFile.

   if (fMmallocDesc) {

      ((TMapFile*)this)->AcquireSemaphore();

      Printf("%-20s %-20s %-10s", "Object", "Class", "Size");
      if (!fFirst)
         Printf("*** no objects stored in memory mapped file ***");

      TMapRec *mr = GetFirst();
      while (OrgAddress(mr)) {
         Printf("%-20s %-20s %-10d", mr->GetName(fOffset),
                mr->GetClassName(fOffset), mr->fBufSize);
         mr = mr->GetNext(fOffset);
      }

      ((TMapFile*)this)->ReleaseSemaphore();

   }
}

//_______________________________________________________________________
void TMapFile::SumBuffer(Int_t bufsize)
{
   // Increment statistics for buffer sizes of objects in this file.

   fWritten++;
   fSumBuffer  += bufsize;
   fSum2Buffer += bufsize*bufsize;
}

//______________________________________________________________________________
Int_t TMapFile::GetBestBuffer()
{
   // Return the best buffer size for objects in this file.
   //
   // The best buffer size is estimated based on the current mean value
   // and standard deviation of all objects written so far to this file.
   // Returns mean value + one standard deviation.

   if (!fWritten) return TBuffer::kMinimalSize;
   Double_t mean = fSumBuffer/fWritten;
   Double_t rms2 = TMath::Abs(fSum2Buffer/fSumBuffer - mean*mean);
   return (Int_t)(mean + std::sqrt(rms2));
}


//______________________________________________________________________________
TMapFile *TMapFile::Create(const char *name, Option_t *option, Int_t size,
                           const char *title)
{
   // Create a memory mapped file. This opens a file (to which the
   // memory will be mapped) and attaches a memory region to it.
   // Option can be either: "NEW", "CREATE", "RECREATE", "UPDATE"
   // or "READ" (see TFile). The default open mode is "READ". The size
   // argument specifies the maximum size of shared memory file in bytes.
   // TMapFile's can only be created via this method. Create() enforces that
   // a TMapFile is always on the memory mapped heap (when "NEW", "CREATE"
   // or "RECREATE" are used).

   TMapFile *newMapFile;
   new TMapFile(name, title, option, size, newMapFile);

   return newMapFile;
}

//______________________________________________________________________________
void TMapFile::SetMapAddress(Long_t addr)
{
   // Set preferred map address. Find out preferred map address as follows:
   // 1) Run consumer program to find the preferred map address:
   //       $ root
   //       root [0] m = TMapFile::Create("dummy.map", "recreate", 10000000);
   //       root [1] m.Print()
   //       Memory mapped file:   dummy.map
   //       Title:
   //       Option:               CREATE
   //       Mapped Memory region: 0x40b4c000 - 0x40d95f00 (2.29 MB)
   //       Current breakval:     0x40b53000
   //       root [2] .q
   //       $ rm dummy.map
   //    Remember begin of mapped region, i.e. 0x40b4c000
   //
   // 2) Add to producer program, just before creating the TMapFile:
   //       TMapFile::SetMapAddress(0x40b4c000);
   //
   // Repeat this if more than one map file is being used.
   //
   // The above procedure allow programs using, e.g., different number of
   // shared libraries (that cause the default mapping address to be
   // different) to create shared memory regions in the same location
   // without overwriting a shared library. The above assumes the consumer
   // program is larger (i.e. has more shared memory occupied) than the
   // producer. If this is not true inverse the procedure.

   fgMapAddress = addr;
}

//______________________________________________________________________________
void *TMapFile::MapToAddress()
{
   // Return the base address at which we would like the next TMapFile's
   // mapped data to start.
   //
   // For now, we let the system decide (start address 0). There are
   // a lot of issues to deal with here to make this work reasonably,
   // including:
   //
   // - Avoid memory collisions with existing mapped address spaces
   //
   // - Reclaim address spaces when their mmalloc heaps are unmapped
   //
   // - When mmalloc heaps are shared between processes they have to be
   //   mapped at the same addresses in each
   //
   // Once created, a mmalloc heap that is to be mapped back in must be
   // mapped at the original address.  I.E. each TMapFile will expect
   // to be remapped at it's original address. This becomes a problem if
   // the desired address is already in use.

#ifdef R__HAVE_MMAP
   if (TStorage::HasCustomNewDelete())
      return (void *)fgMapAddress;
   else
      return (void *)-1;
#else
   return (void *)-1;
#endif
}

//______________________________________________________________________________
void TMapFile::operator delete(void *ptr)
{
   // Need special "operator delete" in which we close the shared memory.
   // This has to be done after the dtor chain has been finished.

   mmalloc_detach(fgMmallocDesc);
   fgMmallocDesc = 0;

   TObject::operator delete(ptr);
}
