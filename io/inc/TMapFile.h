// @(#)root/io:$Id$
// Author: Fons Rademakers   08/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMapFile
#define ROOT_TMapFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMapFile                                                             //
//                                                                      //
// This class implements a shared memory region mapped to a file.       //
// Objects can be placed into this shared memory area using the Add()   //
// member function. Whenever the mapped object(s) change(s) call        //
// Update() to put a fresh copy in the shared memory. This extra        //
// step is necessary since it is not possible to share objects with     //
// virtual pointers between processes (the vtbl ptr points to the       //
// originators unique address space and can not be used by the          //
// consumer process(es)). Consumer processes can map the memory region  //
// from this file and access the objects stored in it via the Get()     //
// method (which returns a copy of the object stored in the shared      //
// memory with correct vtbl ptr set). Only objects of classes with a    //
// Streamer() member function defined can be shared.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#include "Windows4Root.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif
#if !defined(__MMPRIVATE_H) && !defined(__CINT__)
#include "mmprivate.h"
#endif


class TBrowser;
class TDirectory;
class TList;
class TMapRec;

class TMapFile : public TObject {

friend class TMapRec;

private:
   Int_t       fFd;             //Descriptor of mapped file
   Int_t       fVersion;        //ROOT version (or -1 for shadow map file)
   char       *fName;           //Name of mapped file
   char       *fTitle;          //Title of mapped file
   char       *fOption;         //Directory creation options
   void       *fMmallocDesc;    //Pointer to mmalloc descriptor
   ULong_t     fBaseAddr;       //Base address of mapped memory region
   Int_t       fSize;           //Original start size of memory mapped region
   TMapRec    *fFirst;          //List of streamed objects is shared memory
   TMapRec    *fLast;           //Last object in list of shared objects
   Long_t      fOffset;         //Offset in bytes for region mapped by reader
   TDirectory *fDirectory;      //Pointer to directory associated to this mapfile
   TList      *fBrowseList;     //List of KeyMapFile objects
   Bool_t      fWritable;       //TRUE if mapped file opened in RDWR mode
   Int_t       fSemaphore;      //Modification semaphore (or getpid() for WIN32)
   ULong_t     fhSemaphore;     //HANDLE of WIN32 Mutex object to implement semaphore
   TObject    *fGetting;        //Don't deadlock in update mode, when from Get() Add() is called
   Int_t       fWritten;        //Number of objects written sofar
   Double_t    fSumBuffer;      //Sum of buffer sizes of objects written sofar
   Double_t    fSum2Buffer;     //Sum of squares of buffer sizes of objects written so far

   static Long_t fgMapAddress;  //Map to this address, set address via SetMapAddress()
   static void  *fgMmallocDesc; //Used in Close() and operator delete()

protected:
   TMapFile();
   TMapFile(const char *name, const char *title, Option_t *option, Int_t size, TMapFile *&newMapFile);
   TMapFile(const TMapFile &f, Long_t offset = 0);
   void       operator=(const TMapFile &rhs);  // not implemented

   TMapFile  *FindShadowMapFile();
   void       InitDirectory();
   TObject   *Remove(TObject *obj, Bool_t lock);
   TObject   *Remove(const char *name, Bool_t lock);
   void       SumBuffer(Int_t bufsize);
   Int_t      GetBestBuffer();

   void   CreateSemaphore(Int_t pid=0);
   Int_t  AcquireSemaphore();
   Int_t  ReleaseSemaphore();
   void   DeleteSemaphore();

   static void *MapToAddress();

public:
   enum { kDefaultMapSize = 0x80000 }; // default size of mapped heap is 500 KB

   // Should both be protected (waiting for cint)
   virtual ~TMapFile();
   void     operator delete(void *vp);

   void          Browse(TBrowser *b);
   void          Close(Option_t *option = "");
   void         *GetBaseAddr() const { return (void *)fBaseAddr; }
   void         *GetBreakval() const;
   TDirectory   *GetDirectory() const {return fDirectory;}
   Int_t         GetFd() const { return fFd; }
   void         *GetMmallocDesc() const { return fMmallocDesc; }
   const char   *GetName() const { return fName; }
   Int_t         GetSize() const { return fSize; }
   const char   *GetOption() const { return fOption; }
   const char   *GetTitle() const { return fTitle; }
   TMapRec      *GetFirst() const { return (TMapRec*)((Long_t) fFirst + fOffset); }
   TMapRec      *GetLast() const { return (TMapRec*)((Long_t) fLast + fOffset); }
   Bool_t        IsFolder() const;
   Bool_t        IsWritable() const { return fWritable; }
   void         *OrgAddress(void *addr) const { return (void *)((Long_t)addr - fOffset); }
   void          Print(Option_t *option="") const;
   void          ls(Option_t *option="") const;
   Bool_t        cd(const char *path = 0);

   void          Add(const TObject *obj, const char *name = "");
   void          Update(TObject *obj = 0);
   TObject      *Remove(TObject *obj) { return Remove(obj, kTRUE); }
   TObject      *Remove(const char *name) { return Remove(name, kTRUE); }
   void          RemoveAll();
   TObject      *Get(const char *name, TObject *retObj = 0);

   static TMapFile *Create(const char *name, Option_t *option="READ", Int_t size=kDefaultMapSize, const char *title="");
   static TMapFile *WhichMapFile(void *addr);
   static void      SetMapAddress(Long_t addr);

   ClassDef(TMapFile,0)  // Memory mapped directory structure
};



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMapRec                                                              //
//                                                                      //
// A TMapFile contains a list of TMapRec objects which keep track of    //
// the actual objects stored in the mapped file.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMapRec {

friend class TMapFile;

private:
   char            *fName;       // object name
   char            *fClassName;  // class name
   TObject         *fObject;     // pointer to original object
   void            *fBuffer;     // buffer containing object of class name
   Int_t            fBufSize;    // buffer size
   TMapRec         *fNext;       // next MapRec in list

public:
   TMapRec(const char *name, const TObject *obj, Int_t size, void *buf);
   ~TMapRec();
   const char   *GetName(Long_t offset = 0) const { return (char *)((Long_t) fName + offset); }
   const char   *GetClassName(Long_t offset = 0) const { return (char *)((Long_t) fClassName + offset); }
   void         *GetBuffer(Long_t offset = 0) const { return (void *)((Long_t) fBuffer + offset); }
   Int_t         GetBufSize() const { return fBufSize; }
   TObject      *GetObject() const;
   TMapRec      *GetNext(Long_t offset = 0) const { return (TMapRec *)((Long_t) fNext + offset); }
};


//______________________________________________________________________________
inline void *TMapFile::GetBreakval() const
{
   // Return the current location in the memory region for this malloc heap which
   // represents the end of memory in use. Returns 0 if map file was closed.

   if (!fMmallocDesc) return 0;
   return (void *)((struct mdesc *)fMmallocDesc)->breakval;
}

//______________________________________________________________________________
inline TMapFile *TMapFile::WhichMapFile(void *addr)
{
   if (!gROOT || !gROOT->GetListOfMappedFiles()) return 0;

   TObjLink *lnk = ((TList *)gROOT->GetListOfMappedFiles())->LastLink();
   while (lnk) {
      TMapFile *mf = (TMapFile*)lnk->GetObject();
      if (!mf) return 0;
      if ((ULong_t)addr >= mf->fBaseAddr + mf->fOffset &&
          (ULong_t)addr <  (ULong_t)mf->GetBreakval() + mf->fOffset)
         return mf;
      lnk = lnk->Prev();
   }
   return 0;
}

R__EXTERN void *gMmallocDesc;  //is initialized in TClass.cxx

#endif
