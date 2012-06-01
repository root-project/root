// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov 5/12/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#define NDEBUG

#include <stdexcept>
#include <vector>
#include <map>
#include <set>

#import <Cocoa/Cocoa.h>

#include "TSeqCollection.h"
#include "TMacOSXSystem.h"
#include "CocoaUtils.h"
#include "TVirtualX.h"
#include "TError.h"

//
//Stuff which I have to copy from TUnixSystem. Find a better way to organize code.
//Fortunately, this does not violate ODR, but still UGLY.
//

//------------------- Unix TFdSet ----------------------------------------------
#ifndef HOWMANY
#   define HOWMANY(x, y)   (((x)+((y)-1))/(y))
#endif

const Int_t kNFDBITS = (sizeof(Long_t) * 8);  // 8 bits per byte
#ifdef FD_SETSIZE
const Int_t kFDSETSIZE = FD_SETSIZE;          // Linux = 1024 file descriptors
#else
const Int_t kFDSETSIZE = 256;                 // upto 256 file descriptors
#endif


class TFdSet {
private:
   ULong_t fds_bits[HOWMANY(kFDSETSIZE, kNFDBITS)];
public:
   TFdSet() { memset(fds_bits, 0, sizeof(fds_bits)); }
   TFdSet(const TFdSet &org) { memcpy(fds_bits, org.fds_bits, sizeof(org.fds_bits)); }
   TFdSet &operator=(const TFdSet &rhs) { if (this != &rhs) { memcpy(fds_bits, rhs.fds_bits, sizeof(rhs.fds_bits));} return *this; }
   void   Zero() { memset(fds_bits, 0, sizeof(fds_bits)); }
   void   Set(Int_t n)
   {
      if (n >= 0 && n < kFDSETSIZE) {
         fds_bits[n/kNFDBITS] |= (1UL << (n % kNFDBITS));
      } else {
         ::Fatal("TFdSet::Set","fd (%d) out of range [0..%d]", n, kFDSETSIZE-1);
      }
   }
   void   Clr(Int_t n)
   {
      if (n >= 0 && n < kFDSETSIZE) {
         fds_bits[n/kNFDBITS] &= ~(1UL << (n % kNFDBITS));
      } else {
         ::Fatal("TFdSet::Clr","fd (%d) out of range [0..%d]", n, kFDSETSIZE-1);
      }
   }
   Int_t  IsSet(Int_t n)
   {
      if (n >= 0 && n < kFDSETSIZE) {
         return (fds_bits[n/kNFDBITS] & (1UL << (n % kNFDBITS))) != 0;
      } else {
         ::Fatal("TFdSet::IsSet","fd (%d) out of range [0..%d]", n, kFDSETSIZE-1);
         return 0;
      }
   }
   ULong_t *GetBits() { return (ULong_t *)fds_bits; }
};

namespace ROOT {
namespace MacOSX {
namespace Detail {

class MacOSXSystem {
public:
   enum DescriptorType {
      kDTWrite,
      kDTRead
   };

   MacOSXSystem();
   ~MacOSXSystem();


   void AddFileHandler(TFileHandler *fh);
   void RemoveFileHandler(TFileHandler *fh);

   bool SetFileDescriptors();
   void UnregisterFileDescriptor(CFFileDescriptorRef fd);
   void CloseFileDescriptors();

   std::map<int, DescriptorType> fFileDescriptors;
   std::set<CFFileDescriptorRef> fCFFileDescriptors;   
   const ROOT::MacOSX::Util::AutoreleasePool fPool;

   static MacOSXSystem *fgInstance;
};

MacOSXSystem *MacOSXSystem::fgInstance = nullptr;

extern "C" {

//______________________________________________________________________________
void TMacOSXSystem_ReadCallback(CFFileDescriptorRef fdref, CFOptionFlags /*callBackTypes*/, void * /*info*/)
{
   //Native descriptor.
   const int nativeFD = CFFileDescriptorGetNativeDescriptor(fdref);

   //We do not need this descriptor anymore.
   assert(MacOSXSystem::fgInstance != nullptr && "TMacOSXSystem_ReadCallback, MacOSXSystem's singleton is null");   
   MacOSXSystem::fgInstance->UnregisterFileDescriptor(fdref);
   
   CFFileDescriptorInvalidate(fdref);
   CFRelease(fdref);

   NSEvent *fdEvent = [NSEvent otherEventWithType : NSApplicationDefined location : NSMakePoint(0, 0) modifierFlags : 0
                       timestamp: 0. windowNumber : 0 context : nil subtype : 0 data1 : nativeFD data2 : 0];
   [NSApp postEvent : fdEvent atStart : NO];
}

//______________________________________________________________________________
void TMacOSXSystem_WriteCallback(CFFileDescriptorRef fdref, CFOptionFlags /*callBackTypes*/, void * /*info*/)
{
   //Native descriptor.
   const int nativeFD = CFFileDescriptorGetNativeDescriptor(fdref);

   //We do not need this descriptor anymore.
   assert(MacOSXSystem::fgInstance != nullptr && "TMacOSXSystem_WriteCallback, MacOSXSystem's singleton is null");   
   MacOSXSystem::fgInstance->UnregisterFileDescriptor(fdref);

   CFFileDescriptorInvalidate(fdref);
   CFRelease(fdref);

   NSEvent *fdEvent = [NSEvent otherEventWithType : NSApplicationDefined location : NSMakePoint(0, 0) modifierFlags : 0
                       timestamp: 0. windowNumber : 0 context : nil subtype : 0 data1 : nativeFD data2 : 0];
   [NSApp postEvent : fdEvent atStart : NO];
}

}

//______________________________________________________________________________
MacOSXSystem::MacOSXSystem()
{
   assert(fgInstance == 0 && "MacOSXSystem, fgInstance was initialized already");

   fgInstance = this;
}

//______________________________________________________________________________
MacOSXSystem::~MacOSXSystem()
{
   CloseFileDescriptors();
}

//______________________________________________________________________________
void MacOSXSystem::AddFileHandler(TFileHandler *fh)
{
   //Can throw std::bad_alloc. I'm not allocating any resources here, so I'm not going to catch here.
   assert(fh != 0 && "AddFileHandler, fh parameter is null");
   assert(fFileDescriptors.find(fh->GetFd()) == fFileDescriptors.end() && "AddFileHandler, file descriptor was registered already");

   fFileDescriptors[fh->GetFd()] = fh->HasReadInterest() ? kDTRead : kDTWrite;
}

//______________________________________________________________________________
void MacOSXSystem::RemoveFileHandler(TFileHandler *fh)
{
   //Can not throw.
   assert(fh != 0 && "RemoveFileHandler, fh parameter is null");
   std::map<int, DescriptorType>::iterator fdIter = fFileDescriptors.find(fh->GetFd());

   assert(fdIter != fFileDescriptors.end() && "RemoveFileHandler, file handler was not found");
   fFileDescriptors.erase(fdIter);
}

//______________________________________________________________________________
bool MacOSXSystem::SetFileDescriptors()
{
   //Allocates some resources and can throw.
   //So, make sure resources are freed correctly
   //in case of exception (std::bad_alloc) and
   //return false. Return true if everything is ok.

   try {
      for (std::map<int, DescriptorType>::iterator fdIter = fFileDescriptors.begin(), end = fFileDescriptors.end(); fdIter != end; ++fdIter) {
         const bool read = fdIter->second == kDTRead;
         CFFileDescriptorRef fdref = CFFileDescriptorCreate(kCFAllocatorDefault, fdIter->first, false, read ? TMacOSXSystem_ReadCallback : TMacOSXSystem_WriteCallback, 0);

         if (!fdref)
            throw std::runtime_error("MacOSXSystem::SetFileDescriptors: CFFileDescriptorCreate failed");

         CFFileDescriptorEnableCallBacks(fdref, read ? kCFFileDescriptorReadCallBack : kCFFileDescriptorWriteCallBack);
      
         CFRunLoopSourceRef runLoopSource = CFFileDescriptorCreateRunLoopSource(kCFAllocatorDefault, fdref, 0);
         
         if (!runLoopSource) {
            CFRelease(fdref);
            throw std::runtime_error("MacOSXSystem::SetFileDescriptors: CFFileDescriptorCreateRunLoopSource failed");
         }

         CFRunLoopAddSource(CFRunLoopGetMain(), runLoopSource, kCFRunLoopDefaultMode);
         CFRelease(runLoopSource);

         fCFFileDescriptors.insert(fdref);
      }
   } catch (const std::exception &e) {
      CloseFileDescriptors();
      return false;
   }

   return true;
}

//______________________________________________________________________________
void MacOSXSystem::UnregisterFileDescriptor(CFFileDescriptorRef fd)
{
   //This function does not touch file descriptor (it's invalidated and released externally),
   //just remove pointer.

   std::set<CFFileDescriptorRef>::iterator fdIter = fCFFileDescriptors.find(fd);
   assert(fdIter != fCFFileDescriptors.end() && "InvalidateFileDescriptor, file descriptor was not found");
   fCFFileDescriptors.erase(fdIter);
}

//______________________________________________________________________________
void MacOSXSystem::CloseFileDescriptors()
{
   for (std::set<CFFileDescriptorRef>::iterator fdIter = fCFFileDescriptors.begin(), end = fCFFileDescriptors.end(); fdIter != end; ++fdIter) {
      CFFileDescriptorInvalidate(*fdIter);
      CFRelease(*fdIter);
   }   

   fCFFileDescriptors.clear();
}

}//Detail
}//MacOSX
}//ROOT

namespace Private = ROOT::MacOSX::Detail;

ClassImp(TMacOSXSystem)

//______________________________________________________________________________
TMacOSXSystem::TMacOSXSystem()
                  : fPimpl(new Private::MacOSXSystem)
{
   [NSApplication sharedApplication];
}

//______________________________________________________________________________
TMacOSXSystem::~TMacOSXSystem()
{
}

//______________________________________________________________________________
void TMacOSXSystem::ProcessApplicationDefinedEvent(void *e)
{
   //Right now I have app. defined events only
   //for file descriptors. This can change in a future.
   NSEvent *event = (NSEvent *)e;

   assert(event != nil && "AddFileActivityEvent, event parameter is nil");
   assert(event.type == NSApplicationDefined && "AddFileActivityEvent, event parameter has wrong type");
   
   if (event.data2) {
      gVirtualX->DispatchClientMessage(event.data2);
   } else {
      std::map<int, Private::MacOSXSystem::DescriptorType>::iterator fdIter = fPimpl->fFileDescriptors.find(event.data1);
      assert(fdIter != fPimpl->fFileDescriptors.end() && "WaitForAllEvents, file descriptor from NSEvent not found");
      if (fdIter->second == Private::MacOSXSystem::kDTRead)
         fReadready->Set(event.data1);
      else
         fWriteready->Set(event.data1);
      
      ++fNfd;
   }
}

//______________________________________________________________________________
void TMacOSXSystem::WaitEvents(Long_t nextto)
{
   //Wait for GUI/Non-GUI events.

   if (!fPimpl->SetFileDescriptors()) {
      //I consider this error as fatal.
      Fatal("WaitForAllEvents", "SetFileDesciptors failed");
   }

   NSDate *untilDate = nil;
   if (nextto >= 0)//0 also means non-blocking call.
      untilDate = [NSDate dateWithTimeIntervalSinceNow : nextto / 1000.];
   else
      untilDate = [NSDate distantFuture];

   fReadready->Zero();
   fWriteready->Zero();
   fNfd = 0;

   NSEvent *event = [NSApp nextEventMatchingMask : NSAnyEventMask untilDate : untilDate inMode : NSDefaultRunLoopMode dequeue : YES];

   if (event.type == NSApplicationDefined)
      ProcessApplicationDefinedEvent(event);
   else
      [NSApp sendEvent : event];

   while ((event = [NSApp nextEventMatchingMask : NSAnyEventMask untilDate : nil inMode : NSDefaultRunLoopMode dequeue : YES])) {
      if (event.type == NSApplicationDefined)
         ProcessApplicationDefinedEvent(event);
      else
         [NSApp sendEvent : event];
   }

   fPimpl->CloseFileDescriptors();

   gVirtualX->Update(2);
   gVirtualX->Update(3);
}

//______________________________________________________________________________
void TMacOSXSystem::DispatchOneEvent(Bool_t pendingOnly)
{
   //Here I try to emulate TUnixSystem's behavior, which is quite twisted.
   //I'm not even sure, I need all this code :)   
   Bool_t pollOnce = pendingOnly;

   while (true) {
      const ROOT::MacOSX::Util::AutoreleasePool pool;

      //Check for file descriptors ready for reading/writing.
      if (fNfd > 0 && fFileHandler && fFileHandler->GetSize() > 0)
         if (CheckDescriptors())
            if (!pendingOnly)
               return;

      fNfd = 0;
      fReadready->Zero();
      fWriteready->Zero();

      if (pendingOnly && !pollOnce)
         return;

      // check synchronous signals
      if (fSigcnt > 0 && fSignalHandler->GetSize() > 0)
         if (CheckSignals(kTRUE))
            if (!pendingOnly) return;

      fSigcnt = 0;
      fSignals->Zero();

      // check synchronous timers
      Long_t nextto = 0;
      if (fTimers && fTimers->GetSize() > 0) {
         if (DispatchTimers(kTRUE)) {
            // prevent timers from blocking file descriptor monitoring
            nextto = NextTimeOut(kTRUE);
            if (nextto > kItimerResolution || nextto == -1)
               return;
         }
      }

      // if in pendingOnly mode poll once file descriptor activity
      nextto = NextTimeOut(kTRUE);
      
      if (pendingOnly) {
         if (fFileHandler && !fFileHandler->GetSize())
            return;
         nextto = 0;
         pollOnce = kFALSE;
      }

      //Wait for GUI events and for something else, like read/write from stdin/stdout (?).
      WaitEvents(nextto);
      if (pendingOnly)
         return;
   }
}

//______________________________________________________________________________
void TMacOSXSystem::AddFileHandler(TFileHandler *fh)
{
   fPimpl->AddFileHandler(fh);
   TUnixSystem::AddFileHandler(fh);
}

//______________________________________________________________________________
TFileHandler *TMacOSXSystem::RemoveFileHandler(TFileHandler *fh)
{
   fPimpl->RemoveFileHandler(fh);
   return TUnixSystem::RemoveFileHandler(fh);
}
