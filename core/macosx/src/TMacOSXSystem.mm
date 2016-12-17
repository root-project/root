// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov 5/12/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMacOSXSystem.h"

#include <Cocoa/Cocoa.h>

#include "CocoaUtils.h"
#include "TVirtualX.h"
#include "TError.h"
#include "TROOT.h"

#include <stdexcept>
#include <vector>
#include <map>
#include <set>

class TSeqCollection;
//Handle deprecated symbols
namespace ROOT {
namespace MacOSX {
namespace Details {
#ifdef MAC_OS_X_VERSION_10_12
const NSUInteger kEventMaskAny = NSEventMaskAny;
const NSEventType kApplicationDefined = NSEventTypeApplicationDefined;
#else
const NSUInteger kEventMaskAny = NSAnyEventMask;
const NSEventType kApplicationDefined = NSApplicationDefined;
#endif
}
}
}


//The special class to perform a selector to stop a -run: method.
@interface RunStopper : NSObject
@end

@implementation RunStopper

//We attach this delegate only once, when trying to initialize
//NSApplication (by calling its -run method).
//______________________________________________________________________________
- (void) stopRun
{
   [NSApp stop : nil];
   //This is not enough to stop, from docs:
   //This method notifies the application that you want to exit the current run loop
   //as soon as it finishes processing the current NSEvent object.
   //This method does not forcibly exit the current run loop. Instead it sets a flag
   //that the application checks only after it finishes dispatching an actual event object.


   //I'm sending a fake event, to stop.
   NSEvent* stopEvent = [NSEvent otherEventWithType : ROOT::MacOSX::Details::kApplicationDefined
                         location : NSMakePoint(0., 0.) modifierFlags : 0 timestamp : 0.
                                 windowNumber : 0 context : nil subtype: 0 data1 : 0 data2 : 0];
   [NSApp postEvent : stopEvent atStart : true];
}

@end


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
   TFdSet &operator=(const TFdSet &rhs)
   {
      if (this != &rhs) {
         memcpy(fds_bits, rhs.fds_bits, sizeof(rhs.fds_bits));
      }
      return *this;
   }
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
namespace Details {

class MacOSXSystem {
public:
   MacOSXSystem();
   ~MacOSXSystem();

   void InitializeCocoa();

   bool SetFileDescriptors(const TSeqCollection *fileHandlers);
   void UnregisterFileDescriptor(CFFileDescriptorRef fd);
   void CloseFileDescriptors();

   enum DescriptorType {
      kDTWrite,
      kDTRead
   };

   //Before I had C++11 and auto, now I have ugly typedefs.
   void SetFileDescriptor(int fd, DescriptorType fdType);

   std::set<CFFileDescriptorRef> fCFFileDescriptors;

   ROOT::MacOSX::Util::AutoreleasePool fPool;
   bool fCocoaInitialized;

   static MacOSXSystem *fgInstance;
};

MacOSXSystem *MacOSXSystem::fgInstance = 0;

extern "C" {

//______________________________________________________________________________
void TMacOSXSystem_ReadCallback(CFFileDescriptorRef fdref, CFOptionFlags /*callBackTypes*/, void * /*info*/)
{
   //Native descriptor.
   const int nativeFD = CFFileDescriptorGetNativeDescriptor(fdref);

   //We do not need this descriptor anymore.
   assert(MacOSXSystem::fgInstance != 0 && "TMacOSXSystem_ReadCallback, MacOSXSystem's singleton is null");
   MacOSXSystem::fgInstance->UnregisterFileDescriptor(fdref);

   CFFileDescriptorInvalidate(fdref);
   CFRelease(fdref);

   NSEvent *fdEvent = [NSEvent otherEventWithType : kApplicationDefined
                       location : NSMakePoint(0, 0) modifierFlags : 0
                       timestamp: 0. windowNumber : 0 context : nil
                       subtype : 0 data1 : nativeFD data2 : 0];
   [NSApp postEvent : fdEvent atStart : NO];
}

//______________________________________________________________________________
void TMacOSXSystem_WriteCallback(CFFileDescriptorRef fdref, CFOptionFlags /*callBackTypes*/, void * /*info*/)
{
   //Native descriptor.
   const int nativeFD = CFFileDescriptorGetNativeDescriptor(fdref);

   //We do not need this descriptor anymore.
   assert(MacOSXSystem::fgInstance != 0 && "TMacOSXSystem_WriteCallback, MacOSXSystem's singleton is null");
   MacOSXSystem::fgInstance->UnregisterFileDescriptor(fdref);

   CFFileDescriptorInvalidate(fdref);
   CFRelease(fdref);

   NSEvent *fdEvent = [NSEvent otherEventWithType : kApplicationDefined
                       location : NSMakePoint(0, 0) modifierFlags : 0
                       timestamp: 0. windowNumber : 0 context : nil
                       subtype : 0 data1 : nativeFD data2 : 0];
   [NSApp postEvent : fdEvent atStart : NO];
}

}

//______________________________________________________________________________
MacOSXSystem::MacOSXSystem()
                : fPool(true), //Delay the pool creation!
                  fCocoaInitialized(false)
{
   assert(fgInstance == 0 && "MacOSXSystem, fgInstance was initialized already");
   fgInstance = this;
}

//______________________________________________________________________________
MacOSXSystem::~MacOSXSystem()
{
   if (fCocoaInitialized)
      CloseFileDescriptors();
}

//______________________________________________________________________________
void MacOSXSystem::InitializeCocoa()
{
   assert(fCocoaInitialized == false && "InitializeCocoa, Cocoa was initialized already");

   [NSApplication sharedApplication];
   fPool.Reset();//TODO: test, should it be BEFORE shared application???
   fCocoaInitialized = true;
}

//______________________________________________________________________________
bool MacOSXSystem::SetFileDescriptors(const TSeqCollection *fileHandlers)
{
   //Allocates some resources and can throw.
   //So, make sure resources are freed correctly
   //in case of exception (std::bad_alloc) and
   //return false. Return true if everything is ok.

   assert(fileHandlers != 0 && "SetFileDescriptors, parameter 'fileHandlers' is null");

   try {
      //This iteration is really stupid: add a null pointer into the middle of collection
      //and it will stop in the middle! AddFileHandler has a check for this.

      TIter next(fileHandlers);
      while (TFileHandler * const handler = static_cast<TFileHandler *>(next())) {
         assert(handler->GetFd() != -1 && "SetFileDescriptors, invalid file descriptor");

         if (handler->HasReadInterest())
            SetFileDescriptor(handler->GetFd(), kDTRead);

         if (handler->HasWriteInterest())
            SetFileDescriptor(handler->GetFd(), kDTWrite);
      }
   } catch (const std::exception &) {
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
   //While Core Foundation is not Cocoa, it still should not be used if we are not initializing Cocoa.
   assert(fCocoaInitialized == true && "CloseFileDescriptors, Cocoa was not initialized");

   std::set<CFFileDescriptorRef>::iterator fdIter = fCFFileDescriptors.begin(), end = fCFFileDescriptors.end();

   for (; fdIter != end; ++fdIter) {
      CFFileDescriptorInvalidate(*fdIter);
      CFRelease(*fdIter);
   }

   fCFFileDescriptors.clear();
}

//______________________________________________________________________________
void MacOSXSystem::SetFileDescriptor(int fd, DescriptorType fdType)
{
   //While CoreFoundation is not Cocoa, it still should not be used if we are not initializing Cocoa.
   assert(fCocoaInitialized == true && "SetFileDescriptors, Cocoa was not initialized");
   //-1 can come from TSysEvtHandler's ctor.
   assert(fd != -1 && "SetFileDescriptor, invalid file descriptor");

   const bool read = fdType == kDTRead;
   CFFileDescriptorRef fdref = CFFileDescriptorCreate(kCFAllocatorDefault, fd, false,
                               read ? TMacOSXSystem_ReadCallback : TMacOSXSystem_WriteCallback, 0);

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

}//Details
}//MacOSX
}//ROOT

namespace Private = ROOT::MacOSX::Details;

ClassImp(TMacOSXSystem)

//______________________________________________________________________________
TMacOSXSystem::TMacOSXSystem()
                  : fPimpl(new Private::MacOSXSystem),
                    fCocoaInitialized(false),
                    fFirstDispatch(true)
{
}

//______________________________________________________________________________
TMacOSXSystem::~TMacOSXSystem()
{
}

//______________________________________________________________________________
void TMacOSXSystem::DispatchOneEvent(Bool_t pendingOnly)
{
   //Here I try to emulate TUnixSystem's behavior, which is quite twisted.
   //I'm not even sure, I need all this code :)

   if (fFirstDispatch) {
      if (!fCocoaInitialized && !gROOT->IsBatch())
         InitializeCocoa();

      fFirstDispatch = false;
   }

   if (!fCocoaInitialized)//We are in a batch mode (or 'batch').
      return TUnixSystem::DispatchOneEvent(pendingOnly);

   Bool_t pollOnce = pendingOnly;

   while (true) {
      const ROOT::MacOSX::Util::AutoreleasePool pool;

      if (ProcessPendingEvents()) {
         //If we had some events in a system queue, probably,
         //now we have some events in our own event-queue.
         if (gXDisplay && gXDisplay->Notify()) {
            gVirtualX->Update(2);
            gVirtualX->Update(3);
            if (!pendingOnly)
               return;
         }
      }

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
         //if (fFileHandler && !fFileHandler->GetSize())
         //   return;
         nextto = 0;
         pollOnce = kFALSE;
      }

      //Wait for GUI events and for something else, like read/write from stdin/stdout (?).
      WaitEvents(nextto);

      if (gXDisplay && gXDisplay->Notify()) {
         gVirtualX->Update(2);
         gVirtualX->Update(3);
         if (!pendingOnly)
            return;
      }

      if (pendingOnly)
         return;
   }
}

//______________________________________________________________________________
bool TMacOSXSystem::CocoaInitialized() const
{
   return fCocoaInitialized;
}

//______________________________________________________________________________
void TMacOSXSystem::InitializeCocoa()
{
   if (fCocoaInitialized)
      return;

   //TODO: add error handling and results check.

   fPimpl->InitializeCocoa();

   const ROOT::MacOSX::Util::AutoreleasePool pool;

   //[NSApplication sharedApplication];//TODO: clean-up this mess with pools and sharedApplication
   //Documentation says, that +sharedApplication, initializes the app. But this is not true,
   //it's still not really initialized, part of initialization is done by -run method.

   //If you call run, it never returns unless app is finished. I have to stop Cocoa's event loop
   //processing, since we have our own event loop.

   const ROOT::MacOSX::Util::NSScopeGuard<RunStopper> stopper([[RunStopper alloc] init]);

   //Delay? What it should be?
   [stopper.Get() performSelector : @selector(stopRun) withObject : nil afterDelay : 0.05];
   [NSApp run];

   fCocoaInitialized = true;
}

//______________________________________________________________________________
bool TMacOSXSystem::ProcessPendingEvents()
{
   assert(fCocoaInitialized == true && "ProcessPendingEvents, called while Cocoa was not initialized");

   bool processed = false;
   while (NSEvent *event = [NSApp nextEventMatchingMask : Private::kEventMaskAny
                            untilDate : nil inMode : NSDefaultRunLoopMode dequeue : YES]) {
      [NSApp sendEvent : event];
      processed = true;
   }
   return processed;
}

//______________________________________________________________________________
void TMacOSXSystem::WaitEvents(Long_t nextto)
{
   //Wait for GUI/Non-GUI events.

   assert(fCocoaInitialized == true && "WaitEvents, called while Cocoa was not initialized");

   if (fFileHandler && !fPimpl->SetFileDescriptors(fFileHandler)) {
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

   NSEvent *event = [NSApp nextEventMatchingMask : Private::kEventMaskAny
                     untilDate : untilDate inMode : NSDefaultRunLoopMode dequeue : YES];
   if (event) {
      if (event.type == Private::kApplicationDefined)
         ProcessApplicationDefinedEvent(event);
      else
         [NSApp sendEvent : event];
   }

   while ((event = [NSApp nextEventMatchingMask : Private::kEventMaskAny
          untilDate : nil inMode : NSDefaultRunLoopMode dequeue : YES]))
   {
      if (event.type == Private::kApplicationDefined)
         ProcessApplicationDefinedEvent(event);
      else
         [NSApp sendEvent : event];
   }

   fPimpl->CloseFileDescriptors();

   gVirtualX->Update(2);
   gVirtualX->Update(3);
}

//______________________________________________________________________________
void TMacOSXSystem::AddFileHandler(TFileHandler *fh)
{
   if (fh) {
      if (fh->GetFd() == -1)//I do not need this crap!
         Error("AddFileHandler", "invalid file descriptor");
      else
         TUnixSystem::AddFileHandler(fh);
   }
}

//______________________________________________________________________________
TFileHandler *TMacOSXSystem::RemoveFileHandler(TFileHandler *fh)
{
   if (fh)
      return TUnixSystem::RemoveFileHandler(fh);

   return 0;
}

//______________________________________________________________________________
void TMacOSXSystem::ProcessApplicationDefinedEvent(void *e)
{
   //Right now I have app. defined events only
   //for file descriptors. This can change in a future.

   assert(fCocoaInitialized == true &&
          "ProcessApplicationDefinedEvent, called while Cocoa was not initialized");

   NSEvent *event = (NSEvent *)e;
   assert(event != nil &&
          "ProcessApplicationDefinedEvent, event parameter is nil");
   assert(event.type == Private::kApplicationDefined &&
          "ProcessApplicationDefinedEvent, event parameter has wrong type");

   bool descriptorFound = false;

   if (fReadmask->IsSet(event.data1)) {
      fReadready->Set(event.data1);
      descriptorFound = true;
   }

   if (fWritemask->IsSet(event.data1)) {
      fWriteready->Set(event.data1);
      descriptorFound = true;
   }

   if (!descriptorFound) {
      Error("ProcessApplicationDefinedEvent", "file descriptor %d was not found", int(event.data1));
      return;
   }

   ++fNfd;
}

