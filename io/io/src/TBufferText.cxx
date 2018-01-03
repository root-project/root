// $Id$
// Author: Sergey Linev  21.12.2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TBufferText
\ingroup IO

Base class for text-based streamers like TBufferJSON or TBufferXML
Special actions list will use methods, introduced in this class.

Idea to have equivalent methods names in TBufferFile and TBufferText, that
actions list for both are the same.
*/

#include "TBufferText.h"

#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TFile.h"
#include "TRefTable.h"
#include "TProcessID.h"
#include "TVirtualMutex.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TArrayC.h"
#include "TClonesArray.h"
#include "TExMap.h"
#include "TStreamerInfoActions.h"
#include "TError.h"

ClassImp(TBufferText);

const char *TBufferText::fgFloatFmt = "%e";
const char *TBufferText::fgDoubleFmt = "%.14e";

Int_t TBufferText::fgMapSize = kMapSize;
const UInt_t kNullTag = 0;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TBufferText::TBufferText()
   : TBuffer(), fPidOffset(0), fMapCount(0), fMapSize(0), fDisplacement(0), fMap(nullptr), fClassMap(nullptr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor

TBufferText::TBufferText(TBuffer::EMode mode, TObject *parent)
   : TBuffer(mode), fPidOffset(0), fMapCount(0), fMapSize(0), fDisplacement(0), fMap(nullptr), fClassMap(nullptr)
{
   fBufSize = 1000000000;

   fMapSize = fgMapSize;

   SetParent(parent);
   SetBit(kCannotHandleMemberWiseStreaming);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TBufferText::~TBufferText()
{
   delete fMap;
   delete fClassMap;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the last TProcessID in the file.

TProcessID *TBufferText::GetLastProcessID(TRefTable *reftable) const
{
   TFile *file = (TFile *)GetParent();
   // warn if the file contains > 1 PID (i.e. if we might have ambiguity)
   if (file && !reftable->TestBit(TRefTable::kHaveWarnedReadingOld) && file->GetNProcessIDs() > 1) {
      Warning("ReadBuffer", "The file was written during several processes with an "
                            "older ROOT version; the TRefTable entries might be inconsistent.");
      reftable->SetBit(TRefTable::kHaveWarnedReadingOld);
   }

   // the file's last PID is the relevant one, all others might have their tables overwritten
   TProcessID *fileProcessID = TProcessID::GetProcessID(0);
   if (file && file->GetNProcessIDs() > 0) {
      // take the last loaded PID
      fileProcessID = (TProcessID *)file->GetListOfProcessIDs()->Last();
   }
   return fileProcessID;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the exec id stored in the current TStreamerInfo element.
/// The execid has been saved in the unique id of the TStreamerElement
/// being read by TStreamerElement::Streamer.
/// The current element (fgElement) is set as a static global
/// by TStreamerInfo::ReadBuffer (Clones) when reading this TRef.

UInt_t TBufferText::GetTRefExecId()
{
   return TStreamerInfo::GetCurrentElement()->GetUniqueID();
}

////////////////////////////////////////////////////////////////////////////////
/// The TProcessID with number pidf is read from file.
/// If the object is not already entered in the gROOT list, it is added.

TProcessID *TBufferText::ReadProcessID(UShort_t pidf)
{
   TFile *file = (TFile *)GetParent();
   if (!file) {
      if (!pidf)
         return TProcessID::GetPID(); // may happen when cloning an object
      return 0;
   }

   TProcessID *pid = nullptr;
   {
      R__LOCKGUARD_IMT(gInterpreterMutex); // Lock for parallel TTree I/O
      pid = file->ReadProcessID(pidf);
   }

   return pid;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the ProcessID pid is already in the file.
/// If not, add it and return the index number in the local file list.

UShort_t TBufferText::WriteProcessID(TProcessID *pid)
{
   TFile *file = (TFile *)GetParent();
   if (!file)
      return 0;
   return file->WriteProcessID(pid);
}

////////////////////////////////////////////////////////////////////////////////
/// Mark the classindex of the current file as using this TStreamerInfo

void TBufferText::TagStreamerInfo(TVirtualStreamerInfo *info)
{
   TFile *file = (TFile *)GetParent();
   if (file) {
      TArrayC *cindex = file->GetClassIndex();
      Int_t nindex = cindex->GetSize();
      Int_t number = info->GetNumber();
      if (number < 0 || number >= nindex) {
         Error("TagStreamerInfo", "StreamerInfo: %s number: %d out of range[0,%d] in file: %s", info->GetName(), number,
               nindex, file->GetName());
         return;
      }
      if (cindex->fArray[number] == 0) {
         cindex->fArray[0] = 1;
         cindex->fArray[number] = 1;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// force writing the TStreamerInfo to the file

void TBufferText::ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force)
{
   if (info)
      info->ForceWriteInfo((TFile *)GetParent(), force);
}

////////////////////////////////////////////////////////////////////////////////
/// Make sure TStreamerInfo is not optimized, otherwise it will not be
/// possible to support schema evolution in read mode.
/// In case the StreamerInfo has already been computed and optimized,
/// one must disable the option BypassStreamer.

void TBufferText::ForceWriteInfoClones(TClonesArray *a)
{
   TStreamerInfo *sinfo = (TStreamerInfo *)a->GetClass()->GetStreamerInfo();
   ForceWriteInfo(sinfo, kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to TStreamerInfo::ReadBufferClones.

Int_t TBufferText::ReadClones(TClonesArray *a, Int_t nobjects, Version_t objvers)
{
   char **arr = (char **)a->GetObjectRef(0);
   char **end = arr + nobjects;
   // a->GetClass()->GetStreamerInfo()->ReadBufferClones(*this,a,nobjects,-1,0);
   TStreamerInfo *info = (TStreamerInfo *)a->GetClass()->GetStreamerInfo(objvers);
   // return info->ReadBuffer(*this,arr,-1,nobjects,0,1);
   return ApplySequenceVecPtr(*(info->GetReadMemberWiseActions(kTRUE)), arr, end);
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to TStreamerInfo::WriteBufferClones.

Int_t TBufferText::WriteClones(TClonesArray *a, Int_t nobjects)
{
   char **arr = reinterpret_cast<char **>(a->GetObjectRef(0));
   // a->GetClass()->GetStreamerInfo()->WriteBufferClones(*this,(TClonesArray*)a,nobjects,-1,0);
   TStreamerInfo *info = (TStreamerInfo *)a->GetClass()->GetStreamerInfo();
   // return info->WriteBufferAux(*this,arr,-1,nobjects,0,1);
   char **end = arr + nobjects;
   // No need to tell call ForceWriteInfo as it by ForceWriteInfoClones.
   return ApplySequenceVecPtr(*(info->GetWriteMemberWiseActions(kTRUE)), arr, end);
}

////////////////////////////////////////////////////////////////////////////////
/// Write object in buffer
/// !!! Should be used only by TBufferText itself.

void TBufferText::WriteObject(const TObject *obj, Bool_t cacheReuse /* = kTRUE */)
{
   WriteObjectAny(obj, TObject::Class(), cacheReuse);
}

////////////////////////////////////////////////////////////////////////////////

namespace {
struct DynamicType {
   // Helper class to enable typeid on any address
   // Used in code similar to:
   //    typeid( * (DynamicType*) void_ptr );
   virtual ~DynamicType() {}
};
} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Write object to I/O buffer.
/// This function assumes that the value in 'obj' is the value stored in
/// a pointer to a "ptrClass". The actual type of the object pointed to
/// can be any class derived from "ptrClass".
/// Return:
///  - 0: failure
///  - 1: success
///  - 2: truncated success (i.e actual class is missing. Only ptrClass saved.)
///
/// If 'cacheReuse' is true (default) upon seeing an object address a second time,
/// we record the offset where its was written the first time rather than streaming
/// the object a second time.
/// If 'cacheReuse' is false, we always stream the object.  This allows the (re)use
/// of temporary object to store different data in the same buffer.

Int_t TBufferText::WriteObjectAny(const void *obj, const TClass *ptrClass, Bool_t cacheReuse /* = kTRUE */)
{
   if (!obj) {
      WriteObjectClass(nullptr, nullptr, kTRUE);
      return 1;
   }

   if (!ptrClass) {
      Error("WriteObjectAny", "ptrClass argument may not be 0");
      return 0;
   }

   TClass *clActual = ptrClass->GetActualClass(obj);

   if (!clActual) {
      // The ptrClass is a class with a virtual table and we have no
      // TClass with the actual type_info in memory.

      DynamicType *d_ptr = (DynamicType *)obj;
      Warning("WriteObjectAny",
              "An object of type %s (from type_info) passed through a %s pointer was truncated (due "
              "a missing dictionary)!!!",
              typeid(*d_ptr).name(), ptrClass->GetName());
      WriteObjectClass(obj, ptrClass, cacheReuse);
      return 2;
   } else if (clActual && (clActual != ptrClass)) {
      const char *temp = (const char *)obj;
      temp -= clActual->GetBaseClassOffset(ptrClass);
      WriteObjectClass(temp, clActual, cacheReuse);
      return 1;
   } else {
      WriteObjectClass(obj, ptrClass, cacheReuse);
      return 1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.
/// The collection needs to be a split TClonesArray or a split vector of pointers.

Int_t TBufferText::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *obj)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this, obj);
         (*iter)(*this, obj);
      }
   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, obj);
      }
   }
   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.
/// The collection needs to be a split TClonesArray or a split vector of pointers.

Int_t TBufferText::ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
                                       void *end_collection)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(
            *this, *(char **)start_collection); // Warning: This limits us to TClonesArray and vector of pointers.
         (*iter)(*this, start_collection, end_collection);
      }
   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, start_collection, end_collection);
      }
   }
   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.

Int_t TBufferText::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
                                 void *end_collection)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   TStreamerInfoActions::TLoopConfiguration *loopconfig = sequence.fLoopConfig;
   if (gDebug) {

      // Get the address of the first item for the PrintDebug.
      // (Performance is not essential here since we are going to print to
      // the screen anyway).
      void *arr0 = loopconfig->GetFirstAddress(start_collection, end_collection);
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this, arr0);
         (*iter)(*this, start_collection, end_collection, loopconfig);
      }
   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, start_collection, end_collection, loopconfig);
      }
   }
   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Function called by the Streamer functions to serialize object at p
/// to buffer b. The optional argument info may be specified to give an
/// alternative StreamerInfo instead of using the default StreamerInfo
/// automatically built from the class definition.
/// For more information, see class TStreamerInfo.

Int_t TBufferText::WriteClassBuffer(const TClass *cl, void *pointer)
{
   // build the StreamerInfo if first time for the class
   TStreamerInfo *sinfo = (TStreamerInfo *)const_cast<TClass *>(cl)->GetCurrentStreamerInfo();
   if (!sinfo) {
      // Have to be sure between the check and the taking of the lock if the current streamer has changed
      R__LOCKGUARD(gInterpreterMutex);
      sinfo = (TStreamerInfo *)const_cast<TClass *>(cl)->GetCurrentStreamerInfo();
      if (!sinfo) {
         const_cast<TClass *>(cl)->BuildRealData(pointer);
         sinfo = new TStreamerInfo(const_cast<TClass *>(cl));
         const_cast<TClass *>(cl)->SetCurrentStreamerInfo(sinfo);
         const_cast<TClass *>(cl)->RegisterStreamerInfo(sinfo);
         if (gDebug > 0)
            Info("WriteClassBuffer", "Creating StreamerInfo for class: %s, version: %d", cl->GetName(),
                 cl->GetClassVersion());
         sinfo->Build();
      }
   } else if (!sinfo->IsCompiled()) {
      R__LOCKGUARD(gInterpreterMutex);
      // Redo the test in case we have been victim of a data race on fIsCompiled.
      if (!sinfo->IsCompiled()) {
         const_cast<TClass *>(cl)->BuildRealData(pointer);
         sinfo->BuildOld();
      }
   }

   // write the class version number and reserve space for the byte count
   UInt_t R__c = WriteVersion(cl, kTRUE);

   // NOTE: In the future Philippe wants this to happen via a custom action
   TagStreamerInfo(sinfo);
   ApplySequence(*(sinfo->GetWriteTextActions()), (char *)pointer);

   // write the byte count at the start of the buffer
   SetByteCount(R__c, kTRUE);

   if (gDebug > 2)
      Info("WriteClassBuffer", "class: %s version %d has written %d bytes", cl->GetName(), cl->GetClassVersion(),
           UInt_t(fBufCur - fBuffer) - R__c - (UInt_t)sizeof(UInt_t));
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Deserialize information from a buffer into an object.
///
/// Note: This function is called by the xxx::Streamer() functions in
/// rootcint-generated dictionaries.
/// This function assumes that the class version and the byte count
/// information have been read.
///
/// \param[in] version The version number of the class
/// \param[in] start   The starting position in the buffer b
/// \param[in] count   The number of bytes for this object in the buffer
///

Int_t TBufferText::ReadClassBuffer(const TClass *cl, void *pointer, Int_t version, UInt_t start, UInt_t count,
                                   const TClass *onFileClass)
{

   //---------------------------------------------------------------------------
   // The ondisk class has been specified so get foreign streamer info
   /////////////////////////////////////////////////////////////////////////////

   TStreamerInfo *sinfo = nullptr;
   if (onFileClass) {
      sinfo = (TStreamerInfo *)cl->GetConversionStreamerInfo(onFileClass, version);
      if (!sinfo) {
         Error("ReadClassBuffer",
               "Could not find the right streamer info to convert %s version %d into a %s, object skipped at offset %d",
               onFileClass->GetName(), version, cl->GetName(), Length());
         CheckByteCount(start, count, onFileClass);
         return 0;
      }
   }
   //---------------------------------------------------------------------------
   // Get local streamer info
   /////////////////////////////////////////////////////////////////////////////
   /// The StreamerInfo should exist at this point.

   else {
      R__LOCKGUARD(gInterpreterMutex);
      auto infos = cl->GetStreamerInfos();
      auto ninfos = infos->GetSize();
      if (version < -1 || version >= ninfos) {
         Error("ReadBuffer1", "class: %s, attempting to access a wrong version: %d, object skipped at offset %d",
               cl->GetName(), version, Length());
         CheckByteCount(start, count, cl);
         return 0;
      }
      sinfo = (TStreamerInfo *)infos->At(version);
      if (!sinfo) {
         // Unless the data is coming via a socket connection from with schema evolution
         // (tracking) was not enabled.  So let's create the StreamerInfo if it is the
         // one for the current version, otherwise let's complain ...
         // We could also get here if there old class version was '1' and the new class version is higher than 1
         // AND the checksum is the same.
         if (version == cl->GetClassVersion() || version == 1) {
            const_cast<TClass *>(cl)->BuildRealData(pointer);
            // This creation is alright since we just checked within the
            // current 'locked' section.
            sinfo = new TStreamerInfo(const_cast<TClass *>(cl));
            const_cast<TClass *>(cl)->RegisterStreamerInfo(sinfo);
            if (gDebug > 0)
               Info("ReadClassBuffer", "Creating StreamerInfo for class: %s, version: %d", cl->GetName(), version);
            sinfo->Build();
         } else if (version == 0) {
            // When the object was written the class was version zero, so
            // there is no StreamerInfo to be found.
            // Check that the buffer position corresponds to the byte count.
            CheckByteCount(start, count, cl);
            return 0;
         } else {
            Error("ReadClassBuffer",
                  "Could not find the StreamerInfo for version %d of the class %s, object skipped at offset %d",
                  version, cl->GetName(), Length());
            CheckByteCount(start, count, cl);
            return 0;
         }
      } else if (!sinfo->IsCompiled()) { // Note this read is protected by the above lock.
         // Streamer info has not been compiled, but exists.
         // Therefore it was read in from a file and we have to do schema evolution.
         const_cast<TClass *>(cl)->BuildRealData(pointer);
         sinfo->BuildOld();
      }
   }

   // Deserialize the object.
   ApplySequence(*(sinfo->GetReadTextActions()), (char *)pointer);
   if (sinfo->IsRecovered())
      count = 0;

   // Check that the buffer position corresponds to the byte count.
   CheckByteCount(start, count, cl);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Deserialize information from a buffer into an object.
///
/// Note: This function is called by the xxx::Streamer()
/// functions in rootcint-generated dictionaries.
///

Int_t TBufferText::ReadClassBuffer(const TClass *cl, void *pointer, const TClass *onFileClass)
{
   // Read the class version from the buffer.
   UInt_t R__s = 0; // Start of object.
   UInt_t R__c = 0; // Count of bytes.
   Version_t version;

   if (onFileClass)
      version = ReadVersion(&R__s, &R__c, onFileClass);
   else
      version = ReadVersion(&R__s, &R__c, cl);

   Bool_t v2file = kFALSE;
   TFile *file = (TFile *)GetParent();
   if (file && file->GetVersion() < 30000) {
      version = -1; // This is old file
      v2file = kTRUE;
   }

   //---------------------------------------------------------------------------
   // The ondisk class has been specified so get foreign streamer info
   /////////////////////////////////////////////////////////////////////////////

   TStreamerInfo *sinfo = nullptr;
   if (onFileClass) {
      sinfo = (TStreamerInfo *)cl->GetConversionStreamerInfo(onFileClass, version);
      if (!sinfo) {
         Error("ReadClassBuffer",
               "Could not find the right streamer info to convert %s version %d into a %s, object skipped at offset %d",
               onFileClass->GetName(), version, cl->GetName(), Length());
         CheckByteCount(R__s, R__c, onFileClass);
         return 0;
      }
   }
   //---------------------------------------------------------------------------
   // Get local streamer info
   /////////////////////////////////////////////////////////////////////////////
   /// The StreamerInfo should exist at this point.

   else {
      TStreamerInfo *guess = (TStreamerInfo *)cl->GetLastReadInfo();
      if (guess && guess->GetClassVersion() == version) {
         sinfo = guess;
      } else {
         // The last one is not the one we are looking for.
         {
            R__LOCKGUARD(gInterpreterMutex);

            const TObjArray *infos = cl->GetStreamerInfos();
            Int_t infocapacity = infos->Capacity();
            if (infocapacity) {
               if (version < -1 || version >= infocapacity) {
                  Error("ReadClassBuffer",
                        "class: %s, attempting to access a wrong version: %d, object skipped at offset %d",
                        cl->GetName(), version, Length());
                  CheckByteCount(R__s, R__c, cl);
                  return 0;
               }
               sinfo = (TStreamerInfo *)infos->UncheckedAt(version);
               if (sinfo) {
                  if (!sinfo->IsCompiled()) {
                     // Streamer info has not been compiled, but exists.
                     // Therefore it was read in from a file and we have to do schema evolution?
                     R__LOCKGUARD(gInterpreterMutex);
                     const_cast<TClass *>(cl)->BuildRealData(pointer);
                     sinfo->BuildOld();
                  }
                  // If the compilation succeeded, remember this StreamerInfo.
                  // const_cast okay because of the lock on gInterpreterMutex.
                  if (sinfo->IsCompiled())
                     const_cast<TClass *>(cl)->SetLastReadInfo(sinfo);
               }
            }
         }

         if (!sinfo) {
            // Unless the data is coming via a socket connection from with schema evolution
            // (tracking) was not enabled.  So let's create the StreamerInfo if it is the
            // one for the current version, otherwise let's complain ...
            // We could also get here when reading a file prior to the introduction of StreamerInfo.
            // We could also get here if there old class version was '1' and the new class version is higher than 1
            // AND the checksum is the same.
            if (v2file || version == cl->GetClassVersion() || version == 1) {
               R__LOCKGUARD(gInterpreterMutex);

               // We need to check if another thread did not get here first
               // and did the StreamerInfo creation already.
               auto infos = cl->GetStreamerInfos();
               auto ninfos = infos->GetSize();
               if (!(version < -1 || version >= ninfos)) {
                  sinfo = (TStreamerInfo *)infos->At(version);
               }
               if (!sinfo) {
                  const_cast<TClass *>(cl)->BuildRealData(pointer);
                  sinfo = new TStreamerInfo(const_cast<TClass *>(cl));
                  sinfo->SetClassVersion(version);
                  const_cast<TClass *>(cl)->RegisterStreamerInfo(sinfo);
                  if (gDebug > 0)
                     Info("ReadClassBuffer", "Creating StreamerInfo for class: %s, version: %d", cl->GetName(),
                          version);
                  if (v2file) {
                     sinfo->Build();             // Get the elements.
                     sinfo->Clear("build");      // Undo compilation.
                     sinfo->BuildEmulated(file); // Fix the types and redo compilation.
                  } else {
                     sinfo->Build();
                  }
               }
            } else if (version == 0) {
               // When the object was written the class was version zero, so
               // there is no StreamerInfo to be found.
               // Check that the buffer position corresponds to the byte count.
               CheckByteCount(R__s, R__c, cl);
               return 0;
            } else {
               Error("ReadClassBuffer",
                     "Could not find the StreamerInfo for version %d of the class %s, object skipped at offset %d",
                     version, cl->GetName(), Length());
               CheckByteCount(R__s, R__c, cl);
               return 0;
            }
         }
      }
   }

   // deserialize the object
   ApplySequence(*(sinfo->GetReadTextActions()), (char *)pointer);
   if (sinfo->TStreamerInfo::IsRecovered())
      R__c = 0; // 'TStreamerInfo::' avoids going via a virtual function.

   // Check that the buffer position corresponds to the byte count.
   CheckByteCount(R__s, R__c, cl);

   if (gDebug > 2)
      Info("ReadClassBuffer", "for class: %s has read %d bytes", cl->GetName(), R__c);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// stream object to/from buffer

void TBufferText::StreamObject(void *obj, const std::type_info &typeinfo, const TClass * /* onFileClass */)
{
   StreamObject(obj, TClass::GetClass(typeinfo));
}

////////////////////////////////////////////////////////////////////////////////
/// stream object to/from buffer

void TBufferText::StreamObject(void *obj, const char *className, const TClass * /* onFileClass */)
{
   StreamObject(obj, TClass::GetClass(className));
}

void TBufferText::StreamObject(TObject *obj)
{
   // stream object to/from buffer

   StreamObject(obj, obj ? obj->IsA() : TObject::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// read a Float16_t from the buffer

void TBufferText::ReadFloat16(Float_t *f, TStreamerElement * /*ele*/)
{
   ReadFloat(*f);
}

////////////////////////////////////////////////////////////////////////////////
/// read a Double32_t from the buffer

void TBufferText::ReadDouble32(Double_t *d, TStreamerElement * /*ele*/)
{
   ReadDouble(*d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Double32_t from the buffer when the factor and minimun value have
/// been specified
/// see comments about Double32_t encoding at TBufferFile::WriteDouble32().
/// Currently TBufferText does not optimize space in this case.

void TBufferText::ReadWithFactor(Float_t *f, Double_t /* factor */, Double_t /* minvalue */)
{
   ReadFloat(*f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Float16_t from the buffer when the number of bits is specified
/// (explicitly or not)
/// see comments about Float16_t encoding at TBufferFile::WriteFloat16().
/// Currently TBufferText does not optimize space in this case.

void TBufferText::ReadWithNbits(Float_t *f, Int_t /* nbits */)
{
   ReadFloat(*f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Double32_t from the buffer when the factor and minimun value have
/// been specified
/// see comments about Double32_t encoding at TBufferFile::WriteDouble32().
/// Currently TBufferText does not optimize space in this case.

void TBufferText::ReadWithFactor(Double_t *d, Double_t /* factor */, Double_t /* minvalue */)
{
   ReadDouble(*d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Double32_t from the buffer when the number of bits is specified
/// (explicitly or not)
/// see comments about Double32_t encoding at TBufferFile::WriteDouble32().
/// Currently TBufferText does not optimize space in this case.

void TBufferText::ReadWithNbits(Double_t *d, Int_t /* nbits */)
{
   ReadDouble(*d);
}

////////////////////////////////////////////////////////////////////////////////
/// write a Float16_t to the buffer

void TBufferText::WriteFloat16(Float_t *f, TStreamerElement * /*ele*/)
{
   WriteFloat(*f);
}

////////////////////////////////////////////////////////////////////////////////
/// write a Double32_t to the buffer

void TBufferText::WriteDouble32(Double_t *d, TStreamerElement * /*ele*/)
{
   WriteDouble(*d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float16_t from buffer

Int_t TBufferText::ReadArrayFloat16(Float_t *&f, TStreamerElement * /*ele*/)
{
   return ReadArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double32_t from buffer

Int_t TBufferText::ReadArrayDouble32(Double_t *&d, TStreamerElement * /*ele*/)
{
   return ReadArray(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float16_t from buffer

Int_t TBufferText::ReadStaticArrayFloat16(Float_t *f, TStreamerElement * /*ele*/)
{
   return ReadStaticArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double32_t from buffer

Int_t TBufferText::ReadStaticArrayDouble32(Double_t *d, TStreamerElement * /*ele*/)
{
   return ReadStaticArray(d);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float16_t from buffer

void TBufferText::ReadFastArrayFloat16(Float_t *f, Int_t n, TStreamerElement * /*ele*/)
{
   ReadFastArray(f, n);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float16_t from buffer

void TBufferText::ReadFastArrayWithFactor(Float_t *f, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   ReadFastArray(f, n);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float16_t from buffer

void TBufferText::ReadFastArrayWithNbits(Float_t *f, Int_t n, Int_t /*nbits*/)
{
   ReadFastArray(f, n);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double32_t from buffer

void TBufferText::ReadFastArrayDouble32(Double_t *d, Int_t n, TStreamerElement * /*ele*/)
{
   ReadFastArray(d, n);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double32_t from buffer

void TBufferText::ReadFastArrayWithFactor(Double_t *d, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   ReadFastArray(d, n);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double32_t from buffer

void TBufferText::ReadFastArrayWithNbits(Double_t *d, Int_t n, Int_t /*nbits*/)
{
   ReadFastArray(d, n);
}
////////////////////////////////////////////////////////////////////////////////
/// Write array of Float16_t to buffer

void TBufferText::WriteArrayFloat16(const Float_t *f, Int_t n, TStreamerElement * /*ele*/)
{
   WriteArray(f, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double32_t to buffer

void TBufferText::WriteArrayDouble32(const Double_t *d, Int_t n, TStreamerElement * /*ele*/)
{
   WriteArray(d, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float16_t to buffer

void TBufferText::WriteFastArrayFloat16(const Float_t *f, Int_t n, TStreamerElement * /*ele*/)
{
   WriteFastArray(f, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double32_t to buffer

void TBufferText::WriteFastArrayDouble32(const Double_t *d, Int_t n, TStreamerElement * /*ele*/)
{
   WriteFastArray(d, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Skip class version from I/O buffer.

void TBufferText::SkipVersion(const TClass *cl)
{
   ReadVersion(nullptr, nullptr, cl);
}

////////////////////////////////////////////////////////////////////////////////
/// Write data of base class.

void TBufferText::WriteBaseClass(void *start, TStreamerBase *elem)
{
   elem->WriteBuffer(*this, (char *)start);
}

////////////////////////////////////////////////////////////////////////////////
/// Read data of base class.

void TBufferText::ReadBaseClass(void *start, TStreamerBase *elem)
{
   elem->ReadBuffer(*this, (char *)start);
}

////////////////////////////////////////////////////////////////////////////////
/// method compress float string, excluding exp and/or move float point
///  - 1.000000e-01 -> 0.1
///  - 3.750000e+00 -> 3.75
///  - 3.750000e-03 -> 0.00375
///  - 3.750000e-04 -> 3.75e-4
///  - 1.100000e-10 -> 1.1e-10

void TBufferText::CompactFloatString(char *sbuf, unsigned len)
{
   char *pnt = 0, *exp = 0, *lastdecimal = 0, *s = sbuf;
   bool negative_exp = false;
   int power = 0;
   while (*s && --len) {
      switch (*s) {
      case '.': pnt = s; break;
      case 'E':
      case 'e': exp = s; break;
      case '-':
         if (exp)
            negative_exp = true;
         break;
      case '+': break;
      default: // should be digits from '0' to '9'
         if ((*s < '0') || (*s > '9'))
            return;
         if (exp)
            power = power * 10 + (*s - '0');
         else if (pnt && *s != '0')
            lastdecimal = s;
         break;
      }
      ++s;
   }
   if (*s)
      return; // if end-of-string was not found

   if (!exp) {
      // value without exponent like 123.4569000
      if (pnt) {
         if (lastdecimal)
            *(lastdecimal + 1) = 0;
         else
            *pnt = 0;
      }
   } else if (power == 0) {
      if (lastdecimal)
         *(lastdecimal + 1) = 0;
      else if (pnt)
         *pnt = 0;
   } else if (!negative_exp && pnt && exp && (exp - pnt > power)) {
      // this is case of value 1.23000e+02
      // we can move point and exclude exponent easily
      for (int cnt = 0; cnt < power; ++cnt) {
         char tmp = *pnt;
         *pnt = *(pnt + 1);
         *(++pnt) = tmp;
      }
      if (lastdecimal && (pnt < lastdecimal))
         *(lastdecimal + 1) = 0;
      else
         *pnt = 0;
   } else if (negative_exp && pnt && exp && (power < (s - exp))) {
      // this is small negative exponent like 1.2300e-02
      if (!lastdecimal)
         lastdecimal = pnt;
      *(lastdecimal + 1) = 0;
      // copy most significant digit on the point place
      *pnt = *(pnt - 1);

      for (char *pos = lastdecimal + 1; pos >= pnt; --pos)
         *(pos + power) = *pos;
      *(pnt - 1) = '0';
      *pnt = '.';
      for (int cnt = 1; cnt < power; ++cnt)
         *(pnt + cnt) = '0';
   } else if (pnt && exp) {
      // keep exponent, but non-significant zeros
      if (lastdecimal)
         pnt = lastdecimal + 1;
      // copy exponent sign
      *pnt++ = *exp++;
      if (*exp == '+')
         ++exp;
      else if (*exp == '-')
         *pnt++ = *exp++;
      // exclude zeros in the begin of exponent
      while (*exp == '0')
         ++exp;
      while (*exp)
         *pnt++ = *exp++;
      *pnt = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set printf format for float/double members, default "%e"
/// to change format only for doubles, use SetDoubleFormat

void TBufferText::SetFloatFormat(const char *fmt)
{
   if (!fmt)
      fmt = "%e";
   fgFloatFmt = fmt;
   fgDoubleFmt = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// return current printf format for float members, default "%e"

const char *TBufferText::GetFloatFormat()
{
   return fgFloatFmt;
}

////////////////////////////////////////////////////////////////////////////////
/// set printf format for double members, default "%.14e"
/// use it after SetFloatFormat, which also overwrites format for doubles

void TBufferText::SetDoubleFormat(const char *fmt)
{
   if (!fmt)
      fmt = "%.14e";
   fgDoubleFmt = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// return current printf format for double members, default "%.14e"

const char *TBufferText::GetDoubleFormat()
{
   return fgDoubleFmt;
}

////////////////////////////////////////////////////////////////////////////////
/// convert float to string with configured format

const char *TBufferText::ConvertFloat(Float_t value, char *buf, unsigned len, Bool_t not_optimize)
{
   if (not_optimize) {
      snprintf(buf, len, fgFloatFmt, value);
   } else if ((value == std::nearbyint(value)) && (std::abs(value) < 1e15)) {
      snprintf(buf, len, "%1.0f", value);
   } else {
      snprintf(buf, len, fgFloatFmt, value);
      CompactFloatString(buf, len);
   }
   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// convert float to string with configured format

const char *TBufferText::ConvertDouble(Double_t value, char *buf, unsigned len, Bool_t not_optimize)
{
   if (not_optimize) {
      snprintf(buf, len, fgFloatFmt, value);
   } else if ((value == std::nearbyint(value)) && (std::abs(value) < 1e25)) {
      snprintf(buf, len, "%1.0f", value);
   } else {
      snprintf(buf, len, fgDoubleFmt, value);
      CompactFloatString(buf, len);
   }
   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the initial size of the map used to store object and class
/// references during reading.
///
/// The default size is kMapSize.
/// Increasing the default has the benefit that when reading many
/// small objects the array does not need to be resized too often
/// (the system is always dynamic, even with the default everything
/// will work, only the initial resizing will cost some time).
/// Per TBuffer object this option can be changed using SetReadParam().

void TBufferText::SetGlobalReadParam(Int_t mapsize)
{
   fgMapSize = mapsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the initial size of the map used to store object and class
/// references during reading.
///
/// The default size is kMapSize.
/// Increasing the default has the benefit that when reading many
/// small objects the array does not need to be resized too often
/// (the system is always dynamic, even with the default everything
/// will work, only the initial resizing will cost some time).
/// Per TBuffer object this option can be changed using SetReadParam().

void TBufferText::SetGlobalWriteParam(Int_t mapsize)
{
   fgMapSize = mapsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Get default read map size.

Int_t TBufferText::GetGlobalReadParam()
{
   return fgMapSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Get default write map size.

Int_t TBufferText::GetGlobalWriteParam()
{
   return fgMapSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the fMap container and initialize them
/// with the null object.

void TBufferText::InitMap()
{
   if (IsWriting()) {
      if (!fMap) {
         fMap = new TExMap(fMapSize);
         // No need to keep track of the class in write mode
         // fClassMap = new TExMap(fMapSize);
         fMapCount = 0;
      }
   } else {
      if (!fMap) {
         fMap = new TExMap(fMapSize);
         fMap->Add(0, kNullTag); // put kNullTag in slot 0
         fMapCount = 1;
      } else if (fMapCount == 0) {
         fMap->Add(0, kNullTag); // put kNullTag in slot 0
         fMapCount = 1;
      }
      if (!fClassMap) {
         fClassMap = new TExMap(fMapSize);
         fClassMap->Add(0, kNullTag); // put kNullTag in slot 0
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete existing fMap and reset map counter.

void TBufferText::ResetMap()
{
   if (fMap)
      fMap->Delete();
   if (fClassMap)
      fClassMap->Delete();
   fMapCount = 0;
   fDisplacement = 0;

   // reset user bits
   // ResetBit(kUser1);
   // ResetBit(kUser2);
   // ResetBit(kUser3);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the initial size of the map used to store object and class
/// references during reading. The default size is TBufferFile::kMapSize.
/// Increasing the default has the benefit that when reading many
/// small objects the map does not need to be resized too often
/// (the system is always dynamic, even with the default everything
/// will work, only the initial resizing will cost some time).
/// This method can only be called directly after the creation of
/// the TBuffer, before any reading is done. Globally this option
/// can be changed using SetGlobalReadParam().

void TBufferText::SetReadParam(Int_t mapsize)
{
   R__ASSERT(IsReading());
   R__ASSERT(fMap == 0);

   fMapSize = mapsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the initial size of the hashtable used to store object and class
/// references during writing. The default size is TBufferFile::kMapSize.
/// Increasing the default has the benefit that when writing many
/// small objects the hashtable does not get too many collisions
/// (the system is always dynamic, even with the default everything
/// will work, only a large number of collisions will cost performance).
/// For optimal performance hashsize should always be a prime.
/// This method can only be called directly after the creation of
/// the TBuffer, before any writing is done. Globally this option
/// can be changed using SetGlobalWriteParam().

void TBufferText::SetWriteParam(Int_t mapsize)
{
   R__ASSERT(IsWriting());
   R__ASSERT(fMap == 0);

   fMapSize = mapsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Add object to the fMap container.
///
/// If obj is not 0 add object to the map (in read mode also add 0 objects to
/// the map). This method may only be called outside this class just before
/// calling obj->Streamer() to prevent self reference of obj, in case obj
/// contains (via via) a pointer to itself. In that case offset must be 1
/// (default value for offset).

void TBufferText::MapObject(const TObject *obj, UInt_t offset)
{
   if (IsWriting()) {
      if (!fMap)
         InitMap();

      if (obj) {
         CheckCount(offset);
         ULong_t hash = Void_Hash(obj);
         fMap->Add(hash, (Long_t)obj, offset);
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Long_t)obj, (Long_t)((TObject*)obj)->IsA());
         fMapCount++;
      }
   } else {
      if (!fMap || !fClassMap)
         InitMap();

      fMap->Add(offset, (Long_t)obj);
      fClassMap->Add(offset, (obj && obj != (TObject *)-1) ? (Long_t)((TObject *)obj)->IsA() : 0);
      fMapCount++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add object to the fMap container.
///
/// If obj is not 0 add object to the map (in read mode also add 0 objects to
/// the map). This method may only be called outside this class just before
/// calling obj->Streamer() to prevent self reference of obj, in case obj
/// contains (via via) a pointer to itself. In that case offset must be 1
/// (default value for offset).

void TBufferText::MapObject(const void *obj, const TClass *cl, UInt_t offset)
{
   if (IsWriting()) {
      if (!fMap)
         InitMap();

      if (obj) {
         CheckCount(offset);
         ULong_t hash = Void_Hash(obj);
         fMap->Add(hash, (Long_t)obj, offset);
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Long_t)obj, (Long_t)cl);
         fMapCount++;
      }
   } else {
      if (!fMap || !fClassMap)
         InitMap();

      fMap->Add(offset, (Long_t)obj);
      fClassMap->Add(offset, (Long_t)cl);
      fMapCount++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the object stored in the buffer's object map at 'tag'
/// Set ptr and ClassPtr respectively to the address of the object and
/// a pointer to its TClass.

void TBufferText::GetMappedObject(UInt_t tag, void *&ptr, TClass *&ClassPtr) const
{
   // original code in TBufferFile is wrong, fMap->GetSize() is just number of entries, cannot be used for tag checks

   //  if (tag > (UInt_t)fMap->GetSize()) {
   //     ptr = nullptr;
   //     ClassPtr = nullptr;
   //   } else {
   ptr = (void *)(Long_t)fMap->GetValue(tag);
   ClassPtr = (TClass *)(Long_t)fClassMap->GetValue(tag);
   //  }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the specified object of the specified class is already in
/// the buffer. Returns kTRUE if object already in the buffer,
/// kFALSE otherwise (also if obj is 0 ).

Bool_t TBufferText::CheckObject(const TObject *obj)
{
   return CheckObject(obj, TObject::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the specified object of the specified class is already in
/// the buffer. Returns kTRUE if object already in the buffer,
/// kFALSE otherwise (also if obj is 0 ).

Bool_t TBufferText::CheckObject(const void *obj, const TClass *ptrClass)
{
   if (!obj || !fMap || !ptrClass)
      return kFALSE;

   TClass *clActual = ptrClass->GetActualClass(obj);

   Long64_t idx;

   if (clActual && (ptrClass != clActual)) {
      const char *temp = (const char *)obj;
      temp -= clActual->GetBaseClassOffset(ptrClass);
      idx = GetMapEntry(temp);
   } else {
      idx = GetMapEntry(obj);
   }

   return idx == 0 ? kFALSE : kTRUE;
}

////////////////////////////////////////////////////////////////////////////////////
/// Returns object map entry for specified object

Long64_t TBufferText::GetMapEntry(const void *obj)
{
   if (!obj || !fMap)
      return 0;

   return fMap->GetValue(Void_Hash(obj), (Long_t)obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the version number of the owner file.

Int_t TBufferText::GetVersionOwner() const
{
   TFile *file = (TFile *)GetParent();
   if (file)
      return file->GetVersion();
   else
      return 0;
}
