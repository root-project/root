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
#include "TStreamerInfoActions.h"

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TBufferText::TBufferText() : TBuffer(), fPidOffset(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor

TBufferText::TBufferText(TBuffer::EMode mode, TObject *parent) : TBuffer(mode), fPidOffset(0)
{
   fBufSize = 1000000000;

   SetParent(parent);
   SetBit(kCannotHandleMemberWiseStreaming);
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
}

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
      Warning("WriteObjectAny", "An object of type %s (from type_info) passed through a %s pointer was truncated (due "
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
