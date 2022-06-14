// @(#)root/io:$Id$
// Author: Sergey Linev 21/02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\file TBufferIO.cxx
\class TBufferIO
\ingroup IO

Direct subclass of TBuffer, implements common methods for TBufferFile and TBufferText classes
*/

#include "TBufferIO.h"

#include "TExMap.h"
#include "TClass.h"
#include "TFile.h"
#include "TError.h"
#include "TClonesArray.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TArrayC.h"
#include "TRefTable.h"
#include "TProcessID.h"
#include "TVirtualMutex.h"
#include "TInterpreter.h"
#include "TROOT.h"

Int_t TBufferIO::fgMapSize = kMapSize;

ClassImp(TBufferIO);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TBufferIO::TBufferIO(TBuffer::EMode mode) : TBuffer(mode)
{
   fMapSize = fgMapSize;
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TBufferIO::TBufferIO(TBuffer::EMode mode, Int_t bufsiz) : TBuffer(mode, bufsiz)
{
   fMapSize = fgMapSize;
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TBufferIO::TBufferIO(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt, ReAllocCharFun_t reallocfunc)
   : TBuffer(mode, bufsiz, buf, adopt, reallocfunc)
{
   fMapSize = fgMapSize;
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TBufferIO::~TBufferIO()
{
   delete fMap;
   delete fClassMap;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the version number of the owner file.

Int_t TBufferIO::GetVersionOwner() const
{
   TFile *file = (TFile *)GetParent();
   if (file)
      return file->GetVersion();
   else
      return 0;
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

void TBufferIO::SetReadParam(Int_t mapsize)
{
   R__ASSERT(IsReading());
   R__ASSERT(fMap == nullptr);

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

void TBufferIO::SetWriteParam(Int_t mapsize)
{
   R__ASSERT(IsWriting());
   R__ASSERT(fMap == nullptr);

   fMapSize = mapsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the fMap container and initialize them
/// with the null object.

void TBufferIO::InitMap()
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
/// Add object to the fMap container.
///
/// If obj is not 0 add object to the map (in read mode also add 0 objects to
/// the map). This method may only be called outside this class just before
/// calling obj->Streamer() to prevent self reference of obj, in case obj
/// contains (via via) a pointer to itself. In that case offset must be 1
/// (default value for offset).

void TBufferIO::MapObject(const TObject *obj, UInt_t offset)
{
   if (IsWriting()) {
      if (!fMap)
         InitMap();

      if (obj) {
         CheckCount(offset);
         ULong_t hash = Void_Hash(obj);
         fMap->Add(hash, (Longptr_t)obj, offset);
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Longptr_t)obj, (Longptr_t)((TObject*)obj)->IsA());
         fMapCount++;
      }
   } else {
      if (!fMap || !fClassMap)
         InitMap();

      fMap->Add(offset, (Longptr_t)obj);
      fClassMap->Add(offset, (obj && obj != (TObject *)-1) ? (Longptr_t)((TObject *)obj)->IsA() : 0);
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

void TBufferIO::MapObject(const void *obj, const TClass *cl, UInt_t offset)
{
   if (IsWriting()) {
      if (!fMap)
         InitMap();

      if (obj) {
         CheckCount(offset);
         ULong_t hash = Void_Hash(obj);
         fMap->Add(hash, (Longptr_t)obj, offset);
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Longptr_t)obj, (Longptr_t)cl);
         fMapCount++;
      }
   } else {
      if (!fMap || !fClassMap)
         InitMap();

      fMap->Add(offset, (Longptr_t)obj);
      fClassMap->Add(offset, (Longptr_t)cl);
      fMapCount++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the specified object is already in the buffer.
/// Returns kTRUE if object already in the buffer, kFALSE otherwise
/// (also if obj is 0 or TBuffer not in writing mode).

Bool_t TBufferIO::CheckObject(const TObject *obj)
{
   return CheckObject(obj, TObject::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the specified object of the specified class is already in
/// the buffer. Returns kTRUE if object already in the buffer,
/// kFALSE otherwise (also if obj is 0 ).

Bool_t TBufferIO::CheckObject(const void *obj, const TClass *ptrClass)
{
   if (!obj || !fMap || !ptrClass)
      return kFALSE;

   TClass *clActual = ptrClass->GetActualClass(obj);

   ULongptr_t idx;

   if (clActual && (ptrClass != clActual)) {
      const char *temp = (const char *)obj;
      temp -= clActual->GetBaseClassOffset(ptrClass);
      idx = (ULongptr_t)fMap->GetValue(Void_Hash(temp), (Longptr_t)temp);
   } else {
      idx = (ULongptr_t)fMap->GetValue(Void_Hash(obj), (Longptr_t)obj);
   }

   return idx ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the object stored in the buffer's object map at 'tag'
/// Set ptr and ClassPtr respectively to the address of the object and
/// a pointer to its TClass.

void TBufferIO::GetMappedObject(UInt_t tag, void *&ptr, TClass *&ClassPtr) const
{
   // original code in TBufferFile is wrong, fMap->GetSize() is just number of entries, cannot be used for tag checks

   //  if (tag > (UInt_t)fMap->GetSize()) {
   //     ptr = nullptr;
   //     ClassPtr = nullptr;
   //   } else {
   ptr = (void *)(Longptr_t)fMap->GetValue(tag);
   ClassPtr = (TClass *)(Longptr_t)fClassMap->GetValue(tag);
   //  }
}

////////////////////////////////////////////////////////////////////////////////////
/// Returns tag for specified object from objects map (if exists)
/// Returns 0 if object not included into objects map

Long64_t TBufferIO::GetObjectTag(const void *obj)
{
   if (!obj || !fMap)
      return 0;

   return fMap->GetValue(Void_Hash(obj), (Longptr_t)obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete existing fMap and reset map counter.

void TBufferIO::ResetMap()
{
   if (fMap)
      fMap->Delete();
   if (fClassMap)
      fClassMap->Delete();
   fMapCount = 0;
   fDisplacement = 0;

   // reset user bits
   ResetBit(kUser1);
   ResetBit(kUser2);
   ResetBit(kUser3);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset buffer object. Resets map and buffer offset
void TBufferIO::Reset()
{
   SetBufferOffset();
   ResetMap();
}

////////////////////////////////////////////////////////////////////////////////
/// This offset is used when a key (or basket) is transfered from one
/// file to the other.  In this case the TRef and TObject might have stored a
/// pid index (to retrieve TProcessIDs) which referred to their order on the original
/// file, the fPidOffset is to be added to those values to correctly find the
/// TProcessID.  This fPidOffset needs to be increment if the key/basket is copied
/// and need to be zero for new key/basket.

void TBufferIO::SetPidOffset(UShort_t offset)
{
   fPidOffset = offset;
}

//---- Utilities for TStreamerInfo ----------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// force writing the TStreamerInfo to the file

void TBufferIO::ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force)
{
   if (info)
      info->ForceWriteInfo((TFile *)GetParent(), force);
}

////////////////////////////////////////////////////////////////////////////////
/// Make sure TStreamerInfo is not optimized, otherwise it will not be
/// possible to support schema evolution in read mode.
/// In case the StreamerInfo has already been computed and optimized,
/// one must disable the option BypassStreamer.

void TBufferIO::ForceWriteInfoClones(TClonesArray *a)
{
   TStreamerInfo *sinfo = (TStreamerInfo *)a->GetClass()->GetStreamerInfo();
   ForceWriteInfo(sinfo, kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Mark the classindex of the current file as using this TStreamerInfo

void TBufferIO::TagStreamerInfo(TVirtualStreamerInfo *info)
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
/// Interface to TStreamerInfo::ReadBufferClones.

Int_t TBufferIO::ReadClones(TClonesArray *a, Int_t nobjects, Version_t objvers)
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

Int_t TBufferIO::WriteClones(TClonesArray *a, Int_t nobjects)
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
/// Return the last TProcessID in the file.

TProcessID *TBufferIO::GetLastProcessID(TRefTable *reftable) const
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
/// The TProcessID with number pidf is read from file.
/// If the object is not already entered in the gROOT list, it is added.

TProcessID *TBufferIO::ReadProcessID(UShort_t pidf)
{
   TFile *file = (TFile *)GetParent();
   if (!file) {
      if (!pidf)
         return TProcessID::GetPID(); // may happen when cloning an object
      return nullptr;
   }

   TProcessID *pid = nullptr;
   {
      R__LOCKGUARD_IMT(gInterpreterMutex); // Lock for parallel TTree I/O
      pid = file->ReadProcessID(pidf);
   }

   return pid;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the exec id stored in the current TStreamerInfo element.
/// The execid has been saved in the unique id of the TStreamerElement
/// being read by TStreamerElement::Streamer.
/// The current element (fgElement) is set as a static global
/// by TStreamerInfo::ReadBuffer (Clones) when reading this TRef.

UInt_t TBufferIO::GetTRefExecId()
{
   return TStreamerInfo::GetCurrentElement()->GetUniqueID();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the ProcessID pid is already in the file.
/// If not, add it and return the index number in the local file list.

UShort_t TBufferIO::WriteProcessID(TProcessID *pid)
{
   TFile *file = (TFile *)GetParent();
   if (!file)
      return 0;
   return file->WriteProcessID(pid);
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
///
/// This function assumes that the value in 'obj' is the value stored in
/// a pointer to a "ptrClass". The actual type of the object pointed to
/// can be any class derived from "ptrClass".
/// Return:
///   - 0: failure
///   - 1: success
///   - 2: truncated success (i.e actual class is missing. Only ptrClass saved.)
///
/// If 'cacheReuse' is true (default) upon seeing an object address a second time,
/// we record the offset where its was written the first time rather than streaming
/// the object a second time.
/// If 'cacheReuse' is false, we always stream the object.  This allows the (re)use
/// of temporary object to store different data in the same buffer.

Int_t TBufferIO::WriteObjectAny(const void *obj, const TClass *ptrClass, Bool_t cacheReuse /* = kTRUE */)
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

   if (clActual == 0 || clActual->GetState() == TClass::kForwardDeclared) {
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
/// Write object to I/O buffer.

void TBufferIO::WriteObject(const TObject *obj, Bool_t cacheReuse)
{
   WriteObjectAny(obj, TObject::Class(), cacheReuse);
}

//---- Static functions --------------------------------------------------------

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

void TBufferIO::SetGlobalReadParam(Int_t mapsize)
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

void TBufferIO::SetGlobalWriteParam(Int_t mapsize)
{
   fgMapSize = mapsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Get default read map size.

Int_t TBufferIO::GetGlobalReadParam()
{
   return fgMapSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Get default write map size.

Int_t TBufferIO::GetGlobalWriteParam()
{
   return fgMapSize;
}
