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
