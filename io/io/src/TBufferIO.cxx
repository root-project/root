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
#include "TError.h"


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

TBufferIO::TBufferIO(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt,
                     ReAllocCharFun_t reallocfunc)
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
         fMap->Add(0, kNullTag);      // put kNullTag in slot 0
         fMapCount = 1;
      } else if (fMapCount==0) {
         fMap->Add(0, kNullTag);      // put kNullTag in slot 0
         fMapCount = 1;
      }
      if (!fClassMap) {
         fClassMap = new TExMap(fMapSize);
         fClassMap->Add(0, kNullTag);      // put kNullTag in slot 0
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete existing fMap and reset map counter.

void TBufferIO::ResetMap()
{
   if (fMap) fMap->Delete();
   if (fClassMap) fClassMap->Delete();
   fMapCount     = 0;
   fDisplacement = 0;

   // reset user bits
   ResetBit(kUser1);
   ResetBit(kUser2);
   ResetBit(kUser3);
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
