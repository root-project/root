// @(#)root/io:$Id$
// Author: Sergey Linev  21/02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferIO
#define ROOT_TBufferIO

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferIO                                                            //
//                                                                      //
// Direct subclass of TBuffer, implements common methods for            //
// TBufferFile and TBufferText classes                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBuffer.h"

class TExMap;

class TBufferIO : public TBuffer {

protected:
   Int_t fMapCount{0};         ///< Number of objects or classes in map
   Int_t fMapSize{0};          ///< Default size of map
   Int_t fDisplacement{0};     ///< Value to be added to the map offsets
   UShort_t fPidOffset{0};     ///< Offset to be added to the pid index in this key/buffer.
   TExMap *fMap{nullptr};      ///< Map containing object,offset pairs for reading/writing
   TExMap *fClassMap{nullptr}; ///< Map containing object,class pairs for reading

   static Int_t fgMapSize; ///< Default map size for all TBuffer objects

   TBufferIO() = default;

   TBufferIO(TBuffer::EMode mode);
   TBufferIO(TBuffer::EMode mode, Int_t bufsiz);
   TBufferIO(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE,
             ReAllocCharFun_t reallocfunc = nullptr);

public:
   enum { kMapSize = 503 };

   virtual ~TBufferIO();

   ClassDef(TBufferIO, 0) // base class, share methods for TBufferFile and TBufferText
};

#endif
