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

class TBufferIO : public TBuffer {

protected:
   TBufferIO() = default;

   TBufferIO(TBuffer::EMode mode) : TBuffer(mode) {}
   TBufferIO(TBuffer::EMode mode, Int_t bufsiz) : TBuffer(mode, bufsiz) {}
   TBufferIO(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE, ReAllocCharFun_t reallocfunc = 0)
      : TBuffer(mode, bufsiz, buf, adopt, reallocfunc)
   {
   }

public:
   virtual ~TBufferIO();

   ClassDef(TBufferIO, 0) // base class, share methods for TBufferFile and TBufferText
};

#endif
