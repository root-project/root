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
