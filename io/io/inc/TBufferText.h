// $Id$
// Author: Sergey Linev  21.12.2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferText
#define ROOT_TBufferText

#include "TBuffer.h"

class TBufferText : public TBuffer {

protected:
   TBufferText();
   TBufferText(TBuffer::EMode mode, TObject *parent = nullptr);

public:
   // virtual abstract TBuffer methods, which could be redefined here

   virtual TProcessID *GetLastProcessID(TRefTable *reftable) const;
   virtual UInt_t GetTRefExecId();
   virtual TProcessID *ReadProcessID(UShort_t pidf);
   virtual UShort_t WriteProcessID(TProcessID *pid);

   ClassDef(TBufferText, 0); // a TBuffer subclass for all text-based streamers
};

#endif
