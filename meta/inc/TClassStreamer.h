// @(#)root/base:$Name:  $:$Id: TClassStreamer.h,Exp $
// Author: Victor Perev and Philippe Canal   08/05/02

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClassStreamer_h
#define ROOT_TClassStreamer_h

#include "Rtypes.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassStreamer is used to stream an object of a specific class.      //
//                                                                      //
// The address passed to operator() will be the address of the start    //
// of the  object.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TClassStreamer {
protected:
   TClassStreamer() : fStreamer(0) {};

public:
   TClassStreamer(ClassStreamerFunc_t pointer) : fStreamer(pointer) {};
   TClassStreamer(const TClassStreamer &rhs) : fStreamer(rhs.fStreamer) {};

   virtual  ~TClassStreamer(){};   
   virtual void operator()(TBuffer &b, void *objp)
   {
      // The address passed to operator() will be the address of the start of the
      // object.
      
      (*fStreamer)(b,objp);
   }
   
private:
   ClassStreamerFunc_t fStreamer; 
};

#endif
