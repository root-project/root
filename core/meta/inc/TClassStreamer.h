// @(#)root/base:$Id$
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
#include "TClassRef.h"

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
   TClassStreamer(ClassStreamerFunc_t pointer) : fStreamer(pointer), fOnFileClass() {};
   TClassStreamer(const TClassStreamer &rhs) : fStreamer(rhs.fStreamer), fOnFileClass() {};

   virtual void SetOnFileClass( const TClass* cl ) { fOnFileClass = const_cast<TClass*>(cl); }
   virtual const TClass* GetOnFileClass() const { return fOnFileClass; }

   virtual TClassStreamer *Generate() {
      // Virtual copy constructor.
      return new TClassStreamer(*this); 
   }

   virtual  ~TClassStreamer(){};   
   virtual void operator()(TBuffer &b, void *objp)
   {
      // The address passed to operator() will be the address of the start of the
      // object.
      
      (*fStreamer)(b,objp);
   }
   
private:
   ClassStreamerFunc_t fStreamer;
protected:
   TClassRef           fOnFileClass;
};

#endif
