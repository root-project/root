// @(#)root/cont:$Name:  $:$Id: TObjectRef.h,v 1.4 2001/09/28 07:54:00 brun Exp $
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjectRef
#define ROOT_TObjectRef


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjectRef                                                           //
//                                                                      //
// Persistent Reference link to a TObject                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TProcessID;
class TFile;

class TObjectRef : public TObject {

protected:
   TProcessID     *fPID;   //!Pointer to ProcessID when TObjectRef was written

public:
   TObjectRef() {fPID = 0;}
   TObjectRef(TObject *obj);
   TObjectRef(const TObjectRef &ref);
   void operator=(TObject *obj);
   virtual ~TObjectRef() {;}
           TObject *GetObject(); //or better by simply dereferencing operator ->

   static  void   ReadRef(TObject *obj, TBuffer &R__b, TFile *file);
   static  void   SaveRef(TObject *obj, TBuffer &R__b, TFile *file);

   ClassDef(TObjectRef,1)  //Persistent Reference link to a TObject
};

#endif
