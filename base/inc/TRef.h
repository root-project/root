// @(#)root/cont:$Name:  $:$Id: TRef.h,v 1.2 2001/10/05 16:25:53 brun Exp $
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRef
#define ROOT_TRef


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRef                                                                 //
//                                                                      //
// Persistent Reference link to a TObject                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TProcessID;
class TFile;

class TRef : public TObject {

protected:
   TProcessID     *fPID;   //!Pointer to ProcessID when TRef was written

public:
   TRef() {fPID = 0;}
   TRef(TObject *obj);
   TRef(const TRef &ref);
   void operator=(TObject *obj);
   virtual ~TRef() {;}
         TObject *GetObject() const;
   
   static  void   ReadRef(TObject *obj, TBuffer &R__b, TFile *file);
   static  void   SaveRef(TObject *obj, TBuffer &R__b, TFile *file);

   ClassDef(TRef,1)  //Persistent Reference link to a TObject
};

#endif
