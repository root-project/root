// @(#)root/cont:$Name:  $:$Id: TRef.h,v 1.4 2001/11/28 14:49:01 brun Exp $
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
class TExec;
class TObjArray;

class TRef : public TObject {

protected:
   TProcessID       *fPID;     //!Pointer to ProcessID when TRef was written

   static TObjArray  *fgExecs;  //List of execs
   static TObject    *fgObject; //Pointer to object (set in Action on Demand)
      
public:
   //status bits
   enum { kNotComputed = BIT(12)};

   TRef() {fPID = 0;}
   TRef(TObject *obj);
   TRef(const TRef &ref);
   void operator=(TObject *obj);
   TRef& operator=(const TRef &ref);
   virtual ~TRef() {;}
   static Int_t       AddExec(const char *name);
          TObject    *GetObject() const;
   static TObjArray  *GetListOfExecs();
   virtual void       SetAction(const char *name);
   virtual void       SetAction(TObject *parent);
   static  void       SetObject(TObject *obj);

   ClassDef(TRef,1)  //Persistent Reference link to a TObject
};

#endif
