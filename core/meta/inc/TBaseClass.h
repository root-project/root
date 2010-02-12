// @(#)root/meta:$Id$
// Author: Fons Rademakers   08/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBaseClass
#define ROOT_TBaseClass


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBaseClass                                                           //
//                                                                      //
// Description of a base class.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif
#ifndef ROOT_TClassRef
#include "TClassRef.h"
#endif

class TBrowser;
class TClass;

class TBaseClass : public TDictionary {

private:
   TBaseClass(const TBaseClass &);          // Not implemented
   TBaseClass&operator=(const TBaseClass&); // Not implemented
   
private:
   BaseClassInfo_t   *fInfo;      //pointer to CINT base class info
   TClassRef          fClassPtr;  //pointer to the base class TClass
   TClass            *fClass;     //pointer to parent class

public:
   TBaseClass(BaseClassInfo_t *info = 0, TClass *cl = 0);
   virtual     ~TBaseClass();
   virtual void Browse(TBrowser *b);
   const char  *GetTitle() const;
   TClass      *GetClassPointer(Bool_t load=kTRUE);
   Int_t        GetDelta() const;
   Bool_t       IsFolder() const {return kTRUE;}
   Int_t        IsSTLContainer();
   Long_t       Property() const;

   ClassDef(TBaseClass,0)  //Description of a base class
};

#endif
