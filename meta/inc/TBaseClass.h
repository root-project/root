// @(#)root/meta:$Name$:$Id$
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

class TBrowser;
class TClass;
class G__BaseClassInfo;


class TBaseClass : public TDictionary {

private:
   G__BaseClassInfo  *fInfo;      //pointer to CINT base class info
   TClass            *fClassPtr;  //pointer to the base class TClass
   TClass            *fClass;     //pointer to class

public:
   TBaseClass(G__BaseClassInfo *info = 0, TClass *cl = 0);
   virtual     ~TBaseClass();
   virtual void Browse(TBrowser *b);
   const char  *GetName() const;
   const char  *GetTitle() const;
   TClass      *GetClassPointer();
   Int_t        GetDelta() const;
   Int_t        Compare(TObject *obj);
   ULong_t      Hash();
   Bool_t       IsFolder() {return kTRUE;}
   Long_t       Property() const;

   ClassDef(TBaseClass,0)  //Description of a base class
};

#endif
