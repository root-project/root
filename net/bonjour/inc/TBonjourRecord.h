// @(#)root/bonjour:$Id$
// Author: Fons Rademakers   29/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBonjourRecord
#define ROOT_TBonjourRecord


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBonjourRecord                                                       //
//                                                                      //
// Contains all information concerning a Bonjour entry.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif



class TBonjourRecord : public TObject {

private:
   TString   fServiceName;
   TString   fRegisteredType;
   TString   fReplyDomain;

public:
   TBonjourRecord() { }
   TBonjourRecord(const char *name, const char *regType, const char *domain) :
      fServiceName(name), fRegisteredType(regType), fReplyDomain(domain) { }
   virtual ~TBonjourRecord() { }

   Bool_t operator==(const TBonjourRecord &other) const {
      return fServiceName == other.fServiceName &&
             fRegisteredType == other.fRegisteredType &&
             fReplyDomain == other.fReplyDomain;
   }

   Bool_t IsEqual(const TObject *obj) const { return *this == *(TBonjourRecord*)obj; }

   const char *GetServiceName() const { return fServiceName; }
   const char *GetRegisteredType() const { return fRegisteredType; }
   const char *GetReplyDomain() const { return fReplyDomain; }

   void Print(Option_t *opt = "") const;

   ClassDef(TBonjourRecord,0)  // Bonjour information record
};

#endif
