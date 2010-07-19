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
   TString   fTXTRecords;

public:
   TBonjourRecord() { }
   TBonjourRecord(const char *name, const char *regType, const char *domain) :
      fServiceName(name), fRegisteredType(regType), fReplyDomain(domain),
      fTXTRecords("") { }
   TBonjourRecord(const char *name, const char *regType, const char *domain,
                  const char *txt) :
      fServiceName(name), fRegisteredType(regType),
      fReplyDomain(domain), fTXTRecords(txt) { }
   virtual ~TBonjourRecord() { }

   Bool_t operator==(const TBonjourRecord &other) const {
      return fServiceName == other.fServiceName &&
             fRegisteredType == other.fRegisteredType &&
             fReplyDomain == other.fReplyDomain &&
             fTXTRecords == other.fTXTRecords;
   }

   Bool_t IsEqual(const TObject *obj) const { return *this == *(TBonjourRecord*)obj; }

   const char *GetServiceName() const { return fServiceName; }
   const char *GetRegisteredType() const { return fRegisteredType; }
   const char *GetReplyDomain() const { return fReplyDomain; }
   const char *GetTXTRecords() const { return fTXTRecords; }
   Int_t GetTXTRecordsLength() const { return fTXTRecords.Length(); }

   void AddTXTRecord(const char *record);
   void AddTXTRecord(const TString &record);

   void Print(Option_t *opt = "") const;

   ClassDef(TBonjourRecord,0)  // Bonjour information record
};

#endif
