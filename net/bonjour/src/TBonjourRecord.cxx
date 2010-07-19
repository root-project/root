// @(#)root/bonjour:$Id$
// Author: Fons Rademakers   29/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBonjourRecord                                                       //
//                                                                      //
// Contains all information concerning a Bonjour entry.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBonjourRecord.h"
#include "Riostream.h"


ClassImp(TBonjourRecord)

//______________________________________________________________________________
void TBonjourRecord::AddTXTRecord(const char * record)
{
   TString rec(record);
   AddTXTRecord(rec);
}

//______________________________________________________________________________
void TBonjourRecord::AddTXTRecord(const TString &record)
{
   // This methods adds the length before the data for compliance with the
   // mDNS records standard.

   fTXTRecords.Append((char)record.Length());
   fTXTRecords.Append(record);
}

//______________________________________________________________________________
void TBonjourRecord::Print(Option_t *) const
{
   // Print TBonjourRecord.

   cout << "TBonjourRecord:"
        << "\n\tService name: #" << GetServiceName() << "#"
        << "\n\tRegistered type: #" << GetRegisteredType() << "#"
        << "\n\tDomain: #" << GetReplyDomain() << "#"
        << "\n\tTXT Records (length): #" << GetTXTRecords()
                                  << "# (" << GetTXTRecordsLength() << ")"
        << endl;
}
