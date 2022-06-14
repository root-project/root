// @(#)root/io:$Id$
// Author: Amit Bashyal, August 2018

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPIClientInfo
#define ROOT_TMPIClientInfo

#include "TFile.h"
#include "TTimeStamp.h"

class TMPIClientInfo {

private:
   TFile *fFile;
   TString fLocalName;
   UInt_t fContactsCount;
   TTimeStamp fLastContact;
   Double_t fTimeSincePrevContact;

   static void MigrateKey(TDirectory *destination, TDirectory *source);

public:
   TMPIClientInfo();                                      // default constructor
   TMPIClientInfo(const char *filename, UInt_t clientID); // another constructor
   virtual ~TMPIClientInfo();

   TFile *GetFile() const { return fFile; }
   TString GetLocalName() const { return fLocalName; }
   Double_t GetTimeSincePrevContact() const { return fTimeSincePrevContact; }

   void SetFile(TFile *file);

   ClassDef(TMPIClientInfo, 0);
};
#endif
