// @(#)root/io:$Id$
// Author: Amit Bashyal, August 2018

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClientInfo
#define ROOT_TClientInfo

#include "TFile.h"
#include "TTimeStamp.h"

class TClientInfo {

private:
  TFile *fFile;
  TString fLocalName;
  UInt_t fContactsCount;
  TTimeStamp fLastContact;
  Double_t fTimeSincePrevContact;

public:
  TClientInfo();                                      // default constructor
  TClientInfo(const char *filename, UInt_t clientID); // another constructor
  virtual ~TClientInfo();

  TFile *GetFile() const {return fFile;}
  TString GetLocalName() const {return fLocalName;}
  Double_t GetTimeSincePrevContact() const {return fTimeSincePrevContact;}

  void SetFile(TFile *file);

  void R__MigrateKey(TDirectory *destination, TDirectory *source);
  void R__DeleteObject(TDirectory *dir, Bool_t withReset);

  ClassDef(TClientInfo, 0);
};
#endif
