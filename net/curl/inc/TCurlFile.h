// @(#)root/net:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCurlFile
#define ROOT_TCurlFile

#include "ROOT/RCurlConnection.hxx"
#include "TFile.h"

#include <memory>

class TCurlFile : public TFile {
   std::unique_ptr<ROOT::Internal::RCurlConnection> fConnection;

public:
   TCurlFile(const char *url, Option_t *option = "");

   Long64_t GetSize() const override;
   void Seek(Long64_t offset, ERelativeTo pos = kBeg) override;
   Bool_t ReadBuffer(char *buf, Int_t len) override;
   Bool_t ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
   Bool_t ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf) override;

   ClassDefOverride(TCurlFile, 0)
};

#endif
