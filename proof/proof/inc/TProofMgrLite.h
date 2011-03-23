// @(#)root/proofx:$Id$
// Author: G. Ganis, Apr 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofMgrLite
#define ROOT_TProofMgrLite


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofMgrLite                                                        //
//                                                                      //
// Basic functionality implementtaion in the case of Lite sessions      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProofMgr
#include "TProofMgr.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TProofMgrLite : public TProofMgr {

public:
   TProofMgrLite(const char *url, Int_t loglevel = -1, const char *alias = "");
   virtual ~TProofMgrLite() { }

   TProof     *CreateSession(const char * = 0, const char * = 0, Int_t = -1);
   TProofLog  *GetSessionLogs(Int_t ridx = 0, const char *stag = 0,
                              const char *pattern = "-v | SvcMsg", Bool_t rescan = kFALSE);
   TObjString *ReadBuffer(const char *file, Long64_t ofs, Int_t len);
   TObjString *ReadBuffer(const char *file, const char *pattern);

   ClassDef(TProofMgrLite,0)  // XrdProofd PROOF manager interface
};

#endif
