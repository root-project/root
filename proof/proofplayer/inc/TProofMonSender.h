// @(#)root/proofplayer:$Id$
// Author: G.Ganis July 2011

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofMonSender
#define ROOT_TProofMonSender

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofMonSender                                                      //
//                                                                      //
// Provides the interface for PROOF monitoring to different writers.    //
// Allows to decouple the information sent from the backend.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TDSet;
class TList;
class TPerfStat;

class TProofMonSender : public TNamed {

protected:
   Int_t        fSummaryVrs;           // Version of the summary 'table'
   Int_t        fDataSetInfoVrs;       // Version of the dataset info 'table'
   Int_t        fFileInfoVrs;          // Version of the file info 'table'
   enum EConfigBits {                  // TProofMonSender status/config bits
      kSendSummary        = BIT(15),   // Toggle sending of summary
      kSendDataSetInfo    = BIT(16),   // Toggle sending of dataset info
      kSendFileInfo       = BIT(17)    // Toggle sending of files info
   };

   // Auxilliary class describing dataset multiplets
   class TDSetPlet : public TNamed {
   public:
      Int_t    fFiles;
      Int_t    fMissing;
      TDSet   *fDSet;
      TDSetPlet(const char *name, TDSet *ds = 0) :
         TNamed(name, ""), fFiles(0), fMissing(0), fDSet(ds) { }
      virtual ~TDSetPlet() { }
   };
    
public:

   TProofMonSender(const char *n = "Abstract",
                   const char *t = "ProofMonSender") : TNamed(n,t),
                   fSummaryVrs(2), fDataSetInfoVrs(1), fFileInfoVrs(1)
                   { SetBit(TObject::kInvalidObject);
                     SetBit(kSendSummary);
                     SetBit(kSendDataSetInfo);
                     ResetBit(kSendFileInfo); }
   virtual ~TProofMonSender() { }

   // This changes the send control options
   Int_t SetSendOptions(const char *);

   // Object validity
   Bool_t IsValid() const { return (TestBit(TObject::kInvalidObject)) ? kFALSE : kTRUE; }

   // Summary record
   virtual Int_t SendSummary(TList *, const char *) = 0;

   // Information about the dataset(s) processed
   virtual Int_t SendDataSetInfo(TDSet *, TList *, const char *, const char *) = 0;

   // Detailed information about files
   virtual Int_t SendFileInfo(TDSet *, TList *, const char *, const char *) = 0;
   
   ClassDef(TProofMonSender,0); // Interface for PROOF monitoring
};

#endif
