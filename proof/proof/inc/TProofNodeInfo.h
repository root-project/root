// @(#)root/proof:$Id$
// Author: Paul Nilsson   7/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofNodeInfo
#define ROOT_TProofNodeInfo


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofNodeInfo                                                       //
//                                                                      //
// Implementation of PROOF node info.                                   //
// The purpose of this class is to provide a complete node description  //
// for masters, submasters and workers.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"

class TProofResourcesStatic;


class TProofNodeInfo : public TObject {

friend class TProofResourcesStatic;

public:
   enum ENodeType { kMaster, kSubMaster, kWorker };

private:
   ENodeType fNodeType;  // Distinction between master, submaster and worker
   TString   fName;      // Tag of the node (name:port)
   TString   fNodeName;  // Name of the node
   TString   fWorkDir;   // Working directory
   TString   fOrdinal;   // Worker ordinal number
   TString   fImage;     // File system image
   TString   fId;        // Id number
   TString   fConfig;    // Configuration file name [for submasters]
   TString   fMsd;       // Msd value [for submasters]
   Int_t     fPort;      // Port number
   Int_t     fPerfIndex; // Performance index
   Int_t     fNWrks;     // Number of workers (when kSubMaster)

   void operator=(const TProofNodeInfo &);    // idem

public:
   TProofNodeInfo();
   TProofNodeInfo(const char *str);
   TProofNodeInfo(const TProofNodeInfo &nodeInfo);
   ~TProofNodeInfo() { }

   const char    *GetName() const { return fName; }
   ENodeType      GetNodeType() const { return fNodeType; }
   const TString &GetNodeName() const { return fNodeName; }
   const TString &GetWorkDir() const { return fWorkDir; }
   const TString &GetOrdinal() const { return fOrdinal; }
   const TString &GetImage() const { return fImage; }
   const TString &GetId() const { return fId; }
   const TString &GetConfig() const { return fConfig; }
   const TString &GetMsd() const { return fMsd; }
   Int_t          GetPort() const { return fPort; }
   Int_t          GetPerfIndex() const { return fPerfIndex; }
   Int_t          GetNWrks() const { return fNWrks; }

   Bool_t         IsMaster() const { return (fNodeType == kMaster) ? kTRUE : kFALSE; }
   Bool_t         IsSubMaster() const { return (fNodeType == kSubMaster) ? kTRUE : kFALSE; }
   Bool_t         IsWorker() const { return (fNodeType == kWorker) ? kTRUE : kFALSE; }

   void           SetNodeType(ENodeType nt) { fNodeType = nt; }
   void           SetNWrks(Int_t nw) { fNWrks = nw; }

   void Assign(const TProofNodeInfo &n);

   void Print(const Option_t *) const;

   static ENodeType GetNodeType(const TString &type);

   ClassDef(TProofNodeInfo,1) // Class describing a PROOF node
};

#endif
