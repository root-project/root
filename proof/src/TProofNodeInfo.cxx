// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.71 2005/10/27 23:28:33 rdm Exp $
// Author: Paul Nilsson   7/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofNodeInfo                                                       //
//                                                                      //
// Implementation of PROOF node info.                                   //
// The purpose of this class is to provide a complete node description  //
// for masters, submasters and workers.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofNodeInfo.h"

ClassImp(TProofNodeInfo)


//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo():
   fNodeType(kWorker),
   fPort(-1),
   fPerfIndex(100)
{
   // Constructor.
}

//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo(const TProofNodeInfo &nodeInfo) : TObject(nodeInfo)
{
   // Copy constructor.

   fNodeType  = nodeInfo.fNodeType;
   fNodeName  = nodeInfo.fNodeName;
   fWorkDir   = nodeInfo.fWorkDir;
   fOrdinal   = nodeInfo.fOrdinal;
   fImage     = nodeInfo.fImage;
   fId        = nodeInfo.fId;
   fConfig    = nodeInfo.fConfig;
   fMsd       = nodeInfo.fMsd;
   fPort      = nodeInfo.fPort;
   fPerfIndex = nodeInfo.fPerfIndex;
}

//______________________________________________________________________________
void TProofNodeInfo::Print(const Option_t *) const
{
   // Print the TProofNodeInfo structure.

   Printf("fNodeType:  %d", fNodeType);
   Printf("fNodeName:  %s", fNodeName.Data());
   Printf("fWorkDir:   %s", fWorkDir.Data());
   Printf("fOrdinal:   %s", fOrdinal.Data());
   Printf("fImage:     %s", fImage.Data());
   Printf("fId:        %s", fId.Data());
   Printf("fConfig:    %s", fConfig.Data());
   Printf("fMsd:       %s", fMsd.Data());
   Printf("fPort:      %d", fPort);
   Printf("fPerfIndex: %d\n", fPerfIndex);
}

//______________________________________________________________________________
TProofNodeInfo::ENodeType TProofNodeInfo::GetNodeType(const TString &type)
{
   // Static method returning node type. Allowed input: "master", "submaster",
   // or anything else which will be interpreted as worker.

   ENodeType enType;

   if (type == "master") {
      enType = kMaster;
   }
   else if (type == "submaster") {
      enType = kSubMaster;
   }
   else { // [worker/slave or condorworker]
      enType = kWorker;
   }

   return enType;
}
