// @(#)root/proof:$Id$
// Author: Sangsu Ryu 2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofNodes
#define ROOT_TProofNodes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofNodes                                                          //
//                                                                      //
// PROOF worker nodes information                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TProof;
class TMap;

class TProofNodes: public TObject {
private:
   TProof *fProof;
   TMap   *fNodes;        // Map of node infos
   TMap   *fActiveNodes;  // Map of active node infos
   Int_t   fMaxWrksNode;  // Max number of workers per node
   Int_t   fMinWrksNode;  // Min number of workers per node
   Int_t   fNNodes;       // Number of nodes
   Int_t   fNWrks;        // Number of workers
   Int_t   fNActiveWrks;  // Number of active workers
   Int_t   fNCores;       // Number of total cores

   void Build();
public:
   TProofNodes(TProof* proof);

   virtual ~TProofNodes();
   Int_t ActivateWorkers(Int_t nwrks);
   Int_t ActivateWorkers(const char *workers);
   Int_t GetMaxWrksPerNode() const { return fMaxWrksNode; }
   Int_t GetNWorkersCluster() const { return fNWrks; }
   Int_t GetNNodes() const { return fNNodes; }
   Int_t GetNCores() const { return fNCores; }
   Int_t GetMinWrksPerNode() const { return fMinWrksNode; }
   Int_t GetNActives() const { return fNActiveWrks; }
   TMap* GetMapOfNodes() const { return fNodes; }
   TMap* GetMapOfActiveNodes() const { return fActiveNodes; }
   void Print(Option_t* option="") const;

   ClassDef(TProofNodes, 0) //Node and worker information
};

#endif
