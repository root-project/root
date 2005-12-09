// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.71 2005/10/27 23:28:33 rdm Exp $
// Author: Paul Nilsson   7/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofResourcesStatic
#define ROOT_TProofResourcesStatic

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofResourcesStatic                                                //
//                                                                      //
// Implementation of PROOF static resources.                            //
// The purpose of this class is to provide a standard interface to      //
// static config files. It interprets Proof config files (proof.conf)   //
// and sorts the contents into TProofNodeInfo objects. Master info will //
// be placed in fMaster (of type TProofNodeInfo). Submaster info will   //
// be put in fSubmasterList (a TList of TProofNodeInfo objects), while  //
// workers (and condorworkers) will be placed in fWorkerList (a TList   //
// of TProofNodeInfo objects).                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProofResources
#include "TProofResources.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TList;
class TProofNodeInfo;


class TProofResourcesStatic : public TProofResources {

public:
   enum EInfoType { kNodeType, kOption, kHost };

private:
   TProofNodeInfo *fMaster;           // Master node info
   TList          *fSubmasterList;    // Node info list with all submasters
   TList          *fWorkerList;       // Node info list with all workers
   Bool_t          fFoundMaster;      // kTRUE if config file has master info
   TString         fFileName;         // Config file name

   void             InitResources();
   Bool_t           ReadConfigFile(const TString &confDir, const TString &fileName);

   static EInfoType GetInfoType(const TString &word);
   static void      SetOption(TProofNodeInfo *nodeinfo, const TString &option,
                              const TString &value);
   static TProofNodeInfo *CreateNodeInfo(const TString &name);

public:
   TProofResourcesStatic();
   TProofResourcesStatic(const TString &confDir, const TString &fileName);
   virtual ~TProofResourcesStatic();

   TProofNodeInfo *GetMaster();
   TList          *GetSubmasters();
   TList          *GetWorkers();
   TString         GetFileName() const { return fFileName; }
   TString         GetWorkDir(const TString &confDir, const TString &fileName);

   ClassDef(TProofResourcesStatic,0) // Class to handle PROOF static config
};

#endif

