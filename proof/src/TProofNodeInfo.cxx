// @(#)root/proof:$Id$
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

#include "TObjArray.h"
#include "TObjString.h"
#include "TProofNodeInfo.h"

ClassImp(TProofNodeInfo)

// Macros used in decoding serialized info
#define PNISETANY(a) \
  { if (os->String() != "-") { a; } \
    if (!(os = (TObjString *) nxos())) return; }
#define PNISETSTRING(s) PNISETANY(s = os->GetName())
#define PNISETINT(i) PNISETANY(i = os->String().Atoi())

//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo():
   fNodeType(kWorker),
   fPort(-1),
   fPerfIndex(100)
{
   // Constructor.
}

//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo(const char *str)
               : fNodeType(kWorker), fPort(-1), fPerfIndex(100)
{
   // Constructor from a string containing all the information in a serialized
   // way. Used to decode thr information coming from the coordinator
   // <type>|<host@user>|<port>|<ord>|<id>|<perfidx>|<img>|<workdir>|<msd>|<cfg>

   // Needs a non empty string to do something
   if (!str || strlen(str) <= 0)
      return;

   // Tokenize
   TString ss(str);
   TObjArray *oa = ss.Tokenize("|");
   if (!oa)
      return;
   TIter nxos(oa);
   TObjString *os = (TObjString *) nxos();
   if (!os)
      return;

   // Node type
   PNISETANY(fNodeType = GetNodeType(os->GetName()));

   // Host and user name
   PNISETSTRING(fNodeName);

   // Port
   PNISETINT(fPort);

   // Ordinal
   PNISETSTRING(fOrdinal);

   // ID string
   PNISETSTRING(fId);

   // Performance index
   PNISETINT(fPerfIndex);

   // Image
   PNISETSTRING(fImage);

   // Working dir
   PNISETSTRING(fWorkDir);

   // Mass Storage Domain
   PNISETSTRING(fMsd);

   // Config file (master or submaster; for backward compatibility)
   PNISETSTRING(fConfig);
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
void TProofNodeInfo::Assign(const TProofNodeInfo &n)
{
   // Asssign content of node n to this node

   fNodeType  = n.fNodeType;
   fNodeName  = n.fNodeName;
   fWorkDir   = n.fWorkDir;
   fOrdinal   = n.fOrdinal;
   fImage     = n.fImage;
   fId        = n.fId;
   fConfig    = n.fConfig;
   fMsd       = n.fMsd;
   fPort      = n.fPort;
   fPerfIndex = n.fPerfIndex;
}

//______________________________________________________________________________
void TProofNodeInfo::Print(const Option_t *opt) const
{
   // Print the TProofNodeInfo structure.

   if (opt[0] == 'c' || opt[0] == 'C') {
      Printf("%d %s:%d %s %s", fNodeType, fNodeName.Data(), fPort,
                               fOrdinal.Data(), fWorkDir.Data());
   } else {
      Printf(" NodeType:      %d", fNodeType);
      Printf(" NodeName:      %s", fNodeName.Data());
      Printf(" WorkDir:       %s", fWorkDir.Data());
      Printf(" Ordinal:       %s", fOrdinal.Data());
      Printf(" Image:         %s", fImage.Data());
      Printf(" Id:            %s", fId.Data());
      Printf(" Config:        %s", fConfig.Data());
      Printf(" Msd:           %s", fMsd.Data());
      Printf(" Port:          %d", fPort);
      Printf(" Performance:   %d\n", fPerfIndex);
   }
}

//______________________________________________________________________________
TProofNodeInfo::ENodeType TProofNodeInfo::GetNodeType(const TString &type)
{
   // Static method returning node type. Allowed input: "master", "submaster",
   // or anything else which will be interpreted as worker.

   ENodeType enType;

   if (type == "M" || type == "master") {
      enType = kMaster;
   }
   else if (type == "S" || type == "submaster") {
      enType = kSubMaster;
   }
   else { // [worker/slave or condorworker]
      enType = kWorker;
   }

   return enType;
}
