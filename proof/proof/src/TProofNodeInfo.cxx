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

//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo()
               : fNodeType(kWorker), fPort(-1), fPerfIndex(100), fNWrks(1)
{
   // Default constructor.
}

//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo(const char *str)
               : fNodeType(kWorker), fPort(-1), fPerfIndex(100), fNWrks(1)
{
   // Constructor from a string containing all the information in a serialized
   // way. Used to decode thr information coming from the coordinator
   // <type>|<host@user>|<port>|<ord>|<id>|<perfidx>|<img>|<workdir>|<msd>|<cfg>

   // Needs a non empty string to do something
   if (!str || strlen(str) <= 0)
      return;

   TString ss(str), s;
   Ssiz_t from = 0;
   // NodeType
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fNodeType = GetNodeType(s);
   // NodeName
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fNodeName = s;
   // Port
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      if (s.IsDigit()) fPort = s.Atoi();
   // Ordinal
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fOrdinal = s;
   // ID string
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fId = s;
   // Performance
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      if (s.IsDigit()) fPerfIndex = s.Atoi();
   // Image
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fImage = s;
   // Working dir
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fWorkDir = s;
   // Mass Storage Domain
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fMsd = s;
   // Config file (master or submaster; for backward compatibility)
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      fConfig = s;
   // Number of workers
   if (ss.Tokenize(s, from, "|") && !s.IsNull() && s != "-")
      if (s.IsDigit()) fNWrks = s.Atoi();

   // Set the name
   fName.Form("%s:%d", fNodeName.Data(), fPort);
}

//______________________________________________________________________________
TProofNodeInfo::TProofNodeInfo(const TProofNodeInfo &nodeInfo) : TObject(nodeInfo)
{
   // Copy constructor.

   fName      = nodeInfo.fName;
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
   fNWrks     = nodeInfo.fNWrks;
}

//______________________________________________________________________________
void TProofNodeInfo::Assign(const TProofNodeInfo &n)
{
   // Asssign content of node n to this node

   fName      = n.fName;
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
   fNWrks     = n.fNWrks;
}

//______________________________________________________________________________
void TProofNodeInfo::Print(const Option_t *opt) const
{
   // Print the TProofNodeInfo structure.

   if (opt[0] == 'c' || opt[0] == 'C') {
      Printf("%d %s:%d %s %s", fNodeType, fNodeName.Data(), fPort,
                               fOrdinal.Data(), fWorkDir.Data());
   } else {
      Printf(" +++ TProofNodeInfo: %s +++", fName.Data());
      Printf(" NodeName: %s, Port: %d, NodeType: %d, Ordinal: %s",
             fNodeName.Data(), fPort, fNodeType, fOrdinal.Data());
      Printf(" WorkDir: %s, Image: %s", fWorkDir.Data(), fImage.Data());
      Printf(" Id: %s, Config: %s", fId.Data(), fConfig.Data());
      Printf(" Msd: %s", fMsd.Data());
      Printf(" Performance:   %d", fPerfIndex);
      Printf(" NumberOfWrks:  %d", fNWrks);
      Printf("+++++++++++++++++++++++++++++++++++++++++++");
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
