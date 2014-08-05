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

#include "Riostream.h"
#include "TProofResourcesStatic.h"
#include "TSystem.h"
#include "TInetAddress.h"
#include "TProofNodeInfo.h"
#include "TProofDebug.h"
#include "TUrl.h"
#include "TList.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TError.h"

ClassImp(TProofResourcesStatic)


//______________________________________________________________________________
TProofResourcesStatic::TProofResourcesStatic()
{
   // This ctor is used in TProofServ::Setup() in combination with GetWorkDir()
   // for a quick scan of the config file to retrieve the work directory.

   // Create master node info and submaster/worker lists, and set default values
   InitResources();
}

//______________________________________________________________________________
TProofResourcesStatic::TProofResourcesStatic(const char *confDir,
                                             const char *fileName)
{
   // Using this ctor will retrieve all information in the config file
   // and store it in fMaster, fSubmasterList and fWorkerList,
   // condorworkers will be stored in the fWorkerList.

   // Create master node info and submaster/worker lists, and set default values
   InitResources();

   // Open and read the PROOF config file
   if (!ReadConfigFile(confDir, fileName)) {
      PDB(kAll,1)
         Info("TProofResourcesStatic", "error encountered while reading config file");
      fValid = kFALSE;
   }
}

//______________________________________________________________________________
TProofResourcesStatic::~TProofResourcesStatic()
{
   // Destructor.

   delete fSubmasterList;
   delete fWorkerList;
   delete fMaster;
}

//______________________________________________________________________________
void TProofResourcesStatic::InitResources()
{
   // Create master node info and submaster/worker lists,
   // and set default values.

   // Create master
   fMaster = new TProofNodeInfo();
   fMaster->fNodeType = TProofNodeInfo::GetNodeType("master");
   fFoundMaster = kFALSE; // Set to kTRUE if the config file contains master info

   // Create workers
   fWorkerList = new TList();
   fWorkerList->SetOwner();

   // Create submaster
   fSubmasterList = new TList();
   fSubmasterList->SetOwner();

   // Assume that the config file will be ok
   fValid = kTRUE;
}

//______________________________________________________________________________
TProofNodeInfo *TProofResourcesStatic::GetMaster()
{
   // Get the master node. Only return the master info if it was set
   // in the config file.

   if (fFoundMaster)
      return fMaster;

   return 0;
}

//______________________________________________________________________________
TList *TProofResourcesStatic::GetSubmasters()
{
   // Get the list of submaster nodes.

   return fSubmasterList;
}

//______________________________________________________________________________
TList *TProofResourcesStatic::GetWorkers(void)
{
   // Get the list of worker nodes.

   return fWorkerList;
}

//______________________________________________________________________________
Bool_t TProofResourcesStatic::ReadConfigFile(const char *confDir,
                                             const char *fileName)
{
   // Read the PROOF config file and fill the master and worker list.

   Bool_t status = kTRUE;

   // Skip prefix (e.g. "sm:") if any
   const char *p = (const char *) strstr(fileName,":");
   if (p)
      fileName = p+1;

   // Use file specified by the cluster administrator, if any
   const char *cf = gSystem->Getenv("ROOTPROOFCONF");
   if (cf && !(gSystem->AccessPathName(cf, kReadPermission))) {
      fFileName = cf;
   } else {
      if (cf)
         PDB(kGlobal,1)
            Info("ReadConfigFile", "file %s cannot be read:"
                 " check existence and/or permissions", cf);
      if (fileName && strlen(fileName) > 0) {
         // Use user defined file or default
         // Add a proper path to the file name
         fFileName.Form("%s/.%s", gSystem->HomeDirectory(), fileName);
         PDB(kGlobal,2)
            Info("ReadConfigFile", "checking PROOF config file %s", fFileName.Data());
         if (gSystem->AccessPathName(fFileName, kReadPermission)) {
            fFileName.Form("%s/etc/proof/%s", confDir, fileName);
            PDB(kGlobal,2)
               Info("ReadConfigFile", "checking PROOF config file %s", fFileName.Data());
            if (gSystem->AccessPathName(fFileName, kReadPermission)) {
               PDB(kAll,1)
                  Info("ReadConfigFile", "no PROOF config file found");
               return kFALSE;
            }
         }
      } else {
         PDB(kAll,1)
            Info("ReadConfigFile", "no PROOF config file specified");
         return kFALSE;
      }
   }
   PDB(kGlobal,1)
      Info("ReadConfigFile", "using PROOF config file: %s", fFileName.Data());

   // Open the config file
   std::fstream infile(fFileName.Data(), std::ios::in);
   if (infile.is_open()) {
      Bool_t isMaster = kFALSE;
      Bool_t isSubmaster = kFALSE;
      Bool_t isWorker = kFALSE;

      // Each line in the file consists of several 'keywords', e.g.
      //   line = "master mypc image=local"
      //     keyword[0] = "master"
      //     keyword[1] = "mypc"
      //     keyword[2] = "image=local"
      // The last keyword has an option "image" with value "local"
      TString line = "";
      TString keyword = "";

      // Read the entire file into the allLines object
      TString allLines = "";
      allLines.ReadString(infile);
      TObjArray *lines = allLines.Tokenize("\n");
      Int_t numberOfLines = lines->GetEntries();

      // Process one line at the time
      for (Int_t j = 0; j < numberOfLines; j++) {
         TObjString *objLine = (TObjString *)lines->At(j);
         line = objLine->GetString();
         line = line.Strip(TString::kBoth);

         // Unless this line was empty or a comment, interpret the line
         if ( !((line(0,1) == "#") || (line == "")) ) {
            TProofNodeInfo *nodeinfo = 0;

            // Reset boolean (condorworkers are treated as a workers)
            isMaster = kFALSE;
            isSubmaster = kFALSE;
            isWorker = kFALSE;

            // Extract all words in the current line
            TObjArray *tokens = line.Tokenize(" ");
            Int_t n = tokens->GetEntries();
            TString option;
            TString value;
            for (Int_t i = 0; i < n; i++) {

               // Extrace one word from the current line
               keyword = ((TObjString *)tokens->At(i))->GetString();

               // Interpret this keyword
               switch (GetInfoType(keyword)) {
               case kNodeType: {
                  if (keyword == "master" || keyword == "node") {
                     nodeinfo = CreateNodeInfo(keyword);
                     isMaster = kTRUE;     // will be reset
                  }
                  // [either submaster, worker or condorworker]
                  else if (keyword == "submaster") {
                     // Get a submaster info node
                     nodeinfo = CreateNodeInfo(keyword);
                     isSubmaster = kTRUE;
                  } else {
                     // Get a worker or condorworker info node
                     nodeinfo = CreateNodeInfo(keyword);
                     isWorker = kTRUE;
                  }
                  break;
               }
               case kHost: {
                  // Store the host name
                  if (nodeinfo) {
                     nodeinfo->fNodeName = keyword;

                     // Set default image
                     if (isMaster) {
                        TString node = TUrl(nodeinfo->fNodeName).GetHost();
                        nodeinfo->fImage = strstr(nodeinfo->fNodeName, node.Data());
                     } else {
                        // If the node name contains an '@' sign, it should be removed
                        // before copying it into the [default] image info
                        // On what position is the '@' sign? (if there is one)
                        TString tmp = nodeinfo->fNodeName;
                        const Ssiz_t equalPosition = tmp.Index("@", 1, 0, TString::kExact);

                        // Extract the host
                        nodeinfo->fImage = tmp(equalPosition + 1, tmp.Length());
                     }
                  } else {
                     Error("ReadConfigFile","Command not recognized: %s (ignored)",
                           keyword.Data());
                  }
                  break;
               }
               case kOption: {
                  // On what position is the '=' sign?
                  const Ssiz_t equalPosition =
                     keyword.Index("=", 1, 0, TString::kExact);

                  // Extract the option and its value
                  TString tmp = keyword;
                  option = tmp(0, equalPosition);
                  value = tmp(equalPosition + 1, tmp.Length());

                  // Set the node info options
                  SetOption(nodeinfo, option, value);
                  break;
               }
               default:
                  break;
               } // end switch

            } // end if

            // Check if we found a good master
            if (isMaster) {
               // Check if the master can run on this node
               TString node = TUrl(nodeinfo->fNodeName).GetHost();
               TString host = gSystem->GetHostByName(gSystem->HostName()).GetHostName();
               TInetAddress inetaddr = gSystem->GetHostByName(node);
               if (!host.CompareTo(inetaddr.GetHostName()) || (node == "localhost")) {
                  fFoundMaster = kTRUE;
                  fMaster->Assign(*nodeinfo);
               }
            }

            // Store the submaster, worker or condorworker
            if (isWorker) {
               fWorkerList->Add(nodeinfo);
            }
            else if (isSubmaster) {
               fSubmasterList->Add(nodeinfo);
            }
         } // else

      } // while (! infile.eof() )
      infile.close();

      // Did the config file contain appropriate master information?
      if (!fFoundMaster) {
         Error("ReadConfigFile","No master info found in config file");
         status = kFALSE;
      }
   } // end if (infile.is_open())
   else {
      // Error: could not open file
      status = kFALSE;
   }

   return status;
}


//______________________________________________________________________________
void TProofResourcesStatic::SetOption(TProofNodeInfo *nodeinfo,
                                      const TString &option,
                                      const TString &value)
{
   // Static method to set the node info options.

   if (!nodeinfo) return;

   if (option == "workdir") {
      nodeinfo->fWorkDir = value;
   } else if (option == "image") {
      nodeinfo->fImage = value;
   } else if (option == "perf") {
      nodeinfo->fPerfIndex = value.Atoi();
   } else if (option == "config") {
      nodeinfo->fConfig = value;
   } else if (option == "msd") {
      nodeinfo->fMsd = value;
   } else if (option == "port") {
      nodeinfo->fPort = value.Atoi();
   } else {
      ::Error("SetOption","No such option [%s=%s]",option.Data(),value.Data());
   }
}

//______________________________________________________________________________
TProofResourcesStatic::EInfoType TProofResourcesStatic::GetInfoType(const TString &word)
{
   // Static method to determine the info type.

   EInfoType type = kNodeType;

   if ((word == "node") || (word == "master") || (word == "submaster") ||
       (word == "worker") || (word == "slave") ||
       (word == "condorworker") || (word == "condorslave")) {
      type = kNodeType;
   }
   else if (word.Contains("=", TString::kExact)) {
      type = kOption;
   } else {
      type = kHost;
   }

   return type;
}

//______________________________________________________________________________
TProofNodeInfo *TProofResourcesStatic::CreateNodeInfo(const TString &name)
{
   // Fill out the preliminary TProofNodeInfo structure.

   TProofNodeInfo *nodeInfo = new TProofNodeInfo();
   nodeInfo->fNodeType  = TProofNodeInfo::GetNodeType(name);
   nodeInfo->fNodeName  = name;
   nodeInfo->fPort      = -1;
   nodeInfo->fPerfIndex = 100;

   return nodeInfo;
}
