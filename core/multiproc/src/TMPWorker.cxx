/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "MPCode.h"
#include "MPSendRecv.h"
#include "TEnv.h"
#include "TError.h"
#include "TMPWorker.h"
#include "TSystem.h"
#include <memory> //unique_ptr
#include <string>

#include <iostream>

//////////////////////////////////////////////////////////////////////////
///
/// \class TMPWorker
///
/// This class works in conjuction with TMPClient, reacting to messages
/// received from it as specified by the Notify and HandleInput methods.
/// When TMPClient::Fork is called, a TMPWorker instance is passed to it
/// which will take control of the ROOT session in the children processes.
///
/// After forking, every time a message is sent or broadcast to the workers,
/// TMPWorker::Notify is called and the message is retrieved.
/// Messages exchanged between TMPClient and TMPWorker should be sent with
/// the MPSend() standalone function.\n
/// If the code of the message received is above 1000 (i.e. it is an MPCode)
/// the qualified TMPWorker::HandleInput method is called, that takes care
/// of handling the most generic type of messages. Otherwise the unqualified
/// (possibly overridden) version of HandleInput is called, allowing classes
/// that inherit from TMPWorker to manage their own protocol.\n
/// An application's worker class should inherit from TMPWorker and implement
/// a HandleInput method that overrides TMPWorker's.\n
///
//////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
/// Class constructors.
/// Note that this does not set variables like fPid or fS (worker's socket).\n
/// These operations are handled by the Init method, which is called after
/// forking.\n
/// This separation is in place because the instantiation of a worker
/// must be done once _before_ forking, while the initialization of the
/// members must be done _after_ forking by each of the children processes.
TMPWorker::TMPWorker()
          : fFileNames(), fTreeName(), fTree(nullptr), fFile(nullptr),
            fNWorkers(0), fMaxNEntries(0),
            fProcessedEntries(0), fS(), fPid(0), fNWorker(0),
            fTreeCache(0), fTreeCacheIsLearning(kFALSE),
            fUseTreeCache(kTRUE), fCacheSize(-1)
{
   Setup();
}

TMPWorker::TMPWorker(const std::vector<std::string>& fileNames,
                     const std::string& treeName,
                     unsigned nWorkers, ULong64_t maxEntries)
          : fFileNames(fileNames), fTreeName(treeName), fTree(nullptr), fFile(nullptr),
            fNWorkers(nWorkers), fMaxNEntries(maxEntries),
            fProcessedEntries(0), fS(), fPid(0), fNWorker(0),
            fTreeCache(0), fTreeCacheIsLearning(kFALSE),
            fUseTreeCache(kTRUE), fCacheSize(-1)
{
   Setup();
}

TMPWorker::TMPWorker(TTree *tree, unsigned nWorkers, ULong64_t maxEntries)
          : fFileNames(), fTreeName(), fTree(tree), fFile(nullptr),
            fNWorkers(nWorkers), fMaxNEntries(maxEntries),
            fProcessedEntries(0), fS(), fPid(0), fNWorker(0),
            fTreeCache(0), fTreeCacheIsLearning(kFALSE),
            fUseTreeCache(kTRUE), fCacheSize(-1)
{
   Setup();
}

TMPWorker::~TMPWorker()
{
   // Properly close the open file, if any
   CloseFile();
}

//////////////////////////////////////////////////////////////////////////
/// Auxilliary method for common initializations
void TMPWorker::Setup()
{
  Int_t uc = gEnv->GetValue("MultiProc.UseTreeCache", 0);
  if (uc != 1) fUseTreeCache = kFALSE;
  fCacheSize = gEnv->GetValue("MultiProc.CacheSize", -1);
}

//////////////////////////////////////////////////////////////////////////
/// This method is called by children processes right after forking.
/// Initialization of worker properties that must be delayed until after
/// forking must be done here.\n
/// For example, Init saves the pid into fPid, and adds the TMPWorker to
/// the main eventloop (as a TFileHandler).\n
/// Make sure this operations are performed also by overriding implementations,
/// e.g. by calling TMPWorker::Init explicitly.
void TMPWorker::Init(int fd, unsigned workerN)
{
   fS.reset(new TSocket(fd, "MPsock")); //TSocket's constructor with this signature seems much faster than TSocket(int fd)
   fPid = getpid();
   fNWorker = workerN;
   fId = "W" + std::to_string(GetNWorker()) + "|P" + std::to_string(GetPid());
}


void TMPWorker::Run()
{
   while(true) {
      MPCodeBufPair msg = MPRecv(fS.get());
      if (msg.first == MPCode::kRecvError) {
         Error("TMPWorker::Run", "Lost connection to client\n");
         gSystem->Exit(0);
      }

      if (msg.first < 1000)
         HandleInput(msg); //call overridden method
      else
         TMPWorker::HandleInput(msg); //call this class' method
  }
}


//////////////////////////////////////////////////////////////////////////
/// Handle a message with an EMPCode.
/// This method is called upon receiving a message with a code >= 1000 (i.e.
/// EMPCode). It handles the most generic types of messages.\n
/// Classes inheriting from TMPWorker should implement their own HandleInput
/// function, that should be able to handle codes specific to that application.\n
/// The appropriate version of the HandleInput method (TMPWorker's or the
/// overriding version) is automatically called depending on the message code.
void TMPWorker::HandleInput(MPCodeBufPair &msg)
{
   unsigned code = msg.first;

   std::string reply = fId;
   if (code == MPCode::kMessage) {
      //general message, ignore it
      reply += ": ok";
      MPSend(fS.get(), MPCode::kMessage, reply.data());
   } else if (code == MPCode::kError) {
      //general error, ignore it
      reply += ": ko";
      MPSend(fS.get(), MPCode::kMessage, reply.data());
   } else if (code == MPCode::kShutdownOrder || code == MPCode::kFatalError) {
      //client is asking the server to shutdown or client is dying
      MPSend(fS.get(), MPCode::kShutdownNotice, reply.data());
      gSystem->Exit(0);
   } else {
      reply += ": unknown code received. code=" + std::to_string(code);
      MPSend(fS.get(), MPCode::kError, reply.data());
   }
}


//////////////////////////////////////////////////////////////////////////
/// Handle file closing.

void TMPWorker::CloseFile()
{
   // Avoid destroying the cache; must be placed before deleting the trees
   if (fFile) {
      if (fTree) fFile->SetCacheRead(0, fTree);
      delete fFile ;
      fFile = 0;
   }
}

//////////////////////////////////////////////////////////////////////////
/// Handle file opening.

TFile *TMPWorker::OpenFile(const std::string& fileName)
{

   TFile *fp = TFile::Open(fileName.c_str());
   if (fp == nullptr || fp->IsZombie()) {
      std::stringstream ss;
      ss << "could not open file " << fileName;
      std::string errmsg = ss.str();
      SendError(errmsg, PoolCode::kProcError);
      return nullptr;
   }

   return fp;
}


//////////////////////////////////////////////////////////////////////////
/// Retrieve a tree from an open file.

TTree *TMPWorker::RetrieveTree(TFile *fp)
{
   //retrieve the TTree with the specified name from file
   //we are not the owner of the TTree object, the file is!
   TTree *tree = nullptr;
   if(fTreeName == "") {
      // retrieve the first TTree
      // (re-adapted from TEventIter.cxx)
      if (fp->GetListOfKeys()) {
         for(auto k : *fp->GetListOfKeys()) {
            TKey *key = static_cast<TKey*>(k);
            if (!strcmp(key->GetClassName(), "TTree") || !strcmp(key->GetClassName(), "TNtuple"))
               tree = static_cast<TTree*>(fp->Get(key->GetName()));
         }
      }    
   } else {
      tree = static_cast<TTree*>(fp->Get(fTreeName.c_str()));
   }
   if (tree == nullptr) {
      std::stringstream ss;
      ss << "cannot find tree with name " << fTreeName << " in file " << fp->GetName();
      std::string errmsg = ss.str();
      SendError(errmsg, PoolCode::kProcError);
      return nullptr;
   }

   return tree;
}

//////////////////////////////////////////////////////////////////////////
/// Tree cache handling

void TMPWorker::SetupTreeCache(TTree *tree)
{
   if (fUseTreeCache) {
      TFile *curfile = tree->GetCurrentFile();
      if (curfile) {
         if (!fTreeCache) {
            tree->SetCacheSize(fCacheSize);
            fTreeCache = (TTreeCache *)curfile->GetCacheRead(tree);
            if (fCacheSize < 0) fCacheSize = tree->GetCacheSize();
         } else {
            fTreeCache->ResetCache();
            curfile->SetCacheRead(fTreeCache, tree);
            fTreeCache->UpdateBranches(tree);
         }
         if (fTreeCache) {
            fTreeCacheIsLearning = fTreeCache->IsLearning();
            if (fTreeCacheIsLearning)
               Info("SetupTreeCache","the tree cache is in learning phase");
         }
      } else {
         Warning("SetupTreeCache", "default tree does not have a file attached: corruption? Tree cache untouched");
      }
   } else {
      // Disable the cache
      tree->SetCacheSize(0);
   }
}


//////////////////////////////////////////////////////////////////////////
/// Error sender

void TMPWorker::SendError(const std::string& errmsg, unsigned int errcode)
{
   std::string reply = fId + ": " + errmsg;
   MPSend(GetSocket(), errcode, reply.data());
}
