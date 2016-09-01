#include <iostream>

#include "TPoolPlayer.h"
#include "PoolUtils.h"
#include "MPCode.h"

#include "TFile.h"

void TPoolPlayer::Init(int fd, unsigned nWorkers) {
   TMPWorker::Init(fd, nWorkers);
}

void TPoolPlayer::HandleInput(MPCodeBufPair& msg)
{
   unsigned code = msg.first;

   if (code == PoolCode::kProcTree) {
      ProcTree(msg);
   } else if (code == PoolCode::kProcRange || code == PoolCode::kProcFile) {
      ProcDataSet(code, msg);
   } else if (code == PoolCode::kSendResult){
      fSelector.SlaveTerminate();
      MPSend(GetSocket(), PoolCode::kProcResult, fSelector.GetOutputList());
   }else {
      //unknown code received
      std::string reply = "S" + std::to_string(GetPid());
      reply += ": unknown code received: " + std::to_string(code);
      MPSend(GetSocket(), MPCode::kError, reply.data());
   }
}

void TPoolPlayer::ProcDataSet(unsigned int code, MPCodeBufPair& msg)
{
   //evaluate the index of the file to process in fFileNames
   //(we actually don't need the parameter if code == kProcTree)
   unsigned fileN = 0;
   unsigned nProcessed = 0;

   if (code == PoolCode::kProcRange) {
      //retrieve the total number of entries ranges processed so far by TPool
      nProcessed = ReadBuffer<unsigned>(msg.second.get());
      //evaluate the file and the entries range to process
      fileN = nProcessed / fNWorkers;
   } else {
      //evaluate the file and the entries range to process
      fileN = ReadBuffer<unsigned>(msg.second.get());
   }

   // Open the file
   fFile = OpenFile(fFileNames[fileN]);
   if (fFile == nullptr) {
      //errors are handled inside OpenFile
      return;
   }

   //retrieve the TTree with the specified name from file
   //we are not the owner of the TTree object, the file is!
   fTree = RetrieveTree(fFile);
   if (fTree == nullptr) {
      //errors are handled inside RetrieveTree
      return;
   }
   TTree *tree = fTree;

   // Setup the cache, if required
   SetupTreeCache(fTree);

   //create entries range
   Long64_t start = 0;
   Long64_t finish = 0;
   if (code == PoolCode::kProcRange) {
      //example: for 21 entries, 4 workers we want ranges 0-5, 5-10, 10-15, 15-21
      //and this worker must take the rangeN-th range
      unsigned nEntries = tree->GetEntries();
      unsigned nBunch = nEntries / fNWorkers;
      if(nEntries % fNWorkers)
        nBunch++;
      unsigned rangeN = nProcessed % fNWorkers;
      start = rangeN*nBunch + 1;
      if(rangeN < (fNWorkers-1))
         finish = (rangeN+1)*nBunch;
      else
         finish = nEntries;
   } else {
      start = 0;
      finish = tree->GetEntries();
   }

   //check if we are going to reach the max of entries
   //change finish accordingly
   if (fMaxNEntries)
      if (fProcessedEntries + finish - start > fMaxNEntries)
         finish = start + fMaxNEntries - fProcessedEntries;

   if(fFirstEntry){
     fSelector.SlaveBegin(nullptr);
     fFirstEntry = false;
   }

   fSelector.Init(tree);
   fSelector.Notify();
   for(Long64_t entry = start; entry<finish; ++entry) {
      fSelector.Process(entry);
   }

   //update the number of processed entries
   fProcessedEntries += finish - start;

   MPSend(GetSocket(), PoolCode::kIdling);

   return;
}

void TPoolPlayer::ProcTree(MPCodeBufPair& msg)
{

   // The tree must be defined at this level
   if(fTree == nullptr) {
      std::cout << "tree undefined!\n" ;
      //errors are handled inside RetrieveTree
      return;
   }

   //evaluate the index of the file to process in fFileNames
   //(we actually don't need the parameter if code == kProcTree)
   unsigned nProcessed = 0;
   //retrieve the total number of entries ranges processed so far by TPool
   nProcessed = ReadBuffer<unsigned>(msg.second.get());

   //create entries range
   Long64_t start = 0;
   Long64_t finish = 0;
   //example: for 21 entries, 4 workers we want ranges 0-5, 5-10, 10-15, 15-21
   //and this worker must take the rangeN-th range
   unsigned nEntries = fTree->GetEntries();
   unsigned nBunch = nEntries / fNWorkers;
   unsigned rangeN = nProcessed % fNWorkers;
   start = rangeN*nBunch + 1;
   if(rangeN < (fNWorkers-1))
      finish = (rangeN+1)*nBunch;
   else
      finish = nEntries;

   //check if we are going to reach the max of entries
   //change finish accordingly
   if (fMaxNEntries)
      if (fProcessedEntries + finish - start > fMaxNEntries)
         finish = start + fMaxNEntries - fProcessedEntries;

   //process tree
   TTree *tree = fTree;
   CloseFile(); // May not be needed
   if (fTree->GetCurrentFile()) {
      // We need to reopen the file locally (TODO: to understand and fix this)
      fFile = TFile::Open(fTree->GetCurrentFile()->GetName());
      tree = (TTree *) fFile->Get(fTree->GetName());
      fTree = tree;
   }

   // Setup the cache, if required
   SetupTreeCache(fTree);

   if(fFirstEntry){
     fSelector.SlaveBegin(nullptr);
     fFirstEntry = false;
   }

   fSelector.Init(tree);
   fSelector.Notify();

   for(Long64_t entry = start; entry<finish; ++entry) {
      fSelector.Process(entry);
   }

   //update the number of processed entries
   fProcessedEntries += finish - start;

   MPSend(GetSocket(), PoolCode::kIdling);

   return;
}
