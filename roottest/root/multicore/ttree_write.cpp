#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TClass.h"
#include "TROOT.h"
#include "TList.h"
#include "TMap.h"
#include "TObjString.h"
#include "TH1F.h"


#include <memory>
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>
#include <sstream>

std::atomic<bool> waitToStart{true};
std::atomic<unsigned int> countdownToWrite{0};

void printHelp(const char* iName, int iDefaultNThreads)
{
   std::cout << iName <<" [number of threads] [gDebug value]\n\n"
             <<"[number of threads] number of threads to use in test\n"
             <<"[gDebug value] value of gDebug to pass to ROOT (gDebug=1 is useful)\n"
             <<"If no arguments are given "<<iDefaultNThreads<<" threads will be used"<<std::endl;
}

std::pair<int,int> parseOptionsForNumberOfThreadsAndgDebug(int argc, char** argv)
{
   constexpr int kDefaultNThreads = 4;
   int kDefaultgDebug = gDebug;
   int nThreads = kDefaultNThreads;
   int newGDebug = kDefaultgDebug;
   if (argc >= 2) {
      if(strcmp("-h",argv[1]) ==0) {
         printHelp(argv[0],kDefaultNThreads);
         exit( 0 );
      }

      nThreads = atoi(argv[1]);
   }
   if (argc == 3) {
      newGDebug = atoi(argv[2]);
   }

   if (argc > 3) {
      printHelp(argv[0],kDefaultNThreads);
      exit(1);
   }
   return std::make_pair(nThreads, newGDebug) ;
}

TList* createList() {
   auto returnValue = new TList();

   for(unsigned int i=0; i<10;++i) {
      returnValue->Add(new TList());
      returnValue->Add(new TMap());
      returnValue->Add(new TObjString());
      returnValue->Add(new TH1F());
      returnValue->Add(new TH1D());
   }

   return returnValue;
}

int main(int argc, char** argv) {

   auto values = parseOptionsForNumberOfThreadsAndgDebug(argc,argv);

   const int kNThreads = values.first;
   countdownToWrite = kNThreads;

   gDebug = values.second;

   //Tell Root we want to be multi-threaded
   ROOT::EnableThreadSafety();
   //When threading, also have to keep ROOT from logging all TObjects into a list
   TObject::SetObjectStat(false);

   std::vector<std::shared_ptr<std::thread>> threads;
   threads.reserve(kNThreads);

   // Enable multithreading of the FlushBaskets.
   ROOT::EnableImplicitMT(kNThreads);

   for(int i=0; i< kNThreads; ++i) {
      threads.push_back(std::make_shared<std::thread>( std::thread([i]() {
         std::stringstream nameStream;
         nameStream <<"write_thread_"<<i<<".root";

         auto theList = createList();

         while(waitToStart) ;
         TFile f(nameStream.str().c_str(), "RECREATE","test");

         auto listTree = new TTree("TheList","TheList");
         listTree->Branch("theList","TList",&theList);

         --countdownToWrite;
         while(countdownToWrite !=0) ;

         for(unsigned int entry = 0; entry<10000;++entry) {
            listTree->Fill();
         }
         f.Write();
         f.Close();
      }) ) );

   }
   waitToStart = false;
   for(auto& t : threads) {
      t->join();
   }

   return 0;
}
