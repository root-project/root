#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TClass.h"
#include "TROOT.h"

#include "TList.h"
#include "TMap.h"
#include "TObjString.h"
#include "TH1F.h"

//#include "FWCore/FWLite/interface/AutoLibraryLoader.h"

#include <memory>
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<bool> waitToStart{true};

void printHelp(const char* iName, int iDefaultNThreads)
{
  std::cout << iName <<" [number of threads] [filename] [gDebug value]\n\n"
            <<"[number of threads] number of threads to use in test\n"
            <<"[filename] name of CMSSW file to read\n"
            <<"[gDebug value] value of gDebug to pass to ROOT (gDebug=1 is useful)\n"
            <<"If no arguments are given "<<iDefaultNThreads<<" threads will be used and a dummy file will be created."<<std::endl;
}

const std::string kDefaultFileName("ttree_read.root");

std::tuple<int,std::string,int> parseOptions(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  int kDefaultgDebug = gDebug;
  int nThreads = kDefaultNThreads;
  int newGDebug = kDefaultgDebug;
  std::string fileName(kDefaultFileName);
  if( argc >= 2 ) {
    if(strcmp("-h",argv[1]) ==0) {
      printHelp(argv[0],kDefaultNThreads);
      exit( 0 );
    }

    nThreads = atoi(argv[1]);
  }
  if(argc >=3) {
    fileName = argv[2];
  }
  if(argc == 4) {
    newGDebug = atoi(argv[3]);
  }

  if( argc > 4) {
    printHelp(argv[0],kDefaultNThreads);
    exit(1);
  }
  return std::make_tuple(nThreads, fileName, newGDebug) ;
}

void createDummyFile(int tid) {
  auto theList = new TList();

  for(unsigned int i=0; i<10;++i) {
    theList->Add(new TList());
    theList->Add(new TMap());
    theList->Add(new TObjString());
    theList->Add(new TH1F());
    theList->Add(new TH1D());
  }

  std::string name = std::to_string(tid) + kDefaultFileName;
  TFile f(name.c_str(), "RECREATE","test");

  auto listTree = new TTree("Events","TheList");
  listTree->Branch("theList","TList",&theList);

  for(unsigned int i = 0; i<100;++i) {
    listTree->Fill();
  }
  f.Write();
  f.Close();
}


int main(int argc, char** argv) {

  auto options = parseOptions(argc,argv);

  const int kNThreads = std::get<0>(options);

  auto const kFileName = std::get<1>(options);

  gDebug = std::get<2>(options);

//  AutoLibraryLoader::enable();

  //Tell Root we want to be multi-threaded
  ROOT::EnableThreadSafety();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);

  if(kFileName == kDefaultFileName) {
    for(int i=0; i< kNThreads; ++i) {
        createDummyFile(i);
    }
  }

  std::vector<std::shared_ptr<std::thread>> threads;
  threads.reserve(kNThreads);

  for(int i=0; i< kNThreads; ++i) {
    threads.push_back(std::make_shared<std::thread>( std::thread([&kFileName, i]() {
	while(waitToStart) {} ;
        std::string name = std::to_string(i) + kFileName;
        std::unique_ptr<TFile> f{ TFile::Open(name.c_str()) };
	assert(f.get());

	TTree* eventTree = dynamic_cast<TTree*>(f.get()->Get("Events"));
	assert(eventTree);

	for(Long64_t j = 0, iEnd = eventTree->GetEntries();
	    j != iEnd;
	    ++j) {
	  eventTree->GetEntry(j,1);
	}

	f.get()->Close();
	  }) ) );

  }
  waitToStart = false;
  for(auto& t : threads) {
    t->join();
  }

  return 0;
}

