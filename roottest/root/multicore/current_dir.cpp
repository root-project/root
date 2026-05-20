#include "TFile.h"
#include "TROOT.h"
#include "TDirectory.h"

#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

#define NUM_ITER 100000

std::atomic<bool> waitToStart{true};

void printHelp(const char* iName, int iDefaultNThreads)
{
  std::cout << iName <<" [number of threads]\n\n"
            <<"[number of threads] number of threads to use in test\n"
            <<"If no arguments are given "<<iDefaultNThreads<<" threads will be used."<<std::endl;
}

std::tuple<int> parseOptions(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  int nThreads = kDefaultNThreads;
  if( argc == 2 ) {
    if(strcmp("-h",argv[1]) ==0) {
      printHelp(argv[0],kDefaultNThreads);
      exit( 0 );
    }

    nThreads = atoi(argv[1]);
  }

  if( argc > 2) {
    printHelp(argv[0],kDefaultNThreads);
    exit(1);
  }
  return std::make_tuple(nThreads) ;
}

void createEmptyFile(int tid, std::string fileName) {
  std::string name = std::to_string(tid) + fileName;
  TFile f(name.c_str(), "RECREATE","test");
  f.Close();
}


int main(int argc, char** argv) {

  auto options = parseOptions(argc,argv);

  const int kNThreads = std::get<0>(options);

  std::string fileName("empty_file.root");

  //Tell Root we want to be multi-threaded
  ROOT::EnableThreadSafety();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);

  for(int i=0; i< kNThreads; ++i) {
      createEmptyFile(i, fileName);
  }

  std::vector<std::shared_ptr<std::thread>> threads;
  threads.reserve(kNThreads);

  printf("Running current dir on multiple threads...\n");
  for(int i=0; i< kNThreads; ++i) {
    threads.push_back(std::make_shared<std::thread>( std::thread([&fileName, i]() {
        while(waitToStart) ;
        std::string name = std::to_string(i) + fileName;
        std::unique_ptr<TFile> f{ TFile::Open(name.c_str()) };
	assert(f.get());

	TDirectory *mydir = TDirectory::CurrentDirectory();
        for (int j = 0; j < NUM_ITER; ++j) {
            TDirectory *tmpdir = TDirectory::CurrentDirectory();
            if (mydir != tmpdir) {
              printf("Error: current directory has changed, thread %d: INITIAL(%s), CURRENT(%s)\n", i, mydir->GetName(), tmpdir->GetName());
              break;
            }
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

