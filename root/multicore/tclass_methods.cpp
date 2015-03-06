#include "TClass.h"
#include "TThread.h"
#include "TObject.h"
#include "TVirtualStreamerInfo.h"
#include <thread>
#include <memory>
#include <atomic>
#include <cassert>
#include <iostream>

void printHelp(const char* iName, int iDefaultNThreads)
{
  std::cout << iName <<" [number of threads] \n\n"
            <<"If no arguments are given "<<iDefaultNThreads<<" threads will be used"<<std::endl;
}

int parseOptionsForNumberOfThreads(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  int returnValue = kDefaultNThreads;
  if( argc == 2 ) {
    if(strcmp("-h",argv[1]) ==0) {
      printHelp(argv[0],kDefaultNThreads);
      exit( 0 );
    }

    returnValue = atoi(argv[1]);
 }

  if( argc > 2) {
    printHelp(argv[0],kDefaultNThreads);
    exit(1);
  }
  return returnValue ;
}

int main(int argc, char** argv)
{
  const int kNThreads = parseOptionsForNumberOfThreads(argc,argv);

  std::atomic<bool> canStart{false};
  std::vector<std::thread> threads;

  std::atomic<int> classWasGotten{0};
  std::atomic<int> firstMethodGotten{0};

  TThread::Initialize();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);

  //Have to avoid having Streamers modify themselves after they have been used
  TVirtualStreamerInfo::Optimize(false);


  for(int i=0; i<kNThreads; ++i) {
    threads.emplace_back([i,&canStart,&classWasGotten,&firstMethodGotten]() {
        thread_local TThread guard;
        ++classWasGotten;
        ++firstMethodGotten;
        while(not canStart) {}
        auto thingClass = TClass::GetClass("edmtest::Simple");
        --classWasGotten;
        while(classWasGotten !=0) {}

        TMethod* method = thingClass->GetMethodWithPrototype("id","",true /*is const*/, ROOT::kConversionMatch);
        --firstMethodGotten;
        while(firstMethodGotten !=0) {}
        TMethod* method2 = thingClass->GetMethodWithPrototype("operator=","edmtest::Simple const&",false /*is const*/, ROOT::kConversionMatch);

        assert(nullptr != method);
        assert(nullptr != method2);
      });
  }
  canStart = true;

  for(auto& thread: threads) {
    thread.join();
  }

  return 0;
}
