#include "TROOT.h"
#include "ROOT/TTreeProcessorMT.hxx"
#include "TLorentzVector.h"
#include "Math/Vector4D.h"
#include "TChain.h"
#include <iostream>

void printHelp(const char* iName, int iDefaultNThreads)
{
  std::cout << iName <<" [number of threads] [filename] [treename] [constructor] [gDebug value]\n\n"
            <<"[number of threads] number of threads to use in test\n"
            <<"[filename] name of file to read\n"
            <<"[treename] name of the tree to process\n"
            <<"[constructor] TTreeProcessorMT constructor to test\n"
            <<"[gDebug value] value of gDebug to pass to ROOT (gDebug=1 is useful)\n"
            <<"If no arguments are given "<<iDefaultNThreads<<" threads will be used, and a default file name and tree name will be used."<<std::endl;
}

const std::string kDefaultFileName("tp_process_imt.root");
const std::string kDefaultTreeName("events");
const std::string kDefaultConstructor("filename");

std::tuple<int,std::string,std::string,std::string,int> parseOptions(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  int kDefaultgDebug = gDebug;
  int nThreads = kDefaultNThreads;
  int newGDebug = kDefaultgDebug;
  std::string fileName(kDefaultFileName);
  std::string treeName(kDefaultTreeName);
  std::string constructor(kDefaultConstructor);
  if( argc >= 2 ) {
    if(strcmp("-h",argv[1]) ==0) {
      printHelp(argv[0], kDefaultNThreads);
      exit( 0 );
    }
    nThreads = atoi(argv[1]);
  }
  if(argc >=3) {
    fileName = argv[2];
  }
  if(argc >=4) {
    treeName = argv[3];
  }
  if(argc >=5) {
    constructor = argv[4];
  }
  if(argc == 6) {
    newGDebug = atoi(argv[5]);
  }
  if( argc > 6) {
    printHelp(argv[0],kDefaultNThreads);
    exit(1);
  }
  return std::make_tuple(nThreads, fileName, treeName, constructor, newGDebug) ;
}


int main(int argc, char** argv) {

  // We do not want to limit the number of tasks for this particular test:
  ROOT::TTreeProcessorMT::SetTasksPerWorkerHint(1024);

  auto options = parseOptions(argc,argv);
  const int kNThreads = std::get<0>(options);
  auto const kFileName = std::get<1>(options);
  auto const kTreeName = std::get<2>(options);
  auto const kConstructor = std::get<3>(options);
  gDebug = std::get<4>(options);

  std::mutex mutex;
  double globalMaxPt = -1.;

  std::cout << "[IMT] TTreeProcessorMT::Process with " << kNThreads << " threads" << std::endl;

  // Tell ROOT we want to use implicit multi-threading
  ROOT::EnableImplicitMT(kNThreads);

  // Create TTreeProcessorMT depending on constructor option
  std::unique_ptr<ROOT::TTreeProcessorMT> tp;
  if (kConstructor == "filename") {
     tp.reset(new ROOT::TTreeProcessorMT(kFileName, kTreeName));
  }
  else if (kConstructor == "collection") {
     tp.reset(new ROOT::TTreeProcessorMT({kFileName, kFileName}, kTreeName));
  }
  else if (kConstructor == "chain") {
     TChain chain(kTreeName.c_str());
     chain.Add(kFileName.c_str());
     chain.Add(kFileName.c_str());
     tp.reset(new ROOT::TTreeProcessorMT(chain));
  }
  else if (kConstructor == "entrylist") {
     TChain chain(kTreeName.c_str());
     chain.Add(kFileName.c_str());
     chain.Add(kFileName.c_str());

     // Select only 3 entries of different clusters in the chain.
     // Two are located in the first tree, one in the second
     TEntryList entries;
     entries.Enter(0); entries.Enter(123); entries.Enter(307);

     tp.reset(new ROOT::TTreeProcessorMT(chain, entries));
  }
  else if (kConstructor == "friends") {
     TChain chain(kTreeName.c_str());
     chain.Add(kFileName.c_str());
     chain.Add(kFileName.c_str());

     // Create an identical friend with an alias
     TChain friendChain(kTreeName.c_str());
     friendChain.Add(kFileName.c_str());
     friendChain.Add(kFileName.c_str());
     chain.AddFriend(&friendChain, "myfriend");

     tp.reset(new ROOT::TTreeProcessorMT(chain));
  }
  else {
     std::cerr << "Error: " << kConstructor << " is not a valid constructor name. Please choose filename, collection, chain, entrylist or friend" << std::endl;
     return 1;
  }
  
  // Request the processing of the tree
  tp->Process([&](TTreeReader &myReader) {
     TTreeReaderValue<std::vector<ROOT::Math::PxPyPzEVector>> tracksRV(myReader, "tracks");
     TTreeReaderValue<int> eventNumRV(myReader, "evtNum");
     std::unique_ptr<TTreeReaderValue<int>> eventNumFriendRVP;
     bool hasFriend = kConstructor == "friends";
     if (hasFriend)
        eventNumFriendRVP.reset(new TTreeReaderValue<int>(myReader, "myfriend.evtNum"));

     auto start = myReader.GetCurrentEntry();

     int numEntries = 0;
     double maxPt = -1.;
     while(myReader.Next()) {
        auto evtNum (*eventNumRV);
        if (hasFriend) {
           auto evtNumFriend (**eventNumFriendRVP);
           if (evtNum != evtNumFriend) {
              std::cerr << "Error checking event number " << evtNum << ", friend's is " << evtNumFriend << std::endl;
              return;
           }
        } 

        auto tracks = *tracksRV;
        for (auto&& track : tracks){
           auto pt = track.Pt();
           if (pt>maxPt) maxPt = pt;
        }

        ++numEntries;
     }

     auto end = myReader.GetCurrentEntry();

     std::lock_guard<std::mutex> lock(mutex);
     if (maxPt > globalMaxPt) globalMaxPt = maxPt;
     std::cout << "[IMT] Finished task with range " << start << "-" << end << std::endl;

     if (kConstructor == "entrylist")
        std::cout << "[IMT] Processed " << numEntries << " entries" << std::endl;
  });

  std::cout << "[IMT] Global max pt is " << globalMaxPt << std::endl; 
}

