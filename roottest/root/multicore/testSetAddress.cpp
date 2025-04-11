#include "Math/Vector4D.h"
#include "TBranchElement.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TROOT.h"
#include "TTree.h"
#include "TTreeReader.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

using T = std::vector<ROOT::Math::PxPyPzEVector>;

int test_branch() {
  const int kNThreads = 9;
  std::string const kFileName = "just1entry.root";
  std::string const kTreeName = "events";

  std::mutex mutex;

  std::cout << "Parallel processing with " << kNThreads << " threads"
            << std::endl;

  std::atomic<bool> barrier(true);
  std::atomic<int> counter(0);
  std::atomic<bool> barrier1(true);
  std::atomic<int> counter1(0);

  std::atomic<bool> retval(false);

  ROOT::EnableThreadSafety();

  std::vector<std::thread> threads;
  for (int i = 0; i < kNThreads; ++i) {
    threads.emplace_back(([&kTreeName, &kFileName, &mutex, &barrier, &counter,
                           &barrier1, &counter1, &retval]() {
      TFile *floc;
      TTree *tloc;
      TBranchElement *br;

      {
        std::lock_guard<std::mutex> lock(mutex);
        floc = TFile::Open(kFileName.c_str());
        tloc = (TTree *)floc->Get(kTreeName.c_str());
        br = (TBranchElement *)tloc->GetBranch("tracks");
      }

      auto tracksp = new T;

      counter++;
      while (barrier) {
      }

      br->SetAddress(tracksp);

      counter1++;
      while (barrier1) {
      }

      std::lock_guard<std::mutex> lock(mutex);
      br->GetEntry(0);
      auto ptrToPtr = (T **)br->GetAddress();
      auto ptr = *ptrToPtr;
      auto px = ptr->at(0).Px();

      if (px == 0) {
        std::cout << "PX is " << px << std::endl;
        retval = true;
      }
      delete floc;

    }));
  }

  while (counter < kNThreads) {
  }
  barrier = false;

  while (counter1 < kNThreads) {
  }
  barrier1 = false;

  for (auto &&t : threads)
    t.join();

  return retval;
}

int main() { return test_branch(); }
