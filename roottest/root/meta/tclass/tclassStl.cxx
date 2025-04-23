#include <thread>
#include <vector>
#include "TBufferFile.h"
#include "TROOT.h"
#include <iostream>
#include "TClassRef.h"

#ifndef __CLING__

std::vector<std::vector<std::vector<std::pair<int, int>>>> vec;

void write_vector(int n = 10)
{
   static TClassRef cl("std::vector<std::vector<std::vector<std::pair<int, int>>>>");
   TBufferFile buf(TBuffer::kWrite);
   for (int i = 0; i < n; ++i) {
      cl->Property(); // Force a call at each iteration to increase reproducibility of race.
      cl->Streamer(&vec, buf);
      buf.Reset();
   }
   // std::cout << "Thread done\n";
}

void tclassStlImpl(int repeat = 10, int nThreads = 100, int internalRepeat = 10, int nElem = 100)
{
   ROOT::EnableThreadSafety();
   vec.resize(nElem);
   for (auto &inner : vec) {
      inner.resize(nElem);
      for (auto &vp : inner)
         vp.resize(nElem / 10);
   }

   for (int r = 0; r < repeat; ++r) {
      std::vector<std::thread> ths;
      for (int i = 0; i < nThreads; ++i)
         ths.emplace_back(write_vector, internalRepeat);
      for (int i = 0; i < nThreads; ++i)
         ths[i].join();
      std::cout << "Main loop done: " << r << "\n";
   }
}

void tclassStl(int repeat = 10, int nThreads = 100, int internalRepeat = 10, int nElem = 100)
{
   tclassStlImpl(repeat, nThreads, internalRepeat, nElem);
   // Force  instantiation of the std::vector
   gROOT->ProcessLine("std::vector<std::vector<std::vector<std::pair<int, int>>>> vec;");
   tclassStlImpl(repeat, nThreads, internalRepeat, nElem);
}

#else
void tclassStl(int repeat = 10, int nThreads = 100, int internalRepeat = 10, int nElem = 100);
#endif
