#include <atomic>
#include <stdio.h>
#include <future>
#include <thread>

#include "TMapFile.h"

#include "TError.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TSystem.h"

static const char *gFileName = "tmapfile-test.map";
std::atomic<bool> gRunWrite = true;
std::atomic<bool> gRunRead = true;

static TString gExecLocation;

int writer(int iterations)
{
   // Create a new memory mapped file. The memory mapped file can be
   // opened in an other process on the same machine and the objects
   // stored in it can be accessed.

   // TMapFile::SetMapAddress(0x125000000);
   auto mfile = TMapFile::Create(gFileName, "RECREATE", 1000000,
                                 "Test memory mapped file with histograms");

   // Create a 1d, a 2d and a profile histogram. These objects will
   // be automatically added to the current directory, i.e. mfile.
   auto hpx = new TH1F("hpx", "This is the px distribution", 100, -4, 4);
   auto hpxpy = new TH2F("hpxpy", "py vs px", 40, -4, 4, 40, -4, 4);
   auto hprof =
      new TProfile("hprof", "Profile of pz versus px", 100, -4, 4, 0, 32);

   // Print status of mapped file
   mfile->Print();

   // Endless loop filling histograms with random numbers
   Float_t px, py, pz;
   int ii = 0;
   while (gRunWrite.load()) {
      gRandom->Rannor(px, py);
      pz = px * px + py * py;
      hpx->Fill(px);
      hpxpy->Fill(px, py);
      auto e = hprof->GetEntries();
      // We need to have all the entries counted, for a TProfile values
      // outside the range are simply ignored.
      if (pz >= 32)
         pz = 16;
      hprof->Fill(px, pz);
      if (e == hprof->GetEntries())
         fprintf(
            stderr,
            "DEBUG ERROR: hprof not moving for entry %lld px=%g py=%g pz=%g\n",
            (long long)e, px, py, pz);
      if (!(ii % 10)) {
         mfile->Update(); // updates all objects in shared memory
         if (!ii)
            mfile->ls();  // print contents of mapped file after first update
      }
      ii++;
      if (ii >= iterations)
         break;
   }
   mfile->Update();
   printf("Written Entries, hpx=%lld, Mean=%g, RMS=%g\n",
          (Long64_t)hpx->GetEntries(), hpx->GetMean(), hpx->GetRMS());
   printf("Written Entries, hprof=%lld, Mean=%g, RMS=%g\n",
          (Long64_t)hprof->GetEntries(), hprof->GetMean(), hprof->GetRMS());
   fprintf(stderr, "Ending at %d iterations\n", ii);
   return 0;
}

void defaultwriter()
{
   writer(std::numeric_limits<int>::max());
}

template <typename T>
struct HistoCheck {
   HistoCheck(const char *what) : name(what) {}

   const char *name = "";
   T *ptr = nullptr;
   Double_t entries = -1;

   T *operator->() { return ptr; }

   int Check(TMapFile *mfile, int i, double maxentries)
   {
      ptr = dynamic_cast<T *>(mfile->Get(name, ptr));
      if (!ptr) {
         Error("reader", "Could not read histogram %s", name);
         return 1;
      }
      double newentries = ptr->GetEntries();
      if (newentries == 0.0) {
         Error("reader", "Histogram %s has no entries.", name);
         return 2;
      }
      if (maxentries && newentries != maxentries && newentries == entries) {
         Error("reader",
               "Histogram %s has made no progress since last iteration (%d) "
               "with %lld entries.",
               name, i - 1, (Long64_t)newentries);
         return 2;
      }
      if (maxentries && newentries > maxentries) {
         Error("reader",
               "Histogram %s has too many entries: %lld at %d iterations", name,
               (Long64_t)newentries, i);
         return 3;
      }
      entries = newentries;
      return 0;
   }

   int FinalCheck(double maxentries)
   {
      if (ptr->GetEntries() != maxentries) {
         Error("FinalCheck", "%s has only %lld entries instead of %lld", name,
               (Long64_t)ptr->GetEntries(), (Long64_t)maxentries);
         return 4;
      }
      return 0;
   }

   void Print(int i)
   {
      printf("%6s Iteration %d, Entries=%lld, Mean=%g, RMS=%g\n", name, i,
             (Long64_t)ptr->GetEntries(), ptr->GetMean(), ptr->GetRMS());
   }
};

int reader(int maxiterations = 0)
{
   // Starts only when the file exists
   while (gRunRead && gSystem->AccessPathName(gFileName, kFileExists))
      gSystem->Sleep(100); // sleep for 0.1 seconds

   // Open the memory mapped file "hsimple.map" in "READ" (default) mode.
   auto mfile = TMapFile::Create(gFileName);

   // Print status of mapped file and list its contents
   mfile->Print();
   mfile->ls();

   // Create pointers to the objects in shared memory.
   HistoCheck<TH1F> hpx("hpx");
   HistoCheck<TH2F> hpxpy("hpxpy");
   HistoCheck<TProfile> hprof("hprof");

   auto ReadCheck = [&hpx, &hpxpy, &hprof](TMapFile *file, int i,
                                           double maxentries) -> int {
      auto res = hpx.Check(file, i, maxentries);
      if (res)
         return res;
      res = hpxpy.Check(file, i, maxentries);
      if (res)
         return res;
      res = hprof.Check(file, i, maxentries);
      if (res)
         return res;
      return 0;
   };

   // Loop displaying the histograms. Once the producer stops this
   // script will break out of the loop.
   int i = 0;
   for (; i < maxiterations; ++i) {
      auto res = ReadCheck(mfile, i, maxiterations);
      if (res)
         return res;
      if (!(i % 100)) {
         hpx.Print(i);
         hpxpy.Print(i);
         hprof.Print(i);
      }
      gSystem->Sleep(100); // sleep for 0.1 seconds
      if (!gRunRead || hpx->GetEntries() == maxiterations) {
         ++i;
         break;
      }
   }

   auto res = ReadCheck(mfile, i, maxiterations);
   hpx.Print(i);
   hpxpy.Print(i);
   hprof.Print(i);

   printf("Final Entries, hpx=%lld, Mean=%g, RMS=%g\n",
          (Long64_t)hpx->GetEntries(), hpx->GetMean(), hpx->GetRMS());

   res += hpx.FinalCheck(maxiterations);
   res += hpxpy.FinalCheck(maxiterations);
   res += hprof.FinalCheck(maxiterations);

   return res;
}

int TMapFileTestWrite(int maxiterations = 1000000)
{
   std::thread write(defaultwriter);

   TString cmd;
#if defined(__ACLIC__)
   cmd.Form("rootn.exe -e \".L TMapFileTest.C\" -e \"reader(%d);\" -b -l -q",
            maxiterations);
#elif defined(__CLING__)
   cmd.Form("rootn.exe -e \".L TMapFileTest.C\" -e \"reader(%d);\" -b -l -q",
            maxiterations);
#else
   cmd.Form("%s read %d", gExecLocation.Data(), maxiterations);
#endif
   auto eres = gSystem->Exec(cmd);
   if (eres != 0) {
      fprintf(stderr, "%s error %d in executing reader\n", gExecLocation.Data(),
              eres);
      gRunWrite = false;
      return eres;
   }

   gRunWrite = false;
   write.join();
   return 0;
}

int TMapFileTest(int maxiterations = 1000000)
{
   gSystem->Unlink(gFileName);

   std::future<int> result =
      std::async([maxiterations]() { return reader(maxiterations); });

   TString cmd;
#if defined(__ACLIC__)
   cmd.Form("rootn.exe -e \".L TMapFileTest.C+\" -e \"writer(%d);\" -b -l -q",
            maxiterations);
#elif defined(__CLING__)
   cmd.Form("rootn.exe -e \".L TMapFileTest.C\" -e \"writer(%d);\" -b -l -q",
            maxiterations);
#else
   cmd.Form("%s write %d", gExecLocation.Data(), maxiterations);
#endif
   auto eres = gSystem->Exec(cmd);
   if (eres != 0) {
      fprintf(stderr, "%s error %d in executing writer\n", gExecLocation.Data(),
              eres);
      gRunWrite = false;
      gRunRead = false;
      return eres;
   }

   gRunWrite = false;
   gRunRead = false;

   auto res = result.get();

   return res;
}

void error_help(int argc, char **argv)
{
   fprintf(stderr, "Invalid aguments:");
   for (int i = 1; i < argc; ++i)
      fprintf(stderr, " %s", argv[i]);
   fprintf(stderr, "\n");
   fprintf(
      stderr,
      "Expected arguments [complete|read|write|testwrite] max_iterations\n");
}

#ifndef __CLING__
int main(int argc, char **argv)
{
   gExecLocation = argv[0];

   int maxiterations = 1000000;
   if (argc > 3) {
      maxiterations = atoi(argv[3]);
      if (maxiterations == 0) {
         error_help(argc, argv);
         return 1;
      }
   }

   if (argc < 2 || strcmp(argv[1], "complete") == 0)
      return TMapFileTest(maxiterations);

   if (strcmp(argv[1], "testwrite") == 0)
      return TMapFileTestWrite(maxiterations);

   if (strcmp(argv[1], "read") == 0)
      return reader(maxiterations);

   if (strcmp(argv[1], "write") == 0)
      return writer(maxiterations);

   error_help(argc, argv);
   return 1;
}
#endif
