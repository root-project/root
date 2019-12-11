/// \file
/// \ingroup tutorial_io
/// This macro shows the usage of TMPIFile to simulate event
/// reconstruction and merging them in parallel.
/// The JetEvent class is in $ROOTSYS/tutorials/tree/JetEvent.h,cxx
///
/// To run this macro do the following:
/// ~~~{.bash}
/// mpirun -np 4 root -b -l -q testTMPIFile.C
/// ~~~
///
/// \macro_code
///
/// \author Taylor Childers, Yunsong Wang

#include "TMPIFile.h"

R__LOAD_LIBRARY(RMPI)  // Work around autoloading issue when ROOT modules are enabled 

#ifdef TMPI_SECOND_RUN

#include <chrono>
#include <sstream>

R__LOAD_LIBRARY(JetEvent_cxx)

/* ---------------------------------------------------------------------------

The idea of TMPIFile is to run N MPI ranks where some ranks are
producing data (called workers), while other ranks are collecting data and
writing it to disk (called collectors). The number of collectors can be
configured and this should be optimized for each workflow and data size.

This example uses a typical event processing loop, where every N events the
TMPIFile::Sync() function is called. This call triggers the local TTree data
to be sent via MPI to the collector rank where it is merged with all the
other worker rank data and written to a TFile.

An MPI Sub-Communictor is created for each collector which equally distributes
the remaining ranks to distribute the workers among collectors.

--------------------------------------------------------------------------- */

void test_tmpi()
{

   Int_t N_collectors = 2;    // specify how many collectors to run
   Int_t sync_rate = 2;       // workers sync every sync_rate events
   Int_t events_per_rank = 6; // total events each rank will produce then exit
   Int_t sleep_mean = 5;      // simulate compute time for event processing
   Int_t sleep_sigma = 2;     // variation in compute time

   // using JetEvent generator to create a data structure
   // these parameters control this generator
   Int_t jetm = 25;
   Int_t trackm = 60;
   Int_t hitam = 200;
   Int_t hitbm = 100;

   std::string treename = "test_tmpi";
   std::string branchname = "event";

   // set output filename
   std::stringstream smpifname;
   smpifname << "/tmp/merged_output_" << getpid() << ".root";

   // Create new TMPIFile, passing the filename, setting read/write permissions
   // and setting the number of collectors.
   // If MPI_INIT has not been called already, the constructor of TMPIFile
   // will call this.
   TMPIFile *newfile = new TMPIFile(smpifname.str().c_str(), "RECREATE", N_collectors);
   // set random number seed that is based on MPI rank
   // this avoids producing the same events in each MPI rank
   gRandom->SetSeed(gRandom->GetSeed() + newfile->GetMPIGlobalRank());

   // only print log messages in MPI Rank 0
   if (newfile->GetMPIGlobalRank() == 0) {
      Info("test_tmpi", " running with parallel ranks:   %d", newfile->GetMPIGlobalSize());
      Info("test_tmpi", " running with collecting ranks: %d", N_collectors);
      Info("test_tmpi", " running with working ranks:    %d", (newfile->GetMPIGlobalSize() - N_collectors));
      Info("test_tmpi", " running with sync rate:        %d", sync_rate);
      Info("test_tmpi", " running with events per rank:  %d", events_per_rank);
      Info("test_tmpi", " running with sleep mean:       %d", sleep_mean);
      Info("test_tmpi", " running with sleep sigma:      %d", sleep_sigma);
      Info("test_tmpi", " running with seed:             %d", gRandom->GetSeed());
   }

   // print filename for each collector Rank
   if (newfile->IsCollector()) {
      Info("Collector", "[%d]\troot output filename = %s", newfile->GetMPIGlobalRank(), smpifname.str().c_str());
   }

   // This if statement splits the run-time functionality of
   // workers and collectors.
   if (newfile->IsCollector()) {
      // Run by collector ranks
      // This will run until all workers have exited
      newfile->RunCollector();
   } else {
      // Run by worker ranks
      // these ranks generate data to be written to TMPIFile

      // create a TTree to store event data
      TTree *tree = new TTree(treename.c_str(), "Event example with Jets");
      // set the AutoFlush rate to be the same as the sync_rate
      // this synchronizes the TTree branch compression
      tree->SetAutoFlush(sync_rate);

      // Create our fake event data generator
      JetEvent *event = new JetEvent;

      // add our branch to the TTree
      tree->Branch(branchname.c_str(), "JetEvent", &event, 8000, 2);

      // monitor timing
      auto sync_start = std::chrono::high_resolution_clock::now();

      // generate the specified number of events
      for (int i = 0; i < events_per_rank; i++) {

         auto start = std::chrono::high_resolution_clock::now();
         // Generate one event
         event->Build(jetm, trackm, hitam, hitbm);

         auto evt_built = std::chrono::high_resolution_clock::now();
         double build_time = std::chrono::duration_cast<std::chrono::duration<double>>(evt_built - start).count();

         Info("Rank", "[%d] [%d]\tevt = %d;\tbuild_time = %f", newfile->GetMPIColor(), newfile->GetMPILocalRank(), i,
              build_time);

         // if our build time was significant, subtract that from the sleep time
         auto adjusted_sleep = (int)(sleep_mean - build_time);
         auto sleep = abs(gRandom->Gaus(adjusted_sleep, sleep_sigma));

         // simulate the time taken by more complicated event generation
         std::this_thread::sleep_for(std::chrono::seconds(int(sleep)));

         // Fill the tree
         tree->Fill();

         // every sync_rate events, call the TMPIFile::Sync() function
         // to trigger MPI collection of local data
         if ((i + 1) % sync_rate == 0) {
            // call TMPIFile::Sync()
            newfile->Sync();

            auto end = std::chrono::high_resolution_clock::now();
            double sync_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - sync_start).count();
            Info("Rank", "[%d] [%d]\tevent collection time: %f", newfile->GetMPIColor(), newfile->GetMPILocalRank(),
                 sync_time);
            sync_start = std::chrono::high_resolution_clock::now();
         }
      }

      // synchronize any left over events
      if (events_per_rank % sync_rate != 0) {
         newfile->Sync();
      }
   }

   // call Close on the file for clean exit.
   Info("Rank", "[%d] [%d]\tclosing file", newfile->GetMPIColor(), newfile->GetMPILocalRank());
   newfile->Close();

   // open file and test contents
   if (newfile->GetMPILocalRank() == 0) {
      TString filename = newfile->GetMPIFilename();
      Info("Rank", "[%d] [%d]\topening file: %s", newfile->GetMPIColor(), newfile->GetMPILocalRank(), filename.Data());
      TFile file(filename.Data());
      if (file.IsOpen()) {
         file.ls();
         TTree *tree = (TTree *)file.Get(treename.c_str());
         if (tree)
            tree->Print();

         Info("Rank", "[%d] [%d]\tfile should have %d events and has %lld", newfile->GetMPIColor(),
              newfile->GetMPILocalRank(), (newfile->GetMPILocalSize() - 1) * events_per_rank, tree->GetEntries());
      }
   }
}

void testTMPIFile(Bool_t secRun)
{
   auto start = std::chrono::high_resolution_clock::now();

   test_tmpi();

   auto end = std::chrono::high_resolution_clock::now();
   double time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
   std::string msg = "Total elapsed time: ";
   msg += std::to_string(time);
   Info("testTMPIFile", "%s", msg.c_str());
   Info("testTMPIFile", "exiting");
}

#else

void testTMPIFile()
{
   Int_t flag;
   MPI_Initialized(&flag);
   if (!flag) {
      MPI_Init(NULL, NULL);
   }

   // Get rank and size
   Int_t rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);


   // Procecss 0 generates JetEvent library
   if (rank == 0) {
      TString tutdir = gROOT->GetTutorialDir();
      gSystem->Exec("cp " + tutdir + "/tree/JetEvent* .");
      gROOT->ProcessLine(".L JetEvent.cxx+");
   }
   // Wait until it's done
   MPI_Barrier(MPI_COMM_WORLD);
   
   gROOT->ProcessLine("#define TMPI_SECOND_RUN yes");
   gROOT->ProcessLine("#include \"" __FILE__ "\"");
   gROOT->ProcessLine("testTMPIFile(true)");
   
   // TMPIFile will do MPI_Finalize() when closing the file
   Int_t finalized = 0;
   MPI_Finalized(&finalized);
   if (!finalized) {
      MPI_Finalize();
   }
}

#endif
