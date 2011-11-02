//
// Macro to run examples of analysis on PROOF, corresponding to the TSelector
// implementations found under <ROOTSYS>/tutorials/proof .
// This macro uses an existing PROOF session or starts one at the indicated URL.
// In the case non existing PROOF session is found and no URL is given, the
// macro tries to start a local PROOF session.
//
// To run the macro:
//
//   root[] .L proof/runProof.C+
//   root[] runProof("<analysis>")
//
//   Currently available analysis are (to see how all this really works check
//   the scope for the specified option inside the macro):
//
//   1. "simple"
//
//      Selector: ProofSimple.h.C
//
//      root[] runProof("simple")
//
//      This will create a local PROOF session and run an analysis filling
//      100 histos with 100000 gaussian random numbers, and displaying them
//      in a canvas with 100 pads (10x10).
//      The number of histograms can be passed as argument 'nhist' to 'simple',
//      e.g. to fill 16 histos with 1000000 entries use
//
//      root[] runProof("simple(nevt=1000000,nhist=16)")
//
//      The argument nhist3 controls the creation of 3d histos to simulate
//      merging load. By default, no 3D hitogram is created.
//
//   2. "h1"
//
//      Selector: tutorials/tree/h1analysis.h.C
//
//      root[] runProof("h1")
//
//      This runs the 'famous' H1 analysis from $ROOTSYS/tree/h1analysis.C.h.
//      By default the data are read from the HTTP server at root.cern.ch,
//      the data source can be changed via the argument 'h1src', e.g.
//
//      root[] runProof("h1,h1src=/data/h1")
//
//      (the directory specified must contain the 4 H1 files).
//
//      The 'h1' example is also used to show how to use entry-lists in PROOF.
//      To fill the list for the events used for the final plots add the option
//      'fillList':
//
//      root[] runProof("h1,fillList")
//
//      To use the list previously created for the events used for the 
//      final plots add the option 'useList':
//
//      root[] runProof("h1,useList")
//
//  3. "event"
//
//      Selector: ProofEvent.h,.C
//
//      This is an example of using PROOF par files.
//      It runs event generation and simple analysis based on the 'Event'
//      class found under test.
//
//      root[] runProof("event")
//
//  4. "eventproc"
//
//      Selector: ProofEventProc.h.C
//
//      This is an example of using PROOF par files and process 'event'
//      data from the ROOT HTTP server. It runs the ProofEventProc selector
//      which is derived from the EventTree_Proc one found under
//      test/ProofBench. The following specific arguments are available:
//      - 'readall'  to read the whole event, by default only the branches
//                   needed by the analysis are read (read 25% more bytes)
//      - 'datasrc=<dir-with-files>' to read the files from another server,
//                   the files must be named 'event_<num>.root' where <num>=1,2,...
//        or
//      - 'datasrc=<file-with-files>' to take the file content from a text file,
//                   specified one file per line; usefull when testing differences
//                   between several sources and distributions
//      - 'files=N'  to change the number of files to be analysed (default
//                   is 10, max is 50 for the HTTP server).
//      - 'uneven'   to process uneven entries from files following the scheme
//                   {50000,5000,5000,5000,5000} and so on
//
//      root[] runProof("eventproc")
//
//  5. "pythia8"
//
//      Selector: ProofPythia.h.C
//
//      This runs Pythia8 generation based on main03.cc example in Pythia 8.1 
//
//      To run this analysis ROOT must be configured with pythia8.
//
//      Note that before executing this analysis, the env variable PYTHIA8
//      must point to the pythia8100 (or newer) directory, in particular,
//      $PYTHIA8/xmldoc must contain the file Index.xml. The tutorial assumes
//      that the Pythia8 directory is the same on all machines, i.e. local
//      and worker ones.
//
//      root[] runProof("pythia8")
//
//  6. "ntuple"
//
//      Selector: ProofNtuple.h.C
//
//      This is an example of final merging via files created on the workers,
//      using TProofOutputFile. The final file is called ProofNtuple.root
//      and it is created in the directory where the tutorial is run. If
//      the PROOF cluster is remote, the file is received by a local xrootd
//      daemon started for the purpose. Because of this, this example can be
//      run only on unix clients.
//
//      root[] runProof("ntuple")
//
//  7. "dataset"
//
//      Selector: ProofNtuple.h.C
//
//      This is an example of automatic creation of a dataset from files
//      created on the workers, using TProofOutputFile. The dataset is
//      called testNtuple and it is automatically registered and verified.
//      The files contain the same ntuple as in previous example/tutorial 6
//      (the same selector ProofNTuple is used with a slightly different
//      configuration). The dataset is then used to produce the same plot
//      as in 5 but using the DrawSelect methods of PROOF, which also show
//      how to set style, color and other drawing attributes in PROOF.
//      Depending on the relative worker perforance, some of the produced
//      files may result in having no entries. If this happens, the file
//      will be added to the missing (skipped) file list. Increasing the
//      number of events (via nevt=...) typically solves this issue.
//
//      root[] runProof("dataset")
//
//  8. "friends"
//
//      Selectors: ProofFriends.h(.C), ProofAux.h(.C)
//
//      This is an example of TTree friend processing in PROOF. It also shows
//      how to use the TPacketizerFile to steer creation of files.
//
//      root[] runProof("friends")
//
//      The trees are by default created in separate files; to create
//      them in the same file use option 'samefile', e.g.
//
//      root[] runProof("friends(samefile)")
//
//   9. "simplefile"
//
//      Selector: ProofSimpleFile.h.C
//
//      root[] runProof("simplefile")
//
//      This will create a local PROOF session and run an analysis filling
//      16+16 histos with 100000 gaussian random numbers. The merging of
//      these histos goes via file; 16 histos are saved in the top directory,
//      the other 16 into a subdirectory called 'blue'. The final display
//      is done in two canvanses, one for each set of histograms and with
//      16 pads each (4x4).
//      The number of histograms in each set can be passed as argument
//      'nhist' to 'simplefile', e.g. to fill 25 histos with 1000000 entries use
//
//      root[] runProof("simplefile(nevt=1000000,nhist=25)")
//
//
//   General arguments
//   -----------------
//
//   The following arguments are valid for all examples (the ones specific
//   to each tutorial have been explained above)
//
//   0. ACLiC mode
//      By default all processing is done with ACLiC mode '+', i.e. compile
//      if changed. However, this may lead to problems if the available
//      selector libs were compiled in previous sessions with a different
//      set of loaded libraries (this is a general problem in ROOT). When
//      this happens the best solution is to force recompilation (ACLiC
//      mode '++'). To do this just add one or more '+' to the name of the
//      tutorial, e.g. runProof("simple++")
//
//   1. debug=[what:]level
//
//      Controls verbosity; 'level' is an integer number and the optional string
//      'what' one or more of the enum names in TProofDebug.h .
//      e.g. runProof("eventproc(debug=kPacketizer|kCollect:2)") runs 'eventproc' enabling
//           all printouts matching TProofDebug::kPacketizer and having level
//           equal or larger than 2 .
//
//   2. nevt=N
//
//      Set the number of entries to N
//      e.g. runProof("simple(nevt=1000000000)") runs simple with 1000000000
//
//   3. asyn
//
//      Run in non blocking mode
//      e.g. root[] runProof("h1(asyn)")
//
//   4. nwrk=N
//
//      Set the number of active workers to N, usefull to test performance
//      on a remote cluster where control about the number of workers is
//      not possible, e.g. runProof("event(nwrk=2)") runs 'event' with
//      2 workers.
//
//   5. punzip
//
//      Use parallel unzipping in reading files where relevant
//      e.g. root[] runProof("eventproc(punzip)")
//
//   6. cache=<bytes> (or <kbytes>K or <mbytes>M) 
//
//      Change the size of the tree cache; 0 or <0 disables the cache,
//      value cane be in bytes (no suffix), kilobytes (suffix 'K') or
//      megabytes (suffix 'M'), e.g. root[] runProof("eventproc(cache=0)") 
//
//   7. submergers[=S]
//
//      Enabling merging via S submergers or the optimal number if S is
//      not specified, e.g. root[] runProof("simple(hist=1000,submergers)") 
//
//   8. rateest=average
//
//      Enable processed entries estimation for constant progress reporting based on
//      the measured average. This may screw up the progress bar in some cases, which
//      is the reason why it is not on by default .
//      e.g. root[] runProof("eventproc(rateest=average)")
//
//   In all cases, to run on a remote PROOF cluster, the master URL must
//   be passed as second argument; e.g.
//
//      root[] runProof("simple","master.do.main")
//
//   In the case of local running it is possible to specify the number of
//   workers to start as third argument (the default is the number of cores
//   of the machine), e.g.
//
//      root[] runProof("simple",0,4)
//
//   will start 4 workers. Note that the real number of workers is changed
//   only the first time you call runProof into a ROOT session. Following
//   calls can reduce the number of active workers, but not increase it.
//   For example, in the same session of the call above starting 4 workers,
//   this
//
//   root[] runProof("simple",0,8)
//
//   will still use 4 workers, while this
//
//   root[] runProof("simple",0,2)
//
//   will disable 2 workers and use the other 2.
//


#include "TCanvas.h"
#include "TChain.h"
#include "TDrawFeedback.h"
#include "TDSet.h"
#include "TEnv.h"
#include "TEntryList.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFrame.h"
#include "THashList.h"
#include "TList.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TProof.h"
#include "TProofDebug.h"
#include "TString.h"

#include "getProof.C"
void plotNtuple(TProof *p, const char *ds, const char *ntptitle);
int getDebugEnum(const char *what);

TDrawFeedback *fb = 0;

// Variable used to locate the Pythia8 directory for the Pythia8 example
const char *pythia8dir = 0;
const char *pythia8data = 0;

void runProof(const char *what = "simple",
              const char *url = "proof://localhost:40000",
              Int_t nwrks = -1)
{
#ifdef __CINT__
   Printf("runProof: this script can only be executed via ACliC:");
   Printf("runProof:      root [] .x <path>/runProof.C+");
   Printf("runProof: or   root [] .L <path>/runProof.C+");
   Printf("runProof:      root [] runProof(...)");
   return;
#endif
   gEnv->SetValue("Proof.StatsHist",1);

   // Temp dir for PROOF tutorials
   // Force "/tmp/<user>" whenever possible to avoid length problems on MacOsX
   TString tmpdir("/tmp");
   if (gSystem->AccessPathName(tmpdir, kWritePermission)) tmpdir = gSystem->TempDirectory();
   TString us;
   UserGroup_t *ug = gSystem->GetUserInfo(gSystem->GetUid());
   if (!ug) {
      Printf("runProof: could not get user info");
      return;
   }
   us.Form("/%s", ug->fUser.Data());
   if (!tmpdir.EndsWith(us.Data())) tmpdir += us;
   gSystem->mkdir(tmpdir.Data(), kTRUE);
   if (gSystem->AccessPathName(tmpdir, kWritePermission)) {
      Printf("runProof: unable to get a writable tutorial directory (tried: %s)"
             " - cannot continue", tmpdir.Data());
      return;
   }
   TString tutdir = Form("%s/.proof-tutorial", tmpdir.Data());
   if (gSystem->AccessPathName(tutdir)) {
      Printf("runProof: creating the temporary directory"
                " for the tutorial (%s) ... ", tutdir.Data());
      if (gSystem->mkdir(tutdir, kTRUE) != 0) {
         Printf("runProof: could not assert / create the temporary directory"
                " for the tutorial (%s)", tutdir.Data());
         return;
      }
   }

   // For the Pythia8 example we need to set some environment variable;
   // This must be done BEFORE starting the PROOF session
   if (what && !strncmp(what, "pythia8", 7)) {
      // We assume that the remote location of Pythia8 is the same as the local one
      pythia8dir = gSystem->Getenv("PYTHIA8");
      if (!pythia8dir || strlen(pythia8dir) <= 0) {
         Printf("runProof: pythia8: environment variable PYTHIA8 undefined:"
                  " it must contain the path to pythia81xx root directory (local and remote) !");
         return;
      }
      pythia8data = gSystem->Getenv("PYTHIA8DATA");
      if (!pythia8data || strlen(pythia8data) <= 0) {
         gSystem->Setenv("PYTHIA8DATA", Form("%s/xmldoc", pythia8dir));
         pythia8data = gSystem->Getenv("PYTHIA8DATA");
         if (!pythia8data || strlen(pythia8data) <= 0) {
            Printf("runProof: pythia8: environment variable PYTHIA8DATA undefined:"
                   " it one must contain the path to pythia81xx/xmldoc"
                   " subdirectory (local and remote) !");
            return;
         }
      }
      TString env = Form("echo export PYTHIA8=%s; export PYTHIA8DATA=%s",
                         pythia8dir, pythia8data);
      TProof::AddEnvVar("PROOF_INITCMD", env.Data());
   }

   Printf("tutorial dir:\t%s", tutdir.Data());

   // Get the PROOF Session
   TProof *proof = getProof(url, nwrks, tutdir.Data(), "ask");
   if (!proof) {
      Printf("runProof: could not start/attach a PROOF session");
      return;
   }

   // Determine locality of this session
   Bool_t isProofLocal = kFALSE;
   TUrl uu(url);
   if (!strcmp(uu.GetHost(), "localhost") || proof->IsLite() ||
       !strcmp(uu.GetHostFQDN(), TUrl(gSystem->HostName()).GetHostFQDN())) {
      isProofLocal = kTRUE;
   }
#ifdef WIN32
   if (isProofLocal && what && !strcmp(what, "ntuple", 6)) {
      // Not support on windows
      Printf("runProof: the 'ntuple' example needs to run xrootd to receive the output file, \n"
             "          but xrootd is not supported on Windows - cannot continue");
      return;
   }
#endif

   TString proofsessions(Form("%s/sessions",tutdir.Data()));
   // Save tag of the used session
   FILE *fs = fopen(proofsessions.Data(), "a");
   if (!fs) {
      Printf("runProof: could not create files for sessions tags");
   } else {
      fprintf(fs,"session-%s\n", proof->GetSessionTag());
      fclose(fs);
   }
   if (!proof) {
      Printf("runProof: could not start/attach a PROOF session");
      return;
   }

   // Set the number of workers (may only reduce the number of active workers
   // in the session)
   if (nwrks > 0)
      proof->SetParallel(nwrks);

   // Where is the code to run
   char *rootbin = gSystem->Which(gSystem->Getenv("PATH"), "root.exe", kExecutePermission);
   if (!rootbin) {
      Printf("runProof: root.exe not found: please check the environment!");
      return;
   }
   TString rootsys(gSystem->DirName(rootbin));
   rootsys = gSystem->DirName(rootsys);
   TString tutorials(Form("%s/tutorials", rootsys.Data()));
   delete[] rootbin;

   // Create feedback displayer
   if (!fb) {
      new TCanvas("PROOF_EventsHist_canvas", "Events per Worker", 100, 600, 600, 400);
      fb = new TDrawFeedback(proof);
   }
   if (!proof->GetFeedbackList() || !proof->GetFeedbackList()->FindObject("PROOF_EventsHist")) {
      // Number of events per worker
      proof->AddFeedback("PROOF_EventsHist");
   }

   // Parse 'what'; it is in the form 'analysis(arg1,arg2,...)'
   TString args(what);
   args.ReplaceAll("("," "); 
   args.ReplaceAll(")"," "); 
   args.ReplaceAll(","," "); 
   Ssiz_t from = 0;
   TString act, tok;
   if (!args.Tokenize(act, from, " ")) {
      // Cannot continue
      Printf("runProof: action not found: check your arguments (%s)", what);
      return;
   }
   // Extract ACLiC mode
   TString aMode = "+";
   if (act.EndsWith("+")) {
      aMode += "+";
      while (act.EndsWith("+")) { act.Remove(TString::kTrailing,'+'); }
   }
   Printf("runProof: %s: ACLiC mode: '%s'", act.Data(), aMode.Data());

   // Parse out number of events and  'asyn' option, used almost by every test
   TString aNevt, aNwrk, opt, sel, punzip("off"), aCache, aH1Src("http://root.cern.ch/files/h1"),
           aDebug, aDebugEnum, aRateEst;
   Long64_t suf = 1;
   Int_t aSubMg = -1;
   Bool_t fillList = kFALSE, useList = kFALSE;
   while (args.Tokenize(tok, from, " ")) {
      // Debug controllers
      if (tok.BeginsWith("debug=")) {
         aDebug = tok;
         aDebug.ReplaceAll("debug=","");
         Int_t icol = kNPOS;
         if ((icol = aDebug.Index(":")) != kNPOS) {
            aDebugEnum = aDebug(0, icol);
            aDebug.Remove(0, icol+1);
         }
         if (!aDebug.IsDigit()) {
            Printf("runProof: %s: error parsing the 'debug=' option (%s) - ignoring", act.Data(), tok.Data());
            aDebug = "";
            aDebugEnum = "";
         }
      }
      // Number of events
      if (tok.BeginsWith("nevt=")) {
         aNevt = tok;
         aNevt.ReplaceAll("nevt=","");
         if (!aNevt.IsDigit()) {
            Printf("runProof: %s: error parsing the 'nevt=' option (%s) - ignoring", act.Data(), tok.Data());
            aNevt = "";
         }
      }
      // Sync or async ?
      if (tok.BeginsWith("asyn"))
         opt = "ASYN";
      // Number of workers
      if (tok.BeginsWith("nwrk=")) {
         aNwrk = tok;
         aNwrk.ReplaceAll("nwrk=","");
         if (!aNwrk.IsDigit()) {
            Printf("runProof: %s: error parsing the 'nwrk=' option (%s) - ignoring", act.Data(), tok.Data());
            aNwrk = "";
         }
      }
      // Parallel unzipping ?
      if (tok.BeginsWith("punzip"))
         punzip = "on";
      // Number of workers
      if (tok.BeginsWith("cache=")) {
         aCache = tok;
         aCache.ReplaceAll("cache=","");
         if (aCache.EndsWith("k")) { aCache.Remove(TString::kTrailing, 'k'); suf = 1024; }
         if (aCache.EndsWith("K")) { aCache.Remove(TString::kTrailing, 'K'); suf = 1024; }
         if (aCache.EndsWith("M")) { aCache.Remove(TString::kTrailing, 'M'); suf = 1024*1024; }
         if (!aCache.IsDigit()) {
            Printf("runProof: %s: error parsing the 'cache=' option (%s) - ignoring", act.Data(), tok.Data());
            aCache = "";
         }
      }
      // Use submergers?
      if (tok.BeginsWith("submergers")) {
         tok.ReplaceAll("submergers","");
         aSubMg = 0;
         if (tok.BeginsWith("=")) {
            tok.ReplaceAll("=","");
            if (tok.IsDigit()) aSubMg = tok.Atoi();
         }
      }
      // H1: use entry-lists ?
      if (tok.BeginsWith("useList")) {
         useList = kTRUE;
      }
      if (tok.BeginsWith("fillList")) {
         fillList = kTRUE;
         opt += "fillList";
      }
      // H1: change location of files?
      if (tok.BeginsWith("h1src=")) {
         tok.ReplaceAll("h1src=","");
         if (!(tok.IsNull())) aH1Src = tok;
         Printf("runProof: %s: reading data files from '%s'", act.Data(), aH1Src.Data());
      }
      // Rate estimation technique
      if (tok.BeginsWith("rateest=")) {
         tok.ReplaceAll("rateest=","");
         if (!(tok.IsNull())) aRateEst = tok;
         Printf("runProof: %s: progress-bar rate estimation option: '%s'", act.Data(), aRateEst.Data());
      }
   }
   Long64_t nevt = (aNevt.IsNull()) ? -1 : aNevt.Atoi();
   Long64_t nwrk = (aNwrk.IsNull()) ? -1 : aNwrk.Atoi();
   from = 0;

   // Set number workers
   if (nwrk > 0) {
      if (proof->GetParallel() < nwrk) {
         Printf("runProof: %s: request for a number of workers larger then available - ignored", act.Data());
      } else {
         proof->SetParallel(nwrk);
      }
   }

   // Debug controllers
   if (!aDebug.IsNull()) {
      Int_t dbg = aDebug.Atoi();
      Int_t scope = TProofDebug::kAll;
      if (!aDebugEnum.IsNull()) scope = getDebugEnum(aDebugEnum.Data());
      proof->SetLogLevel(dbg, scope);
      Printf("runProof: %s: verbose mode for '%s'; level: %d", act.Data(), aDebugEnum.Data(), dbg);
   }

   // Have constant progress reporting based on estimated info
   // (NB: may screw up the progress bar in some cases)
   if (aRateEst == "average")
      proof->SetParameter("PROOF_RateEstimation", aRateEst);

   // Parallel unzip
   if (punzip == "on") {
      proof->SetParameter("PROOF_UseParallelUnzip", (Int_t)1);
      Printf("runProof: %s: parallel unzip enabled", act.Data());
   } else {
      proof->SetParameter("PROOF_UseParallelUnzip", (Int_t)0);
   }

   // Tree cache
   if (!aCache.IsNull()) {
      Long64_t cachesz = aCache.Atoi() * suf;
      if (cachesz <= 0) {
         proof->SetParameter("PROOF_UseTreeCache", (Int_t)0);
         Printf("runProof: %s: disabling tree cache", act.Data());
      } else {
         proof->SetParameter("PROOF_UseTreeCache", (Int_t)1);
         proof->SetParameter("PROOF_CacheSize", cachesz);
         Printf("runProof: %s: setting cache size to %lld", act.Data(), cachesz);
      }
   } else {
      // Use defaults
      proof->DeleteParameters("PROOF_UseTreeCache");
      proof->DeleteParameters("PROOF_CacheSize");
   }

   // Enable submergers, if required
   if (aSubMg >= 0) {
      gProof->SetParameter("PROOF_UseMergers", aSubMg);
      if (aSubMg > 0) {
         Printf("runProof: %s: enabling merging via %d sub-mergers", act.Data(), aSubMg);
      } else {
         Printf("runProof: %s: enabling merging via sub-mergers (optimal number)", act.Data());
      }
   } else {
      gProof->DeleteParameters("PROOF_UseMergers");
   }

   // Action
   if (act == "simple") {
      // ProofSimple is an example of non-data driven analysis; it
      // creates and fills with random numbers a given number of histos

      // Default 10000 events
      nevt = (nevt < 0) ? 100000 : nevt;
      // Find out the number of histograms
      TString aNhist, aNhist3;
      while (args.Tokenize(tok, from, " ")) {
         // Number of histos
         if (tok.BeginsWith("nhist=")) {
            aNhist = tok;
            aNhist.ReplaceAll("nhist=","");
            if (!aNhist.IsDigit()) {
               Printf("runProof: error parsing the 'nhist=' option (%s) - ignoring", tok.Data());
               aNhist = "";
            }
         } else if (tok.BeginsWith("nhist3=")) {
            aNhist3 = tok;
            aNhist3.ReplaceAll("nhist3=","");
            if (!aNhist3.IsDigit()) {
               Printf("runProof: error parsing the 'nhist3=' option (%s) - ignoring", tok.Data());
               aNhist3 = "";
            }
         }
      }
      Int_t nhist = (aNhist.IsNull()) ? 100 : aNhist.Atoi();
      Int_t nhist3 = (aNhist3.IsNull()) ? -1 : aNhist3.Atoi();
      Printf("\nrunProof: running \"simple\" with nhist= %d, nhist3=%d and nevt= %lld\n", nhist, nhist3, nevt);

      // The number of histograms is added as parameter in the input list
      proof->SetParameter("ProofSimple_NHist", (Long_t)nhist);
      // The number of histograms is added as parameter in the input list
      if (nhist3 > 0) proof->SetParameter("ProofSimple_NHist3", (Long_t)nhist3);
      // The selector string
      sel.Form("%s/proof/ProofSimple.C%s", tutorials.Data(), aMode.Data());
      //
      // Run it for nevt times
      proof->Process(sel.Data(), nevt, opt);

   } else if (act == "h1") {
      // This is the famous 'h1' example analysis run on Proof reading the
      // data from the ROOT http server.

      // Create the chain
      TChain *chain = new TChain("h42");
      chain->Add(TString::Format("%s/dstarmb.root", aH1Src.Data()));
      chain->Add(TString::Format("%s/dstarp1a.root", aH1Src.Data()));
      chain->Add(TString::Format("%s/dstarp1b.root", aH1Src.Data()));
      chain->Add(TString::Format("%s/dstarp2.root", aH1Src.Data()));
      chain->ls();
      // We run on Proof
      chain->SetProof();
      // Set entrylist, if required
      if (useList) {
         TString eln("elist"), elfn("elist.root");
         if (gSystem->AccessPathName(elfn)) {
            Printf("\nrunProof: asked to use an entry list but '%s' not found or not readable", elfn.Data());
            Printf("\nrunProof: did you forget to run with 'fillList=%s'?\n", elfn.Data());
         } else {
            TFile f(elfn);
            if (!(f.IsZombie())) {
               TEntryList *elist = (TEntryList *)f.Get(eln);
               if (elist) {
                  elist->SetDirectory(0); //otherwise the file destructor will delete elist
                  chain->SetEntryList(elist);
               } else {
                  Printf("\nrunProof: could not find entry-list '%s' in file '%s': ignoring",
                         eln.Data(), elfn.Data());
               }
            } else {
               Printf("\nrunProof: requested entry-list file '%s' not existing (or not readable):"
                      " ignoring", elfn.Data());
            }
         }
      }
      // The selector
      sel.Form("%s/tree/h1analysis.C%s", tutorials.Data(), aMode.Data());
      // Run it 
      Printf("\nrunProof: running \"h1\"\n");
      chain->Process(sel.Data(),opt);
      // Cleanup the input list
      gProof->ClearInputData("elist");
      gProof->ClearInputData("elist.root");
      TIter nxi(gProof->GetInputList());
      TObject *o = 0;
      while ((o = nxi())) {
         if (!strncmp(o->GetName(), "elist", 5)) {
            gProof->GetInputList()->Remove(o);
            delete o;
         }
      }

   } else if (act == "pythia8") {

      TString path(Form("%s/Index.xml", pythia8data));
      gSystem->ExpandPathName(path);
      if (gSystem->AccessPathName(path)) {
         Printf("runProof: pythia8: PYTHIA8DATA directory (%s) must"
                " contain the Index.xml file !", pythia8data);
         return;
      }
      TString pythia8par = TString::Format("%s/proof/pythia8.par", tutorials.Data());
      if (gSystem->AccessPathName(pythia8par.Data())) {
         Printf("runProof: pythia8: par file not found (tried %s)", pythia8par.Data());
         return;
      }
      proof->UploadPackage(pythia8par);
      proof->EnablePackage("pythia8");
      // Show enabled packages
      proof->ShowEnabledPackages();
      Printf("runProof: pythia8: check settings:");
      proof->Exec(".!echo hostname = `hostname`; echo \"ls pythia8:\"; ls pythia8");
      // Loading libraries needed
      if (gSystem->Load("libEG.so") < 0) {
         Printf("runProof: pythia8: libEG not found \n");
         return;
      }
      if (gSystem->Load("libEGPythia8.so") < 0) {
         Printf("runProof: pythia8: libEGPythia8 not found \n");
         return;
      }
      // Setting the default number of events, if needed
      nevt = (nevt < 0) ? 100 : nevt;
      Printf("\nrunProof: running \"Pythia01\" nevt= %lld\n", nevt);
      // The selector string
      sel.Form("%s/proof/ProofPythia.C%s", tutorials.Data(), aMode.Data());
      // Run it for nevt times
      proof->Process(sel.Data(), nevt);

  } else if (act == "event") {

      TString eventpar = TString::Format("%s/proof/event.par", tutorials.Data());
      if (gSystem->AccessPathName(eventpar.Data())) {
         Printf("runProof: event: par file not found (tried %s)", eventpar.Data());
         return;
      }

      proof->UploadPackage(eventpar);
      proof->EnablePackage("event");
      Printf("Enabled packages...\n");
      proof->ShowEnabledPackages(); 

      // Setting the default number of events, if needed
      nevt = (nevt < 0) ? 100 : nevt;
      Printf("\nrunProof: running \"event\" nevt= %lld\n", nevt);
      // The selector string
      sel.Form("%s/proof/ProofEvent.C%s", tutorials.Data(), aMode.Data());
      // Run it for nevt times
      proof->Process(sel.Data(), nevt);

  } else if (act == "eventproc") {

      TString eventpar = TString::Format("%s/proof/event.par", tutorials.Data());
      gSystem->ExpandPathName(eventpar);
      if (gSystem->AccessPathName(eventpar.Data())) {
         Printf("runProof: event: par file not found (tried %s)", eventpar.Data());
         return;
      }

      proof->UploadPackage(eventpar);
      proof->EnablePackage("event");
      Printf("Enabled packages...\n");
      proof->ShowEnabledPackages(); 

      // Extract the number of files to process, data source and
      // other parameters controlling the run ...
      Bool_t uneven = kFALSE;
      TString aFiles, aDataSrc("http://root.cern.ch/files/data"), aPartitions;
      proof->SetParameter("ProofEventProc_Read", "optimized");
      while (args.Tokenize(tok, from, " ")) {
         // Number of events
         if (tok.BeginsWith("files=")) {
            aFiles = tok;
            aFiles.ReplaceAll("files=","");
            if (!aFiles.IsDigit()) {
               Printf("runProof: error parsing the 'files=' option (%s) - ignoring", tok.Data());
               aFiles = "";
            }
         } else if (tok.BeginsWith("datasrc=")) {
            tok.ReplaceAll("datasrc=","");
            if (tok.IsDigit()) {
               Printf("runProof: error parsing the 'datasrc=' option (%s) - ignoring", tok.Data());
            } else {
               aDataSrc = tok;
               Printf("runProof: reading files from: %s", aDataSrc.Data());
            }
         } else if (tok == "readall") {
            proof->SetParameter("ProofEventProc_Read", "readall");
            Printf("runProof: eventproc: reading the full event");
         } else if (tok == "uneven") {
            uneven = kTRUE;
         } else if (tok.BeginsWith("partitions=")) {
            tok.ReplaceAll("partitions=","");
            if (tok.IsDigit()) {
               Printf("runProof: error parsing the 'partitions=' option (%s) - ignoring", tok.Data());
            } else {
               aPartitions = tok;
               Printf("runProof: partitions: %s included in packetizer operations", aPartitions.Data());
            }
         }
      }
      Int_t nFiles = (aFiles.IsNull()) ? 10 : aFiles.Atoi();
         Printf("runProof: found aFiles: '%s', nFiles: %d", aFiles.Data(), nFiles);
      if (nFiles > 50) {
         Printf("runProof: max number of files is 50 - resizing request");
         nFiles = 50;
      }

      // We create the chain now
      TChain *c = new TChain("EventTree");

      FileStat_t fst;
      if (gSystem->GetPathInfo(aDataSrc, fst) == 0 && R_ISREG(fst.fMode) &&
         !gSystem->AccessPathName(aDataSrc, kReadPermission)) {
         // It is a local file, we get the TFileCollection and we inject it into the chain
         TFileCollection *fc = new TFileCollection("", "", aDataSrc, nFiles);
         c->AddFileInfoList(fc->GetList());
         delete fc;

      } else {

         // Tokenize the source: if more than 1 we rotate the assignment. More sources can be specified
         // separating them by a '|'
         TObjArray *dsrcs = aDataSrc.Tokenize("|");
         Int_t nds = dsrcs->GetEntries();

         // Fill the chain
         Int_t i = 1, k = 0;
         TString fn;
         for (i = 1; i <= nFiles; i++) {
            k = (i - 1) % nds;
            TObjString *os = (TObjString *) (*dsrcs)[k];
            if (os) {
               fn.Form("%s/event_%d.root", os->GetName(), i);
               if (uneven) {
                  if ((i - 1)%5 == 0)
                     c->AddFile(fn.Data(), 50000);
                  else
                     c->AddFile(fn.Data(), 5000);
               } else {
                  c->AddFile(fn.Data());
               }
            }
         }
         dsrcs->SetOwner();
         delete dsrcs;
      }
      // Show the chain
      c->ls();
      c->SetProof();

      // Only validate the files really needed for the analysis
      proof->SetParameter("PROOF_ValidateByFile", 1);

      // Send over the  partition information, if any
      if (!aPartitions.IsNull()) {
         aPartitions.ReplaceAll("|", ",");
         proof->SetParameter("PROOF_PacketizerPartitions", aPartitions);
      }

      // The selector
      sel.Form("%s/proof/ProofEventProc.C%s", tutorials.Data(), aMode.Data());
      // Run it
      Printf("\nrunProof: running \"eventproc\"\n");
      c->Process(sel.Data(), opt, nevt);

   } else if (act == "ntuple") {

      // ProofNtuple is an example of non-data driven analysis; it
      // creates and fills a disk resident ntuple with automatic file merging 

      // Set the default number of events, if needed
      nevt = (nevt < 0) ? 1000 : nevt;
      Printf("\nrunProof: running \"ntuple\" with nevt= %lld\n", nevt);

      // Output file
      TString fout = TString::Format("%s/ProofNtuple.root", gSystem->WorkingDirectory());
      // Cleanup any existing instance of the output file
      gSystem->Unlink(fout);

      if (!isProofLocal) {
         // Setup a local basic xrootd to receive the file
         Bool_t xrdok = kFALSE;
         Int_t port = 9000;
         while (port < 9010) {
            if (checkXrootdAt(port) != 1) {
               if (startXrootdAt(port, gSystem->WorkingDirectory(), kTRUE) == 0) {
                  xrdok = kTRUE;
                  break;
               }
            }
            port++;
         }
         if (!xrdok) {
            Printf("runProof: could not start basic xrootd on ports 9000-9009 - cannot continue");
            return;
         }
         fout.Insert(0, TString::Format("root://%s:%d/", TUrl(gSystem->HostName()).GetHostFQDN(), port));
         // Make a copy of the files on the master before merging
         proof->AddInput(new TNamed("PROOF_OUTPUTFILE_LOCATION", "LOCAL"));
      }
      proof->AddInput(new TNamed("PROOF_OUTPUTFILE", fout.Data()));

      // The selector string
      sel.Form("%s/proof/ProofNtuple.C%s", tutorials.Data(), aMode.Data());

      // Run it for nevt times
      proof->Process(sel.Data(), nevt, opt);

   } else if (act == "dataset") {

      // This is an example of analysis creating data files on each node which are
      // automatically registered as dataset; the newly created dataset is used to create
      // the final plots. The data are of the same type as for the 'ntuple' example.
      // Selector used: ProofNtuple

      // Set the default number of events, if needed
      nevt = (nevt < 0) ? 1000000 : nevt;
      Printf("\nrunProof: running \"dataset\" with nevt= %lld\n", nevt);

      // Ask for registration of the dataset (the default is the the TFileCollection is return
      // without registration; the name of the TFileCollection is the name of the dataset
      proof->SetParameter("SimpleNtuple.root","testNtuple");

      // Do not plot the ntuple at this level
      proof->SetParameter("PROOF_NTUPLE_DONT_PLOT", "");

      // The selector string
      sel.Form("%s/proof/ProofNtuple.C%s", tutorials.Data(), aMode.Data());
      //
      // Run it for nevt times
      proof->Process(sel.Data(), nevt, opt);

      // The TFileCollection must be in the output
      if (proof->GetOutputList()->FindObject("testNtuple")) {

         // Plot the ntuple via PROOF (example of drawing PROOF actions)
         plotNtuple(proof, "testNtuple", "proof ntuple from dataset");

      } else {
         Printf("runProof: dataset 'testNtuple' not found in the output list");
      }
      // Do not plot the ntuple at this level
      proof->DeleteParameters("PROOF_NTUPLE_DONT_PLOT");
      proof->DeleteParameters("SimpleNtuple.root");

   } else if (act == "friends") {

      // This is an example of analysis creating two data files on each node (the main tree
      // and its friend) which are then processed as 'friends' to create the final plots.
      // Selector used: ProofFriends, ProofAux

      // Find out whether to use the same file or separate files
      Bool_t sameFile = kFALSE;
      while (args.Tokenize(tok, from, " ")) {
         // Number of histos
         if (tok == "samefile") {
            sameFile = kTRUE;
            break;
         }
      }

      // File generation: we use TPacketizerFile in here to create two files per node
      TList *wrks = proof->GetListOfSlaveInfos();
      if (!wrks) {
         Printf("runProof: could not get the list of information about the workers");
         return;
      }
      // Create the map
      TString fntree;
      TMap *files = new TMap;
      files->SetName("PROOF_FilesToProcess");
      TIter nxwi(wrks);
      TSlaveInfo *wi = 0;
      while ((wi = (TSlaveInfo *) nxwi())) {
         fntree.Form("tree_%s.root", wi->GetOrdinal());
         THashList *wrklist = (THashList *) files->GetValue(wi->GetName());
         if (!wrklist) {
            wrklist = new THashList;
            wrklist->SetName(wi->GetName());
            files->Add(new TObjString(wi->GetName()), wrklist);
         }
         wrklist->Add(new TObjString(fntree));
      }

      // Generate the files
      proof->AddInput(files);
      if (sameFile) {
         Printf("runProof: friend tree stored in the same file as the main tree");
         proof->SetParameter("ProofAux_Action", "GenerateTreesSameFile");
      } else {
         proof->SetParameter("ProofAux_Action", "GenerateTrees");
      }
      // Default 1000 events
      nevt = (nevt < 0) ? 10000 : nevt;
      proof->SetParameter("ProofAux_NEvents", (Long64_t)nevt);
      // Special Packetizer
      proof->SetParameter("PROOF_Packetizer", "TPacketizerFile");
      // Now process
      sel.Form("%s/proof/ProofAux.C%s", tutorials.Data(), aMode.Data());
      proof->Process(sel.Data(), 1);
      // Remove the packetizer specifications
      proof->DeleteParameters("PROOF_Packetizer");

      // Print the lists and create the TDSet objects
      TDSet *dset = new TDSet("Tmain", "Tmain");
      TDSet *dsetf = new TDSet("Tfrnd", "Tfrnd");
      if (proof->GetOutputList()) {
         TIter nxo(proof->GetOutputList());
         TObject *o = 0;
         TObjString *os = 0;
         while ((o = nxo())) {
            TList *l = dynamic_cast<TList *> (o);
            if (l && !strncmp(l->GetName(), "MainList-", 9)) {
               TIter nxf(l);
               while ((os = (TObjString *) nxf()))
                  dset->Add(os->GetName());
            }
         }
         nxo.Reset();
         while ((o = nxo())) {
            TList *l = dynamic_cast<TList *> (o);
            if (l && !strncmp(l->GetName(), "FriendList-", 11)) {
               TIter nxf(l);
               while ((os = (TObjString *) nxf()))
                  dsetf->Add(os->GetName());
            }
         }
      }
      // Process with friends
      dset->AddFriend(dsetf, "friend");
      sel.Form("%s/proof/ProofFriends.C%s", tutorials.Data(), aMode.Data());
      dset->Process(sel);
      // Clear the files created by this run
      proof->ClearData(TProof::kUnregistered | TProof::kForceClear);

   } else if (act == "simplefile") {
      
      // ProofSimpleFile is an example of non-data driven analysis with merging
      // via file and objcets saved in different directories; it creates and
      // fills with random numbers two sets of a given number of histos

      // Default 100000 events
      nevt = (nevt < 0) ? 1000000 : nevt;
      // Find out the number of histograms
      TString aNhist;
      while (args.Tokenize(tok, from, " ")) {
         // Number of histos
         if (tok.BeginsWith("nhist=")) {
            aNhist = tok;
            aNhist.ReplaceAll("nhist=","");
            if (!aNhist.IsDigit()) {
               Printf("runProof: error parsing the 'nhist=' option (%s) - ignoring", tok.Data());
               aNhist = "";
            }
         }
      }
      Int_t nhist = (aNhist.IsNull()) ? 16 : aNhist.Atoi();
      Printf("\nrunProof: running \"simplefile\" with nhist= %d and nevt= %lld\n", nhist, nevt);

      // The number of histograms is added as parameter in the input list
      proof->SetParameter("ProofSimple_NHist", (Long_t)nhist);
      
      // Output file
      TString fout = TString::Format("%s/SimpleFile.root", gSystem->WorkingDirectory());
      // Cleanup any existing instance of the output file
      gSystem->Unlink(fout);

      if (!isProofLocal) {
         // Setup a local basic xrootd to receive the file
         Bool_t xrdok = kFALSE;
         Int_t port = 9000;
         while (port < 9010) {
            if (checkXrootdAt(port) != 1) {
               if (startXrootdAt(port, gSystem->WorkingDirectory(), kTRUE) == 0) {
                  xrdok = kTRUE;
                  break;
               }
            }
            port++;
         }
         if (!xrdok) {
            Printf("runProof: could not start basic xrootd on ports 9000-9009 - cannot continue");
            return;
         }
         fout.Insert(0, TString::Format("root://%s:%d/", TUrl(gSystem->HostName()).GetHostFQDN(), port));
         // Make a copy of the files on the master before merging
         proof->AddInput(new TNamed("PROOF_OUTPUTFILE_LOCATION", "LOCAL"));
      }
      proof->AddInput(new TNamed("PROOF_OUTPUTFILE", fout.Data()));
      
      // The selector string
      sel.Form("%s/proof/ProofSimpleFile.C%s", tutorials.Data(), aMode.Data());
      //
      // Run it for nevt times
      proof->Process(sel.Data(), nevt, opt);

   } else {
      // Do not know what to run
      Printf("runProof: unknown tutorial: %s", what);
   }
}

//_______________________________________________________________________________________
void plotNtuple(TProof *p, const char *ds, const char *ntptitle)
{
   // Make some plots from the ntuple 'ntp' via PROOF

   //
   // Create a canvas, with 2 pads
   //
   TCanvas *c1 = new TCanvas(Form("cv-%s", ds), ntptitle,800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   TPad *pad2 = (TPad *) c1->GetPad(2);
   //
   // Display a function of one ntuple column imposing a condition
   // on another column.
   pad1->cd();
   pad1->SetGrid();
   pad1->SetLogy();
   pad1->GetFrame()->SetFillColor(15);

   p->SetParameter("PROOF_LineColor", (Int_t)1);
   p->SetParameter("PROOF_FillStyle", (Int_t)1001);
   p->SetParameter("PROOF_FillColor", (Int_t)45);
   p->DrawSelect(ds, "3*px+2","px**2+py**2>1");
   p->SetParameter("PROOF_FillColor", (Int_t)38);
   p->DrawSelect(ds, "2*px+2","pz>2","same");
   p->SetParameter("PROOF_FillColor", (Int_t)5);
   p->DrawSelect(ds, "1.3*px+2","(px^2+py^2>4) && py>0","same");
   pad1->RedrawAxis();

   //
   // Display a 3-D scatter plot of 3 columns. Superimpose a different selection.
   pad2->cd();
   p->DrawSelect(ds, "pz:py:px","(pz<10 && pz>6)+(pz<4 && pz>3)");
   p->SetParameter("PROOF_MarkerColor", (Int_t)4);
   p->DrawSelect(ds, "pz:py:px","pz<6 && pz>4","same");
   p->SetParameter("PROOF_MarkerColor", (Int_t)5);
   p->DrawSelect(ds, "pz:py:px","pz<4 && pz>3","same");
   TPaveText *l2 = new TPaveText(0.,0.6,0.9,0.95);
   l2->SetFillColor(42);
   l2->SetTextAlign(12);
   l2->AddText("You can interactively rotate this view in 2 ways:");
   l2->AddText("  - With the RotateCube in clicking in this pad");
   l2->AddText("  - Selecting View with x3d in the View menu");
   l2->Draw();

   // Final update
   c1->cd();
   c1->Update();

   // Clear parameters used for the plots
   p->DeleteParameters("PROOF_*Color");
   p->DeleteParameters("PROOF_*Style");
}

int getDebugEnum(const char *what)
{
   // Check if 'what' matches one of the TProofDebug enum and return the corresponding
   // integer. Relies on a perfect synchronization with the content of TProofDebug.h .

   TString sws(what), sw;
   int rcmask = 0;
   int from = 0;
   while (sws.Tokenize(sw, from , "|")) {
      if (sw.BeginsWith("k")) sw.Remove(0,1);

      if (sw == "None") {
         rcmask |= TProofDebug::kNone;
      } else if (sw == "Packetizer") {
         rcmask |= TProofDebug::kPacketizer;
      } else if (sw == "Loop") {
         rcmask |= TProofDebug::kLoop;
      } else if (sw == "Selector") {
         rcmask |= TProofDebug::kSelector;
      } else if (sw == "Output") {
         rcmask |= TProofDebug::kOutput;
      } else if (sw == "Input") {
         rcmask |= TProofDebug::kInput;
      } else if (sw == "Global") {
         rcmask |= TProofDebug::kGlobal;
      } else if (sw == "Package") {
         rcmask |= TProofDebug::kPackage;
      } else if (sw == "Feedback") {
         rcmask |= TProofDebug::kFeedback;
      } else if (sw == "Condor") {
         rcmask |= TProofDebug::kCondor;
      } else if (sw == "Draw") {
         rcmask |= TProofDebug::kDraw;
      } else if (sw == "Asyn") {
         rcmask |= TProofDebug::kAsyn;
      } else if (sw == "Cache") {
         rcmask |= TProofDebug::kCache;
      } else if (sw == "Collect") {
         rcmask |= TProofDebug::kCollect;
      } else if (sw == "Dataset") {
         rcmask |= TProofDebug::kDataset;
      } else if (sw == "Submerger") {
         rcmask |= TProofDebug::kSubmerger;
      } else if (sw == "All") {
         rcmask |= TProofDebug::kAll;
      } else if (!sw.IsNull()) {
         Printf("WARNING: requested debug enum name '%s' does not exist: assuming 'All'", sw.Data());
         rcmask |= TProofDebug::kAll;
      }
   }
   // Done
   return rcmask;
}
