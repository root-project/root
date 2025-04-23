#include <errno.h>
#include <iostream>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "TAxis.h"
#include "TCanvas.h"
#include "TError.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TH1.h"
#include "TMath.h"
#include "TMultiGraph.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"
#include "TText.h"
#include "TTree.h"

#include "pt_data.h"

using namespace std;

enum EMeasurement {
   kMemLeak,
   kMemPeak,
   kMemAlloc,
   kCPUTime,
   kNumMeasurements
};

struct PTMeasurement {
   long memory[4]; // alloc, leak, peak, tag
   double utime;
   double stime;
   double wtime;
};

class PTGraph {
public:
   PTGraph(Long64_t n): nGood(0), nBad(0), nLimit(0) {
      good = new TGraphErrors(n);
      bad = new TGraphErrors(n);
      limit = new TGraphAsymmErrors(n);
   }
   ~PTGraph() {
      // Don't delete; owned by TMultiGraph
      // delete good;
      // delete bad;
      // delete limit;
   }
   int nGood;
   int nBad;
   int nLimit;

   TGraphErrors* good;
   TGraphErrors* bad;
   TGraphAsymmErrors* limit;
};

class PTGraphColl {
public:
   PTGraphColl(Long64_t n) {
      for (int i = 0; i < kNumMeasurements; ++i)
         gr[i] = new PTGraph(n);
   }
   ~PTGraphColl() {
      for (int i = 0; i < kNumMeasurements; ++i)
         delete gr[i];
   }

   PTGraph* gr[kNumMeasurements];
};

const double zLimits[] = {
   3.5 /*memleak*/,
   4.5 /*mempeak*/,
   5.0 /*memalloc*/,
   5.0 /*cputime*/
};

const double uncertainty[] = {
   0.50 /*memleak*/,
   1.00 /*mempeak*/,
   1.00 /*memalloc*/,
   0.05 /*cputime*/
};

const char* measurementNames[kNumMeasurements] = {
   "memory leaks (kB)",
   "peak memory usage (kB)",
   "sum of memory allocations (kB)",
   "cpu time (s)"
};

//______________________________________________________________________________
void InvokeChild(char** argv, const TString& roottestHome){
   // We are the fork's child. Convert ourselves into root.exe (or whatever else was argv[2])

   setenv("LD_PRELOAD", roottestHome + "/scripts/ptpreload.so", 1);
   execvp(argv[0], argv);
}

//______________________________________________________________________________
PTMeasurement ReceiveResults(const TString& fifoName) {
   // Retrieve the measurements from the FIFO and from the child's usage data.

   int fd = open(fifoName, O_RDONLY);
   if (fd < 0) {
      printf("Error pt_collector: opening FIFO %s, %s\n", fifoName.Data(), strerror(errno));
      exit(1);
   }

   // read child performance information
   int status;
   wait(&status);
   if (status != 0){
      unlink(fifoName);
      exit(status); // test failed
   }

   // get memory
   PTMeasurement results;
   read(fd, &results, 4*sizeof(long));
   unlink(fifoName);
   if (results.memory[3] != 699692586){
      printf("Error pt_collector: could not read memory usage from FIFO %s\n", fifoName.Data());;
      exit(1);
   }

   // get cpu time
   struct rusage usage;
   getrusage(RUSAGE_CHILDREN, &usage);
   results.utime = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1000000.;
   results.stime = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1000000.;
   return results;
}

//______________________________________________________________________________
TTree* GetTree(TString& fileName, TString& testName, int argc, char** argv, const TString& cwd, const TString& roottestHome) {
   TString lastArg(argv[argc-1]);

   // build test name
   testName = cwd + "/" + lastArg;
   if (testName.BeginsWith(roottestHome)) {
      testName.Remove(0, roottestHome.Length());
      if (testName[0] == '/') testName.Remove(0, 1);
   }

   // build file name
   fileName = testName;
   fileName.ReplaceAll("/", "");
   Ssiz_t posDot = fileName.Index('.');
   if (posDot != kNPOS) {
      fileName.Remove(posDot); // cut file name after first '.'
   }
   fileName.Prepend("pt_");
   fileName += testName.MD5();
   fileName += ".root";

   TFile* file = TFile::Open(fileName, "UPDATE");
   if (!file || file->IsZombie()) {
      printf("Error pt_collector: could not open data file %s\n", fileName.Data());
      exit(1);
   }

   TTree* tree = 0;
   file->GetObject("PerftrackTree", tree);
   if (!tree) {
      PTData* data = 0;
      tree = new TTree("PerftrackTree", "Performance tracking data");
      tree->Branch("event", &data);
      tree->GetUserInfo()->Add(new TObjString(testName));
   } else {
      TObject* userInfoTest = tree->GetUserInfo()->FindObject(testName);
      if (!userInfoTest) {
         // We have a hash collision: there is another test with a
         // different name writing into the same ROOT file as the
         // current test.
         // We can't do much about it; just ignore this test for
         // performance tracking.
         exit(0);
      }
   }
   return tree;
}

//______________________________________________________________________________
void FillData(const PTMeasurement& results, TTree* tree, PTData& prevdata, PTData& newdata) {

   // Whether the previous is outlier or not, it contains the relevant
   // average / variance etc data.

   time_t rawtime;
   time(&rawtime);
   newdata.date = ctime(&rawtime);

   double resdata[4] = {
      results.memory[kMemLeak]/1024.,
      results.memory[kMemPeak]/1024.,
      results.memory[kMemAlloc]/1024., // in kilobyte
      results.utime + results.stime // in seconds
   };

   newdata.statEntries = prevdata.statEntries + 1;
   newdata.historyThinningCounter = prevdata.historyThinningCounter + 1;
   for (int i = 0; i < kNumMeasurements; ++i) {
      newdata.pval[i]->Set(resdata[i], *prevdata.pval[i], newdata.statEntries);
   }
   newdata.svn = gROOT->GetSvnRevision();
   newdata.outlier = 0;
}

//______________________________________________________________________________
void FillGraphEntry(int i, PTGraph& graph, const PTVal& val,
                    unsigned int statEntries, unsigned int rev, int outlier) {
   TGraphErrors* gr = graph.good;
   int *n = &graph.nGood;
   double var = val.fVar;
   double mean = val.fMean;

   bool ioutlier = outlier & (1 << i);
   if (ioutlier) {
      gr = graph.bad;
      n = &graph.nBad;
      mean = mean * statEntries + val.fVal;
      double sumVal2 = val.fSumVal2 + val.fVal * val.fVal;
      double var2 = sumVal2 - mean * mean;
      if (var2 > 0.) {
         var = sqrt(var2) / (statEntries + 1.);
      } else {
         var = 0.;
      }
      mean /= statEntries + 1.;
   }

   var += uncertainty[i];

   gr->SetPoint(*n, (double)rev, val.fVal);
   gr->SetPointError(*n, 0., var);
   ++(*n);
   graph.limit->SetPoint(graph.nLimit, (double)rev,  mean);
   graph.limit->SetPointError(graph.nLimit++, 0.5, 0.5,
                              0. /*"mean" would zoom y axis out completely*/,
                              zLimits[i] * var);
}

//______________________________________________________________________________
PTGraphColl* CreateOldGraphs(TTree* tree, PTData &olddata) {
   // Allocate the graphs, fill old data

   Long64_t entries = tree->GetEntries();
   PTGraphColl* graphs = new PTGraphColl(entries + 1);
   if (!entries) return graphs;

   PTData *branchdata = &olddata;
   tree->SetBranchAddress("event", &branchdata);

   for (Long64_t entry = 0; entry < entries; ++entry) {
      tree->GetEntry(entry);
      for (int i = 0; i < kNumMeasurements; ++i) {
         FillGraphEntry(i, *(graphs->gr[i]), *(branchdata->pval[i]),
                        branchdata->statEntries,
                        branchdata->svn, branchdata->outlier);
      }
   }
   tree->ResetBranchAddresses();
   return graphs;
}

//______________________________________________________________________________
bool CheckPerformance(PTData& newdata) {
   // Check whether the new performance measurements are within allowed parameters.
   // Return false on test failure (i.e. a significant performance decrease).

   if (newdata.statEntries < 2) return true;

   for (int i = 0; i < kNumMeasurements; ++i) {
      PTVal* val = newdata.pval[i];
      double var = val->fVar + uncertainty[i];
      val->fZ = (val->fVal - val->fMean) / var;
      if (val->fVal > val->fMean + zLimits[i] * var) {
         newdata.outlier |= 1 << i;
      }
   }
   return newdata.outlier == 0;
}

//______________________________________________________________________________
void UpdateGraphs(PTGraphColl* graphs, const PTData& newdata) {
   for (int i = 0; i < kNumMeasurements; ++i) {
      FillGraphEntry(i, *(graphs->gr[i]), *newdata.pval[i], newdata.statEntries,
                     newdata.svn, newdata.outlier);
      graphs->gr[i]->good->Set(graphs->gr[i]->nGood);
      graphs->gr[i]->bad->Set(graphs->gr[i]->nBad);
      graphs->gr[i]->limit->Set(graphs->gr[i]->nLimit);
   }
}

//______________________________________________________________________________
void RevertOutlierStat(TTree* tree, const PTData& prevdata, PTData& newdata) {
   // The new measurement is an outlier; update its integrated statistics data
   // to not take the current measurement into account.

   // Whether the previous is outlier or not, it contains the relevant
   // average / variance etc data.

   --newdata.statEntries;

   for (int i = 0; i < kNumMeasurements; ++i) {
      PTVal* nval = newdata.pval[i];
      const PTVal* pval = prevdata.pval[i];
      nval->fMean = pval->fMean;
      nval->fVar = pval->fVar;
      nval->fSumVal2 = pval->fSumVal2;
   }
}

//______________________________________________________________________________
void ReportFailures(const PTData& newdata, const TString& testName,
                    const TString& fileName) {

   for (int i = 0; i < kNumMeasurements; ++i) {
      if (newdata.outlier & (1 << i)) {
         cout << "Performance decrease (" << measurementNames[i] << ") for test " << testName << " in file " << fileName << endl;
         cout << "   Measured: " << newdata.pval[i]->fVal << endl
              << "   Mean: " << newdata.pval[i]->fMean << endl
              << "   Variance: " << newdata.pval[i]->fVar << endl
              << "   Delta: " << newdata.pval[i]->fZ << "sigmas" << endl;
      }
   }
}

//______________________________________________________________________________
void UpdateTree(TTree* tree, const PTData& newdata) {
   const PTData *localptr = &newdata;
   tree->SetBranchAddress("event",&localptr);
   tree->Fill();
   tree->Write(0, TObject::kWriteDelete);
   TFile* file = tree->GetCurrentFile();
   delete tree;
   delete file;
}

//______________________________________________________________________________
void DeleteOldEntries(TTree *&tree, unsigned int historyThinningCounter, const TString& fileName)
{
   // Delete old entries, to not collect too much historical data.

   static const int ExpTime=200; // 2*ExpTime values (not including outliers) stored in tree int status;
   if (historyThinningCounter % ExpTime != 0) return;

   PTData *data = 0;
   tree->SetBranchAddress("event",&data);
   TString tmpFileName(fileName);
   tmpFileName.ReplaceAll(".root", "_tmp.root");
   TFile::Open(tmpFileName, "RECREATE");
   TTree* newT = tree->CloneTree(0);
   ULong64_t nevent = tree->GetEntries();
   ULong64_t newStatEntries = 0;
   for (ULong64_t i = 0; i < nevent; i++){
      tree->GetEntry(i);
      if (data->outlier != 0) {
         newT->Fill();
         continue;
      }
      if (newStatEntries % 2 == 0) {
         // Let every 2nd entry survive
         ++newStatEntries;
         // Set stat entries to the current number of entries
         // Gives new entries after history deletion more weight
         data->statEntries = newStatEntries;

         newT->Fill();
      }
   }
   TFile* fileOld = tree->GetCurrentFile();
   delete tree;
   delete fileOld;
   gSystem->Unlink(fileName);
   gSystem->Rename(tmpFileName, fileName);
   tree = newT;
   tree->ResetBranchAddresses();
}

//______________________________________________________________________________
void SaveGraphs(PTGraphColl* graphs, const TString& dataFileName, const TString& testName)
{
   // Save the graphs into a canvas.

   TCanvas *c1 = new TCanvas("c1","Performance Monitoring Plots",1200,800);
   TText* title = new TText(.1, .95, testName);
   title->SetTextFont((title->GetTextFont() / 10) * 10 + 3); // scalable, pixels
   title->SetTextSizePixels(24);
   title->Draw();
   c1->Update();
   UInt_t w, h;
   title->GetBoundingBox(w, h);
   TPad* graphPad = new TPad("graphPad", "graphPad", 0., 0.,
                             1., .92,
                             -1, 0, 0);
   graphPad->Draw();
   graphPad->Divide(2, (kNumMeasurements + 1) / 2, 0.);
   TList listMG;
   listMG.SetOwner();
   for (int i = 0; i < kNumMeasurements; ++i) {
      graphs->gr[i]->good->SetMarkerColor(kBlack);
      graphs->gr[i]->good->SetMarkerStyle(kCircle); // 7 probably faster (not scalable)
      graphs->gr[i]->good->SetLineColor(kBlack);
      graphs->gr[i]->good->SetLineStyle(1);
      graphs->gr[i]->limit->SetLineWidth(3);

      graphs->gr[i]->bad->SetMarkerColor(kRed);
      graphs->gr[i]->bad->SetMarkerStyle(kCircle); // 7 probably faster (not scalable)
      graphs->gr[i]->bad->SetLineColor(kRed);
      graphs->gr[i]->bad->SetLineStyle(1);
      graphs->gr[i]->limit->SetLineWidth(3);

      graphs->gr[i]->limit->SetMarkerColor(kGreen);
      graphs->gr[i]->limit->SetMarkerStyle(kOpenTriangleUp); // 7 probably faster (not scalable)
      graphs->gr[i]->limit->SetLineColor(kGreen);
      graphs->gr[i]->limit->SetFillColor(kGreen);
      graphs->gr[i]->limit->SetLineWidth(1);
      graphs->gr[i]->limit->SetFillStyle(3001);

      TMultiGraph* mg = new TMultiGraph();
      listMG.Add(mg);
      if (graphs->gr[i]->limit->GetN() > 0)
         mg->Add(graphs->gr[i]->limit, "P5");
      if (graphs->gr[i]->bad->GetN() > 0)
         mg->Add(graphs->gr[i]->bad, "P");
      if (graphs->gr[i]->good->GetN() > 0)
         mg->Add(graphs->gr[i]->good, "LP");
      mg->SetTitle(measurementNames[i]);

      graphPad->cd(i + 1);
      mg->Draw("A");
      gPad->Update();
      gPad->SetGrid();

      mg->GetXaxis()->SetTitle("SVN revision");
      mg->GetXaxis()->SetTitleOffset(1.0);
      //mg->GetYaxis()->SetTitle(measurementNames[i]);
      //mg->GetYaxis()->SetTitleOffset(1.5);

      static const double delta = 1E-4;
      double v = (mg->GetHistogram()->GetMinimum() + mg->GetHistogram()->GetMaximum()) / 2.;
      if (mg->GetHistogram()->GetMinimum() > v * (1. - delta)) {
         mg->SetMinimum(v * (1. - delta));
         mg->SetMaximum(v * (1. + delta));
      }
      gPad->Update();
   }
   TString imageName(dataFileName);
   imageName.ReplaceAll(".root", ".gif");
   gErrorIgnoreLevel = 3000; // No "Info:" message
   c1->SaveAs(imageName);
   gErrorIgnoreLevel = 0;
   delete c1;
}

//______________________________________________________________________________
int main(int argc, char** argv)
{
   TH1::AddDirectory(false);

   if (argc < 2) {
      printf("Error: insufficient number of arguments.\n"
             "  pt_collector <ROOTTEST_HOME> program arguments...\n");
      return 1;
   }

   ++argv; // skip program name "pt_collector", previous argv[1] becomes argv[0] etc
   --argc;

   TString roottestHome(argv[0]);
   ++argv;
   --argc;

   TString cwd(gSystem->pwd());

   // build fifo name
   TString fifoName(cwd);
   fifoName += "/pt_fifo_";
   fifoName += (unsigned long)getpid();
   setenv("PT_FIFONAME", fifoName, 1);
   mkfifo(fifoName, 0666);

   pid_t pid=fork();
   if (pid == 0) InvokeChild(argv, roottestHome);
   else {
      PTMeasurement results = ReceiveResults(fifoName);
      TString test;
      TString file;
      TTree* tree = GetTree(file, test, argc, argv, cwd, roottestHome);
      PTData olddata;
      PTGraphColl* graphs = CreateOldGraphs(tree,olddata);
      PTData newdata;
      FillData(results, tree, olddata, newdata);
      CheckPerformance(newdata);

      UpdateGraphs(graphs, newdata);
      SaveGraphs(graphs, file, test);
      delete graphs;

      if (newdata.outlier) {
         ReportFailures(newdata, test, file);
         RevertOutlierStat(tree, olddata, newdata);
      }

      DeleteOldEntries(tree, newdata.historyThinningCounter, file);
      UpdateTree(tree, newdata);
   }
}

