/**************************************************************************
 *
 *  Analyze the running time of roottest (and thus ROOT).
 *  roottest needs to be run with "make TIME=1"
 *  The resulting file roottiming.root will contain a tree with
 *  each branch representing a roottest (sub)directory,
 *
 *  This script produces a graph with a floating average; peaks indicate
 *  changes. It should be run in compiled mode:
 *  root -l roottiming.root
 *  root [] .x roottesttiming.C+
 *
 *  Axel, 2009
 *
 **************************************************************************/


#include <string>
#include <vector>
#include <deque>
#include <map>
#include <iostream>
#include "TDirectory.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TDatime.h"
#include "TGraph.h"
#include "TH1.h"
#include "TStyle.h"

class TestInfo {
public:
   TestInfo(long entries=64):
      fTimingSum(0), fGraph(0),
      fNumDataPoints(0), fDate(entries), fTimingAverage(entries) {};

   ~TestInfo() {
      //delete fGraph;
   }

   void Add(double date, float timing) {
      if (fTimingHistory.size() >= fgHistorySize) {
         fTimingSum -= fTimingHistory.back();
         fTimingHistory.pop_back();
      }
      fTimingSum += timing;
      fTimingHistory.push_front(timing);
      if (fNumDataPoints + 1 > fDate.size()) {
         fDate.resize((fNumDataPoints + 1) * 2);
         fTimingAverage.resize((fNumDataPoints + 1) * 2);
      }
      fDate[fNumDataPoints] = date;
      fTimingAverage[fNumDataPoints] = Ratio(timing);
      ++fNumDataPoints;
   }

   TGraph* GetGraph() {
      if (!fGraph)
         fGraph = new TGraph(fNumDataPoints, &fDate[0], &fTimingAverage[0]);
      return fGraph;
   }

   static void SetHistorySize(unsigned int size) {
      fgHistorySize = size;
   }
   static unsigned int GetHistorySize() {
      return fgHistorySize;
   }

   float Ratio(float v) {
      // ratio of v over unning average
      if (fNumDataPoints) 
         if (fNumDataPoints < fgHistorySize)
            return v / (fTimingSum / fNumDataPoints);
         else
            return v / (fTimingSum / fgHistorySize);
      else return 1.;
   }

private:
   float fTimingSum;
   TGraph* fGraph;
   std::deque<float> fTimingHistory; // filo of timings
   unsigned int fNumDataPoints; // number of filled entries in fDate
   std::vector<double> fDate; // date of a point (yyyymmdd.hh)
   std::vector<double> fTimingAverage; // running average of timings
   static unsigned int fgHistorySize; // length of the running average
};

unsigned int TestInfo::fgHistorySize = 20;

struct TTestResult {
   TDatime         fDate;
   UInt_t          fRunId;
   UInt_t          fTestId;
   TString         fTestName;
   Double32_t      fDuration;
   TString         fHostName;
   TString         fHostModel;
   Double32_t      fHostMhz;
};

typedef TTestResult TreeData;

void roottesttiming() {
   std::map<std::string, TestInfo> tests;
   TestInfo::SetHistorySize(50);
   
   TreeData* data=0;
   TTree* t = (TTree*)gDirectory->Get("timing");
   if (!t) {
      std::cerr << "Can't find tree \"timing\" in current file!" << std::endl;
      return;
   }
   t->SetBranchAddress("test", &data);
   t->SetBranchStatus("fTestId", 0);
   t->SetBranchStatus("fRunId", 0);
   t->SetBranchStatus("fHost*", 0);

   for (long e = 0; e < t->GetEntries(); ++e) {
      t->GetEntry(e);
      /*
      double date = (data->fDate.GetYear() - 2000)*10000;
      date += data->fDate.GetMonth()*100;
      date += data->fDate.GetDay();
      date += data->fDate.GetHour() / 100;
      date += data->fDate.GetMinute() / 100000;
      tests[data->fTestName.Data()].Add(date, data->fDuration);
      */
      tests[data->fTestName.Data()].Add(data->fDate.Convert(), data->fDuration);
      //std::cout << data->fTestName << ": " << data->fDuration << std::endl;
      if (!e) {
         //int X0 = data->fDate.Convert();
         //gStyle->SetTimeOffset(X0);
      }
   }

   TCanvas* canv = new TCanvas();
   for (std::map<std::string, TestInfo>::iterator iResult = tests.begin();
        iResult != tests.end(); ++iResult) {
      const char* opt = "LP SAME";
      TGraph* g = iResult->second.GetGraph();
      g->SetTitle(iResult->first.c_str());
      //std::cout << iResult->first.c_str() << ": " << g->GetN() << std::endl;
      if (iResult == tests.begin()) {
         opt = "ALP";
         TH1* h = g->GetHistogram();
         h->SetMinimum(0.);
         h->SetMinimum(2.);
         //h->GetXaxis()->SetTimeDisplay(1);
         h->SetTitle(Form("ROOTtest performance;Date;Test duration / average of previous %d runs",
                          TestInfo::GetHistorySize()));
      }
      g->Draw(opt);
   }

   canv->SaveAs("roottesttiming.pdf");
   canv->SaveAs("roottesttiming.root");
}
