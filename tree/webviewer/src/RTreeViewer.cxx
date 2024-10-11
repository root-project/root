// Author: Sergey Linev, 7.10.2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RTreeViewer.hxx>

// #include <ROOT/RLogger.hxx>
#include <ROOT/RWebWindow.hxx>

#include "TTree.h"
#include "TVirtualPad.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"
#include "TVirtualMonitoring.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TAxis.h"
#include "TSystem.h"

#include "TBufferJSON.h"

namespace ROOT {

class RTreeDrawMonitoring : public TVirtualMonitoringWriter {

private:

   RTreeDrawMonitoring(const RTreeDrawMonitoring&) = delete;
   RTreeDrawMonitoring& operator=(const RTreeDrawMonitoring&) = delete;

   Int_t fPeriod{100};
   long long fLastProgressSendTm{0};
   RTreeViewer &fViewer;

public:
   RTreeDrawMonitoring(Int_t period, RTreeViewer &viewer)
      : TVirtualMonitoringWriter(), fPeriod(period), fViewer(viewer)
   {
   }

   // TFile related info. In general they are gathered and sent only sometimes as summaries
   bool SendFileCloseEvent(TFile * /*file*/) override { return false; }
   bool SendFileReadProgress(TFile * /*file*/) override { return false; }
   bool SendFileWriteProgress(TFile * /*file*/) override { return false; }

   bool SendParameters(TList * /*valuelist*/, const char * /*identifier*/ = nullptr) override { return false; }
   bool SendInfoTime() override { return false; }
   bool SendInfoUser(const char * /*user*/ = nullptr) override { return false; }
   bool SendInfoDescription(const char * /*jobtag*/) override { return false; }
   bool SendInfoStatus(const char * /*status*/) override { return false; }

   bool SendFileOpenProgress(TFile * /*file*/, TList * /*openphases*/, const char * /*openphasename*/,
                               bool /*forcesend*/ = false) override
   {
      return false;
   }

   bool SendProcessingStatus(const char * /*status*/, bool /*restarttimer*/ = false) override { return false; }
   bool SendProcessingProgress(Double_t nevent, Double_t /*nbytes*/, bool /*force*/ = false) override
   {
      long long millisec = gSystem->Now();

      if (fLastProgressSendTm && (millisec < fLastProgressSendTm + fPeriod))
         return true;

      fLastProgressSendTm = millisec;

      gSystem->ProcessEvents();

      fViewer.SendProgress(nevent);

      return true;
   }
   void SetLogLevel(const char * /*loglevel*/ = "WARNING") override {}
   void Verbose(bool /*onoff*/) override {}
};


class RTreeDrawInvokeTimer : public TTimer {
public:
   RTreeViewer &fViewer;

   /// constructor
   RTreeDrawInvokeTimer(Long_t milliSec, bool mode, RTreeViewer &viewer) : TTimer(milliSec, mode), fViewer(viewer) {}

   /// timeout handler
   /// used to process postponed requests in main ROOT thread
   void Timeout() override { fViewer.InvokeTreeDraw(); }
};


} // namespace ROOT

using namespace ROOT;
using namespace std::string_literals;


/** \class ROOT::RTreeViewer
\ingroup webwidgets

\brief Web-based %ROOT TTree viewer
*/


//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RTreeViewer::RTreeViewer(TTree *tree)
{
   fWebWindow = ROOT::RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/tree/index.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetConnectCallBack([this](unsigned connid) { WebWindowConnect(connid); });
   fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(900, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // allow the only connection
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   if (tree) SetTree(tree);

   fTimer = std::make_unique<RTreeDrawInvokeTimer>(10, true, *this);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RTreeViewer::~RTreeViewer()
{

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// assign new TTree to the viewer

void RTreeViewer::SetTree(TTree *tree)
{
   fTree = tree;

   // reset expression when new tree is assigned
   fCfg.fExprX.clear();
   fCfg.fExprY.clear();
   fCfg.fExprZ.clear();
   fCfg.fExprCut.clear();
   fCfg.fNumber = 0;
   fCfg.fFirst = 0;

   UpdateConfig();

   Update();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Suggest to use leaf in the gui
/// Normally just assign as last edited expression

bool RTreeViewer::SuggestLeaf(const TLeaf *leaf)
{
   const TBranch *branch = leaf ? leaf->GetBranch() : nullptr;

   const TTree *tree = branch ? branch->GetTree() : nullptr;

   if (!tree || (fTree != tree))
      return false;

   if ((const_cast<TBranch *>(branch)->GetListOfBranches()->GetLast() < 0) && (branch->GetNleaves() == 1)) {
      std::string brname = branch->GetName();
      if (brname == leaf->GetName())
         return SuggestBranch(branch);
   }

   fWebWindow->Send(0, "SUGGEST:"s + FormatItemName(leaf->GetFullName().Data()));

   return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Suggest to use branch in the gui
/// Normally just assign as last edited expression

bool RTreeViewer::SuggestBranch(const TBranch *branch)
{
   const TTree *tree = branch ? branch->GetTree() : nullptr;

   if (!tree || (fTree != tree))
      return false;

   fWebWindow->Send(0, "SUGGEST:"s + FormatItemName(branch->GetFullName().Data()));

   return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Suggest to use expression in the gui
/// Normally just assign as last edited expression

bool RTreeViewer::SuggestExpression(const std::string &expr)
{
   if (!fTree || expr.empty())
      return false;

   fWebWindow->Send(0, "SUGGEST:"s + expr);

   return true;
}


/////////////////////////////////////////////////////////////////////////////////
/// Show or update viewer in web window
/// If web browser already started - just refresh drawing like "reload" button does
/// If no web window exists or \param always_start_new_browser configured, starts new window
/// \param args arguments to display

void RTreeViewer::Show(const RWebDisplayArgs &args, bool always_start_new_browser)
{
   std::string user_args = "";
   if (!GetShowHierarchy()) user_args = "{ nobrowser: true }";
   fWebWindow->SetUserArgs(user_args);

   if (args.GetWidgetKind().empty())
      const_cast<RWebDisplayArgs *>(&args)->SetWidgetKind("RTreeViewer");

   if ((fWebWindow->NumConnections(true) == 0) || always_start_new_browser)
      fWebWindow->Show(args);
   else
      Update();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return address of web window used for tree viewer

std::string RTreeViewer::GetWindowAddr() const
{
   return fWebWindow ? fWebWindow->GetAddr() : ""s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return URL of web window used for tree viewer
/// See \ref ROOT::RWebWindow::GetUrl docu for more details

std::string RTreeViewer::GetWindowUrl(bool remote)
{
   return fWebWindow ? fWebWindow->GetUrl(remote) : ""s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Update tree viewer in all web displays

void RTreeViewer::Update()
{
   SendCfg(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Send data for initialize viewer

void RTreeViewer::SendCfg(unsigned connid)
{
   std::string json = "CFG:"s + TBufferJSON::ToJSON(&fCfg, TBufferJSON::kSkipTypeInfo + TBufferJSON::kNoSpaces).Data();

   fWebWindow->Send(connid, json);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// react on new connection

void RTreeViewer::WebWindowConnect(unsigned connid)
{
   SendCfg(connid);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// receive data from client

void RTreeViewer::WebWindowCallback(unsigned connid, const std::string &arg)
{
   if (arg == "GETCFG"s) {

      SendCfg(connid);

   } else if (arg == "QUIT_ROOT"s) {

      fWebWindow->TerminateROOT();

   } if (arg.compare(0, 5, "DRAW:"s) == 0) {

      if (!fTree) return;

      auto newcfg = TBufferJSON::FromJSON<RConfig>(arg.substr(5));

      if (newcfg) {
         fCfg = *newcfg;
         fTimer->TurnOn();
      }
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Format item name used in draw expression

std::string RTreeViewer::FormatItemName(const std::string &name)
{
   std::string res = name;

   std::string from = "/";
   std::string to = "\\/";

   size_t start_pos = 0;
   while((start_pos = res.find(from, start_pos)) != std::string::npos) {
       res.replace(start_pos, from.length(), to);
       start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
   }
   return res;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Add branches to config

void RTreeViewer::AddBranches(TObjArray *branches)
{
   if (!branches || (branches->GetLast() < 0))
      return;

   TIter iter(branches);

   while (auto br = dynamic_cast<TBranch *>(iter())) {

      auto leaves = br->GetListOfLeaves();

      auto subbr = br->GetListOfBranches();

      std::string brname = br->GetName();

      TLeaf *leaf0 = (leaves->GetLast() == 0) ? dynamic_cast<TLeaf *>(leaves->At(0)) : nullptr;

      auto brelem = dynamic_cast<TBranchElement *>(br);

      std::string brfullname = br->GetFullName().Data();

      if ((subbr->GetLast() < 0) && leaf0 && (brname == leaf0->GetName())) {

         // ignore branches containing objects, see TBranchElement::GetTypeName()
         if (brelem && (brelem->GetStreamerType() < 1 || brelem->GetStreamerType() > 59))
            continue;

         fCfg.fBranches.emplace_back(FormatItemName(brfullname), br->GetTitle() + " / "s + leaf0->GetTypeName());
         continue;
      }

      TIter liter(leaves);
      while (auto leaf = dynamic_cast<TLeaf *>(liter())) {

         std::string leaffullname = leaf->GetFullName().Data();

         // ignore counter leaf for STL container
         if (brelem && brelem->GetStreamerType() == TStreamerInfo::kSTL && (leaves->GetLast() == 0) && (leaffullname == brfullname + "_"))
            break;

         fCfg.fBranches.emplace_back(FormatItemName(leaffullname), leaf->GetTitle() + " / "s + leaf->GetTypeName());
      }

      AddBranches(subbr);
   }
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Update RConfig data

void RTreeViewer::UpdateConfig()
{
   fCfg.fBranches.clear();

   if (!fTree) return;

   fCfg.fTreeName = fTree->GetName();

   AddBranches(fTree->GetListOfBranches());

   fCfg.fTreeEntries = fTree->GetEntries();

   fCfg.fStep = 1;
   fCfg.fLargerStep = fCfg.fTreeEntries/100;
   if (fCfg.fLargerStep < 2) fCfg.fLargerStep = 2;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Invoke tree drawing

void RTreeViewer::InvokeTreeDraw()
{

   fTimer->TurnOff();

   UpdateConfig();

   std::string expr = fCfg.fExprX;
   if (!fCfg.fExprY.empty()) {
      expr += ":"s;
      expr += fCfg.fExprY;

      if (!fCfg.fExprZ.empty()) {
         expr += ":"s;
         expr += fCfg.fExprZ;
      }
   }

   Long64_t nentries = (fCfg.fNumber > 0) ? fCfg.fNumber : TTree::kMaxEntries;

   auto old = gMonitoringWriter;
   RTreeDrawMonitoring monitoring(50, *this);
   gMonitoringWriter = &monitoring;

   fLastSendProgress.clear();

   fTree->Draw(expr.c_str(), fCfg.fExprCut.c_str(), fCfg.fOption.c_str(), nentries, fCfg.fFirst);

   gMonitoringWriter = old;

   if (!fLastSendProgress.empty())
      SendProgress(-1.);

   std::string canv_name;

   if (gPad) {
      if ((expr.find('\\') != std::string::npos) || (expr.find('#') != std::string::npos)) {
         auto FixTitle = [](TNamed *obj) {
            if (!obj) return;
            TString title = obj->GetTitle();
            title.ReplaceAll("\\/", "/");
            title.ReplaceAll("#","\\#");
            obj->SetTitle(title.Data());
         };
         TIter iter(gPad->GetListOfPrimitives());
         while (auto obj = iter()) {
            if (expr == obj->GetTitle()) {
               FixTitle(dynamic_cast<TNamed *> (obj));
               TH1 *hist = dynamic_cast<TH1 *> (obj);
               if (hist) {
                  FixTitle(hist->GetXaxis());
                  FixTitle(hist->GetYaxis());
                  FixTitle(hist->GetZaxis());
               }
            }
         }
      }

      gPad->Update();
      canv_name = gPad->GetName();
   }

   // at the end invoke callback
   if (fCallback)
      fCallback(canv_name);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Send progress to the client

void RTreeViewer::SendProgress(Double_t nevent)
{
   std::string progress = "100";

   if (nevent >= 0.) {

      Long64_t first = fCfg.fFirst;
      Long64_t nentries = fTree->GetEntries();
      Long64_t last = nentries;
      if ((fCfg.fNumber > 0) && (first + fCfg.fNumber < nentries))
         last = first + fCfg.fNumber;

      if (last > first) {
         Double_t p = nevent / ( last - first + 0. ) * 100.;
         if (p > 100) p = 100;
         progress = std::to_string(p);
      }
   }

   if (fLastSendProgress == progress)
      return;

   fLastSendProgress = progress;

   if (fWebWindow->CanSend(0, true))
      fWebWindow->Send(0, "PROGRESS:"s + progress);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Create new viewer
/// Method used for plugin

RTreeViewer *RTreeViewer::NewViewer(TTree *t)
{
   auto viewer = new RTreeViewer(t);

   viewer->Show();

   return viewer;
}

