/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RProvider.hxx>

#include "TString.h"
#include "TLeaf.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TBranchBrowsable.h"
#include "TTree.h"
#include "TH1.h"
#include "TDirectory.h"
#include "TSystem.h"
#include "TVirtualMonitoring.h"

using namespace ROOT::Experimental::Browsable;

class TTreeDrawMonitoring : public TVirtualMonitoringWriter {

private:

   TTreeDrawMonitoring(const TTreeDrawMonitoring&) = delete;
   TTreeDrawMonitoring& operator=(const TTreeDrawMonitoring&) = delete;

   Int_t fPeriod{100};
   TTree *fTree{nullptr};
   void *fHandle2{nullptr};
   long long fLastProgressSendTm{0};

public:
   TTreeDrawMonitoring(Int_t period, TTree *tree, void *handle2)
      : TVirtualMonitoringWriter(), fPeriod(period), fTree(tree), fHandle2(handle2)
   {
   }

   // TFile related info. In general they are gathered and sent only sometimes as summaries
   Bool_t SendFileCloseEvent(TFile * /*file*/) override { return kFALSE; }
   Bool_t SendFileReadProgress(TFile * /*file*/) override { return kFALSE; }
   Bool_t SendFileWriteProgress(TFile * /*file*/) override { return kFALSE; }

   Bool_t SendParameters(TList * /*valuelist*/, const char * /*identifier*/ = nullptr) override { return kFALSE; }
   Bool_t SendInfoTime() override { return kFALSE; }
   Bool_t SendInfoUser(const char * /*user*/ = nullptr) override { return kFALSE; }
   Bool_t SendInfoDescription(const char * /*jobtag*/) override { return kFALSE; }
   Bool_t SendInfoStatus(const char * /*status*/) override { return kFALSE; }

   Bool_t SendFileOpenProgress(TFile * /*file*/, TList * /*openphases*/, const char * /*openphasename*/,
                               Bool_t /*forcesend*/ = kFALSE) override
   {
      return kFALSE;
   }

   Bool_t SendProcessingStatus(const char * /*status*/, Bool_t /*restarttimer*/ = kFALSE) override { return kFALSE; }
   Bool_t SendProcessingProgress(Double_t nevent, Double_t /*nbytes*/, Bool_t /*force*/ = kFALSE) override
   {
      long long millisec = gSystem->Now();

      if (fLastProgressSendTm && (millisec < fLastProgressSendTm + fPeriod))
         return kTRUE;

      fLastProgressSendTm = millisec;

      gSystem->ProcessEvents();

      Double_t total = fTree->GetEntries();

      if (total > 0)
         RProvider::ReportProgress(fHandle2, nevent <= total ? nevent / total : 1.);

      return kTRUE;
   }
   void SetLogLevel(const char * /*loglevel*/ = "WARNING") override {}
   void Verbose(Bool_t /*onoff*/) override {}
};

/** Provider for drawing of branches / leafs in the TTree */

class TLeafProvider : public RProvider {

   void *fHandle2{nullptr}; ///<!  used only for reporting progress

public:

   TH1 *DrawTree(TTree *ttree, const std::string &expr, const std::string &hname)
   {
      if (!ttree)
         return nullptr;

      std::string expr2 = expr + ">>htemp_tree_draw";

      auto old = gMonitoringWriter;
      std::unique_ptr<TTreeDrawMonitoring> monitoring;

      if (fHandle2 && RProvider::ReportProgress(fHandle2, 0.)) {
         monitoring = std::make_unique<TTreeDrawMonitoring>(50, ttree, fHandle2);
         gMonitoringWriter = monitoring.get();
      }

      ttree->Draw(expr2.c_str(),"","goff");

      gMonitoringWriter = old;

      if (!gDirectory)
         return nullptr;

      auto htemp = dynamic_cast<TH1*>(gDirectory->FindObject("htemp_tree_draw"));

      if (!htemp)
         return nullptr;

      htemp->SetDirectory(nullptr);
      htemp->SetName(hname.c_str());

      auto FixTitle = [](TNamed *obj) {
         TString title = obj->GetTitle();
         title.ReplaceAll("\\/", "/");
         title.ReplaceAll("#","\\#");
         obj->SetTitle(title.Data());
      };

      FixTitle(htemp);
      FixTitle(htemp->GetXaxis());
      FixTitle(htemp->GetYaxis());
      FixTitle(htemp->GetZaxis());

      htemp->BufferEmpty();

      return htemp;
   }

   void AdjustExpr(TString &expr, TString &name)
   {
      expr.ReplaceAll("/", "\\/");

      auto pos = name.First('[');
      if (pos != kNPOS) {
         name.Remove(pos);
         pos = expr.First('[');
         if (pos != kNPOS) {
            expr.Remove(pos);
            expr.Append("[]");
         }
      }

      if (name.First('@') != 0)
         return;

      name.Remove(0, 1);

      pos = expr.Index(".@");

      if ((pos != kNPOS) && (expr.Index("()", pos) != expr.Length() - 2))
         expr.Append("()");

      if ((pos != kNPOS) && (pos > 1)) {
         expr.Remove(pos+1, 1);
         pos --;
         while ((pos > 0) && (expr[pos] != '.')) pos--;
         if (pos > 0)
            expr.Insert(pos+1, "@");
         else
            expr.Prepend("@");
      }

      expr.ReplaceAll("->@","@->");
   }

   bool GetDrawExpr(const TBranch *tbranch, TString &expr, TString &name)
   {
      if (!tbranch)
         return false;

      // there are many leaves, plain TTree::Draw does not work
      if (tbranch->GetNleaves() > 1)
         return false;

      // there are sub-branches, plain TTree::Draw does not work
      if (const_cast<TBranch *>(tbranch)->GetListOfBranches()->GetEntriesFast() > 0)
         return false;

      name = tbranch->GetName();

      expr = tbranch->GetFullName();

      AdjustExpr(expr, name);

      return true;
   }

   bool GetDrawExpr(const TLeaf *tleaf, TString &expr, TString &name)
   {
      if (!tleaf)
         return false;

      auto tbranch = tleaf->GetBranch();
      if (tbranch && (tbranch->GetNleaves() == 1))
         return GetDrawExpr(tbranch, expr, name);

      name = tleaf->GetName();

      expr = tleaf->GetFullName();

      AdjustExpr(expr, name);

      return true;
   }

   TH1 *DrawBranch(const TBranch *tbranch)
   {
      TString expr, name;
      if (!GetDrawExpr(tbranch, expr, name))
         return nullptr;

      return DrawTree(tbranch->GetTree(), expr.Data(), name.Data());
   }

   TH1 *DrawBranch(std::unique_ptr<RHolder> &obj)
   {
      fHandle2 = obj.get();
      return DrawBranch(obj->get_object<TBranch>());
   }

   TH1 *DrawLeaf(std::unique_ptr<RHolder> &obj)
   {
      fHandle2 = obj.get();

      auto tleaf = obj->get_object<TLeaf>();

      TString expr, name;

      if (!GetDrawExpr(tleaf,expr, name))
         return nullptr;

      return DrawTree(tleaf->GetBranch()->GetTree(), expr.Data(), name.Data());
   }

   bool GetDrawExpr(const TBranchElement *tbranch, TString &expr, TString &name)
   {
      if (!tbranch)
         return false;

      // there are sub-branches, plain TTree::Draw does not work
      if (const_cast<TBranchElement *>(tbranch)->GetListOfBranches()->GetEntriesFast() > 0)
         return false;

      // just copy and paste code from TBranchElement::Browse
      expr = name = tbranch->GetName();

      Int_t pos = expr.First('[');
      if (pos != kNPOS)
         expr.Remove(pos);
      if (tbranch->GetMother()) {
         TString mothername = tbranch->GetMother()->GetName();
         pos = mothername.First('[');
         if (pos != kNPOS) {
            mothername.Remove(pos);
         }
         Int_t len = mothername.Length();
         if (len) {
            if (mothername(len-1) != '.') {
               // We do not know for sure whether the mother's name is
               // already preprended.  So we need to check:
               //    a) it is prepended
               //    b) it is NOT the name of a daugher (i.e. mothername.mothername exist)
               TString doublename = mothername;
               doublename.Append(".");
               Int_t isthere = (expr.Index(doublename) == 0);
               if (!isthere) {
                  expr.Prepend(doublename);
               } else {
                  if (tbranch->GetMother()->FindBranch(mothername)) {
                     doublename.Append(mothername);
                     isthere = (expr.Index(doublename) == 0);
                     if (!isthere) {
                        mothername.Append(".");
                        expr.Prepend(mothername);
                     }
                  } else {
                     // Nothing to do because the mother's name is
                     // already in the name.
                  }
               }
            } else {
               // If the mother's name end with a dot then
               // the daughter probably already contains the mother's name
               if (expr.Index(mothername) == kNPOS) {
                  expr.Prepend(mothername);
               }
            }
         }
      }

      AdjustExpr(expr, name);

      return true;
   }

   TH1 *DrawBranchElement(std::unique_ptr<RHolder> &obj)
   {
      fHandle2 = obj.get();

      auto tbranch = obj->get_object<TBranchElement>();
      TString expr, name;
      if (!GetDrawExpr(tbranch, expr, name))
         return nullptr;

      return DrawTree(tbranch->GetTree(), expr.Data(), name.Data());
   }

   bool GetDrawExpr(const TVirtualBranchBrowsable *browsable, TString &expr, TString &name)
   {
      if (!browsable)
         return false;

      auto cl = browsable->GetClassType();

      bool can_draw  = (!cl || (cl->GetCollectionProxy() && cl->GetCollectionProxy()->GetType() > 0));
      if (!can_draw)
         return false;

      auto br = browsable->GetBranch();
      if (!br)
         return false;

      browsable->GetScope(expr);

      name = browsable->GetName();

      // If this is meant to be run on the collection
      // we need to "move" the "@" from branch.@member
      // to branch@.member
      // fullname.ReplaceAll(".@","@.");

      AdjustExpr(expr, name);

      return true;
   }


   TH1 *DrawBranchBrowsable(std::unique_ptr<RHolder> &obj)
   {
      auto browsable = obj->get_object<TVirtualBranchBrowsable>();

      TString expr, name;

      if (!GetDrawExpr(browsable, expr, name))
         return nullptr;

      return DrawTree(browsable->GetBranch()->GetTree(), expr.Data(), name.Data());
   }

};
