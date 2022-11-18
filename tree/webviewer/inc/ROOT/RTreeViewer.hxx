// Author: Sergey Linev, 7.10.2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RTreeViewer
#define ROOT7_RTreeViewer

#include <ROOT/RWebDisplayArgs.hxx>

#include "Rtypes.h"

#include <memory>
#include <vector>
#include <string>
#include <functional>

class TTree;
class TBranch;
class TLeaf;
class TObjArray;

namespace ROOT {
namespace Experimental {

class RWebWindow;
class TProgressTimer;

class RTreeViewer {

friend class TProgressTimer;

public:

   using PerformDrawCallback_t = std::function<void(const std::string &)>;

   struct RBranchInfo {
      std::string fName, fTitle;
      RBranchInfo() = default;
      RBranchInfo(const std::string &_name, const std::string &_title) : fName(_name), fTitle(_title) {}
   };

   struct RConfig {
      std::string fTreeName, fExprX, fExprY, fExprZ, fExprCut, fOption;
      std::vector<RBranchInfo> fBranches;
      Long64_t fNumber{0}, fFirst{0}, fStep{1}, fLargerStep{2}, fTreeEntries{0};
   };

   RTreeViewer(TTree *tree = nullptr);
   virtual ~RTreeViewer();

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   std::string GetWindowAddr() const;

   void SetTree(TTree *tree);

   bool SuggestLeaf(const TLeaf *leaf);

   bool SuggestBranch(const TBranch *branch);

   bool SuggestExpression(const std::string &expr);

   void SetCallback(PerformDrawCallback_t func) { fCallback = func; }

   /** Configures default hierarchy browser visibility, only has effect before showing web window */
   void SetShowHierarchy(bool on = true) { fShowHierarchy = on; }

   /** Returns default hierarchy browser visibility */
   bool GetShowHierarchy() const { return fShowHierarchy; }

   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   void Update();

   static RTreeViewer *NewViewer(TTree *);

private:

   TTree *fTree{nullptr};                  ///<! TTree to show
   std::string fTitle;                     ///<! title of tree viewer
   std::shared_ptr<RWebWindow> fWebWindow; ///<! web window
   bool fShowHierarchy{false};             ///<! show TTree hierarchy
   RConfig fCfg;                           ///<! configuration, exchanged between client and server
   PerformDrawCallback_t fCallback;        ///<! callback invoked when tree draw performed
   std::unique_ptr<TProgressTimer> fProgrTimer; ///<! timer used to get draw progress
   std::string fLastSendProgress;          ///<! last send progress to client

   void WebWindowConnect(unsigned connid);
   void WebWindowCallback(unsigned connid, const std::string &arg);

   void SendCfg(unsigned connid);

   std::string FormatItemName(const std::string &name);

   void AddBranches(TObjArray *branches);

   void UpdateConfig();

   void SendProgress(bool completed = false);

   void InvokeTreeDraw(const std::string &json);
};

} // namespace Experimental
} // namespace ROOT

#endif
