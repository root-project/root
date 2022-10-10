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

#include <memory>
#include <vector>
#include <string>
#include <functional>

class TTree;

namespace ROOT {
namespace Experimental {

class RWebWindow;
class REveManager;

class RTreeViewer {

public:

   using PerformDrawCallback_t = std::function<void(const std::string &)>;

   struct RBranchInfo {
      std::string fName, fTitle;
      RBranchInfo() = default;
      RBranchInfo(const std::string &_name, const std::string &_title) : fName(_name), fTitle(_title) {}
   };

   struct RConfig {
      std::string fExprX, fExprY, fExprZ, fExprCut;
      std::vector<RBranchInfo> fBranches;
   };

   RTreeViewer(TTree *tree = nullptr);
   virtual ~RTreeViewer();

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   std::string GetWindowAddr() const;

   void SetTree(TTree *tree);

   void SetCallback(PerformDrawCallback_t func) { fCallback = func; }

   /** Configures default hierarchy browser visibility, only has effect before showing web window */
   void SetShowHierarchy(bool on = true) { fShowHierarchy = on; }

   /** Returns default hierarchy browser visibility */
   bool GetShowHierarchy() const { return fShowHierarchy; }

   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   void Update();

private:

   TTree *fTree{nullptr};                  ///<! TTree to show
   std::string fTitle;                     ///<! title of tree viewer
   std::shared_ptr<RWebWindow> fWebWindow; ///<! web window
   bool fShowHierarchy{false};             ///<! show TTree hierarchy
   RConfig fCfg;                           ///<! configuration, exchanged between client and server
   PerformDrawCallback_t fCallback;        ///<! callback invoked when tree draw performed

   void WebWindowConnect(unsigned connid);
   void WebWindowCallback(unsigned connid, const std::string &arg);

   void SendCfg(unsigned connid);

   void UpdateBranchList();

   void InvokeTreeDraw(const std::string &json);
};

} // namespace Experimental
} // namespace ROOT

#endif
