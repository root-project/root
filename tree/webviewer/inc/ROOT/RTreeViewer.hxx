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

class TTree;

namespace ROOT {
namespace Experimental {

class RWebWindow;
class REveManager;

class RTreeViewer {

   TTree *fTree{nullptr};                  ///<! TTree to show
   std::string fTitle;                     ///<! title of tree viewer
   std::shared_ptr<RWebWindow> fWebWindow; ///<! web window
   bool fShowHierarchy{false};             ///<! show TTree hierarchy

   void WebWindowCallback(unsigned connid, const std::string &arg);

   void SendViewerData(unsigned connid);

public:

   RTreeViewer(TTree *tree = nullptr);
   virtual ~RTreeViewer();

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   std::string GetWindowAddr() const;

   void SetTree(TTree *tree);

   /** Configures default hierarchy browser visibility, only has effect before showing web window */
   void SetShowHierarchy(bool on = true) { fShowHierarchy = on; }

   /** Returns default hierarchy browser visibility */
   bool GetShowHierarchy() const { return fShowHierarchy; }

   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   void Update();

};

} // namespace Experimental
} // namespace ROOT

#endif
