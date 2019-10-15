/// \file ROOT/RBrowser.hxx
/// \ingroup WebGui ROOT7
/// \author Bertrand Bellenot <bertrand.bellenot@cern.ch>
/// \date 2019-02-28
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowser
#define ROOT7_RBrowser

#include <ROOT/RWebWindow.hxx>
#include <ROOT/RCanvas.hxx>
#include <ROOT/RBrowserItem.hxx>
#include <ROOT/RBrowsable.hxx>
#include <ROOT/RFileBrowsable.hxx>

#include <vector>
#include <memory>
#include <stdint.h>

class TString;
class TCanvas;
class TFile;

namespace ROOT {
namespace Experimental {

/** Web-based ROOT file browser */

class RBrowser {

protected:

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! default connection id

   std::string fDescPath;                ///<! last scanned directory
   std::vector<RBrowserFileItem> fDesc;     ///<! plain list of current directory
   std::vector<RBrowserFileItem *> fSorted; ///<! current sorted list (no ownership)

   bool fUseRCanvas{false};             ///<!  which canvas should be used
   std::vector<std::unique_ptr<TCanvas>> fCanvases;  ///<! canvases created by browser, should be closed at the end
   std::string fActiveCanvas;            ///<! name of active for RBrowser canvas, not a gPad!
   std::vector<std::shared_ptr<ROOT::Experimental::RCanvas>> fRCanvases; ///<!  ROOT7 canvases

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to browser

   RBrowsable  fBrowsable;                   ///<! central browsing element

   TCanvas *AddCanvas();
   TCanvas *GetActiveCanvas() const;
   TFile *OpenFile(const std::string &fname);
   std::string GetCanvasUrl(TCanvas *canv);
   void CloseCanvas(const std::string &name);

   std::shared_ptr<RCanvas> AddRCanvas();
   std::shared_ptr<RCanvas> GetActiveRCanvas() const;
   std::string GetRCanvasUrl(std::shared_ptr<RCanvas> &canv);

   void AddFolder(const char *name);
   void AddFile(const char *name);
   void Browse(const std::string &path);
   void Build(const std::string &path);
   std::string GetClassIcon(std::string &classname);
   std::string GetFileIcon(TString &name);
   std::string ProcessBrowserRequest(const std::string &msg);
   std::string ProcessDblClick(const std::string &path, const std::string &drawingOptions);
   long ProcessRunCommand(const std::string &file_path);
   bool ProcessSaveFile(const std::string &file_path);
   std::string GetCurrentWorkingDirectory();

   bool IsBuild() const { return fDesc.size() > 0; }

   void SendInitMsg(unsigned connid);
   void WebWindowCallback(unsigned connid, const std::string &arg);

public:
   RBrowser(bool use_rcanvas = true);
   virtual ~RBrowser();

   bool GetUseRCanvas() const { return fUseRCanvas; }
   void SetUseRCanvas(bool on = true) { fUseRCanvas = on; }

   /// show Browser in specified place
   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   /// hide Browser
   void Hide();

};

} // namespace Experimental
} // namespace ROOT

#endif
