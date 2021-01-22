/// \file ROOT/RBrowser.hxx
/// \ingroup rbrowser
/// \author Bertrand Bellenot <bertrand.bellenot@cern.ch>
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-02-28
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowser
#define ROOT7_RBrowser

#include <ROOT/RWebWindow.hxx>
#include <ROOT/RBrowserData.hxx>

#include <vector>
#include <memory>

class TString;
class TCanvas;
class TFile;

namespace ROOT {
namespace Experimental {

class RCanvas;
class RBrowserWidget;

/** Web-based ROOT file browser */

class RBrowser {

protected:

   struct BrowserPage {
      bool fIsEditor{true};   ///<! either editor or image viewer
      std::string fName;
      std::string fTitle;
      std::string fFileName;
      std::string fContent;
      bool fFirstSend{false};  ///<! if editor content was send at least one
      std::string fItemPath;   ///<! item path in the browser
      BrowserPage(bool is_edit) : fIsEditor(is_edit) {}
      std::string GetKind() const { return fIsEditor ? "edit" : "image"; }
   };

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! default connection id

   bool fUseRCanvas{false};             ///<!  which canvas should be used
   std::vector<std::unique_ptr<TCanvas>> fCanvases;  ///<! canvases created by browser, should be closed at the end
   std::string fActiveTab;            ///<! name of active for tab (RCanvas, TCanvas or BrowserPage)
   std::vector<std::shared_ptr<ROOT::Experimental::RCanvas>> fRCanvases; ///<!  ROOT7 canvases
   std::vector<std::shared_ptr<RBrowserWidget>> fWidgets; ///<!  all browser widgets
   std::vector<std::unique_ptr<BrowserPage>> fPages;      ///<! list of text editors
   int fPagesCnt{0};                                     ///<! counter for created editors

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to browser

   RBrowserData  fBrowsable;                   ///<! central browsing element

   TCanvas *AddCanvas();
   TCanvas *GetActiveCanvas() const;
   std::string GetCanvasUrl(TCanvas *);

   std::shared_ptr<RCanvas> AddRCanvas();
   std::shared_ptr<RCanvas> GetActiveRCanvas() const;
   std::string GetRCanvasUrl(std::shared_ptr<RCanvas> &);

   std::shared_ptr<RBrowserWidget> AddWidget(const std::string &kind);
   std::shared_ptr<RBrowserWidget> GetActiveWidget() const;

   BrowserPage *AddPage(bool is_editor);
   BrowserPage *GetPage(const std::string &name) const;
   BrowserPage *GetActivePage() const { return GetPage(fActiveTab); }
   BrowserPage *FindPageFor(const std::string &item_path, bool is_editor = true);

   void CloseTab(const std::string &name);

   std::string ProcessBrowserRequest(const std::string &msg);
   std::string ProcessDblClick(const std::string &path, const std::string &drawingOptions, const std::string &);
   std::string ProcessNewTab(const std::string &msg);
   long ProcessRunMacro(const std::string &file_path);
   void ProcessSaveFile(const std::string &fname, const std::string &content);
   std::string GetCurrentWorkingDirectory();

   std::vector<std::string> GetRootHistory();
   std::vector<std::string> GetRootLogs();

   void SendInitMsg(unsigned connid);
   void ProcessMsg(unsigned connid, const std::string &arg);

   std::string SendPageContent(BrowserPage *editor);

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
