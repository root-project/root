/// \file ROOT/RBrowser.hxx
/// \ingroup rbrowser
/// \author Bertrand Bellenot <bertrand.bellenot@cern.ch>
/// \author Sergey Linev <S.Linev@gsi.de>
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
#include <ROOT/RBrowserData.hxx>

#include <vector>
#include <memory>

class TString;
class TCanvas;
class TFile;

namespace ROOT {
namespace Experimental {

class RCanvas;

/** Web-based ROOT file browser */

class RBrowser {

protected:

   struct EditorPage {
      std::string fName;
      std::string fTitle;
      std::string fFileName;
      std::string fContent;
   };

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! default connection id

   bool fUseRCanvas{false};             ///<!  which canvas should be used
   std::vector<std::unique_ptr<TCanvas>> fCanvases;  ///<! canvases created by browser, should be closed at the end
   std::string fActiveTab;            ///<! name of active for tab (RCanvas, TCanvas or EditorPage)
   std::vector<std::shared_ptr<ROOT::Experimental::RCanvas>> fRCanvases; ///<!  ROOT7 canvases
   std::vector<std::unique_ptr<EditorPage>> fEditors;   ///<! list of text editors
   int fEditorsCnt{0};                                  ///<! counter for created editors

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to browser

   RBrowserData  fBrowsable;                   ///<! central browsing element

   TCanvas *AddCanvas();
   TCanvas *GetActiveCanvas() const;
   std::string GetCanvasUrl(TCanvas *canv);

   std::shared_ptr<RCanvas> AddRCanvas();
   std::shared_ptr<RCanvas> GetActiveRCanvas() const;
   std::string GetRCanvasUrl(std::shared_ptr<RCanvas> &canv);

   EditorPage *AddEditor();
   EditorPage *GetActiveEditor() const;

   void CloseTab(const std::string &name);

   std::string ProcessBrowserRequest(const std::string &msg);
   std::string ProcessDblClick(const std::string &path, const std::string &drawingOptions, const std::string &);
   long ProcessRunCommand(const std::string &file_path);
   void ProcessSaveFile(const std::string &file_path);
   std::string GetCurrentWorkingDirectory();

   void SendInitMsg(unsigned connid);
   void ProcessMsg(unsigned connid, const std::string &arg);

   std::string SendEditorContent(EditorPage *editor);

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
