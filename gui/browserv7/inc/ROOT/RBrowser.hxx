// Authors: Bertrand Bellenot <bertrand.bellenot@cern.ch> Sergey Linev <S.Linev@gsi.de>
// Date: 2019-02-28
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

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

namespace ROOT {

class RBrowserWidget;
class RBrowserTimer;

class RBrowser {

   friend class RBrowserTimer;

protected:

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! default connection id

   bool fUseRCanvas{false};              ///<!  which canvas should be used
   bool fCatchWindowShow{true};          ///<! if arbitrary RWebWindow::Show calls should be catched by browser
   std::string fActiveWidgetName;        ///<! name of active widget
   std::vector<std::shared_ptr<RBrowserWidget>> fWidgets; ///<!  all browser widgets
   int fWidgetCnt{0};                                     ///<! counter for created widgets
   std::string fPromptFileOutput;        ///<! file name for prompt output
   float fLastProgressSend{0};           ///<! last value of send progress
   long long fLastProgressSendTm{0};      ///<! time when last progress message was send

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to browser

   RBrowserData  fBrowsable;                 ///<! central browsing element
   std::unique_ptr<RBrowserTimer>    fTimer; ///<!  timer to handle postponed requests
   std::vector<std::vector<std::string>> fPostponed; ///<! postponed messages, handled in timer

   std::shared_ptr<RBrowserWidget> AddWidget(const std::string &kind);
   std::shared_ptr<RBrowserWidget> AddCatchedWidget(RWebWindow *win, const std::string &kind);
   std::shared_ptr<RBrowserWidget> FindWidget(const std::string &name, const std::string &kind = "") const;
   std::shared_ptr<RBrowserWidget> GetActiveWidget() const { return FindWidget(fActiveWidgetName); }

   void CloseTab(const std::string &name);

   std::string ProcessBrowserRequest(const std::string &msg);
   std::string ProcessDblClick(unsigned connid, std::vector<std::string> &args);
   std::string NewWidgetMsg(std::shared_ptr<RBrowserWidget> &widget);
   void ProcessRunMacro(const std::string &file_path);
   void ProcessSaveFile(const std::string &fname, const std::string &content);
   std::string GetCurrentWorkingDirectory();

   std::vector<std::string> GetRootHistory();
   std::vector<std::string> GetRootLogs();

   void SendInitMsg(unsigned connid);
   void ProcessMsg(unsigned connid, const std::string &arg);
   void SendProgress(unsigned connid, float progr);

   void AddInitWidget(const std::string &kind);

   void CheckWidgtesModified(unsigned connid);

   void ProcessPostponedRequests();

public:
   RBrowser(bool use_rcanvas = false);
   virtual ~RBrowser();

   bool GetUseRCanvas() const { return fUseRCanvas; }
   void SetUseRCanvas(bool on = true) { fUseRCanvas = on; }

   void AddTCanvas() { AddInitWidget("tcanvas"); }
   void AddRCanvas() { AddInitWidget("rcanvas"); }

   /// show Browser in specified place
   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   /// hide Browser
   void Hide();

   std::string GetWindowUrl(bool remote);

   void SetWorkingPath(const std::string &path);

   /// Enable/disable catch of RWebWindow::Show calls to embed created widgets, default on
   void SetCatchWindowShow(bool on = true) { fCatchWindowShow = on; }

   /// Is RWebWindow::Show calls catched for embeding of created widgets
   bool GetCatchWindowShow() const { return fCatchWindowShow; }

   bool ActivateWidget(const std::string &title, const std::string &kind = "");

   void ClearOnClose(const std::shared_ptr<void> &handle);

};

} // namespace ROOT

#endif
