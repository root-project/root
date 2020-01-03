/// \file ROOT/RFileDialog.hxx
/// \ingroup rbrowser
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-10-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFileDialog
#define ROOT7_RFileDialog

#include <ROOT/RWebWindow.hxx>
#include <ROOT/RBrowsable.hxx>

#include <vector>
#include <memory>
#include <stdint.h>

namespace ROOT {
namespace Experimental {


/** Initial message send to client to configure layout */


/// function signature for connect/disconnect call-backs
/// argument is connection id
using RFileDialogCallback_t = std::function<void(const std::string &)>;


/** Web-based FileDialog */

class RFileDialog {
public:

   enum EDialogTypes {
      kOpenFile,
      kSaveAs,
      kNewFile
   };

protected:

   EDialogTypes fKind{kOpenFile};      ///<! dialog kind OpenFile, SaveAs, NewFile
   std::string  fTitle;                ///<! title, when not specified default will be used
   RBrowsable   fBrowsable;            ///<! central browsing element

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window for file dialog

   bool fDidSelect{false};           ///<! true when dialog is selected or closed
   std::string fSelect;              ///<! result of file selection
   RFileDialogCallback_t fCallback;  ///<! function receiving result, called once

   static std::string TypeAsString(EDialogTypes kind);

   void SendInitMsg(unsigned connid);
   void SendChPathMsg(unsigned connid);

   void ProcessMsg(unsigned connid, const std::string &arg);

   void InvokeCallBack();

   static std::string Dialog(EDialogTypes kind, const std::string &title, const std::string &fname);

public:

   RFileDialog(EDialogTypes kind = kOpenFile, const std::string &title = "", const std::string &fname = "");
   virtual ~RFileDialog();

   const EDialogTypes &GetType() const { return fKind; }

   void SetCallback(RFileDialogCallback_t callback);

   void Show(const RWebDisplayArgs &args = "");
   void Hide();

   bool IsCompleted() const { return fDidSelect; }
   const std::string &GetFileName() const { return fSelect; }

   static std::string OpenFile(const std::string &title = "", const std::string &fname = "");
   static std::string SaveAs(const std::string &title = "", const std::string &fname = "");
   static std::string NewFile(const std::string &title = "", const std::string &fname = "");

   static std::shared_ptr<RFileDialog> Embedded(const std::shared_ptr<RWebWindow> &window, const std::string &args);

};

} // namespace Experimental
} // namespace ROOT

#endif
