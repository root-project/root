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

#include <TSystem.h>

#include <vector>
#include <sstream>
#include <iostream>

namespace ROOT {
namespace Experimental {

/** Base description of browser item, required only to build hierarchy */

class RBaseItem {
public:
   int id{0};               ///< node id, index in array
   std::string name;        ///< node name
   std::vector<int> chlds;  ///< list of childs id

   RBaseItem(int _id = 0) : id(_id) {}
   RBaseItem(int _id, const std::string &_name) : id(_id), name(_name) {}
};

/** class ROOT::Experimental::RFileItem
 * \ingroup webdisplay
 * Descriptor for the openui5 File Browser Item, used in RBrowser
 */

class RFileItem : public RBaseItem  {
public:
   int type{0};             ///<! file type
   int uid{0};              ///<! file uid
   int gid{0};              ///<! file gid
   bool islink{false};      ///<! true if symbolic link
   bool isdir{false};       ///<! true if directory
   long modtime;            ///<! modification time
   int64_t size;            ///<! file size
   std::string fsize;       ///< file size
   std::string mtime;       ///< file attributes
   std::string ftype;       ///< file attributes
   std::string fuid;        ///< user id
   std::string fgid;        ///< group id

   RFileItem(int _id, const std::string &_name, FileStat_t &stat);
};

class RBrowserFSDescription {

   std::vector<RFileItem> fDesc;    ///< converted description, send to client

   int fTopNode{0};                 ///<! selected top node

   std::string fDrawJson;           ///<! JSON with main nodes drawn by client
   bool fPreferredOffline{false};   ///<! indicates that full description should be provided to client

   void ResetRndrInfos();

public:
   RBrowserFSDescription() = default;

   void AddFolder(const char *name);
   void AddFile(const char *name);
   void Build(const std::string &path);

   /** Number of unique nodes in the geometry */
   int GetNumNodes() const { return fDesc.size(); }

   bool IsBuild() const { return GetNumNodes() > 0; }

   std::string ProcessBrowserRequest(const std::string &req = "");
};

class RBrowser {

protected:

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! connection id

   RBrowserFSDescription fDesc; ///<! file system decription

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to show geometry

   void WebWindowCallback(unsigned connid, const std::string &arg);

public:
   RBrowser();
   virtual ~RBrowser();

   // method required when any panel want to be inserted into the RCanvas
   std::shared_ptr<RWebWindow> GetWindow();

   /// show Browser in specified place
   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   /// hide Browser
   void Hide();

};

} // namespace Experimental
} // namespace ROOT

#endif
