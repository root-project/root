/// \file ROOT/RBrowser.hxx
/// \ingroup WebGui ROOT7
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

#ifndef ROOT7_RBrowserItem
#define ROOT7_RBrowserItem

namespace ROOT {
namespace Experimental {

/** Request send from client to get content of path element */
class RRootBrowserRequest {
public:
   std::string path;   ///< requested path
   int first{0};       ///< first child to request
   int number{0};      ///< number of childs to request, 0 - all childs
   std::string sort;   ///< kind of sorting
};

/** Representation of single item in the file browser */
class RRootBrowserItem {
public:
   std::string name;     ///< file name
   std::string fsize;    ///< file size
   std::string mtime;    ///< modification time
   std::string ftype;    ///< file attributes
   std::string fuid;     ///< user id
   std::string fgid;     ///< group id
   int nchilds{0};       ///< number of childs
   bool checked{false};  ///< is checked
   bool expanded{false}; ///< is expanded
   RRootBrowserItem() = default;
   RRootBrowserItem(const std::string &_name, const std::string &_fsize, const std::string &_mtime,
                    const std::string &_ftype, const std::string &_fuid, const std::string &_fgid,
                    int _nchilds = 0) : name(_name), fsize(_fsize), mtime(_mtime), ftype(_ftype),
                    fuid(_fuid), fgid(_fgid), nchilds(_nchilds) {}
};

/** Reply on browser request */
class RRootBrowserReply {
public:
   std::string path;     ///< reply path
   int nchilds{0};       ///< total number of childs in the node
   int first{0};         ///< first node in returned list
   std::vector<RRootBrowserItem> nodes; ///< list of nodes
};


} // namespace Experimental
} // namespace ROOT

#endif


