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

#include <string>
#include <vector>
#include <memory>

namespace ROOT {
namespace Experimental {

/** Request send from client to get content of path element */
class RBrowserRequest {
public:
   std::string path; ///< requested path
   int first{0};     ///< first child to request
   int number{0};    ///< number of childs to request, 0 - all childs
   std::string sort; ///< kind of sorting
};

/** Representation of single item in the browser */
class RBrowserItem {
protected:
   std::string name;     ///< item name
   int nchilds{0};       ///< number of childs
   bool checked{false};  ///< is checked
   bool expanded{false}; ///< is expanded
public:
   RBrowserItem() = default;
   RBrowserItem(const std::string &_name, int _nchilds = 0) : name(_name), nchilds(_nchilds) {}

   const std::string &GetName() const { return name; }

   void SetChecked(bool on = true) { checked = on; }
   void SetExpanded(bool on = true) { expanded = on; }

   // must be here, one needs virtual table for correct streaming of RBrowserReply
   virtual ~RBrowserItem() = default;
};

/** Reply on browser request */
class RBrowserReply {
public:
   std::string path;                  ///< reply path
   int nchilds{0};                    ///< total number of childs in the node
   int first{0};                      ///< first node in returned list
   std::vector<RBrowserItem *> nodes; ///< list of pointers, no ownership!
};

/** Reply on browser request, with ownership */
class RBrowserReplyNew {
public:
   std::string path;                  ///< reply path
   int nchilds{0};                    ///< total number of childs in the node
   int first{0};                      ///< first node in returned list
   std::vector<std::unique_ptr<RBrowserItem>> nodes; ///< list of pointers with ownership!
};


} // namespace Experimental
} // namespace ROOT

#endif


