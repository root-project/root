/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowsable
#define ROOT7_RBrowsable

#include <ROOT/RBrowserItem.hxx>

#include <ROOT/Browsable/RHolder.hxx>
#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RWrapper.hxx>
#include <ROOT/Browsable/RGroup.hxx>
#include <ROOT/Browsable/RProvider.hxx>

#include <memory>
#include <string>
#include <vector>

class TObject;


namespace ROOT {
namespace Experimental {

/** \class RBrowsable
\ingroup rbrowser
\brief Way to browse (hopefully) everything in ROOT
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsable {

   std::shared_ptr<Browsable::RElement> fTopElement;    ///<! top element for the RBrowsable

   RElementPath_t  fWorkingPath;                        ///<! path showed in Breadcrumb
   std::shared_ptr<Browsable::RElement> fWorkElement;   ///<! main element used for working in browser dialog

   RElementPath_t fLastPath;                             ///<! path to last used element
   std::shared_ptr<Browsable::RElement> fLastElement;    ///<! last element used in request
   std::vector<std::unique_ptr<RBrowserItem>> fLastItems; ///<! created browser items - used in requests
   bool fLastAllChilds{false};                           ///<! if all chlds were extracted
   std::vector<const RBrowserItem *> fLastSortedItems;   ///<! sorted child items, used in requests
   std::string fLastSortMethod;                          ///<! last sort method

   RElementPath_t DecomposePath(const std::string &path);

   void ResetLastRequest();

   bool ProcessBrowserRequest(const RBrowserRequest &request, RBrowserReply &reply);

public:
   RBrowsable() = default;

   RBrowsable(std::shared_ptr<Browsable::RElement> elem) { SetTopElement(elem); }

   virtual ~RBrowsable() = default;

   void SetTopElement(std::shared_ptr<Browsable::RElement> elem);

   void SetWorkingDirectory(const std::string &strpath);
   void SetWorkingPath(const RElementPath_t &path);

   const RElementPath_t &GetWorkingPath() const { return fWorkingPath; }

   std::string ProcessRequest(const RBrowserRequest &request);

   std::shared_ptr<Browsable::RElement> GetElement(const std::string &str);
   std::shared_ptr<Browsable::RElement> GetElementFromTop(const RElementPath_t &path);
};


} // namespace Experimental
} // namespace ROOT

#endif
