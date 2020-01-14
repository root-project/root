/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RGroup
#define ROOT7_Browsable_RGroup

#include <ROOT/Browsable/RElement.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class RGroup
\ingroup rbrowser
\brief Group of browsable elements - combines several different elements together.
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-11-22
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RGroup : public RElement {

   std::string fName;
   std::string fTitle;
   std::vector<std::shared_ptr<RElement>> fChilds;

public:

   RGroup(const std::string &name, const std::string &title = "") : RElement(), fName(name), fTitle(title) {}

   virtual ~RGroup() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fTitle; }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override;

   void Add(std::shared_ptr<RElement> elem) { fChilds.emplace_back(elem); }

   auto &GetChilds() const { return fChilds; }
};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif
