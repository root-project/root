/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RWrapper
#define ROOT7_Browsable_RWrapper

#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class RWrapper
\ingroup rbrowser
\brief Wrapper for other element - to provide different name
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-11-22
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RWrapper : public RElement {
   std::string fName;
   std::shared_ptr<RElement> fElem;
   bool fExapndByDefault{false};

public:
   RWrapper() = default;

   RWrapper(const std::string &name, std::shared_ptr<RElement> elem) : fName(name), fElem(elem) {}

   virtual ~RWrapper() = default;

   /** Name of element, must be provided in derived classes */
   std::string GetName() const override { return fName; }

   /** Title of element (optional) */
   std::string GetTitle() const override { return fElem->GetTitle(); }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override { return fElem->GetChildsIter(); }

   /** Returns element content, depends from kind. Can be "text" or "image64" */
   std::string GetContent(const std::string &kind = "text") override { return fElem->GetContent(kind); }

   /** Access object */
   std::unique_ptr<RHolder> GetObject() override { return fElem->GetObject(); }

   /** Get default action */
   EActionKind GetDefaultAction() const override { return fElem->GetDefaultAction(); }

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override { return fElem->IsCapable(action); }

   bool IsExpandByDefault() const override { return fExapndByDefault || fElem->IsExpandByDefault(); }
   void SetExpandByDefault(bool on = true) { fExapndByDefault = on; }


};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif
