/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/TObjectElement.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>


#include "RooWorkspace.h"

using namespace ROOT::Browsable;
using namespace std::string_literals;


class xRooBrowsingElement : public TObjectElement {
public:
   xRooBrowsingElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return (action == kActDraw6);
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return kActDraw6;
   }

   std::string GetContent(const std::string &kind) override
   {
      return TObjectElement::GetContent(kind);
   }

};


////////////////////////////////////////////////////////////
/// Representing RooWorkspace in browsables

class xRooWorkspaceElement : public TObjectElement {

public:
   xRooWorkspaceElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   Long64_t GetSize() const override
   {
      auto ws = dynamic_cast<const RooWorkspace *>(CheckObject());
      return ws ? 100 : -1;
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return kActBrowse;
   }

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return action == kActBrowse;
   }

};

// ==============================================================================================

class xRooProvider : public RProvider {

public:
   xRooProvider()
   {
      RegisterBrowse(RooWorkspace::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<xRooWorkspaceElement>(object);
      });
   }

} newxRooProvider;
