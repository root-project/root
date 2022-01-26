/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RItem.hxx>

#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageStorage.hxx>

#include "TClass.h"
#include "RFieldHolder.hxx"


using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;


// ==============================================================================================

/** \class RFieldElement
\ingroup rbrowser
\brief Browsing element representing of RField
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFieldElement : public RElement {
protected:
   std::shared_ptr<ROOT::Experimental::Detail::RPageSource> fNtplSource;

   std::string fParentName;

   ROOT::Experimental::DescriptorId_t fFieldId;

public:

   RFieldElement(std::shared_ptr<ROOT::Experimental::Detail::RPageSource> ntplSource,
                 const std::string &parent_name,
                 const ROOT::Experimental::DescriptorId_t id)
                 : RElement(), fNtplSource(ntplSource), fParentName(parent_name), fFieldId(id) {}

   virtual ~RFieldElement() = default;

   /** Name of RField */
   std::string GetName() const override
   {
      return fNtplSource->GetSharedDescriptorGuard()->GetFieldDescriptor(fFieldId).GetFieldName();
   }

   /** Title of RField */
   std::string GetTitle() const override
   {
      auto descriptorGuard = fNtplSource->GetSharedDescriptorGuard();
      auto &fld = descriptorGuard->GetFieldDescriptor(fFieldId);
      return "RField name "s + fld.GetFieldName() + " type "s + fld.GetTypeName();
   }

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   /** Return class of field  - for a moment using RNTuple class as dummy */
   const TClass *GetClass() const { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }

   std::unique_ptr<RHolder> GetObject() override
   {
      return std::make_unique<RFieldHolder>(fNtplSource, fParentName, fFieldId);
   }

   EActionKind GetDefaultAction() const override
   {
      auto descriptorGuard = fNtplSource->GetSharedDescriptorGuard();
      auto range = descriptorGuard->GetFieldIterable(fFieldId);
      if (range.begin() != range.end()) return kActNone;
      return kActDraw7;
   }

   bool IsCapable(EActionKind kind) const override
   {
      if ((kind == kActDraw6) || (kind == kActDraw7))
         return GetDefaultAction() == kActDraw7;

      return false;
   }

};

// ==============================================================================================

/** \class RNTupleElement
\ingroup rbrowser
\brief Browsing element representing of RNTuple
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleElement : public RElement {
protected:
   std::shared_ptr<ROOT::Experimental::Detail::RPageSource> fNtplSource;

public:
   RNTupleElement(const std::string &ntplName, const std::string &filename)
   {
      ROOT::Experimental::RNTupleReadOptions options;
      fNtplSource = ROOT::Experimental::Detail::RPageSource::Create(ntplName, filename, options);
      fNtplSource->Attach();
   }

   virtual ~RNTupleElement() = default;

   /** Returns true if no ntuple found */
   bool IsNull() const { return !fNtplSource; }

   /** Name of NTuple */
   std::string GetName() const override { return fNtplSource->GetSharedDescriptorGuard()->GetName(); }

   /** Title of NTuple */
   std::string GetTitle() const override { return "RNTuple title"s; }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override;

   const TClass *GetClass() const { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }

   //EActionKind GetDefaultAction() const override;

   //bool IsCapable(EActionKind) const override;
};


// ==============================================================================================

/** \class RFieldsIterator
\ingroup rbrowser
\brief Iterator over RNTuple fields
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RFieldsIterator : public RLevelIter {

   std::shared_ptr<ROOT::Experimental::Detail::RPageSource> fNtplSource;
   std::vector<ROOT::Experimental::DescriptorId_t> fFieldIds;
   std::string fParentName;
   int fCounter{-1};

public:
   RFieldsIterator(std::shared_ptr<ROOT::Experimental::Detail::RPageSource> ntplSource,
                   std::vector<ROOT::Experimental::DescriptorId_t> &&ids,
                   const std::string &parent_name = ""s)
                   : fNtplSource(ntplSource), fFieldIds(ids), fParentName(parent_name)
   {
   }

   virtual ~RFieldsIterator() = default;

   bool Next() override
   {
      return ++fCounter < (int) fFieldIds.size();
   }

   std::string GetItemName() const override
   {
      return fNtplSource->GetSharedDescriptorGuard()->GetFieldDescriptor(fFieldIds[fCounter]).GetFieldName();
   }

   bool CanItemHaveChilds() const override
   {
      auto descriptorGuard = fNtplSource->GetSharedDescriptorGuard();
      auto subrange = descriptorGuard->GetFieldIterable(fFieldIds[fCounter]);
      return subrange.begin() != subrange.end();
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {

      int nchilds = 0;
      std::string fieldName;
      std::string typeName;

      {
         auto descriptorGuard = fNtplSource->GetSharedDescriptorGuard();
         for (auto &sub : descriptorGuard->GetFieldIterable(fFieldIds[fCounter])) {
            (void)sub;
            nchilds++;
         }

         auto &field = descriptorGuard->GetFieldDescriptor(fFieldIds[fCounter]);
         fieldName = field.GetFieldName();
         typeName = field.GetTypeName();
      }

      auto item = std::make_unique<RItem>(fieldName, nchilds, nchilds > 0 ? "sap-icon://split" : "sap-icon://e-care");

      item->SetTitle("RField name "s + fieldName + " type "s + typeName);

      return item;
   }

   std::shared_ptr<RElement> GetElement() override
   {
      return std::make_shared<RFieldElement>(fNtplSource, fParentName, fFieldIds[fCounter]);
   }
};


std::unique_ptr<RLevelIter> RFieldElement::GetChildsIter()
{
   std::vector<ROOT::Experimental::DescriptorId_t> ids;
   std::string prefix;

   {
      auto descriptorGuard = fNtplSource->GetSharedDescriptorGuard();

      for (auto &f : descriptorGuard->GetFieldIterable(fFieldId))
         ids.emplace_back(f.GetId());

      if (ids.size() == 0)
         return nullptr;

      prefix = fParentName;
      auto &fld = descriptorGuard->GetFieldDescriptor(fFieldId);
      prefix.append(fld.GetFieldName());
   }
   prefix.append(".");

   return std::make_unique<RFieldsIterator>(fNtplSource, std::move(ids), prefix);
}

std::unique_ptr<RLevelIter> RNTupleElement::GetChildsIter()
{
   std::vector<ROOT::Experimental::DescriptorId_t> ids;

   {
      auto descriptorGuard = fNtplSource->GetSharedDescriptorGuard();
      for (auto &f : descriptorGuard->GetTopLevelFields())
         ids.emplace_back(f.GetId());
   }

   if (ids.size() == 0) return nullptr;
   return std::make_unique<RFieldsIterator>(fNtplSource, std::move(ids));
}


// ==============================================================================================

/** \class RNTupleBrowseProvider
\ingroup rbrowser
\brief Provider for browsing RNTuple classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleBrowseProvider : public RProvider {

public:

   RNTupleBrowseProvider()
   {
      RegisterNTupleFunc([](const std::string &tuple_name, const std::string &filename) -> std::shared_ptr<RElement> {
         auto elem = std::make_shared<RNTupleElement>(tuple_name, filename);
         return elem->IsNull() ? nullptr : elem;
      });
   }

   virtual ~RNTupleBrowseProvider()
   {
      RegisterNTupleFunc(nullptr);
   }

} newRNTupleBrowseProvider;

