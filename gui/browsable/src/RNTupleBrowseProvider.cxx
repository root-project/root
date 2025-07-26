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

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleBrowseUtils.hxx>
#include <ROOT/RNTupleDescriptor.hxx>

#include "TClass.h"
#include "RFieldHolder.hxx"


using namespace std::string_literals;

using namespace ROOT::Browsable;


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
   std::shared_ptr<ROOT::RNTupleReader> fNtplReader;
   std::string fParentName;
   ROOT::DescriptorId_t fFieldId;
   std::string fDisplayName;

public:
   RFieldElement(std::shared_ptr<ROOT::RNTupleReader> ntplReader, const std::string &parent_name,
                 const ROOT::DescriptorId_t id, const std::string &displayName)
      : RElement(), fNtplReader(ntplReader), fParentName(parent_name), fFieldId(id), fDisplayName(displayName)
   {
   }

   ~RFieldElement() override = default;

   /** Name of RField */
   std::string GetName() const override
   {
      return fNtplReader->GetDescriptor().GetFieldDescriptor(fFieldId).GetFieldName();
   }

   /** Title of RField */
   std::string GetTitle() const override
   {
      auto &fld = fNtplReader->GetDescriptor().GetFieldDescriptor(fFieldId);
      return "RField name "s + fld.GetFieldName() + " type "s + fld.GetTypeName();
   }

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   /** Return class of field  - for a moment using RNTuple class as dummy */
   const TClass *GetClass() const { return TClass::GetClass<ROOT::RNTuple>(); }

   std::unique_ptr<RHolder> GetObject() override
   {
      return std::make_unique<RFieldHolder>(fNtplReader, fParentName, fFieldId, fDisplayName);
   }

   EActionKind GetDefaultAction() const override
   {
      auto range = fNtplReader->GetDescriptor().GetFieldIterable(fFieldId);
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
   std::shared_ptr<ROOT::RNTupleReader> fNtplReader;

public:
   RNTupleElement(const std::string &ntplName, const std::string &filename)
   {
      fNtplReader = ROOT::RNTupleReader::Open(ntplName, filename);
   }

   ~RNTupleElement() override = default;

   /** Returns true if no ntuple found */
   bool IsNull() const { return !fNtplReader; }

   /** Name of NTuple */
   std::string GetName() const override { return fNtplReader->GetDescriptor().GetName(); }

   /** Title of NTuple */
   std::string GetTitle() const override { return "RNTuple title"s; }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override;

   const TClass *GetClass() const { return TClass::GetClass<ROOT::RNTuple>(); }

   std::unique_ptr<RItem> CreateItem() const override
   {
      auto item = std::make_unique<RItem>(GetName(), -1, "sap-icon://table-chart");
      item->SetTitle(GetTitle());
      return item;
   }

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

   std::shared_ptr<ROOT::RNTupleReader> fNtplReader;
   std::vector<ROOT::DescriptorId_t> fProvidedFieldIds;
   std::vector<ROOT::DescriptorId_t> fActualFieldIds;
   std::string fParentName;
   int fCounter{-1};

public:
   RFieldsIterator(std::shared_ptr<ROOT::RNTupleReader> ntplReader, std::vector<ROOT::DescriptorId_t> &&ids,
                   const std::string &parent_name = "")
      : fNtplReader(ntplReader), fProvidedFieldIds(ids), fParentName(parent_name)
   {
      const auto &desc = fNtplReader->GetDescriptor();
      fActualFieldIds.reserve(fProvidedFieldIds.size());
      for (auto fid : fProvidedFieldIds) {
         fActualFieldIds.emplace_back(ROOT::Internal::GetNextBrowsableField(fid, desc));
      }
   }

   ~RFieldsIterator() override = default;

   bool Next() override { return ++fCounter < (int)fProvidedFieldIds.size(); }

   std::string GetItemName() const override
   {
      return fNtplReader->GetDescriptor().GetFieldDescriptor(fProvidedFieldIds[fCounter]).GetFieldName();
   }

   bool CanItemHaveChilds() const override
   {
      auto subrange = fNtplReader->GetDescriptor().GetFieldIterable(fActualFieldIds[fCounter]);
      return subrange.begin() != subrange.end();
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {
      int nchilds = 0;
      for (auto &sub : fNtplReader->GetDescriptor().GetFieldIterable(fActualFieldIds[fCounter])) {
         (void)sub;
         nchilds++;
      }

      const auto &field = fNtplReader->GetDescriptor().GetFieldDescriptor(fProvidedFieldIds[fCounter]);

      auto item =
         std::make_unique<RItem>(field.GetFieldName(), nchilds, nchilds > 0 ? "sap-icon://split" : "sap-icon://e-care");

      item->SetTitle("RField name "s + field.GetFieldName() + " type "s + field.GetTypeName());

      return item;
   }

   std::shared_ptr<RElement> GetElement() override
   {
      const auto name = fNtplReader->GetDescriptor().GetFieldDescriptor(fProvidedFieldIds[fCounter]).GetFieldName();
      return std::make_shared<RFieldElement>(fNtplReader, fParentName, fActualFieldIds[fCounter], name);
   }
};


std::unique_ptr<RLevelIter> RFieldElement::GetChildsIter()
{
   std::vector<ROOT::DescriptorId_t> ids;
   std::string prefix;

   const auto &desc = fNtplReader->GetDescriptor();
   for (auto &f : fNtplReader->GetDescriptor().GetFieldIterable(ROOT::Internal::GetNextBrowsableField(fFieldId, desc)))
      ids.emplace_back(f.GetId());

   if (ids.size() == 0)
      return nullptr;

   prefix = fParentName;
   const auto &fld = desc.GetFieldDescriptor(fFieldId);
   prefix.append(fld.GetFieldName());
   prefix.append(".");

   return std::make_unique<RFieldsIterator>(fNtplReader, std::move(ids), prefix);
}

std::unique_ptr<RLevelIter> RNTupleElement::GetChildsIter()
{
   std::vector<ROOT::DescriptorId_t> ids;

   for (auto &f : fNtplReader->GetDescriptor().GetTopLevelFields())
      ids.emplace_back(f.GetId());

   if (ids.size() == 0) return nullptr;
   return std::make_unique<RFieldsIterator>(fNtplReader, std::move(ids));
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

   ~RNTupleBrowseProvider() override { RegisterNTupleFunc(nullptr); }

} newRNTupleBrowseProvider;

