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

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RMiniFile.hxx>

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
   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNTuple;

   std::string fParentName;

   ROOT::Experimental::DescriptorId_t fFieldId;

public:

   RFieldElement(std::shared_ptr<ROOT::Experimental::RNTupleReader> tuple, const std::string &parent_name, const ROOT::Experimental::DescriptorId_t id) : RElement(), fNTuple(tuple), fParentName(parent_name), fFieldId(id) {}

   virtual ~RFieldElement() = default;

   /** Name of RField */
   std::string GetName() const override { return fNTuple->GetDescriptor().GetFieldDescriptor(fFieldId).GetFieldName(); }

   /** Title of RField */
   std::string GetTitle() const override
   {
      auto &fld = fNTuple->GetDescriptor().GetFieldDescriptor(fFieldId);
      return "RField name "s + fld.GetFieldName() + " type "s + fld.GetTypeName();
   }

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   /** Return class of field  - for a moment using RNTuple class as dummy */
   const TClass *GetClass() const { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }

   std::unique_ptr<RHolder> GetObject() override
   {
      return std::make_unique<RFieldHolder>(fNTuple, fParentName, fFieldId);
   }

   EActionKind GetDefaultAction() const override
   {
      auto range = fNTuple->GetDescriptor().GetFieldRange(fFieldId);
      if (range.begin() != range.end()) return kActNone;

      auto &field = fNTuple->GetDescriptor().GetFieldDescriptor(fFieldId);

      bool supported = (field.GetTypeName() == "double"s) ||  (field.GetTypeName() == "float"s) ||
                       (field.GetTypeName() == "int"s) || (field.GetTypeName() == "std::int32_t"s) ||
                       (field.GetTypeName() == "std::uint32_t"s) || (field.GetTypeName() == "std::string"s);

      if (!supported)
         printf("Field %s type %s not yet supported for drawing\n", field.GetFieldName().c_str(), field.GetTypeName().c_str());

      return supported ? kActDraw7 : kActNone;
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
   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNTuple;

public:
   RNTupleElement(const std::string &tuple_name, const std::string &filename)
   {
      fNTuple = ROOT::Experimental::RNTupleReader::Open(tuple_name, filename);
   }

   virtual ~RNTupleElement() = default;

   /** Returns true if no ntuple found */
   bool IsNull() const { return !fNTuple; }

   /** Name of NTuple */
   std::string GetName() const override { return fNTuple->GetDescriptor().GetName(); }

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

   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNTuple;
   std::vector<ROOT::Experimental::DescriptorId_t> fFieldIds;
   std::string fParentName;
   int fCounter{-1};

public:
   RFieldsIterator(std::shared_ptr<ROOT::Experimental::RNTupleReader> tuple, std::vector<ROOT::Experimental::DescriptorId_t> &&ids, const std::string &parent_name = ""s) : fNTuple(tuple), fFieldIds(ids), fParentName(parent_name)
   {
   }

   virtual ~RFieldsIterator() = default;

   bool Next() override
   {
      return ++fCounter < (int) fFieldIds.size();
   }

   std::string GetItemName() const override
   {
      return fNTuple->GetDescriptor().GetFieldDescriptor(fFieldIds[fCounter]).GetFieldName();
   }

   bool CanItemHaveChilds() const override
   {
      auto subrange = fNTuple->GetDescriptor().GetFieldRange(fFieldIds[fCounter]);
      return subrange.begin() != subrange.end();
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {

      int nchilds = 0;
      for (auto &sub: fNTuple->GetDescriptor().GetFieldRange(fFieldIds[fCounter])) { (void) sub; nchilds++; }

      auto &field = fNTuple->GetDescriptor().GetFieldDescriptor(fFieldIds[fCounter]);

      auto item = std::make_unique<RItem>(field.GetFieldName(), nchilds, nchilds > 0 ? "sap-icon://split" : "sap-icon://e-care");

      item->SetTitle("RField name "s + field.GetFieldName() + " type "s + field.GetTypeName());

      return item;
   }

   std::shared_ptr<RElement> GetElement() override
   {
      return std::make_shared<RFieldElement>(fNTuple, fParentName, fFieldIds[fCounter]);
   }
};


std::unique_ptr<RLevelIter> RFieldElement::GetChildsIter()
{
   std::vector<ROOT::Experimental::DescriptorId_t> ids;

   for (auto &f : fNTuple->GetDescriptor().GetFieldRange(fFieldId))
      ids.emplace_back(f.GetId());

   if (ids.size() == 0) return nullptr;

   std::string prefix = fParentName;
   auto &fld = fNTuple->GetDescriptor().GetFieldDescriptor(fFieldId);
   prefix.append(fld.GetFieldName());
   prefix.append(".");

   return std::make_unique<RFieldsIterator>(fNTuple, std::move(ids), prefix);
}

std::unique_ptr<RLevelIter> RNTupleElement::GetChildsIter()
{
   std::vector<ROOT::Experimental::DescriptorId_t> ids;

   for (auto &f : fNTuple->GetDescriptor().GetTopLevelFields())
      ids.emplace_back(f.GetId());

   if (ids.size() == 0) return nullptr;
   return std::make_unique<RFieldsIterator>(fNTuple, std::move(ids));
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

