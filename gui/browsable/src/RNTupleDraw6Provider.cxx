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
#include "TH1.h"
#include "TVirtualPad.h"

#include "RFieldHolder.hxx"

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;


// ==============================================================================================

/** \class RNTupleDraw6Provider
\ingroup rbrowser
\brief Provider for RNTuple classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleDraw6Provider : public RProvider {

   template<typename T>
   void FillHistogram(std::shared_ptr<ROOT::Experimental::RNTupleReader> &tuple, const std::string &field_name, TH1 *hist)
   {
      auto view = tuple->GetView<T>(field_name);
      for (auto i : tuple->GetEntryRange())
         hist->Fill(view(i));
   }

public:

   RNTupleDraw6Provider()
   {

      RegisterDraw6(TClass::GetClass<ROOT::Experimental::RNTuple>(), [this](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         auto p = dynamic_cast<RFieldHolder*> (obj.get());
         if (!p) return false;

         auto tuple = p->GetNTuple();
         std::string name = p->GetParentName();
         auto id = p->GetId();

         auto &field = tuple->GetDescriptor().GetFieldDescriptor(id);
         name.append(field.GetFieldName());

         std::string title = "Drawing of RField "s + name;

         auto h1 = new TH1F("hdraw", title.c_str(), 100, 0, 0);
         h1->SetDirectory(nullptr);

         if (field.GetTypeName() == "double"s)
            FillHistogram<double>(tuple, name, h1);
         else if (field.GetTypeName() == "float"s)
            FillHistogram<float>(tuple, name, h1);
         else if (field.GetTypeName() == "int"s)
            FillHistogram<int>(tuple, name, h1);
         else if (field.GetTypeName() == "std::int32_t"s)
            FillHistogram<int32_t>(tuple, name, h1);
         else {
            delete h1;
            return false;
         }

         h1->BufferEmpty();

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(h1, opt.c_str());

         return true;
      });


   }


} newRNTupleDraw6Provider;

