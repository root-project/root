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

public:

   RNTupleDraw6Provider()
   {
      RegisterDraw6(TClass::GetClass<ROOT::Experimental::RNTuple>(), [this](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {

         auto p = dynamic_cast<RFieldHolder<void>*> (obj.get());
         if (!p) return false;

         auto tuple = p->GetNTuple();
         std::string name = p->GetParentName();
         auto id = p->GetId();

         auto &field = tuple->GetDescriptor().GetFieldDescriptor(id);
         name.append(field.GetFieldName());

         std::string title = "Drawing of RField "s + name;

         auto h1 = new TH1F("hdraw", title.c_str(), 100, -20, 20);
         h1->SetDirectory(nullptr);

         auto view = tuple->GetView<double>(name);
         for (auto i : tuple->GetEntryRange())
            h1->Fill(view(i));

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(h1, opt.c_str());

         return true;
      });


   }


} newRNTupleDraw6Provider;

