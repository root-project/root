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

#include <ROOT/RNTuple.hxx>
#include <ROOT/RMiniFile.hxx>

#include "TClass.h"

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;



/** \class RNTupleElement
\ingroup rbrowser
\brief Browsing of RNTuple
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleElement : public RElement {
protected:
   std::unique_ptr<ROOT::Experimental::RNTupleReader> fNTuple;

public:
   RNTupleElement(const std::string &tuple_name, const std::string &filename);

   virtual ~RNTupleElement() = default;

   /** Returns true if no ntuple found */
   bool IsNull() const { return !fNTuple; }

   /** Name of NTuple */
   std::string GetName() const override;

   /** Title of NTuple */
   std::string GetTitle() const override;

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override;

   const TClass *GetClass() const { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }

   //EActionKind GetDefaultAction() const override;

   //bool IsCapable(EActionKind) const override;
};


RNTupleElement::RNTupleElement(const std::string &tuple_name, const std::string &filename)
{
   fNTuple = ROOT::Experimental::RNTupleReader::Open(tuple_name, filename);
}

std::string RNTupleElement::GetName() const
{
   return ""s;
}

std::string RNTupleElement::GetTitle() const
{
   return "title"s;
}

std::unique_ptr<RLevelIter> RNTupleElement::GetChildsIter()
{
   return nullptr;
}


// ==============================================================================================

/** \class RNTupleProvider
\ingroup rbrowser
\brief Provider for RNTuple classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleProvider : public RProvider {

public:

   RNTupleProvider()
   {
      RegisterNTupleFunc([](const std::string &tuple_name, const std::string &filename) -> std::shared_ptr<RElement> {
         auto elem = std::make_shared<RNTupleElement>(tuple_name, filename);
         return elem->IsNull() ? nullptr : elem;
      });
   }

   virtual ~RNTupleProvider()
   {
      RegisterNTupleFunc(nullptr);
   }

} newRNTupleProvider;

