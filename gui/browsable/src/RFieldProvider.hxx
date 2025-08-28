/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Browsable_RFieldProvider
#define ROOT_Browsable_RFieldProvider

#include "TH1.h"
#include "TMath.h"
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleDrawVisitor.hxx>
#include <ROOT/RNTupleView.hxx>

#include "RFieldHolder.hxx"

using namespace ROOT::Browsable;

using namespace std::string_literals;

// ==============================================================================================

/** \class RFieldProvider
\ingroup rbrowser
\brief Base class for provider of RNTuple drawing
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFieldProvider : public RProvider {
public:
   // virtual ~RFieldProvider() = default;

   TH1 *DrawField(RFieldHolder *holder)
   {
      if (!holder) return nullptr;

      auto ntplReader = holder->GetNtplReader();

      const auto qualifiedFieldName = ntplReader->GetDescriptor().GetQualifiedFieldName(holder->GetId());
      auto view = ntplReader->GetView<void>(qualifiedFieldName);

      ROOT::Internal::RNTupleDrawVisitor drawVisitor(ntplReader, holder->GetDisplayName());
      view.GetField().AcceptVisitor(drawVisitor);
      return drawVisitor.MoveHist();
   }
};

#endif
