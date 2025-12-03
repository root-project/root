/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPad.h"
#include "TClass.h"

#include "RFieldProvider.hxx"
#include "RVisualizationProvider.hxx"

// ==============================================================================================

/** \class RNTupleDraw6Provider
\ingroup rbrowser
\brief Provider for RNTuple drawing on TCanvas
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/

class RNTupleDraw6Provider : public RProvider {
private:
   RFieldProvider fieldProvider;
   RVisualizationProvider visualizationProvider;

public:
   RNTupleDraw6Provider()
   {
      RegisterDraw6(TClass::GetClass<ROOT::RNTuple>(),
                    [this](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {
                       auto visHolder = dynamic_cast<RVisualizationHolder *>(obj.get());
                       if (visHolder) {
                          auto treeMap = visualizationProvider.CreateTreeMap(visHolder);
                          if (!treeMap)
                             return false;

                          pad->Add(treeMap.release(), opt.c_str());
                          return true;
                       }

                       auto fieldHolder = dynamic_cast<RFieldHolder *>(obj.get());
                       if (fieldHolder) {
                          auto h1 = fieldProvider.DrawField(fieldHolder);
                          if (!h1)
                             return false;

                          pad->Add(h1, opt.c_str());
                          return true;
                       }
                       return false;
                    });
   }
} newRNTupleDraw6Provider;
