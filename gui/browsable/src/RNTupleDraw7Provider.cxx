/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClass.h"

#include <ROOT/RCanvas.hxx>
#include <ROOT/TObjectDrawable.hxx>

#include "RFieldProvider.hxx"
#include "RVisualizationProvider.hxx"

using namespace ROOT::Browsable;

// ==============================================================================================

/** \class RNTupleDraw7Provider
\ingroup rbrowser
\brief Provider for RNTuple drawing on RCanvas
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/

class RNTupleDraw7Provider : public RProvider {
private:
   RFieldProvider fieldProvider;
   RVisualizationProvider visualizationProvider;

public:
   RNTupleDraw7Provider()
   {
      RegisterDraw7(TClass::GetClass<ROOT::RNTuple>(),
                    [this](std::shared_ptr<ROOT::Experimental::RPadBase> &subpad, std::unique_ptr<RHolder> &obj,
                           const std::string &opt) -> bool {
                       auto visHolder = dynamic_cast<RVisualizationHolder *>(obj.get());
                       if (visHolder) {
                          auto treeMap = visualizationProvider.CreateTreeMap(visHolder);
                          if (!treeMap)
                             return false;

                          std::shared_ptr<ROOT::Experimental::RTreeMapPainter> shared;
                          shared.reset(treeMap.release());

                          subpad->Draw<ROOT::Experimental::TObjectDrawable>(shared, opt);
                          return true;
                       }

                       auto fieldHolder = dynamic_cast<RFieldHolder *>(obj.get());
                       if (fieldHolder) {
                          auto h1 = fieldProvider.DrawField(fieldHolder);
                          if (!h1)
                             return false;

                          std::shared_ptr<TH1> shared;
                          shared.reset(h1);

                          subpad->Draw<ROOT::Experimental::TObjectDrawable>(shared, opt);
                          return true;
                       }

                       return false;
                    });
   }
} newRNTupleDraw7Provider;
