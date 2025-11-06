/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RVISUALIZATIONPROVIDER_HXX
#define ROOT_RVISUALIZATIONPROVIDER_HXX

#include <ROOT/Browsable/RProvider.hxx>

#include "RVisualizationHolder.hxx"

#include <ROOT/RNTupleTreeMap.hxx>

/** \class RVisualizationProvider
\ingroup rbrowser
\brief Provider for RNTuple TreeMap visualization on TCanvas
\author Patryk Pilichowski
\date 2025
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/
class RVisualizationProvider : public RProvider {
public:
   /** Create TreeMap visualization for RNTuple */
   std::unique_ptr<ROOT::Experimental::RTreeMapPainter> CreateTreeMap(RVisualizationHolder *holder) const
   {
      if (!holder)
         return nullptr;

      return ROOT::Experimental::CreateTreeMapFromRNTuple(holder->GetFileName(), holder->GetTupleName());
   }
};

#endif // ROOT_RVISUALIZATIONPROVIDER_HXX
