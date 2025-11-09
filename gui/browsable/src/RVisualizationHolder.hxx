/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RVISUALIZATIONHOLDER_HXX
#define ROOT_RVISUALIZATIONHOLDER_HXX

#include <ROOT/Browsable/RHolder.hxx>
#include <ROOT/RNTupleReader.hxx>

/** \class RVisualizationHolder
\ingroup rbrowser
\brief Holder for RNTuple visualization data
\author Patryk Pilichowski
\date 2025
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/

class RVisualizationHolder : public ROOT::Browsable::RHolder {
protected:
   std::shared_ptr<ROOT::RNTupleReader> fNtplReader;
   std::string fFileName;
   std::string fTupleName;

public:
   RVisualizationHolder(std::shared_ptr<ROOT::RNTupleReader> ntplReader, const std::string &fileName,
                        const std::string &tupleName)
      : fNtplReader(ntplReader), fFileName(fileName), fTupleName(tupleName)
   {
   }

   const TClass *GetClass() const override { return TClass::GetClass<ROOT::RNTuple>(); }

   /** Returns direct (temporary) object pointer */
   const void *GetObject() const override { return nullptr; }

   std::shared_ptr<ROOT::RNTupleReader> GetNTupleReader() const { return fNtplReader; }
   const std::string &GetFileName() const { return fFileName; }
   const std::string &GetTupleName() const { return fTupleName; }
};

#endif // ROOT_RVISUALIZATIONHOLDER_HXX
