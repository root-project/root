// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCUSTOMCOLUMNBASE
#define ROOT_RCUSTOMCOLUMNBASE

#include "ROOT/RDF/GraphNode.hxx"
#include "ROOT/RDF/RBookedDefines.hxx"
#include "ROOT/RDF/RSampleInfo.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

class TTreeReader;

namespace ROOT {
namespace RDF {
class RDataSource;
}
namespace Detail {
namespace RDF {

namespace RDFInternal = ROOT::Internal::RDF;

class RDefineBase {
protected:
   const std::string fName; ///< The name of the custom column
   const std::string fType; ///< The type of the custom column as a text string
   std::vector<Long64_t> fLastCheckedEntry;
   RDFInternal::RBookedDefines fDefines;
   std::deque<bool> fIsInitialized; // because vector<bool> is not thread-safe
   const std::map<std::string, std::vector<void *>> &fDSValuePtrs; // reference to RLoopManager's data member
   ROOT::RDF::RDataSource *fDataSource; ///< non-owning ptr to the RDataSource, if any. Used to retrieve column readers.
   const ROOT::RDF::ColumnNames_t fColumnNames;
   /// The nth flag signals whether the nth input column is a custom column or not.
   ROOT::RVecB fIsDefine;

public:
   RDefineBase(std::string_view name, std::string_view type, unsigned int nSlots,
               const RDFInternal::RBookedDefines &defines,
               const std::map<std::string, std::vector<void *>> &DSValuePtrs, ROOT::RDF::RDataSource *ds,
               const ColumnNames_t &columnNames);

   RDefineBase &operator=(const RDefineBase &) = delete;
   RDefineBase &operator=(RDefineBase &&) = delete;
   virtual ~RDefineBase();
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   /// Return the (type-erased) address of the Define'd value for the given processing slot.
   virtual void *GetValuePtr(unsigned int slot) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   std::string GetName() const;
   std::string GetTypeName() const;
   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   virtual void Update(unsigned int slot, Long64_t entry) = 0;
   /// Update function to be called once per sample, used if the derived type is a RDefinePerSample
   virtual void Update(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo &/*id*/) {}
   /// Clean-up operations to be performed at the end of a task.
   virtual void FinaliseSlot(unsigned int slot) = 0;
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RCUSTOMCOLUMNBASE
