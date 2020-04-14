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
#include "ROOT/RDF/RBookedCustomColumns.hxx"

#include <memory>
#include <string>
#include <vector>
#include <deque>

class TTreeReader;

namespace ROOT {
namespace Detail {
namespace RDF {

namespace RDFInternal = ROOT::Internal::RDF;
class RLoopManager;

class RCustomColumnBase {
protected:
   RLoopManager *fLoopManager; ///< A raw pointer to the RLoopManager at the root of this functional graph. It is only
                               /// guaranteed to contain a valid address during an event loop.
   const std::string fName; ///< The name of the custom column
   const std::string fType; ///< The type of the custom column as a text string
   unsigned int fNChildren{0};      ///< number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< number of times that a children node signaled to stop processing entries.
   const unsigned int fNSlots;      ///< number of thread slots used by this node, inherited from parent node.
   const bool fIsDataSourceColumn; ///< does the custom column refer to a data-source column? (or a user-define column?)
   std::vector<Long64_t> fLastCheckedEntry;
   /// A unique ID that identifies this custom column.
   /// Used e.g. to distinguish custom columns with the same name in different branches of the computation graph.
   const unsigned int fID = GetNextID();
   RDFInternal::RBookedCustomColumns fCustomColumns;
   std::deque<bool> fIsInitialized; // because vector<bool> is not thread-safe

   static unsigned int GetNextID();

public:
   RCustomColumnBase(RLoopManager *lm, std::string_view name, std::string_view type, unsigned int nSlots,
                     bool isDSColumn, const RDFInternal::RBookedCustomColumns &customColumns);

   RCustomColumnBase &operator=(const RCustomColumnBase &) = delete;
   RCustomColumnBase &operator=(RCustomColumnBase &&) = delete;
   virtual ~RCustomColumnBase();
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void *GetValuePtr(unsigned int slot) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   RLoopManager *GetLoopManagerUnchecked() const;
   std::string GetName() const;
   std::string GetTypeName() const;
   virtual void Update(unsigned int slot, Long64_t entry) = 0;
   virtual void ClearValueReaders(unsigned int slot) = 0;
   bool IsDataSourceColumn() const { return fIsDataSourceColumn; }
   virtual void InitNode();
   /// Return the unique identifier of this RCustomColumnBase.
   unsigned int GetID() const { return fID; }
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RCUSTOMCOLUMNBASE
