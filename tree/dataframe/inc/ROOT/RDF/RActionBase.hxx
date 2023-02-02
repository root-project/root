// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RACTIONBASE
#define ROOT_RACTIONBASE

#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RSampleInfo.hxx"
#include "ROOT/RDF/Utils.hxx" // ColumnNames_t
#include "RtypesCore.h"

#include <memory>
#include <string>

namespace ROOT {

namespace Detail {
namespace RDF {
class RLoopManager;
class RDefineBase;
class RMergeableValueBase;
} // namespace RDF
} // namespace Detail

namespace Internal {
namespace RDF {
namespace GraphDrawing {
class GraphNode;
}

using namespace ROOT::Detail::RDF;

class RActionBase {
protected:
   /// A raw pointer to the RLoopManager at the root of this functional graph.
   /// Never null: children nodes have shared ownership of parent nodes in the graph.
   RLoopManager *fLoopManager;

private:
   const unsigned int fNSlots; ///< Number of thread slots used by this node.
   bool fHasRun = false;
   const ColumnNames_t fColumnNames;
   /// List of systematic variations that affect the result of this action ("nominal" excluded).
   std::vector<std::string> fVariations;

   RColumnRegister fColRegister;

public:
   RActionBase(RLoopManager *lm, const ColumnNames_t &colNames, const RColumnRegister &colRegister,
               const std::vector<std::string> &prevVariations);
   RActionBase(const RActionBase &) = delete;
   RActionBase &operator=(const RActionBase &) = delete;
   virtual ~RActionBase();

   const ColumnNames_t &GetColumnNames() const { return fColumnNames; }
   RColumnRegister &GetColRegister() { return fColRegister; }
   RLoopManager *GetLoopManager() { return fLoopManager; }
   unsigned int GetNSlots() const { return fNSlots; }
   virtual void Run(unsigned int slot, Long64_t entry) = 0;
   virtual void Initialize() = 0;
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void TriggerChildrenCount() = 0;
   virtual void FinalizeSlot(unsigned int) = 0;
   virtual void Finalize() = 0;
   /// This method is invoked to update a partial result during the event loop, right before passing the result to a
   /// user-defined callback registered via RResultPtr::RegisterCallback
   virtual void *PartialUpdate(unsigned int slot) = 0;

   // overridden by RJittedAction
   virtual bool HasRun() const { return fHasRun; }
   virtual void SetHasRun() { fHasRun = true; }

   virtual std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>> &visitedMap) = 0;

   /**
      Retrieve a wrapper to the result of the action that knows how to merge
      with others of the same type.
   */
   virtual std::unique_ptr<RMergeableValueBase> GetMergeableValue() const = 0;

   virtual ROOT::RDF::SampleCallback_t GetSampleCallback() = 0;

   const std::vector<std::string> &GetVariations() const { return fVariations; }

   virtual std::unique_ptr<RActionBase> MakeVariedAction(std::vector<void *> &&results) = 0;
};
} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RACTIONBASE
