/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_DETAIL_RACTIONIMPL
#define ROOT_RDF_DETAIL_RACTIONIMPL

#include <ROOT/RDF/RSampleInfo.hxx> // SampleCallback_t

#include <memory> // std::unique_ptr
#include <stdexcept> // std::logic_error
#include <utility> // std::declval

namespace ROOT {
namespace Detail {
namespace RDF {

class RMergeableValueBase;

/// Base class for action helpers, see RInterface::Book() for more information.
template <typename Helper>
class RActionImpl {
public:
   virtual ~RActionImpl() = default;
   // call Helper::FinalizeTask if present, do nothing otherwise
   template <typename T = Helper>
   auto CallFinalizeTask(unsigned int slot) -> decltype(std::declval<T>().FinalizeTask(slot))
   {
      static_cast<Helper *>(this)->FinalizeTask(slot);
   }

   template <typename... Args>
   void CallFinalizeTask(unsigned int, Args...) {}

   template <typename H = Helper>
   auto CallPartialUpdate(unsigned int slot) -> decltype(std::declval<H>().PartialUpdate(slot), (void *)(nullptr))
   {
      return &static_cast<Helper *>(this)->PartialUpdate(slot);
   }

   template <typename... Args>
   [[noreturn]] void *CallPartialUpdate(...)
   {
      throw std::logic_error("This action does not support callbacks!");
   }

   template <typename T = Helper>
   auto CallMakeNew(void *typeErasedResSharedPtr) -> decltype(std::declval<T>().MakeNew(typeErasedResSharedPtr))
   {
      return static_cast<Helper *>(this)->MakeNew(typeErasedResSharedPtr);
   }

   template <typename... Args>
   [[noreturn]] Helper CallMakeNew(void *, Args...)
   {
      const auto &actionName = static_cast<Helper *>(this)->GetActionName();
      throw std::logic_error("The MakeNew method is not implemented for this action helper (" + actionName +
                             "). Cannot Vary its result.");
   }

   // Helper functions for RMergeableValue
   virtual std::unique_ptr<RMergeableValueBase> GetMergeableValue() const
   {
      throw std::logic_error("`GetMergeableValue` is not implemented for this type of action.");
   }

   /// Override this method to register a callback that is executed before the processing a new data sample starts.
   /// The callback will be invoked in the same conditions as with DefinePerSample().
   virtual ROOT::RDF::SampleCallback_t GetSampleCallback() { return {}; }
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_RDF_DETAIL_RACTIONIMPL

