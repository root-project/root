// Author: Enrico Guiraud, CERN 11/2021

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RRESULTMAP
#define ROOT_RDF_RRESULTMAP

#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RMergeableValue.hxx"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ROOT {

namespace RDF {

namespace Experimental {
// fwd decl for MakeResultMap
template <typename T>
class RResultMap;
} // namespace Experimental
} // namespace RDF

namespace Detail {
namespace RDF {
template <typename T>
std::unique_ptr<RMergeableVariations<T>> GetMergeableValue(ROOT::RDF::Experimental::RResultMap<T> &rmap);
} // namespace RDF
} // namespace Detail

namespace Internal {
namespace RDF {
template <typename T>
ROOT::RDF::Experimental::RResultMap<T>
MakeResultMap(std::vector<std::shared_ptr<T>> &&results, std::vector<std::string> &&keys, RLoopManager &lm,
              std::shared_ptr<ROOT::Internal::RDF::RActionBase> actionPtr)
{
   return ROOT::RDF::Experimental::RResultMap<T>(std::move(results), std::move(keys), lm, std::move(actionPtr));
}
} // namespace RDF
} // namespace Internal

namespace RDF {

namespace Experimental {

template <typename T>
class RResultMap {

   std::vector<std::string> fKeys;                            // values are the keys available in fMap
   std::unordered_map<std::string, std::shared_ptr<T>> fMap;  // shared_ptrs are never null
   ROOT::Detail::RDF::RLoopManager *fLoopManager;             // never null
   std::shared_ptr<ROOT::Internal::RDF::RActionBase> fAction; // never null

   friend RResultMap ROOT::Internal::RDF::MakeResultMap<T>(std::vector<std::shared_ptr<T>> &&results,
                                                           std::vector<std::string> &&keys,
                                                           ROOT::Detail::RDF::RLoopManager &lm,
                                                           std::shared_ptr<ROOT::Internal::RDF::RActionBase> actionPtr);

   friend std::unique_ptr<ROOT::Detail::RDF::RMergeableVariations<T>>
   ROOT::Detail::RDF::GetMergeableValue<T>(RResultMap<T> &rmap);

   // The preconditions are that results and keys have the same size, are ordered the same way, and keys are unique.
   RResultMap(std::vector<std::shared_ptr<T>> &&results, std::vector<std::string> &&keys,
              ROOT::Detail::RDF::RLoopManager &lm, std::shared_ptr<ROOT::Internal::RDF::RActionBase> actionPtr)
      : fKeys{keys}, fLoopManager(&lm), fAction(std::move(actionPtr))
   {
      R__ASSERT(results.size() == keys.size() && "Keys and values have different sizes!");
      std::size_t i = 0u;
      for (const auto &k : keys) {
         auto it = fMap.insert({k, results[i++]});
         R__ASSERT(it.second &&
                   "Failed to insert an element in RResultMap, maybe a duplicated key? This should never happen.");
      }
   }

public:
   using iterator       = typename decltype(fMap)::iterator;
   using const_iterator = typename decltype(fMap)::const_iterator;

   // TODO: can we use a std::string_view here without having to create a temporary std::string anyway?
   T &operator[](const std::string &key)
   {
      auto it = fMap.find(key);
      if (it == fMap.end())
         throw std::runtime_error("RResultMap: no result with key \"" + key + "\".");

      if (!fAction->HasRun())
         fLoopManager->Run();
      return *it->second;
   }

   /// Iterator to walk through key-value pairs of all variations for a given object.
   /// Runs the event loop before returning if necessary.
   iterator begin()
   {
      if (!fAction->HasRun())
         fLoopManager->Run();
      return fMap.begin();
   }
   const_iterator cbegin()
   {
      if (!fAction->HasRun())
         fLoopManager->Run();
      return fMap.cbegin();
   }
   iterator end() { return fMap.end(); }
   const_iterator end() const { return fMap.cend(); }
   const_iterator cend() const { return fMap.cend(); }

   const std::vector<std::string> &GetKeys() const { return fKeys; }
};

} // namespace Experimental
} // namespace RDF

namespace Detail {
namespace RDF {
////////////////////////////////////////////////////////////////////////////////
/// \brief Retrieve mergeable values after calling ROOT::RDF::VariationsFor .
/// \param[in] rmap lvalue reference of an RResultMap object.
/// \returns A container with the variation names and then variation values.
///
/// This function triggers the execution of the RDataFrame computation graph.
/// Then retrieves an RMergeableVariations object created with the results held
/// by the RResultMap input. The user obtains ownership of the mergeable, which
/// in turn holds a copy variation names and variation results. The RResultMap
/// is not destroyed in the process and will still retain (shared) ownership of
/// the original results.
///
/// Example usage:
/// ~~~{.cpp}
/// auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
/// auto h = df.Vary("x", [](){return ROOT::RVecI{-1, 2};}, {}, 2).Histo1D<int>("x");
/// auto hs = ROOT::RDF::Experimental::VariationsFor(h);
/// std::unique_ptr<RMergeableVariations<T>> m = ROOT::Detail::RDF::GetMergeableValue(hs);
/// ~~~
template <typename T>
std::unique_ptr<RMergeableVariations<T>> GetMergeableValue(ROOT::RDF::Experimental::RResultMap<T> &rmap)
{
   if (!rmap.fAction->HasRun())
      rmap.fLoopManager->Run(); // Prevents from using `const` specifier in parameter

   auto mValueBase = rmap.fAction->GetMergeableValue();
   std::unique_ptr<RMergeableVariationsBase> mVariationsBase{
      static_cast<RMergeableVariationsBase *>(mValueBase.release())};
   return std::make_unique<RMergeableVariations<T>>(std::move(*mVariationsBase));
}
} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif
