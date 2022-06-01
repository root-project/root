// Author: Enrico Guiraud, CERN 11/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RRESULTMAP
#define ROOT_RDF_RRESULTMAP

#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx" // Union

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

namespace Internal {
namespace RDF {
template <typename T>
ROOT::RDF::Experimental::RResultMap<T>
MakeResultMap(std::shared_ptr<T> nominalResult, std::vector<std::shared_ptr<T>> &&variedResults,
              std::vector<std::string> &&keys, RLoopManager &lm,
              std::shared_ptr<ROOT::Internal::RDF::RActionBase> nominalAction,
              std::shared_ptr<ROOT::Internal::RDF::RActionBase> variedAction)
{
   return ROOT::RDF::Experimental::RResultMap<T>(std::move(nominalResult), std::move(variedResults), std::move(keys),
                                                 lm, std::move(nominalAction), std::move(variedAction));
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
   std::shared_ptr<ROOT::Internal::RDF::RActionBase> fNominalAction; // never null
   std::shared_ptr<ROOT::Internal::RDF::RActionBase> fVariedAction;  // might be null if there are no variations

   friend RResultMap
   ROOT::Internal::RDF::MakeResultMap<T>(std::shared_ptr<T> nominalResult,
                                         std::vector<std::shared_ptr<T>> &&variedResults,
                                         std::vector<std::string> &&keys, ROOT::Detail::RDF::RLoopManager &lm,
                                         std::shared_ptr<ROOT::Internal::RDF::RActionBase> nominalAction,
                                         std::shared_ptr<ROOT::Internal::RDF::RActionBase> variedAction);

   // The preconditions are that results and keys have the same size, are ordered the same way, and keys are unique.
   RResultMap(std::shared_ptr<T> &&nominalResult, std::vector<std::shared_ptr<T>> &&variedResults,
              std::vector<std::string> &&keys, ROOT::Detail::RDF::RLoopManager &lm,
              std::shared_ptr<ROOT::Internal::RDF::RActionBase> nominalAction,
              std::shared_ptr<ROOT::Internal::RDF::RActionBase> variedAction)
      : fKeys{ROOT::Internal::RDF::Union({"nominal"}, keys)}, fLoopManager(&lm),
        fNominalAction(std::move(nominalAction)), fVariedAction(std::move(variedAction))
   {
      R__ASSERT(variedResults.size() == keys.size() && "Keys and values have different sizes!");
      std::size_t i = 0u;
      fMap.insert({"nominal", std::move(nominalResult)});
      for (const auto &k : keys) {
         auto it = fMap.insert({k, variedResults[i++]});
         R__ASSERT(it.second &&
                   "Failed to insert an element in RResultMap, maybe a duplicated key? This should never happen.");
      }
   }

public:
   // TODO: can we use a std::string_view here without having to create a temporary std::string anyway?
   T &operator[](const std::string &key)
   {
      auto it = fMap.find(key);
      if (it == fMap.end())
         throw std::runtime_error("RResultMap: no result with key \"" + key + "\".");

      if ((fVariedAction != nullptr && !fVariedAction->HasRun()) || !fNominalAction->HasRun())
         fLoopManager->Run();
      return *it->second;
   }

   const std::vector<std::string> &GetKeys() const { return fKeys; }
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif
