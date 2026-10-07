// Author: Vincenzo Eduardo Padulano CERN 10/2026
// Author: Silia Taider CERN 10/2026

#ifndef ROOT_RDF_CUSTOMLOADERHELPERS
#define ROOT_RDF_CUSTOMLOADERHELPERS

#include <ROOT/RDF/RActionImpl.hxx>
#include <ROOT/RVec.hxx>

#include <memory>
#include <typeinfo>
#include <vector>

class TTreeReader;

namespace ROOT::Internal::RDF {

class R__CLING_PTRCHECK(off) CustomLoaderHelper : public ROOT::Detail::RDF::RActionImpl<CustomLoaderHelper> {
public:
   /// Converts one type-erased column value to float and appends it to the destination vector
   using ColHandler_t = void (*)(ROOT::RVecF &, void *);

private:
   std::shared_ptr<ROOT::RVecF> fLocation;
   /// One handler per input column, resolved once from the column types
   std::vector<ColHandler_t> fColHandlers;

public:
   CustomLoaderHelper(const std::shared_ptr<ROOT::RVecF> &location,
                      const std::vector<const std::type_info *> &colTypeIDs);

   CustomLoaderHelper(CustomLoaderHelper &&) = default;
   CustomLoaderHelper &operator=(CustomLoaderHelper &&) = default;
   CustomLoaderHelper(const CustomLoaderHelper &) = delete;
   CustomLoaderHelper &operator=(const CustomLoaderHelper &) = delete;
   ~CustomLoaderHelper() override = default;

   void InitTask(TTreeReader *, unsigned int) {}
   // TODO(staider) Support multiple slots to also work in IMT mode
   void Exec([[maybe_unused]] unsigned int slot, const std::vector<void *> &values);
   void Initialize() { /* noop */ }
   void Finalize() {}

   std::string GetActionName() { return "CustomLoaderHelper"; }
};
} // namespace ROOT::Internal::RDF

#endif // ROOT_RDF_CUSTOMLOADERHELPERS
