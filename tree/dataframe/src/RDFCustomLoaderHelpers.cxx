#include <ROOT/RDF/CustomLoaderHelpers.hxx>
#include <ROOT/RDF/Utils.hxx> // TypeID2TypeName

#include <algorithm>
#include <stdexcept>
#include <typeinfo>

namespace {

using ROOT::Internal::RDF::CustomLoaderHelper;

template <typename T>
void AppendAs(ROOT::RVecF &dest, void *value, std::size_t /*maxSize*/, float /*padding*/)
{
   dest.push_back(static_cast<float>(*static_cast<T *>(value)));
}

template <typename T>
void AppendVectorAs(ROOT::RVecF &dest, void *value, std::size_t maxSize, float padding)
{
   const auto &vec = *static_cast<ROOT::RVec<T> *>(value);
   const auto size = std::min(vec.size(), maxSize);
   dest.insert(dest.end(), vec.begin(), vec.begin() + size);
   dest.insert(dest.end(), maxSize - size, padding);
}

/// Pick the handler for a column of type T or RVec<T>, flagging vector columns
template <typename T>
bool MatchColHandler(const std::type_info &colType, CustomLoaderHelper::ColHandler_t &handler, bool &isVector)
{
   if (colType == typeid(T)) {
      handler = &AppendAs<T>;
   } else if (colType == typeid(ROOT::RVec<T>)) {
      handler = &AppendVectorAs<T>;
      isVector = true;
   }
   return handler != nullptr;
}

/// Pick the handler matching the type of a column among the supported types and their RVecs
template <typename... Types>
CustomLoaderHelper::ColHandler_t ResolveColHandler(const std::type_info &colType, bool &isVector)
{
   CustomLoaderHelper::ColHandler_t handler = nullptr;
   if ((MatchColHandler<Types>(colType, handler, isVector) || ...))
      return handler;

   throw std::invalid_argument("CustomLoaderHelper: column type '" + ROOT::Internal::RDF::TypeID2TypeName(colType) +
                               "' cannot be converted to float.");
}

} // namespace

ROOT::Internal::RDF::CustomLoaderHelper::CustomLoaderHelper(const std::shared_ptr<ROOT::RVecF> &location,
                                                            const unsigned int nSlots,
                                                            const std::vector<const std::type_info *> &colTypeIDs,
                                                            const std::vector<std::size_t> &vecSizes, float vecPadding)
   : fLocation(location), fNSlots(nSlots), fColTypeIDs(colTypeIDs), fVecSizes(vecSizes), fVecPadding(vecPadding)
{
   fColHandlers.reserve(fColTypeIDs.size());
   fMaxSizes.reserve(fColTypeIDs.size());
   std::size_t vecIdx = 0;
   for (const auto *colType : fColTypeIDs) {
      bool isVector = false;
      fColHandlers.push_back(
         ResolveColHandler<float, double, bool, char, signed char, unsigned char, short, unsigned short, int,
                           unsigned int, long, unsigned long, long long, unsigned long long>(*colType, isVector));
      // Vector columns take the next maximum size, in column order
      fMaxSizes.push_back(isVector ? fVecSizes[vecIdx++] : 1);
   }
}

void ROOT::Internal::RDF::CustomLoaderHelper::Exec(unsigned int /*slot*/, const std::vector<void *> &values)
{
   // The readers deliver the values in column order, the same order the handlers were resolved in
   auto nValues{values.size()};
   for (decltype(nValues) i{}; i < nValues; i++)
      fColHandlers[i](*fLocation, values[i], fMaxSizes[i], fVecPadding);
}
