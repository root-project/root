#ifndef TMVA_RTENSOR_UTILS
#define TMVA_RTENSOR_UTILS

#include <vector>
#include <string>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RInterface.hxx"

namespace TMVA {
namespace Experimental {

/// \brief Convert the content of an RDataFrame to an RTensor
/// \param[in] dataframe RDataFrame node
/// \param[in] columns Vector of column names
/// \param[in] layout Memory layout
/// \return RTensor with content from selected columns
template <typename T, typename U>
RTensor<T>
AsTensor(U &dataframe, std::vector<std::string> columns = {}, MemoryLayout layout = MemoryLayout::RowMajor)
{
   // If no columns are specified, get all columns from dataframe
   if (columns.size() == 0) {
      columns = dataframe.GetColumnNames();
   }

   // Book actions to read-out columns of dataframe in vectors
   using ResultPtr = ROOT::RDF::RResultPtr<std::vector<T>>;
   std::vector<ResultPtr> resultPtrs;
   for (auto &col : columns) {
      resultPtrs.emplace_back(dataframe.template Take<T>(col));
   }

   // Copy data to tensor based on requested memory layout
   const auto numCols = resultPtrs.size();
   const auto numEntries = resultPtrs[0]->size();
   RTensor<T> x({numEntries, numCols}, layout);
   const auto data = x.GetData();
   if (layout == MemoryLayout::RowMajor) {
      for (std::size_t i = 0; i < numEntries; i++) {
         const auto entry = data + numCols * i;
         for (std::size_t j = 0; j < numCols; j++) {
            entry[j] = resultPtrs[j]->at(i);
         }
      }
   } else if (layout == MemoryLayout::ColumnMajor) {
      for (std::size_t i = 0; i < numCols; i++) {
         // TODO: Replace by RVec<T>::insert as soon as available.
         std::memcpy(data + numEntries * i, &resultPtrs[i]->at(0), numEntries * sizeof(T));
      }
   } else {
      throw std::runtime_error("Memory layout is not known.");
   }

   // Remove dimensions of 1
   x.Squeeze();

   return x;
}

} // namespace TMVA::Experimental
} // namespace TMVA

#endif
