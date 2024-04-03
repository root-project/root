#ifndef TMVA_CHUNKLOADER
#define TMVA_CHUNKLOADER

#include <iostream>
#include <vector>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class RChunkLoaderFunctor {

private:
   std::size_t fOffset = 0;
   std::size_t fVecSizeIdx = 0;
   std::vector<std::size_t> fMaxVecSizes;

   float fVecPadding;

   TMVA::Experimental::RTensor<float> &fChunkTensor;

   /// \brief Load the final given value into fChunkTensor
   /// \tparam First_T
   /// \param first
   template <typename First_T>
   void AssignToTensor(First_T first)
   {
      fChunkTensor.GetData()[fOffset++] = first;
   }

   /// \brief Load the final given value into fChunkTensor
   /// \tparam VecType
   /// \param first
   template <typename VecType>
   void AssignToTensor(const ROOT::RVec<VecType> &first)
   {
      AssignVector(first);
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam First_T
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename First_T, typename... Rest_T>
   void AssignToTensor(First_T first, Rest_T... rest)
   {
      fChunkTensor.GetData()[fOffset++] = first;

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam VecType
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename VecType, typename... Rest_T>
   void AssignToTensor(const ROOT::RVec<VecType> &first, Rest_T... rest)
   {
      AssignVector(first);

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Loop through the values of a given vector and load them into the RTensor
   /// Note: the given vec_size does not have to be the same size as the given vector
   ///       If the size is bigger than the given vector, zeros are used as padding.
   ///       If the size is smaller, the remaining values are ignored.
   /// \tparam VecType
   /// \param vec
   template <typename VecType>
   void AssignVector(const ROOT::RVec<VecType> &vec)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();

      for (std::size_t i = 0; i < max_vec_size; i++) {
         if (i < vec_size) {
            fChunkTensor.GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor.GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   RChunkLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor,
                       const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                       const float vecPadding = 0.0)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding)
   {
   }

   /// \brief Loop through all columns of an event and put their values into an RTensor
   /// \param first
   /// \param ...rest
   void operator()(First first, Rest... rest)
   {
      fVecSizeIdx = 0;
      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }
};

template <typename... Args>
class RChunkLoader {

private:
   std::string fTreeName;
   std::vector<std::string> fFileNames;
   std::size_t fChunkSize;
   std::size_t fNumColumns;

   std::vector<std::string> fCols;
   std::string fFilters;

   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;

   ROOT::RDF::RNode f_rdf;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param treeName
   /// \param fileName
   /// \param chunkSize
   /// \param cols
   /// \param filters
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoader(ROOT::RDF::RNode rdf, const std::size_t chunkSize,
                const std::vector<std::string> &cols, const std::string &filters = "",
                const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkSize(chunkSize),
        fCols(cols),
        fFilters(filters),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fNumColumns(cols.size())
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t>
   LoadChunk(TMVA::Experimental::RTensor<float> &chunkTensor, const std::size_t currentRow)
   {
      RChunkLoaderFunctor<Args...> func(chunkTensor, fVecSizes, fVecPadding);

      // Load events if filters are given
      if (fFilters.size() > 0) {
         return loadFiltered(f_rdf, func, currentRow);
      }

      // load events if no filters are given
      return loadNonFiltered(f_rdf, func, currentRow);
   }

private:
   /// \brief Add filters to the RDataFrame and load a chunk of data
   /// \param x_rdf
   /// \param func
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t> loadFiltered(ROOT::RDF::RNode &x_rdf, RChunkLoaderFunctor<Args...> &func, const std::size_t currentRow=0)
   {
      // Add the given filters to the RDataFrame
      auto x_filter = x_rdf.Filter(fFilters, "RBatchGenerator_Filter");

      // add range to the DataFrame
      auto x_ranged = x_filter.Range(currentRow, currentRow + fChunkSize);
      auto myReport = x_ranged.Report();

      // load data
      x_ranged.Foreach(func, fCols);

      // Use the report to gather the number of events processed and passed.
      // passed_events is used to determine the starting event of the next chunk
      // processed_events is used to determine if the end of the database is reached.
      std::size_t processed_events = myReport.begin()->GetAll();
      std::size_t passed_events = (myReport.end() - 1)->GetPass();

      return std::make_pair(processed_events, passed_events);
   }

   /// \brief Loop over the events in the dataframe untill either the end of the dataframe
   /// is reached, or a full chunk is loaded
   /// \param x_rdf
   /// \param func
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t> loadNonFiltered(ROOT::RDF::RNode &x_rdf, RChunkLoaderFunctor<Args...> &func, const std::size_t currentRow=0)
   {
      // add range
      auto x_ranged = x_rdf.Range(currentRow, currentRow + fChunkSize);
      // auto x_ranged = x_rdf.Range(currentRow, currentRow + fChunkSize);
      auto myCount = x_ranged.Count();

      // load data
      x_ranged.Foreach(func, fCols);

      // get loading info
      std::size_t processed_events = myCount.GetValue();
      std::size_t passed_events = myCount.GetValue();
      return std::make_pair(processed_events, passed_events);
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_CHUNKLOADER
