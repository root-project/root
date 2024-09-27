// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RCHUNKLOADER
#define TMVA_RCHUNKLOADER

#include <vector>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename... ColTypes>
class RChunkLoaderFunctor {
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};

   TMVA::Experimental::RTensor<float> &fChunkTensor;

   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec)
   {
      const auto &max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      const auto &vec_size = vec.size();
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.cbegin(), vec.cend(), &fChunkTensor.GetData()[fOffset]);
         std::fill(&fChunkTensor.GetData()[fOffset + vec_size], &fChunkTensor.GetData()[fOffset + max_vec_size],
                   fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.cbegin(), vec.cbegin() + max_vec_size, &fChunkTensor.GetData()[fOffset]);
      }
      fOffset += max_vec_size;
   }

   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val)
   {
      fChunkTensor.GetData()[fOffset++] = val;
   }

public:
   RChunkLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor, const std::vector<std::size_t> &maxVecSizes,
                       float vecPadding)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      (AssignToTensor(cols), ...);
   }
};

template <typename... ColTypes>
class RChunkLoaderFunctorFilters {

private:
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   std::size_t fEntries{};
   std::size_t fChunkSize{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};

   TMVA::Experimental::RTensor<float> &fChunkTensor;
   TMVA::Experimental::RTensor<float> &fRemainderTensor;

   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), &fChunkTensor.GetData()[fOffset]);
         std::fill(&fChunkTensor.GetData()[fOffset + vec_size], &fChunkTensor.GetData()[fOffset + max_vec_size],
                   fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, &fChunkTensor.GetData()[fOffset]);
      }
      fOffset += max_vec_size;
      fEntries++;
   }

   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val)
   {
      fChunkTensor.GetData()[fOffset++] = val;
      fEntries++;
   }

public:
   RChunkLoaderFunctorFilters(TMVA::Experimental::RTensor<float> &chunkTensor,
                              TMVA::Experimental::RTensor<float> &remainderTensor, std::size_t entries,
                              std::size_t chunkSize, std::size_t &&offset,
                              const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                              const float vecPadding = 0.0)
      : fChunkTensor(chunkTensor),
        fRemainderTensor(remainderTensor),
        fEntries(entries),
        fChunkSize(chunkSize),
        fOffset(offset),
        fMaxVecSizes(maxVecSizes),
        fVecPadding(vecPadding)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      if (fEntries == fChunkSize) {
         fChunkTensor = fRemainderTensor;
         fOffset = 0;
      }
      (AssignToTensor(cols), ...);
   }

   std::size_t &SetEntries() { return fEntries; }
   std::size_t &SetOffset() { return fOffset; }
};

template <typename... Args>
class RChunkLoader {

private:
   std::size_t fChunkSize;

   std::vector<std::string> fCols;

   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;

   ROOT::RDF::RNode &f_rdf;
   TMVA::Experimental::RTensor<float> &fChunkTensor;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoader(ROOT::RDF::RNode &rdf, TMVA::Experimental::RTensor<float> &chunkTensor, const std::size_t chunkSize,
                const std::vector<std::string> &cols, const std::vector<std::size_t> &vecSizes = {},
                const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkTensor(chunkTensor),
        fChunkSize(chunkSize),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding)
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return Number of processed events
   std::size_t LoadChunk(const std::size_t currentRow)
   {
      RChunkLoaderFunctor<Args...> func(fChunkTensor, fVecSizes, fVecPadding);

      ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fChunkSize);
      auto myCount = f_rdf.Count();

      // load data
      f_rdf.Foreach(func, fCols);

      // get loading info
      return myCount.GetValue();
   }
};

template <typename... Args>
class RChunkLoaderFilters {

private:
   ROOT::RDF::RNode &f_rdf;
   TMVA::Experimental::RTensor<float> &fChunkTensor;

   std::size_t fChunkSize;
   std::vector<std::string> fCols;
   const std::size_t fNumEntries;
   std::size_t fNumAllEntries;
   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;
   std::size_t fNumColumns;

   const std::size_t fPartOfChunkSize;
   TMVA::Experimental::RTensor<float> fRemainderChunkTensor;
   std::size_t fRemainderChunkTensorRow = 0;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param filters
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoaderFilters(ROOT::RDF::RNode &rdf, TMVA::Experimental::RTensor<float> &chunkTensor,
                       const std::size_t chunkSize, const std::vector<std::string> &cols, std::size_t numEntries,
                       std::size_t numAllEntries, const std::vector<std::size_t> &vecSizes = {},
                       const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkTensor(chunkTensor),
        fChunkSize(chunkSize),
        fCols(cols),
        fNumEntries(numEntries),
        fNumAllEntries(numAllEntries),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fNumColumns(cols.size()),
        fPartOfChunkSize(chunkSize / 5),
        fRemainderChunkTensor(std::vector<std::size_t>{fPartOfChunkSize, fNumColumns})
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t> LoadChunk(std::size_t currentRow)
   {
      for (std::size_t i = 0; i < fRemainderChunkTensorRow; i++) {
         std::copy(fRemainderChunkTensor.GetData() + (i * fNumColumns),
                   fRemainderChunkTensor.GetData() + ((i + 1) * fNumColumns),
                   fChunkTensor.GetData() + (i * fNumColumns));
      }

      RChunkLoaderFunctorFilters<Args...> func(fChunkTensor, fRemainderChunkTensor, fRemainderChunkTensorRow,
                                               fChunkSize, fRemainderChunkTensorRow * fNumColumns, fVecSizes,
                                               fVecPadding);

      std::size_t passedEvents = 0;
      std::size_t processedEvents = 0;

      while ((passedEvents < fChunkSize && passedEvents < fNumEntries) && currentRow < fNumAllEntries) {
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fPartOfChunkSize);
         auto report = f_rdf.Report();

         f_rdf.Foreach(func, fCols);

         processedEvents += report.begin()->GetAll();
         passedEvents += (report.end() - 1)->GetPass();

         currentRow += fPartOfChunkSize;
         func.SetEntries() = passedEvents;
         func.SetOffset() = passedEvents * fNumColumns;
      }

      fRemainderChunkTensorRow = passedEvents > fChunkSize ? passedEvents - fChunkSize : 0;

      return std::make_pair(processedEvents, passedEvents);
   }

   std::size_t LastChunk()
   {
      for (std::size_t i = 0; i < fRemainderChunkTensorRow; i++) {
         std::copy(fRemainderChunkTensor.GetData() + (i * fNumColumns),
                   fRemainderChunkTensor.GetData() + ((i + 1) * fNumColumns),
                   fChunkTensor.GetData() + (i * fNumColumns));
      }

      return fRemainderChunkTensorRow;
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_RCHUNKLOADER
