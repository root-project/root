// Author: Silia Taider, CERN 08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RBATCHSINK
#define ROOT_INTERNAL_ML_RBATCHSINK

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class TFile;
class TTree;
namespace ROOT {
class RNTupleWriter;
}

namespace ROOT::Experimental::Internal::ML {

struct RFlat2DMatrix;

/**
 * \struct RColumnLayout
 * \brief Where one of the loader's columns lives inside a batch-tensor row and how it's shaped.
 *
 * A batch row is a flat span of floats in which a vector column occupies \p fWidth consecutive
 * slots (padded to its specified maximum size). This describes the inverse mapping, so a sink can
 * write each column back under its original name (unexpanded).
 */
struct RColumnLayout {
   std::string fName;   // the column's original name
   std::size_t fOffset; // index of the column's first float within a row
   std::size_t fWidth;  // number of floats the column occupies
   bool fIsVector;      // write as a vector of \p fWidth floats rather than a single float
};

/**
 * \class RBatchSink
 * \brief Writes RFlat2DMatrix batches to disk letting RDataLoaderEngine::Save() stay agnostic to
 * the output format
 */
class RBatchSink {
protected:
   std::vector<RColumnLayout> fLayout;
   std::size_t fRowWidth{};

   explicit RBatchSink(std::vector<RColumnLayout> layout) : fLayout(std::move(layout))
   {
      for (const auto &col : fLayout)
         fRowWidth += col.fWidth;
   }

   /// \brief Copy one row of \p fRowWidth floats into the writer's buffers and write the entry
   virtual void FillRow(const float *row) = 0;

public:
   RBatchSink(const RBatchSink &) = delete;
   RBatchSink &operator=(const RBatchSink &) = delete;
   RBatchSink(RBatchSink &&) = delete;
   RBatchSink &operator=(RBatchSink &&) = delete;
   virtual ~RBatchSink() = default;

   /// \brief Write every row of \p batch.
   void FillBatch(const RFlat2DMatrix &batch);

   /// \brief Flush everything to disk once after the last batch
   virtual void Commit() = 0;
};

/// \brief Writes batches into a TTree, as one float leaf per scalar column and one
/// std::vector<float> branch per vector column, matching the RNTuple sink's schema.
class RTTreeBatchSink final : public RBatchSink {
   std::unique_ptr<TFile> fFile;
   TTree *fTree;
   std::vector<float> fRow;                        // scalar branches are bound into this buffer
   std::vector<std::vector<float>> fVectorBuffers; // one buffer per vector column, in fLayout order

public:
   RTTreeBatchSink(std::string_view dataset_name, std::string_view filename, std::vector<RColumnLayout> layout);
   RTTreeBatchSink(const RTTreeBatchSink &) = delete;
   RTTreeBatchSink &operator=(const RTTreeBatchSink &) = delete;
   RTTreeBatchSink(RTTreeBatchSink &&) = delete;
   RTTreeBatchSink &operator=(RTTreeBatchSink &&) = delete;
   ~RTTreeBatchSink() override;

   void FillRow(const float *row) override;
   void Commit() override;
};

/// \brief Writes batches into an RNTuple, as one float field per scalar column and one
/// std::vector<float> field per vector column.
class RNTupleBatchSink final : public RBatchSink {
   std::vector<std::shared_ptr<float>> fScalarFields;
   std::vector<std::shared_ptr<std::vector<float>>> fVectorFields;
   std::vector<float *> fDestinations; // where each column's floats go
   std::unique_ptr<ROOT::RNTupleWriter> fWriter;

public:
   RNTupleBatchSink(std::string_view dataset_name, std::string_view filename, std::vector<RColumnLayout> layout);
   RNTupleBatchSink(const RNTupleBatchSink &) = delete;
   RNTupleBatchSink &operator=(const RNTupleBatchSink &) = delete;
   RNTupleBatchSink(RNTupleBatchSink &&) = delete;
   RNTupleBatchSink &operator=(RNTupleBatchSink &&) = delete;
   ~RNTupleBatchSink() override;

   void FillRow(const float *row) override;
   void Commit() override;
};

//////////////////////////////////////////////////////////////////////////
/// \brief Create the sink matching \p format.
/// \param dataset_name Name of the output TTree or RNTuple.
/// \param filename Path of the output file, overwritten if it exists.
/// \param layout Where each of the loader's columns sits in a batch row, see RColumnLayout.
/// \param format Either "ttree" or "rntuple".
std::unique_ptr<RBatchSink> CreateBatchSink(std::string_view dataset_name, std::string_view filename,
                                            std::vector<RColumnLayout> layout, std::string_view format);

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RBATCHSINK
