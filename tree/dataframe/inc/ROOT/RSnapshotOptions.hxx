// Author: Guilherme Amadio, Enrico Guiraud, Danilo Piparo CERN  2/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSNAPSHOTOPTIONS
#define ROOT_RSNAPSHOTOPTIONS

#include <Compression.h>
#include <string_view>
#include <string>

namespace ROOT {

namespace RDF {
enum class ESnapshotOutputFormat {
   kDefault,
   kTTree,
   kRNTuple
};

// clang-format off
/**
\struct ROOT::RDF::RSnapshotOptions
\brief A collection of options to steer the creation of the dataset on disk through Snapshot().

Some settings are output format-dependent. Please refer to the table below for an overview of all options, and to which
output format they apply.

Note that for RNTuple, the defaults correspond to those set in RNTupleWriteOptions.

<table>
<thead>
<tr>
<th>Format</th>
<th>Option</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">All</td>
<td><code>fMode</code></td>
<td><code>std::string</code></td>
<td>&quot;RECREATE&quot;</td>
<td>Creation mode for the output TFile</td>
</tr>
<tr>
<td><code>fCompressionAlgorithm</code></td>
<td><code>ROOT::RCompressionSetting::EAlgorithm</code></td>
<td>Zstd</td>
<td>Compression algorithm for the output dataset</td>
</tr>
<tr>
<td><code>fCompressionLevel</code></td>
<td><code>int</code></td>
<td>5</td>
<td>Compression level for the output dataset</td>
</tr>
<tr>
<td><code>fOutputFormat</code></td>
<td><code>ROOT::RDF::ESnapshotOutputFormat</code></td>
<td>TTree</td>
<td>Which output data format to use</td>
</tr>
<tr>
<td><code>fLazy</code></td>
<td><code>bool</code></td>
<td>False</td>
<td>Whether to immediately start the event loop when Snapshot() is called</td>
</tr>
<tr>
<td><code>fOverwriteIfExists</code></td>
<td><code>bool</code></td>
<td>False</td>
<td>If <code>fMode</code> is &quot;UPDATE&quot;, overwrite the object with the same name in the output file already present</td>
</tr>
<tr>
<td><code>fVector2RVec</code></td>
<td><code>bool</code></td>
<td>True</td>
<td>Store <code>std::vector</code>-type columns as <code>ROOT::RVec</code> in the output</td>
</tr>
<tr>
<td><code>fIncludeInVariations</code></td>
<td><code>bool</code></td>
<td>False</td>
<td>Include columns that result from a Vary() action in the output</td>
</tr>
<tr>
<td rowspan="3">TTree</td>
<td><code>fAutoFlush</code></td>
<td><code>int</code></td>
<td>0</td>
<td>AutoFlush setting for the output (see TTree::SetAutoFlush())</td>
</tr>
<tr>
<td><code>fSplitLevel</code></td>
<td><code>int</code></td>
<td>99</td>
<td>Split level of the output branches</td>
</tr>
<tr>
<td><code>fBasketSize</code></td>
<td><code>int</code></td>
<td>-1</td>
<td>Output basket size (a value of -1 means the TTree default of 32000 B used)</td>
</tr>
<tr>
<td rowspan="6">RNTuple</td>
<td><code>fApproxZippedClusterSize</code></td>
<td><code>std::size_t</code></td>
<td>128 MiB</td>
<td>Approximate output compressed cluster size</td>
</tr>
<tr>
<td><code>fMaxUnzippedClusterSize</code></td>
<td><code>std::size_t</code></td>
<td>1280 MiB</td>
<td>Maximum uncompressed output cluster size</td>
</tr>
<tr>
<td><code>fInitialUnzippedPageSize</code></td>
<td><code>std::size_t</code></td>
<td>256 B</td>
<td>Initial output page size before compression</td>
</tr>
<tr>
<td><code>fMaxUnzippedPageSize</code></td>
<td><code>std::size_t</code></td>
<td>1 MiB</td>
<td>Maximum allowed output page size before compression</td>
</tr>
<tr>
<td><code>fEnablePageChecksums</code></td>
<td><code>bool</code></td>
<td>True</td>
<td>Enable checksumming for output pages</td>
</tr>
<tr>
<td><code>fEnableSamePageMerging</code></td>
<td><code>bool</code></td>
<td>True</td>
<td>Enable identical-page deduplication (requires page checksumming enabled)</td>
</tr>
</tbody>
</table>
*/
// clang-format on
struct RSnapshotOptions {
   using ECAlgo = ROOT::RCompressionSetting::EAlgorithm::EValues;
   RSnapshotOptions() = default;
   RSnapshotOptions(std::string_view mode, ECAlgo comprAlgo, int comprLevel, int autoFlush, int splitLevel, bool lazy,
                    bool overwriteIfExists = false, bool vector2RVec = true, int basketSize = -1,
                    std::size_t approxZippedClusterSize = 128 * 1024 * 1024,
                    std::size_t maxUnzippedClusterSize = 10 * 128 * 1024 * 1024,
                    std::size_t maxUnzippedPageSize = 1024 * 1024, std::size_t initUnzippedPageSize = 256,
                    bool enablePageChecksums = true, bool enableSamePageMerging = true,
                    ESnapshotOutputFormat outputFormat = ESnapshotOutputFormat::kDefault)
      : fMode(mode),
        fOutputFormat(outputFormat),
        fCompressionAlgorithm(comprAlgo),
        fCompressionLevel{comprLevel},
        fLazy(lazy),
        fOverwriteIfExists(overwriteIfExists),
        fVector2RVec(vector2RVec),
        fAutoFlush(autoFlush),
        fSplitLevel(splitLevel),
        fBasketSize(basketSize),
        fApproxZippedClusterSize(approxZippedClusterSize),
        fMaxUnzippedClusterSize(maxUnzippedClusterSize),
        fInitialUnzippedPageSize(initUnzippedPageSize),
        fMaxUnzippedPageSize(maxUnzippedPageSize),
        fEnablePageChecksums(enablePageChecksums),
        fEnableSamePageMerging(enableSamePageMerging)
   {
   }
   std::string fMode = "RECREATE"; ///< Mode of creation of output file
   ESnapshotOutputFormat fOutputFormat = ESnapshotOutputFormat::kDefault; ///< Which data format to write to
   ECAlgo fCompressionAlgorithm =
      ROOT::RCompressionSetting::EAlgorithm::kZSTD; ///< Compression algorithm of output file
   int fCompressionLevel = 5;                       ///< Compression level of output file
   bool fLazy = false;                              ///< Do not start the event loop when Snapshot is called
   bool fOverwriteIfExists = false;  ///< If fMode is "UPDATE", overwrite object in output file if it already exists
   bool fVector2RVec = true;         ///< If set to true will convert std::vector columns to RVec when saving to disk
   bool fIncludeVariations = false;  ///< Include columns that result from a Vary() action

   /// *(TTree only)* AutoFlush value for output tree
   int fAutoFlush = 0;
   /// *(TTree only)* Split level of output tree
   int fSplitLevel = 99;
   /// *(TTree only)* Set a custom basket size option. For more details, see
   /// https://root.cern/manual/trees/#baskets-clusters-and-the-tree-header
   int fBasketSize = -1;

   /// *(RNTuple only)* Approximate target compressed cluster size
   std::size_t fApproxZippedClusterSize = 128 * 1024 * 1024;
   /// *(RNTuple only)* Maximum uncompressed cluster size
   std::size_t fMaxUnzippedClusterSize = 10 * fApproxZippedClusterSize;
   /// *(RNTuple only)* Initial page size before compression
   std::size_t fInitialUnzippedPageSize = 256;
   /// *(RNTuple only)* Maximum allowed page size before compression
   std::size_t fMaxUnzippedPageSize = 1024 * 1024;
   /// *(RNTuple only)* Enable checksumming for pages
   bool fEnablePageChecksums = true;
   /// *(RNTuple only)* Enable identical-page deduplication. Requires page checksumming
   bool fEnableSamePageMerging = true;
};
} // namespace RDF
} // namespace ROOT

#endif
