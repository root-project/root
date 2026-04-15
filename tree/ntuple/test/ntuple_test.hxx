#ifndef ROOT_RNTuple_Test
#define ROOT_RNTuple_Test

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RMiniFile.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleFillStatus.hxx>
#include <ROOT/RNTupleJoinTable.hxx>
#include <ROOT/RNTupleMerger.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleParallelWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriteOptionsDaos.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RRawFile.hxx>
#include <ROOT/TestSupport.hxx>

#include <RZip.h>
#include <TClass.h>
#include <TFile.h>
#include <TROOT.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "CustomStruct.hxx"

#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

using ROOT::EExtraTypeInfoIds;
using ROOT::RNTupleLocalIndex;
using ROOT::RNTupleLocator;
using ROOT::RNTupleLocatorObject64;
using ROOT::Internal::RColumnIndex;
using RClusterDescriptor = ROOT::RClusterDescriptor;
using RClusterDescriptorBuilder = ROOT::Internal::RClusterDescriptorBuilder;
using RClusterGroupDescriptorBuilder = ROOT::Internal::RClusterGroupDescriptorBuilder;
using RColumnDescriptorBuilder = ROOT::Internal::RColumnDescriptorBuilder;
using RColumnElementBase = ROOT::Internal::RColumnElementBase;
using RColumnSwitch = ROOT::Internal::RColumnSwitch;
using ROOT::Internal::RExtraTypeInfoDescriptorBuilder;
using RFieldDescriptorBuilder = ROOT::Internal::RFieldDescriptorBuilder;
template <class T>
using RField = ROOT::RField<T>;
using RFieldBase = ROOT::RFieldBase;
using RFieldDescriptor = ROOT::RFieldDescriptor;
using RMiniFileReader = ROOT::Internal::RMiniFileReader;
using RNTupleAtomicCounter = ROOT::Experimental::Detail::RNTupleAtomicCounter;
using RNTupleAtomicTimer = ROOT::Experimental::Detail::RNTupleAtomicTimer;
using RNTupleCalcPerf = ROOT::Experimental::Detail::RNTupleCalcPerf;
using RNTupleCompressor = ROOT::Internal::RNTupleCompressor;
using RNTupleDecompressor = ROOT::Internal::RNTupleDecompressor;
using RNTupleDescriptor = ROOT::RNTupleDescriptor;
using RNTupleFillStatus = ROOT::RNTupleFillStatus;
using RNTupleDescriptorBuilder = ROOT::Internal::RNTupleDescriptorBuilder;
using RNTupleFileWriter = ROOT::Internal::RNTupleFileWriter;
using RNTupleJoinTable = ROOT::Experimental::Internal::RNTupleJoinTable;
using RNTupleParallelWriter = ROOT::RNTupleParallelWriter;
using RNTupleReader = ROOT::RNTupleReader;
using RNTupleReadOptions = ROOT::RNTupleReadOptions;
using RNTupleWriter = ROOT::RNTupleWriter;
using RNTupleWriteOptions = ROOT::RNTupleWriteOptions;
using RNTupleWriteOptionsDaos = ROOT::Experimental::RNTupleWriteOptionsDaos;
using RNTupleMetrics = ROOT::Experimental::Detail::RNTupleMetrics;
using RNTupleMerger = ROOT::Experimental::Internal::RNTupleMerger;
using RNTupleMergeOptions = ROOT::Experimental::Internal::RNTupleMergeOptions;
using ENTupleMergingMode = ROOT::Experimental::Internal::ENTupleMergingMode;
using RNTupleModel = ROOT::RNTupleModel;
using RNTupleOpenSpec = ROOT::Experimental::RNTupleOpenSpec;
using RNTuplePlainCounter = ROOT::Experimental::Detail::RNTuplePlainCounter;
using RNTuplePlainTimer = ROOT::Experimental::Detail::RNTuplePlainTimer;
using RNTupleProcessor = ROOT::Experimental::RNTupleProcessor;
using RNTupleSerializer = ROOT::Internal::RNTupleSerializer;
using RPage = ROOT::Internal::RPage;
using RPageAllocatorHeap = ROOT::Internal::RPageAllocatorHeap;
using RPagePool = ROOT::Internal::RPagePool;
using RPageSink = ROOT::Internal::RPageSink;
using RPageSinkBuf = ROOT::Internal::RPageSinkBuf;
using RPageSinkFile = ROOT::Internal::RPageSinkFile;
using RPageSource = ROOT::Internal::RPageSource;
using RPageSourceFile = ROOT::Internal::RPageSourceFile;
using RPageStorage = ROOT::Internal::RPageStorage;
using RPrepareVisitor = ROOT::Internal::RPrepareVisitor;
using RPrintSchemaVisitor = ROOT::Internal::RPrintSchemaVisitor;
using RRawFile = ROOT::Internal::RRawFile;
using EContainerFormat = RNTupleFileWriter::EContainerFormat;
template <typename T>
using RNTupleView = ROOT::RNTupleView<T>;

using ROOT::Internal::MakeUninitArray;
using ROOT::TestSupport::FileRaii;

#ifdef R__USE_IMT
struct IMTRAII {
   IMTRAII() { ROOT::EnableImplicitMT(); }
   ~IMTRAII() { ROOT::DisableImplicitMT(); }
};
#endif

/// Creates an uncompressed RNTuple called "ntpl" with three float fields, px, py, pz, with a single entry.
/// The page of px has a wrong checksum. The page of py has corrupted data. The page of pz is valid.
/// The function is backend agnostic (file, DAOS, ...).
void CreateCorruptedRNTuple(const std::string &uri);

enum class EEndianness {
   LE,
   BE
};

/// Given the file at `filePath` containing an UNCOMPRESSED RNTuple, and given the seek/len of a section of this
/// RNTuple, patches a byte range of this section with the given buffer `bytesToWrite`. After doing this, it recomputes
/// and updates the section's checksum.
/// `sectionSeek` must point to the start of the section's payload, excluding the key (and, in case of the Anchor, the
/// first 6 bytes containing the object's version and nbytes).
/// `sectionLen` must refer to the in-memory of the section's payload, excluding the key and the checksum.
/// Note that this is assumed to be equal to the on-disk size since the section must be uncompressed.
///
/// This function does very minimal checks (e.g. it does not verify that the given section is actually correct or in
/// the file at all), so the caller must validate these assumptions.
void PatchRNTupleSection(std::string_view filePath, std::uint64_t sectionSeek, std::uint64_t sectionLen,
                         std::uint64_t patchedOffsetIntoSection, const std::byte *bytesToWrite,
                         std::size_t bytesToWriteLen, EEndianness sectionEndianness);

#endif
