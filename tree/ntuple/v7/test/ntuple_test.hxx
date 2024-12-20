#ifndef ROOT7_RNTuple_Test
#define ROOT7_RNTuple_Test

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RMiniFile.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleFillStatus.hxx>
#include <ROOT/RNTupleIndex.hxx>
#include <ROOT/RNTupleMerger.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleParallelWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleUtil.hxx>
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

using ClusterSize_t = ROOT::Experimental::ClusterSize_t;
using DescriptorId_t = ROOT::Experimental::DescriptorId_t;
using EColumnType = ROOT::Experimental::EColumnType;
using ROOT::Experimental::EExtraTypeInfoIds;
using ENTupleStructure = ROOT::Experimental::ENTupleStructure;
using NTupleSize_t = ROOT::Experimental::NTupleSize_t;
using RClusterIndex = ROOT::Experimental::RClusterIndex;
using RClusterDescriptor = ROOT::Experimental::RClusterDescriptor;
using RClusterDescriptorBuilder = ROOT::Experimental::Internal::RClusterDescriptorBuilder;
using RClusterGroupDescriptorBuilder = ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder;
using RColumnDescriptorBuilder = ROOT::Experimental::Internal::RColumnDescriptorBuilder;
using RColumnElementBase = ROOT::Experimental::Internal::RColumnElementBase;
using RColumnSwitch = ROOT::Experimental::RColumnSwitch;
using ROOT::Experimental::Internal::RExtraTypeInfoDescriptorBuilder;
using RFieldDescriptorBuilder = ROOT::Experimental::Internal::RFieldDescriptorBuilder;
template <class T>
using RField = ROOT::Experimental::RField<T>;
using RFieldBase = ROOT::Experimental::RFieldBase;
using RFieldDescriptor = ROOT::Experimental::RFieldDescriptor;
using RNTupleLocator = ROOT::Experimental::RNTupleLocator;
using RNTupleLocatorObject64 = ROOT::Experimental::RNTupleLocatorObject64;
using RMiniFileReader = ROOT::Experimental::Internal::RMiniFileReader;
using RNTupleAtomicCounter = ROOT::Experimental::Detail::RNTupleAtomicCounter;
using RNTupleAtomicTimer = ROOT::Experimental::Detail::RNTupleAtomicTimer;
using RNTupleCalcPerf = ROOT::Experimental::Detail::RNTupleCalcPerf;
using RNTupleCompressor = ROOT::Experimental::Internal::RNTupleCompressor;
using RNTupleDecompressor = ROOT::Experimental::Internal::RNTupleDecompressor;
using RNTupleDescriptor = ROOT::Experimental::RNTupleDescriptor;
using RNTupleFillStatus = ROOT::Experimental::RNTupleFillStatus;
using RNTupleDescriptorBuilder = ROOT::Experimental::Internal::RNTupleDescriptorBuilder;
using RNTupleFileWriter = ROOT::Experimental::Internal::RNTupleFileWriter;
using RNTupleIndex = ROOT::Experimental::Internal::RNTupleIndex;
using RNTupleParallelWriter = ROOT::Experimental::RNTupleParallelWriter;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleReadOptions = ROOT::Experimental::RNTupleReadOptions;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;
using RNTupleWriteOptionsDaos = ROOT::Experimental::RNTupleWriteOptionsDaos;
using RNTupleMetrics = ROOT::Experimental::Detail::RNTupleMetrics;
using RNTupleMerger = ROOT::Experimental::Internal::RNTupleMerger;
using RNTupleMergeOptions = ROOT::Experimental::Internal::RNTupleMergeOptions;
using ENTupleMergingMode = ROOT::Experimental::Internal::ENTupleMergingMode;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleOpenSpec = ROOT::Experimental::RNTupleOpenSpec;
using RNTuplePlainCounter = ROOT::Experimental::Detail::RNTuplePlainCounter;
using RNTuplePlainTimer = ROOT::Experimental::Detail::RNTuplePlainTimer;
using RNTupleProcessor = ROOT::Experimental::RNTupleProcessor;
using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;
using RPage = ROOT::Experimental::Internal::RPage;
using RPageAllocatorHeap = ROOT::Experimental::Internal::RPageAllocatorHeap;
using RPagePool = ROOT::Experimental::Internal::RPagePool;
using RPageSink = ROOT::Experimental::Internal::RPageSink;
using RPageSinkBuf = ROOT::Experimental::Internal::RPageSinkBuf;
using RPageSinkFile = ROOT::Experimental::Internal::RPageSinkFile;
using RPageSource = ROOT::Experimental::Internal::RPageSource;
using RPageSourceFile = ROOT::Experimental::Internal::RPageSourceFile;
using RPageStorage = ROOT::Experimental::Internal::RPageStorage;
using RPrepareVisitor = ROOT::Experimental::RPrepareVisitor;
using RPrintSchemaVisitor = ROOT::Experimental::RPrintSchemaVisitor;
using RRawFile = ROOT::Internal::RRawFile;
using EContainerFormat = RNTupleFileWriter::EContainerFormat;

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;
   bool fPreserveFile = false;

public:
   explicit FileRaii(const std::string &path) : fPath(path) {}
   FileRaii(const FileRaii &) = delete;
   FileRaii &operator=(const FileRaii &) = delete;
   ~FileRaii()
   {
      if (!fPreserveFile)
         std::remove(fPath.c_str());
   }
   std::string GetPath() const { return fPath; }

   // Useful if you want to keep a test file after the test has finished running
   // for debugging purposes. Should only be used locally and never pushed.
   void PreserveFile() { fPreserveFile = true; }
};

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

#endif
