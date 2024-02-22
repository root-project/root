#ifndef ROOT7_RNTuple_Test
#define ROOT7_RNTuple_Test

#include <ROOT/RColumnModel.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RMiniFile.hxx>
#include <ROOT/RNTupleCollectionWriter.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMerger.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleParallelWriter.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageSourceFriends.hxx>
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

// Backward compatibility for gtest version < 1.10.0
#ifndef TYPED_TEST_SUITE
#define TYPED_TEST_SUITE TYPED_TEST_CASE
#endif

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
using ENTupleStructure = ROOT::Experimental::ENTupleStructure;
using NTupleSize_t = ROOT::Experimental::NTupleSize_t;
using RColumnModel = ROOT::Experimental::RColumnModel;
using RClusterIndex = ROOT::Experimental::RClusterIndex;
using RClusterDescriptorBuilder = ROOT::Experimental::Internal::RClusterDescriptorBuilder;
using RClusterGroupDescriptorBuilder = ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder;
using RColumnDescriptorBuilder = ROOT::Experimental::Internal::RColumnDescriptorBuilder;
using RFieldDescriptorBuilder = ROOT::Experimental::Internal::RFieldDescriptorBuilder;
using RException = ROOT::Experimental::RException;
template <class T>
using RField = ROOT::Experimental::RField<T>;
using RFieldBase = ROOT::Experimental::RFieldBase;
using RFieldDescriptor = ROOT::Experimental::RFieldDescriptor;
using RNTupleLocator = ROOT::Experimental::RNTupleLocator;
using RNTupleLocatorObject64 = ROOT::Experimental::RNTupleLocatorObject64;
using RMiniFileReader = ROOT::Experimental::Internal::RMiniFileReader;
using RNTuple = ROOT::Experimental::RNTuple;
using RNTupleAtomicCounter = ROOT::Experimental::Detail::RNTupleAtomicCounter;
using RNTupleAtomicTimer = ROOT::Experimental::Detail::RNTupleAtomicTimer;
using RNTupleCalcPerf = ROOT::Experimental::Detail::RNTupleCalcPerf;
using RNTupleCompressor = ROOT::Experimental::Internal::RNTupleCompressor;
using RNTupleDecompressor = ROOT::Experimental::Internal::RNTupleDecompressor;
using RNTupleDescriptor = ROOT::Experimental::RNTupleDescriptor;
using RNTupleDescriptorBuilder = ROOT::Experimental::Internal::RNTupleDescriptorBuilder;
using RNTupleFileWriter = ROOT::Experimental::Internal::RNTupleFileWriter;
using RNTupleParallelWriter = ROOT::Experimental::RNTupleParallelWriter;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleReadOptions = ROOT::Experimental::RNTupleReadOptions;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;
using RNTupleWriteOptionsDaos = ROOT::Experimental::RNTupleWriteOptionsDaos;
using RNTupleMetrics = ROOT::Experimental::Detail::RNTupleMetrics;
using RNTupleMerger = ROOT::Experimental::Internal::RNTupleMerger;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTuplePlainCounter = ROOT::Experimental::Detail::RNTuplePlainCounter;
using RNTuplePlainTimer = ROOT::Experimental::Detail::RNTuplePlainTimer;
using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;
using RPage = ROOT::Experimental::Internal::RPage;
using RPageAllocatorHeap = ROOT::Experimental::Internal::RPageAllocatorHeap;
using RPageDeleter = ROOT::Experimental::Internal::RPageDeleter;
using RPagePool = ROOT::Experimental::Internal::RPagePool;
using RPageSink = ROOT::Experimental::Internal::RPageSink;
using RPageSinkBuf = ROOT::Experimental::Internal::RPageSinkBuf;
using RPageSinkFile = ROOT::Experimental::Internal::RPageSinkFile;
using RPageSource = ROOT::Experimental::Internal::RPageSource;
using RPageSourceFile = ROOT::Experimental::Internal::RPageSourceFile;
using RPageSourceFriends = ROOT::Experimental::Internal::RPageSourceFriends;
using RPageStorage = ROOT::Experimental::Internal::RPageStorage;
using RPrepareVisitor = ROOT::Experimental::RPrepareVisitor;
using RPrintSchemaVisitor = ROOT::Experimental::RPrintSchemaVisitor;
using RRawFile = ROOT::Internal::RRawFile;
template <class T>
using RResult = ROOT::Experimental::RResult<T>;

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;
public:
   explicit FileRaii(const std::string &path) : fPath(path) { }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   ~FileRaii() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

#endif
