#ifndef ROOT7_RNTupleUtil_Test
#define ROOT7_RNTupleUtil_Test

#include "gtest/gtest.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleImporter.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>

#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "CustomStructUtil.hxx"

using ROOT::Experimental::RNTupleImporter;
using ROOT::Experimental::RNTupleInspector;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleReader;
using ROOT::Experimental::RNTupleWriteOptions;
using ROOT::Experimental::RNTupleWriter;

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the
 * guarded file when the wrapper object goes out of scope.
 */
class FileRaii {
private:
   static constexpr bool kDebug = false; // if true, don't delete the file on destruction
   std::string fPath;

public:
   explicit FileRaii(const std::string &path) : fPath(path) {}
   FileRaii(const FileRaii &) = delete;
   FileRaii &operator=(const FileRaii &) = delete;
   ~FileRaii()
   {
      if (!kDebug)
         std::remove(fPath.c_str());
   }
   std::string GetPath() const { return fPath; }
};

#endif // ROOT7_RNTupleUtil_Test
