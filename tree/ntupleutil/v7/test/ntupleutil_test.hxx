#ifndef ROOT7_RNTupleUtil_Test
#define ROOT7_RNTupleUtil_Test

#include "gtest/gtest.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>

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
