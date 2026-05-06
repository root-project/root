#include <TPython.h>

#include <any>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "gtest/gtest.h"

namespace {

void task1(int i)
{
   std::stringstream code;
   code << "arr.append(" << i << ")";
   TPython::Exec(code.str().c_str());
}

} // namespace

// Test TPython::Exec from multiple threads.
TEST(TPython, ExecMultithreading)
{
   int nThreads = 100;

   // Concurrently append to this array
   TPython::Exec("arr = []");

   std::vector<std::thread> threads;
   for (int i = 0; i < nThreads; i++) {
      threads.emplace_back(task1, i);
   }

   for (decltype(threads.size()) i = 0; i < threads.size(); i++) {
      threads[i].join();
   }

   // In the end, let's check if the size is correct.
   std::any len;
   TPython::Exec("_anyresult = ROOT.std.make_any['int'](len(arr))", &len);
   EXPECT_EQ(std::any_cast<int>(len), nThreads);
}

namespace {

// Write a tiny Python macro that sets the variable `tpython_loadmacro_marker` to `marker`.
// to a temp file whose name contains `nameSubstr`. Returns the path used.
std::filesystem::path writeMarkerMacro(const std::string &nameSubstr, int marker)
{
   auto path = std::filesystem::temp_directory_path() / (std::string("tpython_loadmacro_") + nameSubstr + "_test.py");
   std::ofstream{path} << "tpython_loadmacro_marker = " << marker << "\n";
   return path;
}

int readMarker()
{
   std::any out;
   TPython::Exec("_anyresult = ROOT.std.make_any['int'](tpython_loadmacro_marker)", &out);
   return std::any_cast<int>(out);
}

} // namespace

// LoadMacro must accept filenames containing a single quote character.
TEST(TPython, LoadMacroSingleQuoteInName)
{
   auto path = writeMarkerMacro("with'quote", 11);
   TPython::Exec("tpython_loadmacro_marker = 0");
   TPython::LoadMacro(path.string().c_str());
   EXPECT_EQ(readMarker(), 11);
   std::filesystem::remove(path);
}

#ifndef _MSC_VER
// On platforms whose filesystems allow it (i.e. not Windows), LoadMacro must
// also accept filenames containing a double quote character.
TEST(TPython, LoadMacroDoubleQuoteInName)
{
   auto path = writeMarkerMacro("with\"quote", 22);
   TPython::Exec("tpython_loadmacro_marker = 0");
   TPython::LoadMacro(path.string().c_str());
   EXPECT_EQ(readMarker(), 22);
   std::filesystem::remove(path);
}

// Same idea for newline / carriage-return characters in the file name: these
// are legal on POSIX filesystems and used to break the Python source across
// lines, turning it into a SyntaxError.
TEST(TPython, LoadMacroNewlineInName)
{
   auto path = writeMarkerMacro("with\nnewline", 33);
   TPython::Exec("tpython_loadmacro_marker = 0");
   TPython::LoadMacro(path.string().c_str());
   EXPECT_EQ(readMarker(), 33);
   std::filesystem::remove(path);
}

TEST(TPython, LoadMacroCarriageReturnInName)
{
   auto path = writeMarkerMacro("with\rcarriagereturn", 44);
   TPython::Exec("tpython_loadmacro_marker = 0");
   TPython::LoadMacro(path.string().c_str());
   EXPECT_EQ(readMarker(), 44);
   std::filesystem::remove(path);
}
#endif
