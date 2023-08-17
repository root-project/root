#include <cstdio>
#include <fstream>

#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>

#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"

#if defined(_WIN32)
   const char *suffix = ".dll";
#else
   const char *suffix = ".so";
#endif

const static std::string code_template{R"(
#ifndef TCLINGLOADUNLOADFILETESTS_HELPER_%s
#define TCLINGLOADUNLOADFILETESTS_HELPER_%s
bool %s_%s(double x, double val)
{
    return x > val;
}
#endif
)"};

void write_n_helpers(const std::string &basename, unsigned int n)
{
   for (unsigned int i = 0; i < n; i++) {
      char buffer[300];
      const auto itos = std::to_string(i);
      std::snprintf(buffer, 300, code_template.c_str(), itos.c_str(), itos.c_str(), basename.c_str(), itos.c_str());
      std::string library_file{basename + "_" + itos + ".cpp"};
      std::ofstream myfile{library_file};
      myfile << buffer;
   }
}

void compile_n_macros(const std::string &basename, unsigned int n)
{
   for (unsigned int i = 0; i < n; i++) {
      std::string library_file{basename + "_" + std::to_string(i) + ".cpp"};
      gSystem->CompileMacro(library_file.c_str(), "O");
   }
}

void load_unload(const std::string &basename, unsigned int n)
{
   std::string library_so{basename + "_" + std::to_string(n) + "_cpp" + suffix};

   for (int i = 0; i < 100; i++) {
      EXPECT_EQ(gInterpreter->LoadFile(library_so.c_str()), 0) << "Failed to load " << library_so << std::endl;
      EXPECT_EQ(gInterpreter->UnloadFile(library_so.c_str()), 0) << "Failed to unload " << library_so << std::endl;
   }
}

void remove_n_library_artifacts(const std::string &basename, unsigned int n)
{
   for (unsigned int i = 0; i < n; i++) {
      std::string library_basename{basename + "_" + std::to_string(i)};
      std::string library_file{library_basename + ".cpp"};
      std::string library_d{library_basename + "_cpp.d"};
      std::string library_pcm{library_basename + "_cpp_ACLiC_dict_rdict.pcm"};

      std::remove(library_file.c_str());
      std::remove(library_d.c_str());
      std::remove(library_pcm.c_str());
   }
}

TEST(TClingLoadUnloadFile, ConcurrentLoadUnloadSameLib)
{
   ROOT::EnableThreadSafety();

   std::string basename{"concurrent_load_unload_same_lib"};
   // All threads will load/unload the same library
   unsigned int n_libraries = 1;
   write_n_helpers(basename, n_libraries);

   compile_n_macros(basename, n_libraries);

   unsigned int n_threads = 5;
   std::vector<std::thread> threads;
   threads.reserve(n_threads);
   for (unsigned int i = 0; i < n_threads; i++) {
      threads.emplace_back(load_unload, basename, 0); // Only one shared library for all threads
   }
   for (auto &thread : threads) {
      thread.join();
   }

   remove_n_library_artifacts(basename, n_libraries);
}

TEST(TClingLoadUnloadFile, ConcurrentLoadUnloadOneLibPerThread)
{
   ROOT::EnableThreadSafety();

   std::string basename{"concurrent_load_unload_one_lib_per_thread"};
   // Every thread loads/unloads a different library
   unsigned int n_libraries = 5;
   write_n_helpers(basename, n_libraries);

   compile_n_macros(basename, n_libraries);

   std::vector<std::thread> threads;
   threads.reserve(n_libraries);
   for (unsigned int i = 0; i < n_libraries; i++) {
      threads.emplace_back(load_unload, basename, i);
   }
   for (auto &thread : threads) {
      thread.join();
   }

   remove_n_library_artifacts(basename, n_libraries);
}
