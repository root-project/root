#include <cstdio>
#include <fstream>

#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>

#include "TROOT.h"
#include "TSystem.h"

#if defined(_WIN32)
const char *suffix = ".dll";
#else
const char *suffix = ".so";
#endif

const static std::string code_template{R"(
#ifndef TSYSTEMCOMPILEMACROTHREADSAFETY_HELPER_%s
#define TSYSTEMCOMPILEMACROTHREADSAFETY_HELPER_%s
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

void compile_macro_n_times(const std::string &filename, unsigned int n)
{
   for (unsigned int i = 0; i < n; i++) {
      gSystem->CompileMacro(filename.c_str(), "O");
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

TEST(TSystemCompileMacro, ConcurrentCompileMacroSameFile)
{
   ROOT::EnableThreadSafety();

   std::string basename{"compile_macro_lib"};
   // All threads will compile the same library
   unsigned int n_libraries = 1;
   write_n_helpers(basename, n_libraries);

   std::string library_file{basename + "_0.cpp"};

   unsigned int n_threads = 5;
   std::vector<std::thread> threads;
   threads.reserve(n_threads);
   unsigned int n_compilations{10};
   for (unsigned int i = 0; i < n_threads; i++) {
      threads.emplace_back(compile_macro_n_times, library_file,
                           n_compilations); // Only one shared library for all threads
   }
   for (auto &thread : threads) {
      thread.join();
   }

   remove_n_library_artifacts(basename, n_libraries);
}

TEST(TSystemCompileMacro, ConcurrentCompileMacroOneFilePerThread)
{
   ROOT::EnableThreadSafety();

   std::string basename{"compile_macro_one_lib_per_thread"};
   // Every thread compiles a different library
   unsigned int n_libraries = 5;
   write_n_helpers(basename, n_libraries);

   std::vector<std::string> library_names;
   library_names.reserve(n_libraries);
   for (std::size_t i = 0; i < n_libraries; i++) {
      library_names.emplace_back(basename + "_" + std::to_string(i) + ".cpp");
   }

   std::vector<std::thread> threads;
   threads.reserve(n_libraries);
   unsigned int n_compilations{10};
   for (std::size_t i = 0; i < n_libraries; i++) {
      threads.emplace_back(compile_macro_n_times, library_names[i], n_compilations);
   }
   for (auto &thread : threads) {
      thread.join();
   }

   remove_n_library_artifacts(basename, n_libraries);
}
