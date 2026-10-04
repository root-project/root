/// \file
/// Scenarios of the LLVM isolation tests, with the libraries named by the environment
/// variables LLVM_ISOLATION_*:
///   "jit":   interpreter and JIT work after loading LOAD, if set, into the global
///            scope (as gSystem->Load does).
///   "names": the static LLVM libraries in ARCHIVES (separated by ':') define every
///            symbol in LLVMIsolationCanary.h, so that these cannot silently become
///            stale.

#include "LLVMIsolationCanary.h"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

/// Add the names in the symbol table of a GNU ar archive: its first member, named "/"
/// or "/SYM64/", with the big-endian number of symbols, their offsets and their names.
bool LLVMIsolationArchiveSymbols(const std::string &path, std::set<std::string> &symbols)
{
   std::ifstream file(path, std::ios::binary);
   char header[8 + 60] = {};
   if (!file.read(header, sizeof(header)) ||
       (std::string(header, 8) != "!<arch>\n" && std::string(header, 8) != "!<thin>\n"))
      return false;
   size_t width = std::strncmp(header + 8, "/ ", 2) == 0 ? 4 : std::strncmp(header + 8, "/SYM64/ ", 8) == 0 ? 8 : 0;
   std::string table(std::strtoull(header + 8 + 48, nullptr, 10), '\0');
   if (!width || !file.read(&table[0], table.size()))
      return false;
   uint64_t count = 0;
   for (size_t i = 0; i < width; ++i)
      count = count << 8 | static_cast<unsigned char>(table[i]);
   for (size_t i = 0, pos = width * (count + 1); i < count && pos < table.size(); ++i) {
      std::string name(table.c_str() + pos);
      pos += name.size() + 1;
      symbols.insert(name);
   }
   return true;
}

void LLVMIsolation(const char *scenario = "jit")
{
   const std::string which = scenario;
   if (which == "jit") {
      const char *library = gSystem->Getenv("LLVM_ISOLATION_LOAD");
      if (library && gSystem->Load(library) < 0) {
         Error("LLVMIsolation", "cannot load %s", library);
         return;
      }
      gInterpreter->Declare("template <class T> T LLVMIsolationTwice(T a) { return a + a; }");
      auto result = gInterpreter->Calc("LLVMIsolationTwice(21)");
      TClass *cl = TClass::GetClass("TNamed");
      if (result == 42 && cl && cl->GetListOfMethods()->GetSize() > 0)
         printf("LLVM isolation: JIT OK\n");
   } else if (which == "names") {
      std::set<std::string> defined;
      std::stringstream archives(gSystem->Getenv("LLVM_ISOLATION_ARCHIVES"));
      for (std::string archive; std::getline(archives, archive, ':');) {
         if (!LLVMIsolationArchiveSymbols(archive, defined))
            Error("LLVMIsolation", "%s is not an archive with a symbol table", archive.c_str());
      }
#define LLVM_ISOLATION_NAME(name, mangled) mangled,
      const char *names[] = {LLVM_ISOLATION_DEFINED_SYMBOLS(LLVM_ISOLATION_NAME)
                                LLVM_ISOLATION_PROBED_SYMBOLS(LLVM_ISOLATION_NAME)};
      int missing = 0;
      for (const char *name : names) {
         if (!defined.count(name)) {
            Error("LLVMIsolation", "not defined: %s", name);
            ++missing;
         }
      }
      printf("LLVM isolation names: missing=%d of %zu\n", missing, std::size(names));
   }
}
