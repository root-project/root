/// \file
/// Scenarios of the LLVM isolation tests, with the libraries named by the environment
/// variables LLVM_ISOLATION_*:
///   "jit":   interpreter and JIT work after loading LOAD, if set, into the global
///            scope (as gSystem->Load does).
///   "names": the static LLVM libraries in ARCHIVES (separated by ':') define every
///            symbol in LLVMIsolationCanary.h, so that these cannot silently become
///            stale.
///   "exports": on macOS, the image IMAGE (libCling by default, else loaded first)
///              uses two-level namespaces, loads no shared LLVM or Clang library, and
///              exports no LLVM or Clang symbols: dyld merges weak definitions across
///              images by name, and lookups by name, as with -undefined dynamic_lookup
///              or dlsym(), see all of them.
///   "imports": on Windows, the DLL IMAGE (libCling by default) imports no LLVM or
///              Clang DLL: the loader shares a DLL by name with every other module
///              that imports a DLL of that name.

#include "LLVMIsolationCanary.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>

/// Whether `name` is part of LLVM's or libclang's C API, or an Itanium-mangled name
/// involving the namespaces llvm or clang outside the cling API.
bool LLVMIsolationIsLLVMSymbol(const char *name)
{
   if (std::strncmp(name, "_Z", 2) != 0)
      return (std::strncmp(name, "LLVM", 4) == 0 && std::isupper(static_cast<unsigned char>(name[4]))) ||
             std::strncmp(name, "clang_", 6) == 0;
   const char *p = name + 2;
   while (std::isupper(static_cast<unsigned char>(*p)))
      ++p;
   return std::strncmp(p, "5cling", 6) != 0 && (std::strstr(name, "4llvm") || std::strstr(name, "5clang"));
}

struct LLVMIsolationMachOAudit {
   int exports = 0, llvm = 0, weak = 0, shared = 0;
   bool twoLevel = false;
};

/// Audit a 64-bit Mach-O image: its exported symbols, those involving LLVM or Clang
/// and the weak ones among them, and the shared LLVM or Clang libraries it loads.
/// `linkedit` is where the image's __LINKEDIT contents would start at file offset 0.
LLVMIsolationMachOAudit LLVMIsolationAuditMachO(const mach_header_64 *header, const char *linkedit)
{
   LLVMIsolationMachOAudit audit;
   audit.twoLevel = header->flags & MH_TWOLEVEL;
   const char *command = reinterpret_cast<const char *>(header + 1);
   for (uint32_t i = 0; i < header->ncmds; ++i, command += reinterpret_cast<const load_command *>(command)->cmdsize) {
      uint32_t cmd = reinterpret_cast<const load_command *>(command)->cmd;
      if (cmd == LC_LOAD_DYLIB || cmd == LC_LOAD_WEAK_DYLIB || cmd == LC_REEXPORT_DYLIB) {
         const auto *dylib = reinterpret_cast<const dylib_command *>(command);
         const char *path = command + dylib->dylib.name.offset;
         const char *leaf = std::strrchr(path, '/') ? std::strrchr(path, '/') + 1 : path;
         audit.shared += std::strncmp(leaf, "libLLVM", 7) == 0 || std::strncmp(leaf, "libclang-cpp", 12) == 0;
      } else if (cmd == LC_SYMTAB && linkedit) {
         const auto *symtab = reinterpret_cast<const symtab_command *>(command);
         const auto *symbols = reinterpret_cast<const nlist_64 *>(linkedit + symtab->symoff);
         for (uint32_t s = 0; s < symtab->nsyms; ++s) {
            const nlist_64 &symbol = symbols[s];
            if ((symbol.n_type & N_STAB) || (symbol.n_type & (N_EXT | N_PEXT)) != N_EXT ||
                (symbol.n_type & N_TYPE) != N_SECT)
               continue;
            ++audit.exports;
            // Mach-O prefixes names with an underscore.
            if (LLVMIsolationIsLLVMSymbol(linkedit + symtab->stroff + symbol.n_un.n_strx + 1)) {
               ++audit.llvm;
               audit.weak += (symbol.n_desc & N_WEAK_DEF) != 0;
            }
         }
      }
   }
   return audit;
}
#endif

#ifdef _WIN32
/// Whether `dll` is a shared LLVM or Clang library, such as LLVM-C.dll, libLLVM-22.dll,
/// libclang.dll or clang-cpp.dll, but not clang's runtime libraries.
bool LLVMIsolationIsLLVMDll(std::string dll)
{
   for (char &c : dll)
      c = std::tolower(static_cast<unsigned char>(c));
   for (const char *prefix : {"llvm", "libllvm", "libclang", "clang-cpp"})
      if (dll.rfind(prefix, 0) == 0)
         return true;
   return false;
}

struct LLVMIsolationPEAudit {
   int shared = 0;
   std::vector<std::string> imports;
};

/// Audit a PE image, given as the contents of its file: the DLLs it imports, also
/// with delayed loading, and the LLVM or Clang ones among them.
LLVMIsolationPEAudit LLVMIsolationAuditPE(const std::string &file)
{
   LLVMIsolationPEAudit audit;
   auto u32 = [&](uint32_t offset) {
      uint32_t value = 0;
      if (offset + 4 <= file.size())
         std::memcpy(&value, file.data() + offset, 4);
      return value;
   };
   auto u16 = [&](uint32_t offset) { return u32(offset) & 0xffff; };
   uint32_t nt = u32(0x3c);
   if (u32(nt) != 0x4550) // "PE\0\0"
      return audit;
   uint32_t optional = nt + 24, sections = optional + u16(nt + 20), numSections = u16(nt + 6);
   uint32_t directories = optional + (u16(optional) == 0x20b ? 112 : 96); // PE32+ or PE32
   // The file offset of a relative virtual address, through the section table.
   auto offset = [&](uint32_t rva) {
      for (uint32_t s = 0; s < numSections; ++s) {
         uint32_t header = sections + 40 * s, address = u32(header + 12);
         if (rva >= address && rva < address + (std::max)(u32(header + 8), u32(header + 16)))
            return rva - address + u32(header + 20);
      }
      return rva;
   };
   auto import = [&](uint32_t nameRva) {
      const char *name = offset(nameRva) < file.size() ? file.c_str() + offset(nameRva) : "";
      audit.shared += LLVMIsolationIsLLVMDll(name);
      audit.imports.push_back(name);
   };
   // IMAGE_IMPORT_DESCRIPTOR (20 bytes, Name at 12) and IMAGE_DELAYLOAD_DESCRIPTOR
   // (32 bytes, DllNameRVA at 4), each array ending with a zero name.
   if (uint32_t imports = u32(directories + 8))
      for (uint32_t d = offset(imports); u32(d + 12); d += 20)
         import(u32(d + 12));
   if (uint32_t delayed = u32(directories + 8 * 13))
      for (uint32_t d = offset(delayed); u32(d + 4); d += 32)
         import(u32(d + 4));
   return audit;
}
#endif

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
   } else if (which == "exports") {
#ifdef __APPLE__
      std::string image =
         gSystem->Getenv("LLVM_ISOLATION_IMAGE") ? gSystem->Getenv("LLVM_ISOLATION_IMAGE") : "libCling.so";
      if (image != "libCling.so" && !dlopen(image.c_str(), RTLD_NOW | RTLD_LOCAL)) {
         Error("LLVMIsolation", "%s", dlerror());
         return;
      }
      const std::string leaf = image.substr(image.rfind('/') + 1);
      for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
         const std::string name = _dyld_get_image_name(i);
         if (name.size() < leaf.size() + 1 ||
             name.compare(name.size() - leaf.size() - 1, std::string::npos, "/" + leaf) != 0)
            continue;
         const auto *header = reinterpret_cast<const mach_header_64 *>(_dyld_get_image_header(i));
         const char *linkedit = nullptr;
         const char *command = reinterpret_cast<const char *>(header + 1);
         for (uint32_t c = 0; c < header->ncmds;
              ++c, command += reinterpret_cast<const load_command *>(command)->cmdsize) {
            const auto *segment = reinterpret_cast<const segment_command_64 *>(command);
            if (segment->cmd == LC_SEGMENT_64 && std::strcmp(segment->segname, SEG_LINKEDIT) == 0)
               linkedit =
                  reinterpret_cast<const char *>(_dyld_get_image_vmaddr_slide(i) + segment->vmaddr - segment->fileoff);
         }
         auto audit = LLVMIsolationAuditMachO(header, linkedit);
         printf("LLVM isolation exports of %s: llvm=%d weak=%d shared=%d twolevel=%d exports=%d\n", leaf.c_str(),
                audit.llvm, audit.weak, audit.shared, audit.twoLevel, audit.exports);
         return;
      }
      Error("LLVMIsolation", "%s is not loaded", leaf.c_str());
#endif
   } else if (which == "imports") {
#ifdef _WIN32
      TString image = gSystem->Getenv("LLVM_ISOLATION_IMAGE") ? gSystem->Getenv("LLVM_ISOLATION_IMAGE") : "libCling";
      if (!gSystem->FindDynamicLibrary(image, kTRUE)) {
         Error("LLVMIsolation", "cannot find %s", image.Data());
         return;
      }
      std::ifstream file(image.Data(), std::ios::binary);
      const std::string contents{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
      auto audit = LLVMIsolationAuditPE(contents);
      printf("LLVM isolation imports: reading %s (%zu bytes)\n", image.Data(), contents.size());
      for (const std::string &dll : audit.imports)
         printf("LLVM isolation imports:   %s%s\n", dll.c_str(),
                LLVMIsolationIsLLVMDll(dll) ? "  <- LLVM or Clang" : "");
      printf("LLVM isolation imports of %s: shared=%d dlls=%zu\n", gSystem->BaseName(image.Data()), audit.shared,
             audit.imports.size());
#endif
   }
}
