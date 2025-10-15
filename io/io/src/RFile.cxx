/// \file v7/src/RFile.cxx
/// \ingroup Base ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-03-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include "ROOT/RFile.hxx"

#include <ROOT/StringUtils.hxx>
#include <ROOT/RError.hxx>

#include <Byteswap.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>
#include <TROOT.h>

#include <algorithm>
#include <cstring>

ROOT::RLogChannel &ROOT::Experimental::Internal::RFileLog()
{
   static ROOT::RLogChannel sLog("ROOT.File");
   return sLog;
}

using ROOT::Experimental::RFile;
using ROOT::Experimental::Internal::RFileLog;

static void CheckExtension(std::string_view path)
{
   if (ROOT::EndsWith(path, ".xml")) {
      throw ROOT::RException(R__FAIL("ROOT::RFile doesn't support XML files."));
   }

   if (!ROOT::EndsWith(path, ".root")) {
      R__LOG_INFO(RFileLog()) << "ROOT::RFile only supports ROOT files. The preferred file extension is \".root\"";
   }
}

namespace {
enum class ENameCycleError {
   kNoError,
   kAnyCycle,
   kInvalidSyntax,
   kCycleTooLarge,
   kNameEmpty,
   kCOUNT
};

struct RNameCycleResult {
   std::string fName;
   std::optional<std::int16_t> fCycle;
   ENameCycleError fError;
};
} // namespace

static const char *ToString(ENameCycleError err)
{
   static const char *const kErrorStr[] = {"", "", "invalid syntax", "cycle is too large", "name is empty"};
   static_assert(std::size(kErrorStr) == static_cast<std::size_t>(ENameCycleError::kCOUNT));
   return kErrorStr[static_cast<std::size_t>(err)];
}

static ENameCycleError DecodeNumericCycle(const char *str, std::optional<std::int16_t> &out)
{
   uint32_t res = 0;
   do {
      if (!isdigit(*str))
         return ENameCycleError::kInvalidSyntax;
      if (res * 10 > std::numeric_limits<std::int16_t>::max())
         return ENameCycleError::kCycleTooLarge;
      res *= 10;
      res += *str - '0';
   } while (*++str);

   assert(res < std::numeric_limits<std::int16_t>::max());
   out = static_cast<std::int16_t>(res);

   return ENameCycleError::kNoError;
}

static RNameCycleResult DecodeNameCycle(std::string_view nameCycleRaw)
{
   RNameCycleResult result{};

   if (nameCycleRaw.empty())
      return result;

   // Scan the string to find the name length and the semicolon
   std::size_t semicolonIdx = nameCycleRaw.find_first_of(';');

   if (semicolonIdx == 0) {
      result.fError = ENameCycleError::kNameEmpty;
      return result;
   }

   // Verify that we have at most one ';'
   if (nameCycleRaw.substr(semicolonIdx + 1).find_first_of(';') != std::string_view::npos) {
      result.fError = ENameCycleError::kInvalidSyntax;
      return result;
   }

   result.fName = nameCycleRaw.substr(0, semicolonIdx);
   if (semicolonIdx < std::string_view::npos) {
      if (semicolonIdx == nameCycleRaw.length() - 1 && nameCycleRaw[semicolonIdx] == '*')
         result.fError = ENameCycleError::kAnyCycle;
      else
         result.fError = DecodeNumericCycle(nameCycleRaw.substr(semicolonIdx + 1).data(), result.fCycle);
   }

   return result;
}

/// This function first validates, then normalizes the given path in place.
///
/// Returns an empty string if `path` is a suitable path to store an object into a RFile,
/// otherwise returns a description of why that is not the case.
///
/// A valid object path must:
///   - not be empty
///   - not contain the character '.'
///   - not contain ASCII control characters or whitespace characters (including tab or newline).
///   - not contain more than RFile::kMaxPathNesting path fragments (i.e. more than RFile::kMaxPathNesting - 1 '/')
///   - not end with a '/'
///
/// In addition, when *writing* an object to RFile, the character ';' is also banned.
///
/// Passing an invalid path to Put will cause it to throw an exception, and
/// passing an invalid path to Get will always return nullptr.
///
/// If required, `path` is modified to make its hierarchy-related meaning consistent. This entails:
/// - combining any consecutive '/' into a single one;
/// - stripping any leading '/'.
///
static std::string ValidateAndNormalizePath(std::string &path)
{
   ////// First, validate path.

   if (path.empty())
      return "path cannot be empty";

   if (path.back() == '/')
      return "path cannot end with a '/'";

   bool valid = true;
   for (char ch : path) {
      // Disallow control characters, tabs, newlines, whitespace and dot.
      // NOTE: not short-circuiting or early returning to enable loop vectorization.
      valid &= !(ch < 33 || ch == '.');
   }
   if (!valid)
      return "path cannot contain control characters, whitespaces or dots";

   //// Path is valid so far, normalize it.

   // Strip all leading '/'
   {
      auto nToStrip = 0u;
      const auto len = path.length();
      while (nToStrip < len && path[nToStrip] == '/')
         ++nToStrip;

      if (nToStrip > 0)
         path.erase(0, nToStrip);
   }

   // Remove duplicate consecutive '/'
   const auto it = std::unique(path.begin(), path.end(), [](char a, char b) { return (a == '/' && b == '/'); });
   path.erase(it, path.end());

   //// After the path has been normalized, check the nesting level by counting how many slashes it contains.
   const auto nesting = std::count(path.begin(), path.end(), '/');
   if (nesting > RFile::kMaxPathNesting)
      return "pathView contains too many levels of nesting";

   return "";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

RFile::RFile(std::unique_ptr<TFile> file) : fFile(std::move(file)) {}

RFile::~RFile() = default;

std::unique_ptr<RFile> RFile::Open(std::string_view path)
{
   CheckExtension(path);

   TDirectory::TContext ctx(nullptr); // XXX: probably not thread safe?
   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   if (!tfile || tfile->IsZombie())
      throw ROOT::RException(R__FAIL("failed to open file " + std::string(path) + " for reading"));

   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

std::unique_ptr<RFile> RFile::Update(std::string_view path)
{
   CheckExtension(path);

   TDirectory::TContext ctx(nullptr); // XXX: probably not thread safe?
   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "UPDATE_WITHOUT_GLOBALREGISTRATION"));
   if (!tfile || tfile->IsZombie())
      throw ROOT::RException(R__FAIL("failed to open file " + std::string(path) + " for updating"));

   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

std::unique_ptr<RFile> RFile::Recreate(std::string_view path)
{
   CheckExtension(path);

   TDirectory::TContext ctx(nullptr); // XXX: probably not thread safe?
   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "RECREATE_WITHOUT_GLOBALREGISTRATION"));
   if (!tfile || tfile->IsZombie())
      throw ROOT::RException(R__FAIL("failed to open file " + std::string(path) + " for writing"));

   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

TKey *RFile::GetTKey(std::string_view path) const
{
   // In RFile, differently from TFile, when dealing with a path like "a/b/c", we always consider it to mean
   // "object 'c' in subdirectory 'b' of directory 'a'". We don't try to get any other of the possible combinations,
   // including the object called "a/b/c".
   std::string fullPath = std::string(path);
   char *dirName = fullPath.data();
   char *restOfPath = strchr(dirName, '/');
   TDirectory *dir = fFile.get();
   while (restOfPath) {
      // Truncate `dirName` to the position of this '/'.
      *restOfPath = 0;
      ++restOfPath;
      // `restOfPath` should always be a non-empty string unless `path` ends with '/' (which it shouldn't, as we are
      // supposed to have normalized it before calling this function).
      assert(*restOfPath);

      dir = dir->GetDirectory(dirName);
      if (!dir)
         return nullptr;

      dirName = restOfPath;
      restOfPath = strchr(restOfPath, '/');
   }
   // NOTE: after this loop `dirName` contains the base name of the object.

   // Get the leaf object from the innermost directory.
   TKey *key = dir->FindKey(dirName);
   if (key) {
      // For some reason, FindKey will not return nullptr if we asked for a specific cycle and that cycle
      // doesn't exist. It will instead return any key whose cycle is *at most* the requested one.
      // This is very confusing, so in RFile we actually return null if the requested cycle is not there.
      RNameCycleResult res = DecodeNameCycle(dirName);
      if (res.fError != ENameCycleError::kAnyCycle) {
         if (res.fError != ENameCycleError::kNoError) {
            R__LOG_ERROR(RFileLog()) << "error decoding namecycle '" << dirName << "': " << ToString(res.fError);
            key = nullptr;
         } else if (res.fCycle && *res.fCycle != key->GetCycle()) {
            key = nullptr;
         }
      }
   }
   return key;
}

void *RFile::GetUntyped(std::string_view pathSV, const std::type_info &type) const
{
   if (!fFile)
      throw ROOT::RException(R__FAIL("File has been closed"));

   std::string path{pathSV};

   const TClass *cls = TClass::GetClass(type);
   if (!cls)
      throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + path));

   if (auto err = ValidateAndNormalizePath(path); !err.empty())
      throw RException(R__FAIL("Invalid object path '" + path + "': " + err));

   TKey *key = GetTKey(path);
   void *obj = key ? key->ReadObjectAny(cls) : nullptr;

   if (obj) {
      // Disavow any ownership on `obj`
      if (auto autoAddFunc = cls->GetDirectoryAutoAdd(); autoAddFunc) {
         autoAddFunc(obj, nullptr);
      }
   } else if (key && !GetROOT()->IsBatch()) {
      R__LOG_WARNING(RFileLog()) << "Tried to get object '" << path << "' of type " << cls->GetName()
                                 << " but that path contains an object of type " << key->GetClassName();
   }

   return obj;
}

void RFile::PutUntyped(std::string_view pathSV, const std::type_info &type, const void *obj, std::uint32_t flags)
{
   const TClass *cls = TClass::GetClass(type);
   if (!cls)
      throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + std::string(pathSV)));

   std::string path{pathSV};
   if (auto err = ValidateAndNormalizePath(path); !err.empty())
      throw RException(R__FAIL("Invalid object path '" + path + "': " + err));

   if (path.find_first_of(';') != std::string_view::npos) {
      throw RException(
         R__FAIL("Invalid object path '" + path +
                 "': character ';' is used to specify an object cycle, which only makes sense when reading."));
   }

   if (!fFile)
      throw ROOT::RException(R__FAIL("File has been closed"));

   if (!fFile->IsWritable())
      throw ROOT::RException(R__FAIL("File is not writable"));

   // If `path` refers to a subdirectory, make sure we always write in an actual TDirectory,
   // otherwise we may have a mix of top-level objects called "a/b/c" and actual directory
   // structures.
   // Sadly, TFile does nothing to prevent this and will happily write "a/b" even if there
   // is already a directory "a" containing an object "b". We don't want that ambiguity here, so we take extra steps
   // to ensure it doesn't happen.
   const auto tokens = ROOT::Split(path, "/");
   const auto FullPathUntil = [&tokens](auto idx) {
      return ROOT::Join("/", std::span<const std::string>{tokens.data(), idx + 1});
   };
   TDirectory *dir = fFile.get();
   for (auto tokIdx = 0u; tokIdx < tokens.size() - 1; ++tokIdx) {
      // Alas, not only does mkdir not fail if the file already contains an object "a/b" and you try
      // to create dir "a", but even when it does fail it doesn't tell you why.
      // We obviously don't want to allow the coexistence of regular object named "a/b" and the directory
      // named "a", so we manually check if each level of nesting doesn't exist already as a non-directory.
      const TKey *existing = dir->GetKey(tokens[tokIdx].c_str());
      if (existing && strcmp(existing->GetClassName(), "TDirectory") != 0 &&
          strcmp(existing->GetClassName(), "TDirectoryFile") != 0) {
         throw ROOT::RException(R__FAIL("error adding object '" + path + "': failed to create directory '" +
                                        FullPathUntil(tokIdx) + "': name already taken by an object of type '" +
                                        existing->GetClassName() + "'"));
      }
      dir = dir->mkdir(tokens[tokIdx].c_str(), "", true);
      if (!dir) {
         throw ROOT::RException(R__FAIL(std::string("failed to create directory ") + FullPathUntil(tokIdx)));
      }
   }

   const bool allowOverwrite = (flags & kPutAllowOverwrite) != 0;
   const bool backupCycle = (flags & kPutOverwriteKeepCycle) != 0;
   const Option_t *writeOpts = "";
   if (!allowOverwrite) {
      const TKey *existing = dir->GetKey(tokens[tokens.size() - 1].c_str());
      if (existing) {
         throw ROOT::RException(R__FAIL(std::string("trying to overwrite object ") + path + " of type " +
                                        existing->GetClassName() + " with another object of type " + cls->GetName()));
      }
   } else if (!backupCycle) {
      writeOpts = "WriteDelete";
   }

   int success = dir->WriteObjectAny(obj, cls, tokens[tokens.size() - 1].c_str(), writeOpts);

   if (!success) {
      throw ROOT::RException(R__FAIL(std::string("Failed to write ") + path + " to file"));
   }
}

size_t RFile::Flush()
{
   return fFile->Write();
}

void RFile::Close()
{
   // NOTE: this also flushes the file internally
   fFile.reset();
}
