#include "ROOT/InternalIOUtils.hxx"

#include "ROOT/RConfig.hxx" // R__UNIX

#ifdef R__UNIX

#include <ROOT/RLogger.hxx>

ROOT::RLogChannel &InternalLogChannel()
{
   static ROOT::RLogChannel sLog("ROOT.Internal");
   return sLog;
}

// getxattr
#ifdef R__FBSD
#include <sys/extattr.h>
#else
#include <sys/xattr.h>
#endif

#ifdef R__MACOSX
/* On macOS getxattr takes two extra arguments that should be set to 0 */
#define getxattr(path, name, value, size) getxattr(path, name, value, size, 0u, 0)
#endif

#ifdef R__FBSD
#define getxattr(path, name, value, size) extattr_get_file(path, EXTATTR_NAMESPACE_USER, name, value, size)
#endif

#include "ROOT/StringUtils.hxx" // ROOT::StartsWith
#include "TEnv.h"               // TEnv::GetValue
#endif

std::optional<std::string>
ROOT::Internal::GetXAttrVal([[maybe_unused]] std::string_view path, [[maybe_unused]] std::string_view xattr)
{
#ifdef R__UNIX
   // First call to getxattr evaluates the length of the extended attribute value
   if (auto len = getxattr(path.data(), xattr.data(), nullptr, 0); len >= 0) {
      std::string xval(len, 0);
      // Second call extracts the extended attribute value, checking it's of the correct length
      if (getxattr(path.data(), xattr.data(), xval.data(), len) == len)
         return xval;
   }
#endif
   return std::nullopt;
}

std::optional<std::string> ROOT::Internal::GetEOSRedirectedXRootURL([[maybe_unused]] std::string_view inputPath)
{
#ifdef R__UNIX
   if (inputPath.empty() || inputPath.back() == '/')
      return std::nullopt;

   if (gEnv->GetValue("TFile.CrossProtocolRedirects", 1) != 1)
      return std::nullopt;

   auto xurl = ROOT::Internal::GetXAttrVal(inputPath, "eos.url.xroot");
   if (!xurl)
      return std::nullopt;

   auto baseName = inputPath.substr(inputPath.find_last_of("/") + 1);
   // Sometimes the `getxattr` call may return an invalid URL due
   // to the POSIX attribute not being yet completely filled by EOS.
   if (!std::equal(baseName.crbegin(), baseName.crend(), xurl->crbegin())) {
      R__LOG_WARNING(InternalLogChannel())
         << "Could not find path base name '" << baseName << "' in redirected URL '" << *xurl << "'.";
      return std::nullopt;
   }

   // Ensure the redirected URL actually starts with the XRootD protocol string
   if (ROOT::StartsWith(*xurl, "root://") || ROOT::StartsWith(*xurl, "xroot://") ||
       ROOT::StartsWith(*xurl, "roots://") || ROOT::StartsWith(*xurl, "xroots://"))
      return xurl;
   else
      R__LOG_WARNING(InternalLogChannel())
         << "Redirected URL '" << *xurl << "' does not begin with any valid XRootD protocol string.";

#endif
   return std::nullopt;
}
