#include "ROOT/InternalIOUtils.hxx"

#include "ROOT/RConfig.hxx" // R__UNIX

#ifdef R__UNIX
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

std::optional<std::string> ROOT::Internal::GetXAttrVal(const char *path, const char *xattr)
{
#ifdef R__UNIX
   // First call to getxattr evaluates the length of the extended attribute value
   if (auto len = getxattr(path, xattr, nullptr, 0); len > 0) {
      std::string xval(len, 0);
      // Second call extracts the extended attribute value, checking it's of the correct length
      if (getxattr(path, xattr, &xval[0], len) == len)
         return xval;
   }
#else
   (void)path;
   (void)xattr;
#endif
   return std::nullopt;
}

std::optional<std::string> ROOT::Internal::GetEOSRedirectedXRootURL(const char *inputURL)
{
#ifdef R__UNIX
   std::string_view inputSV{inputURL};
   if (inputSV.empty() || inputSV.back() == '/')
      return std::nullopt;

   if (gEnv->GetValue("TFile.CrossProtocolRedirects", 1) != 1)
      return std::nullopt;

   auto xurl = ROOT::Internal::GetXAttrVal(inputURL, "eos.url.xroot");
   if (!xurl)
      return std::nullopt;

   auto baseName = inputSV.substr(inputSV.find_last_of("/") + 1);
   // Sometimes the `getxattr` call may return an invalid URL due
   // to the POSIX attribute not being yet completely filled by EOS.
   if (!std::equal(baseName.crbegin(), baseName.crend(), xurl->crbegin()))
      return std::nullopt;

   // Ensure the redirected URL actually starts with the XRootD protocol string
   if (ROOT::StartsWith(*xurl, "root://") || ROOT::StartsWith(*xurl, "xroot://"))
      return xurl;
#else
   (void)inputURL;
#endif
   return std::nullopt;
}
