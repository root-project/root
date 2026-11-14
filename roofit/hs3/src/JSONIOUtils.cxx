#include "JSONIOUtils.h"

#include <RooAbsBinning.h>
#include <RooConstVar.h>

#include <cctype>
#include <sstream>
#include <string>
#include <utility>

using RooFit::Detail::JSONNode;

// Parse the entire string as a number (integer or floating-point). The output is only valid if that succeeds.
ToNumberOutput toNumber(const std::string &str)
{
   // std::from_chars for floating-point types is not portably available on all
   // platforms ROOT supports, so we rely on the stream extraction.
   std::istringstream stream(str);
   ToNumberOutput output;
   output.valid = (stream >> output.value) && stream.eof();
   return output;
}

// True if `arg` is a RooConstVar whose name is a plain number (i.e. a literal constant).
bool isLiteralConstVar(RooAbsArg const &arg)
{
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(&arg);
   return isRooConstVar && toNumber(arg.GetName()).valid;
}

// Write a reference to `arg`: the value if it is a literal constant, and the name otherwise.
void writeReference(JSONNode &node, RooAbsArg const &arg)
{
   if (isLiteralConstVar(arg)) {
      node << static_cast<RooConstVar const &>(arg).getVal();
   } else {
      node << arg.GetName();
   }
}

bool startsWith(std::string_view str, std::string_view prefix)
{
   return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

bool endsWith(std::string_view str, std::string_view suffix)
{
   return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

std::string removeSuffix(std::string_view str, std::string_view suffix)
{
   std::string out;
   out += str;
   out = out.substr(0, out.length() - suffix.length());
   return out;
}

void writeAxisBinning(JSONNode &node, const RooAbsBinning &binning)
{
   node["min"] << binning.lowBound();
   node["max"] << binning.highBound();
   if (binning.isUniform()) {
      node["nbins"] << binning.numBins();
      return;
   }

   auto &edges = node["edges"].set_seq();
   edges.append_child() << binning.binLow(0);
   for (int i = 0; i < binning.numBins(); ++i) {
      edges.append_child() << binning.binHigh(i);
   }
}

// Parse the leading "MAJOR.MINOR" of an HS3 version string like "0.2", "0.2.9" or "0.3".
bool parseHS3Version(std::string const &version, int &major, int &minor)
{
   std::size_t pos = 0;
   auto readNumber = [&](int &out) {
      const std::size_t begin = pos;
      while (pos < version.size() && std::isdigit(static_cast<unsigned char>(version[pos]))) {
         ++pos;
      }
      // Reject empty and absurdly long numbers, which std::stoi could not represent.
      if (pos == begin || pos - begin > 9) {
         return false;
      }
      out = std::stoi(version.substr(begin, pos - begin));
      return true;
   };
   if (!readNumber(major) || pos >= version.size() || version[pos] != '.') {
      return false;
   }
   ++pos;
   return readNumber(minor);
}

// Whether an HS3 version predates `major.minor`. Versions that can't be parsed count as old. That includes the empty
// version of objects that are imported without a document.
bool isHS3VersionBefore(std::string const &version, int major, int minor)
{
   int versionMajor = 0;
   int versionMinor = 0;
   if (!parseHS3Version(version, versionMajor, versionMinor)) {
      return true;
   }
   return std::make_pair(versionMajor, versionMinor) < std::make_pair(major, minor);
}
