#include "Math/Util.h"

ROOT::Math::Util::TimingScope::TimingScope(std::function<void(std::string const &)> printer, std::string const &message)
   : fBegin{std::chrono::steady_clock::now()}, fPrinter{printer}, fMessage{message}
{
}

namespace {

template <class T>
std::string printTime(T duration)
{
   std::stringstream ss;
   // Here, nanoseconds are represented as "long int", so the maximum value
   // corresponds to about 300 years. Nobody will wait that long for a
   // computation to complete, so for timing measurements it's fine to cast to
   // nanoseconds to keep things simple.
   double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
   bool forceSeconds = false;
   // The largest unit that we consider for pretty-printing is days.
   if (ns >= 24 * 60 * 60e9) {
      auto days = std::floor(ns / (24 * 60 * 60e9));
      ss << days << " d ";
      ns -= days * 24 * 60 * 60e9; // subtract the days to print only the rest
      // to avoid printouts like "1 d 4 h 23 min 324 ns", we force to print
      // seconds if we print either days, hours, or minutes
      forceSeconds = true;
   }
   if (ns >= 60 * 60e9) {
      auto hours = std::floor(ns / (60 * 60e9));
      ss << hours << " h ";
      ns -= hours * 60 * 60e9;
      forceSeconds = true;
   }
   if (ns >= 60e9) {
      auto minutes = std::floor(ns / 60e9);
      ss << minutes << " min ";
      ns -= minutes * 60e9;
      forceSeconds = true;
   }
   if (ns >= 1e9 || forceSeconds) {
      ss << (1e-9 * ns) << " s";
   } else if (ns >= 1e6) {
      ss << (1e-6 * ns) << " ms";
   } else if (ns >= 1e3) {
      ss << (1e-3 * ns) << " μs";
   } else {
      ss << ns << " ns";
   }
   return ss.str();
}

} // namespace

ROOT::Math::Util::TimingScope::~TimingScope()
{
   using std::chrono::steady_clock;
   steady_clock::time_point end = steady_clock::now();
   std::stringstream ss;
   ss << fMessage << " " << printTime(end - fBegin);
   fPrinter(ss.str());
}
