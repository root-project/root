///
/// Basic logging utilities meant to be used by binaries.
/// Use instead of RLogger for user-facing messages.
///
#ifndef ROOT_Main_Logging
#define ROOT_Main_Logging

#include <iostream>

namespace Detail {

class NullBuf : public std::streambuf {
public:
   int overflow(int c) final { return c; }
};

// A stream that discards all input.
class NullStream : public std::ostream {
   NullBuf fBuf;

public:
   NullStream() : std::ostream(&fBuf) {}
};

inline const char *gLogName = "";
/// Log verbosity works this way:
/// If it's <= 0, all warnings are suppressed.
/// Additionally, when Info(lv) is called, it is only displayed if lv <= LogVerbosity.
inline int gLogVerbosity = 1;

inline NullStream &GetNullStream()
{
   static NullStream nullStream;
   return nullStream;
}

} // namespace Detail

inline void InitLog(const char *name, int defaultVerbosity = 1)
{
   Detail::gLogName = name;
   Detail::gLogVerbosity = defaultVerbosity;
}

inline void SetLogVerbosity(int verbosity)
{
   Detail::gLogVerbosity = verbosity;
}

inline int GetLogVerbosity()
{
   return Detail::gLogVerbosity;
}

inline std::ostream &Err()
{
   std::cerr << "Error in <" << Detail::gLogName << ">: ";
   return std::cerr;
}

inline std::ostream &Warn()
{
   std::ostream &s = Detail::gLogVerbosity < 1 ? Detail::GetNullStream() : std::cerr;
   s << "Warning in <" << Detail::gLogName << ">: ";
   return s;
}

inline std::ostream &Info(int minLevel)
{
   std::ostream &s = Detail::gLogVerbosity < minLevel ? Detail::GetNullStream() : std::cerr;
   s << "Info in <" << Detail::gLogName << ">: ";
   return s;
}

#endif
