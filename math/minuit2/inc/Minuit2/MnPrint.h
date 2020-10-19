// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnPrint
#define ROOT_Minuit2_MnPrint

#include "Minuit2/MnConfig.h"

#include <sstream>
#include <utility>
#include <cassert>
#include <string>
#include <ios>

namespace ROOT {

   namespace Minuit2 {


/**
    define std::ostream operators for output
*/

class FunctionMinimum;
std::ostream& operator<<(std::ostream&, const FunctionMinimum&);

class MinimumState;
std::ostream& operator<<(std::ostream&, const MinimumState&);

class LAVector;
std::ostream& operator<<(std::ostream&, const LAVector&);

class LASymMatrix;
std::ostream& operator<<(std::ostream&, const LASymMatrix&);

class MnUserParameters;
std::ostream& operator<<(std::ostream&, const MnUserParameters&);

class MnUserCovariance;
std::ostream& operator<<(std::ostream&, const MnUserCovariance&);

class MnGlobalCorrelationCoeff;
std::ostream& operator<<(std::ostream&, const MnGlobalCorrelationCoeff&);

class MnUserParameterState;
std::ostream& operator<<(std::ostream&, const MnUserParameterState&);

class MnMachinePrecision;
std::ostream& operator<<(std::ostream&, const MnMachinePrecision&);

class MinosError;
std::ostream& operator<<(std::ostream&, const MinosError&);

class ContoursError;
std::ostream& operator<<(std::ostream&, const ContoursError&);

// std::pair<double, double> is used by MnContour
std::ostream& operator<<(std::ostream& os, const std::pair<double,double> & point);

/* Design notes: 1) We want to delay the costly conversion from object references to
   strings to a point after we have decided whether or not to
   show that string to the user at all. 2) We want to offer a customization point for
   external libraries that want to replace the MnPrint logging. The actual
   implementation is in a separate file, MnPrintImpl.cxx file that external libraries
   can replace with their own implementation.
*/

// scope guard to set common prefix for all messages in local scope
class MnPrintPrefix {
public:
  MnPrintPrefix(const char* prefix);
  ~MnPrintPrefix();
private:
  const char* fSave;
};

// scope guard to temporarily switch global verbosity level to a local one
class MnPrintLocalLevel {
public:
  MnPrintLocalLevel(int level);
  ~MnPrintLocalLevel();
private:
  int fSave;
};

// static logging class which manages global verbosity level
class MnPrint {

public:
   // want this to be an enum class for strong typing...
   enum class Verbosity {
      Error = 0,
      Warn = 1,
      Info = 2,
      Debug = 3
   };

   // ...but also want the values accessible from MnPrint scope for convenience
   static constexpr auto eError = Verbosity::Error;
   static constexpr auto eWarn = Verbosity::Warn;
   static constexpr auto eInfo = Verbosity::Info;
   static constexpr auto eDebug = Verbosity::Debug;

   // used for one-line printing of fcn minimum state
   class Oneline {
   public:
     Oneline(double fcn, double edm, int ncalls, int iter=-1);
     Oneline(const MinimumState & state, int iter=-1);
     Oneline(const FunctionMinimum& fmin, int iter=-1);

   private:
     double fFcn, fEdm;
     int fNcalls, fIter;

     friend std::ostream& operator<<(std::ostream& os, const Oneline& x);
   };

   // set print level and return the previous one
   static int SetLevel(int level);

   // return current level
   static int Level();

   template <class... Ts>
   static void Log(Verbosity level, const char* prefix, const Ts&... args) {
     if (Level() < static_cast<int>(level))
       return;
     std::ostringstream os;
     assert(prefix); // prefix must be always set
     os << prefix;
     if (sizeof...(Ts))
        os << ":";
     StreamArgs(os, args...);
     Impl(level, os.str());
   }

   template <class... Ts>
   static void Error(const Ts&... args) {
     Log(eError, Prefix(), args...);
   }

   template <class... Ts>
   static void Warn(const Ts&... args) {
     Log(eWarn, Prefix(), args...);
   }

   template <class... Ts>
   static void Info(const Ts&... args) {
     Log(eInfo, Prefix(), args...);
   }

   template <class... Ts>
   static void Debug(const Ts&... args) {
     Log(eDebug, Prefix(), args...);
   }

 private:

   static const char* Prefix();

   // see MnPrintImpl.cxx
   static void Impl(Verbosity level, const std::string& s);

   // TMP to handle lambda argument correctly, exploiting overload resolution rules
   template <class T>
   static auto HandleLambda(std::ostream& os, const T& t, int) -> decltype(t(os), void()) { t(os); }

   template <class T>
   static void HandleLambda(std::ostream& os, const T& t, float) {
     os << t;
   }

   static void StreamArgs(std::ostringstream&) {}

   // end of recursion
   template <class T>
   static void StreamArgs(std::ostringstream& os, const T& t) {
     os << " ";
     HandleLambda(os, t, 0);
   }

   template <class T, class... Ts>
   static void StreamArgs(std::ostringstream& os, const T& t, const Ts&... ts) {
     os << " " << t;
     StreamArgs(os, ts...);
   }

   friend class MnPrintPrefix;
};

} // ns Minuit2
} // ns ROOT

#endif  // ROOT_Minuit2_MnPrint
