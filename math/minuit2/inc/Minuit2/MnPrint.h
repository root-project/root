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

//#define DEBUG
//#define WARNINGMSG

#include <iostream>
#include <string>

#ifdef DEBUG
#ifndef WARNINGMSG
#define WARNINGMSG
#endif
#endif




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


// class define static print level values

class MnPrint {

public:
   // set print level and return the previous one
   static int SetLevel(int level);

   // return current level
   static int Level();

   // print current minimization state
   static void PrintState(std::ostream & os, const MinimumState & state, const char * msg, int iter = -1);

   // print current minimization state
   static void PrintState(std::ostream & os, double fcn, double edm, int ncalls, const char * msg, int iter = -1);

   // print FCN value with right precision adding optionally end line
   static void PrintFcn(std::ostream & os, double value, bool endline = true);
};



  }  // namespace Minuit2

}  // namespace ROOT


// macro to report messages

#ifndef USE_ROOT_ERROR

#ifndef MNLOG
#define MN_OS std::cerr
#else
#define MN_OS MNLOG
#endif

#define MN_INFO_MSG(str) \
   if (MnPrint::Level() > 0) MN_OS << "Info: " << str    \
       << std::endl;
#define MN_ERROR_MSG(str) \
   if (MnPrint::Level() >= 0) MN_OS << "Error: " << str \
       << std::endl;
# define MN_INFO_VAL(x) \
   if (MnPrint::Level() > 0) MN_OS << "Info: " << #x << " = " << (x) << std::endl;
# define MN_ERROR_VAL(x) \
   if (MnPrint::Level() >= 0) MN_OS << "Error: " << #x << " = " << (x) << std::endl;


// same giving a location

#define MN_INFO_MSG2(loc,str) \
  if (MnPrint::Level() > 0) MN_OS << "Info in " << loc << " : " << str \
       << std::endl;
#define MN_ERROR_MSG2(loc,str) \
   if (MnPrint::Level() >= 0) MN_OS << "Error in " << loc << " : " << str \
       << std::endl;
# define MN_INFO_VAL2(loc,x) \
   if (MnPrint::Level() > 0) MN_OS << "Info in " << loc << " : " << #x << " = " << (x) << std::endl;
# define MN_ERROR_VAL2(loc,x) \
   if (MnPrint::Level() >= 0) MN_OS << "Error in " << loc << " : " << #x << " = " << (x) << std::endl;



#else
// use ROOT error reporting system
#include "TError.h"
#include "Math/Util.h"

// this first two should be used only with string literals to
// avoid warning produced by the format in TError
#define  MN_INFO_MSG(str) \
   ::Info("Minuit2",str);
#define  MN_ERROR_MSG(str) \
   ::Error("Minuit2",str);
# define MN_INFO_VAL(x) \
   {std::string str = std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
      ::Info("Minuit2","%s",str.c_str() );}
# define MN_ERROR_VAL(x) \
   {std::string str = std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Error("Minuit2","%s",str.c_str() );}

# define MN_INFO_MSG2(loc,txt) \
   {std::string str = std::string(loc) + std::string(" : ") + std::string(txt); \
   ::Info("Minuit2","%s",str.c_str() );}
# define MN_ERROR_MSG2(loc,txt) \
   {std::string str = std::string(loc) + std::string(" : ") + std::string(txt); \
   ::Error("Minuit2","%s",str.c_str() );}

# define MN_INFO_VAL2(loc,x) \
   {std::string str = std::string(loc) + std::string(" : ") + std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Info("Minuit2","%s",str.c_str() );}
# define MN_ERROR_VAL2(loc,x) \
   {std::string str = std::string(loc) + std::string(" : ") + std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Error("Minuit2","%s",str.c_str() );}



#endif


#endif  // ROOT_Minuit2_MnPrint
