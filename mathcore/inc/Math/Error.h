// @(#)root/mathcore:$Id$
// Authors: L. Moneta

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Math_Error
#define ROOT_Math_Error




#ifdef DEBUG
#ifndef WARNINGMSG
#define WARNINGMSG
#endif
#endif



/** 
   Pre-processor macro to report messages 
   which can be configured to use ROOT error or 
   simply an iostream in case of stan-alone builds
*/

#ifndef USE_ROOT_ERROR

// use iostream instead of ROOT 

#include <iostream>

#ifndef ROOT_MATH_LOG
#define ROOT_MATH_OS std::cerr
#else 
#define ROOT_MATH_LOG
#endif

#define MATH_INFO_MSG(str) \
   ROOT_MATH_OS << "Info in ROOT::Math: " << str \
       << std::endl;
#define MATH_ERROR_MSG(str) \
   ROOT_MATH_OS << "Error in ROOT::Math: " << str \
       << std::endl;
# define MATH_INFO_VAL(x) \
   ROOT_MATH_OS << "Info in ROOT::Math: " << #x << " = " << (x) << std::endl; 
# define MATH_ERROR_VAL(x) \
   ROOT_MATH_OS << "Info in ROOT::Math: " << #x << " = " << (x) << std::endl; 


// same giving a location

#define MATH_INFO_MSG2(loc,str) \
  ROOT_MATH_OS << "Info in ROOT::Math " << loc << " : " << str \
       << std::endl;
#define MATH_ERROR_MSG2(loc,str) \
   ROOT_MATH_OS << "Error in ROOT::Math " << loc << " : " << str \
       << std::endl;
# define MATH_INFO_VAL2(loc,x) \
   ROOT_MATH_OS << "Info in ROOT::Math " << loc  << " : " << #x << " = " << (x) << std::endl;
# define MATH_ERROR_VAL2(loc,x) \
   ROOT_MATH_OS << "Info in ROOT::Math " << loc << " : " << #x << " = " << (x) << std::endl; 



#else
// use ROOT error reporting system 

#include "TError.h"
#include "Math/Util.h"

#define  MATH_INFO_MSG(str) \
   ::Info("ROOT::Math",str);
#define  MATH_ERROR_MSG(str) \
   ::Error("ROOT::Math",str);
# define MATH_INFO_VAL(x) \
   {std::string str = std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Info("ROOT::Math",str.c_str() );} 
# define MATH_ERROR_VAL(x) \
   {std::string str = std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Error("ROOT::Math",str.c_str() );} 

# define MATH_INFO_VAL2(loc,x) \
   {std::string str = std::string(loc) + std::string(" : ") + std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Info("ROOT::Math",str.c_str() );} 
# define MATH_ERROR_VAL2(loc,x) \
   {std::string str = std::string(loc) + std::string(" : ") + std::string(#x) + std::string(" = ") + ROOT::Math::Util::ToString(x); \
   ::Error("ROOT::Math",str.c_str() );} 


#endif


#endif  // ROOT_MATH_Error
