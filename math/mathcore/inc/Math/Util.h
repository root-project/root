// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 15:44:38 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Utility functions for all ROOT Math

#ifndef ROOT_Math_Util
#define ROOT_Math_Util

#include <string> 
#include <sstream> 


namespace ROOT { 

   namespace Math { 



/** 
   namespace defining Utility functions needed by mathcore 
*/ 
namespace Util { 

/**
   Utility function for conversion to strings
*/
template<class T>
std::string ToString(const T& val)
{
   std::ostringstream buf;
   buf << val;
   
   std::string ret = buf.str();
   return ret;
}

}  // end namespace Util


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Util */
