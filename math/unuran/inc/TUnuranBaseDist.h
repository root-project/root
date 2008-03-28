// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranBaseDist


#ifndef ROOT_Math_TUnuranBaseDist
#define ROOT_Math_TUnuranBaseDist

//needed by the ClassDef
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


//______________________________________________________________________
/** 
   TUnuranBaseDist, base class for Unuran distribution classees such as 
   TUnuranContDist (for one-dimension) or TUnuranMultiContDist (multi-dimension)
*/ 
///////////////////////////////////////////////////////////////////////
class TUnuranBaseDist  {

public: 


   /** 
      Destructor (no operations)
   */ 
   virtual ~TUnuranBaseDist () {}

   /**
      Abstract clone method for a deep copy of the derived classes
    */
   virtual TUnuranBaseDist * Clone() const = 0; 


// private: 
//    /**
//       Default constructor
//     */
//    TUnuranBaseDist() {}

//    /** 
//       Copy constructor
//    */ 
//    TUnuranBaseDist(const TUnuranBaseDist & ) {} 

//    /** 
//       Assignment operator
//    */ 
//    TUnuranBaseDist & operator = (const TUnuranBaseDist & ) {
//       return *this;
//    } 


   ClassDef(TUnuranBaseDist,1)  //Base class for Unuran distribution wrappers


}; 



#endif /* ROOT_Math_TUnuranBaseDist */
