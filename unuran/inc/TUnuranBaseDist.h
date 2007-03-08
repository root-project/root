// @(#)root/unuran:$Name:  $:$Id: TUnuranBaseDist.h,v 1.3 2007/02/05 10:24:44 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranBaseDist

//////////////////////////////////////////////////////////////////////
// 
//   TUnuranBaseDistr class 
//   wrapper class for one dimensional continous distribution
// 
///////////////////////////////////////////////////////////////////////

#ifndef ROOT_Math_TUnuranBaseDist
#define ROOT_Math_TUnuranBaseDist


//////////////////////////////////////////////////////////////////////
/** 
   TUnuranBaseDistr class 
   base class for UNuran distribution wrappers
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


//   ClassDef(TUnuranBaseDist,1)  //Base class for Unuran distribution wrappers


}; 



#endif /* ROOT_Math_TUnuranBaseDist */
