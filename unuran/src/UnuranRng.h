// @(#)root/unuran:$Name:  $:$Id: inc/Math/UnuranRng.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Wed Sep 27 11:22:34 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class UnuranRng

#ifndef ROOT_UnuranRng
#define ROOT_UnuranRng


/** 
   UnuranRng class for interface ROOT random generators to Unuran 
*/ 

template<class Random> 
struct UnuranRng { 

public:   

   /// funciton to delete object (not needed)
   static void Delete(void * /* p */ ) { 
//       Random * r = reinterpret_cast<Random *> (p); 
//       delete r; 
   } 

   /// function to set the seed in the random
   static void Seed(void * p, unsigned long seed) { 
      Random * r = reinterpret_cast<Random *> (p); 
      r->SetSeed(seed); 
   }

   /// function to sample random generator
   static double Rndm(void * p) { 
      Random * r = reinterpret_cast<Random *> (p); 
      return r->Rndm();  
   }

 
}; 



#endif /* ROOT_UnuranRng */
