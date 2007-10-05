// @(#)root/mathmore:$Id$
// Author: L. Moneta Fri Aug 24 17:20:45 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class GSLRngWrapper

#ifndef ROOT_Math_GSLRngWrapper
#define ROOT_Math_GSLRngWrapper


namespace ROOT { 

   namespace Math { 


/** 
   GSLRngWrapper class to wrap gsl_rng structure 
*/ 
class GSLRngWrapper {

public: 


   /** 
      Default constructor 
   */ 
   GSLRngWrapper () : 
      fRng(0),
      fRngType(0) 
    {
    }

   /** 
      Constructor with type 
   */ 
   GSLRngWrapper(const gsl_rng_type * type) : 
      fRng(0),
      fRngType(type) 
    {
    }

   /**
      Destructor  (free the rng if not done before)
    */
    ~GSLRngWrapper() { 
       Free();
    } 

    void Allocate() { 
      if (fRngType == 0) SetDefaultType();
      if (fRng != 0) Free(); 
      fRng = gsl_rng_alloc( fRngType );
      //std::cout << " allocate   " << fRng <<  std::endl;
    }

    void Free() { 
      //std::cout << "free gslrng " << fRngType <<  "  " << fRng <<  std::endl;
      if (fRng != 0) gsl_rng_free(fRng);       
      fRng = 0; 
    }


    void SetType(const gsl_rng_type * type) { 
      fRngType = type; 
    }

    void SetDefaultType() { 
      // construct default engine
      gsl_rng_env_setup(); 
      fRngType =  gsl_rng_default; 
    }



    inline gsl_rng * Rng() const { return fRng; } 

private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   GSLRngWrapper(const GSLRngWrapper &) {} 

   /** 
      Assignment operator
   */ 
   GSLRngWrapper & operator = (const GSLRngWrapper & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }


private: 

   gsl_rng * fRng; 
   const gsl_rng_type * fRngType; 
};
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLRngWrapper */
