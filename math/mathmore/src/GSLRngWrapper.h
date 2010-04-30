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
      fOwn(0),
      fRng(0),
      fRngType(0) 
    {
    }

   /** 
      Constructor with type 
   */ 
   GSLRngWrapper(const gsl_rng_type * type) : 
      fOwn(1),
      fRng(0),
      fRngType(type) 
    {
    }

   /** 
       construct from an existing gsl_rng
       it is managed externally - so will not be deleted at the end
   */
   GSLRngWrapper(const gsl_rng * r ) : 
      fOwn(0),
      fRngType(0) 
    {
       fRng = const_cast<gsl_rng *>(r); 
    }

   /** 
      Copy constructor - pass ownership (need not to be const)
      Just copy the pointer and do not manage it 
   */ 
   GSLRngWrapper(GSLRngWrapper & r) :
      fOwn(r.fOwn),
      fRng(r.fRng),
      fRngType(r.fRngType)
   { 
      // in case an rng exists must release it
      if (fRng && fOwn) r.fOwn = false;  
   } 


   /**
      Destructor  (free the rng if not done before)
    */
    ~GSLRngWrapper() { 
       if (fOwn) Free();
    } 

    void Allocate() { 
      if (fRngType == 0) SetDefaultType();
      if (fRng != 0 && fOwn) Free(); 
      fRng = gsl_rng_alloc( fRngType );
      //std::cout << " allocate   " << fRng <<  std::endl;
    }

    void Free() { 
       if (!fOwn) return; // no operation if pointer is not own 
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



    inline gsl_rng * Rng()  { return fRng; } 

    inline const gsl_rng * Rng() const { return fRng; } 

private:
   // usually copying is non trivial, so we make this unaccessible


   /** 
      Assignment operator
      Disable since if don't want to change an already created wrapper
   */ 
   GSLRngWrapper & operator = (const GSLRngWrapper & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }


private: 

   bool fOwn; // ownership of contained pointer
   gsl_rng * fRng; 
   const gsl_rng_type * fRngType; 
};
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLRngWrapper */
