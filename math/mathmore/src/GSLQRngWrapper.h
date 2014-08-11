// @(#)root/mathmore:$Id$
// Author: L. Moneta Fri Aug 24 17:20:45 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class GSLQRngWrapper

#ifndef ROOT_Math_GSLQRngWrapper
#define ROOT_Math_GSLQRngWrapper

#include "gsl/gsl_qrng.h"

namespace ROOT {

   namespace Math {


/**
   GSLQRngWrapper class to wrap gsl_qrng structure
*/
class GSLQRngWrapper {

public:


   /**
      Default constructor
   */
   GSLQRngWrapper () :
      fOwn(0),
      fRng(0),
      fRngType(0)
    {
    }

   /**
      Constructor with type
   */
   GSLQRngWrapper(const gsl_qrng_type * type) :
      fOwn(1),
      fRng(0),
      fRngType(type)
    {
    }

   /**
       construct from an existing gsl_qrng
       it is managed externally - so will not be deleted at the end
   */
   GSLQRngWrapper(const gsl_qrng * r ) :
      fOwn(0),
      fRngType(0)
    {
       fRng = const_cast<gsl_qrng *>(r);
    }

   /**
      Copy constructor - clone the GSL object and manage it
   */
   GSLQRngWrapper(GSLQRngWrapper & r) :
      fOwn(1),
      fRngType(r.fRngType)
   {
      fRng = gsl_qrng_clone(r.fRng);
   }

   /**
      Assignment operator
   */
   GSLQRngWrapper & operator = (const GSLQRngWrapper & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      fRngType = rhs.fRngType;
      int iret = 0;
      if (fRngType == rhs.fRngType) {
         iret = gsl_qrng_memcpy(fRng, rhs.fRng);
         if (!iret) return *this;
      }
      // otherwise create a new copy
      if (fOwn) Free();
      fRng = gsl_qrng_clone(rhs.fRng);
      fOwn = true;
      return *this;
   }

   /**
      Destructor  (free the rng if not done before)
    */
    ~GSLQRngWrapper() {
       if (fOwn) Free();
    }

    void Allocate(unsigned int dimension) {
      if (fRngType == 0) SetDefaultType();
      if (fRng != 0 && fOwn) Free();
      fRng = gsl_qrng_alloc( fRngType, dimension );
      //std::cout << " allocate   " << fRng <<  std::endl;
    }

    void Free() {
       if (!fOwn) return; // no operation if pointer is not own
      //std::cout << "free gslrng " << fRngType <<  "  " << fRng <<  std::endl;
      if (fRng != 0) gsl_qrng_free(fRng);
      fRng = 0;
    }


    void SetType(const gsl_qrng_type * type) {
      fRngType = type;
    }

    void SetDefaultType() {
      // construct default engine (Sobol)
      fRngType =  gsl_qrng_sobol;
    }


   unsigned int Dimension() const { return fRng->dimension; }

    inline gsl_qrng * Rng()  { return fRng; }

    inline const gsl_qrng * Rng() const { return fRng; }



private:

   bool fOwn; // ownership of contained pointer
   gsl_qrng * fRng;
   const gsl_qrng_type * fRngType;
};


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLQRngWrapper */
