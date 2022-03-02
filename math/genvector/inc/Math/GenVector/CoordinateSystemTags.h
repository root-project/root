// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team and                     *
  *                      FNAL LCG ROOT MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header source file for CoordinateSystemTag's classes
//
// Created by: Lorenzo Moneta  at Wed Apr 05 2006
//
//

#ifndef ROOT_Math_GenVector_CoordinateSystemTags
#define ROOT_Math_GenVector_CoordinateSystemTags  1



namespace ROOT {

namespace Math {


//__________________________________________________________________________________________
   /**
      DefaultCoordinateSystemTag
      Default tag for identifying any coordinate system

      @ingroup GenVector

      @sa Overview of the @ref GenVector "physics vector library"
   */

   class  DefaultCoordinateSystemTag {};


//__________________________________________________________________________________________
   /**
      Tag for identifying vectors based on a global coordinate system

      @ingroup GenVector

      @sa Overview of the @ref GenVector "physics vector library"
   */
   class  GlobalCoordinateSystemTag {};

//__________________________________________________________________________________________
   /**
      Tag for identifying vectors based on a local coordinate system

      @ingroup GenVector

      @sa Overview of the @ref GenVector "physics vector library"
   */
   class   LocalCoordinateSystemTag {};


}  // namespace Math

}  // namespace ROOT



#endif
