// @(#)root/mathcore:$Name:  $:$Id: CoordinateSystemTags.h,v 1.1 2005/12/05 08:40:34 moneta Exp $
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


    /**
       Default tag identifying any coordinate system 
     */

    class  DefaultCoordinateSystemTag {}; 


    /**
       Tag for identifying vectors based on a global coordinate system
     */
    class  GlobalCoordinateSystemTag {}; 

    /**
       Tag for identifying vectors based on a local coordinate system
     */
    class   LocalCoordinateSystemTag {}; 


  }  // namespace Math

}  // namespace ROOT



#endif
