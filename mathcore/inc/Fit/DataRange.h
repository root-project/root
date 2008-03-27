// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:05:02 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class DataRange

#ifndef ROOT_Fit_DataRange
#define ROOT_Fit_DataRange

#include <vector>

namespace ROOT { 

   namespace Fit { 


//___________________________________________________________________________________
/** 
   class describing the range in the coordinates 
   it supports  multiple range in a coordinate. 
   The rnage dimension is the dimension of the coordinate, its size is 
   the number of interval for each coordinate. 
   Default range is -inf, inf
   Range can be modified with the add range method

   @ingroup FitData
*/ 
class DataRange {

public: 

   typedef std::vector<std::pair<double,double> > RangeSet;
   typedef std::vector< RangeSet >   RangeIntervals; 

   /** 
      Default constructor (infinite range) 
   */ 
   explicit DataRange (unsigned int dim = 1) :
      fRanges ( std::vector<RangeSet> (dim) )
   {}

   /**
      construct a range for [xmin, xmax] 
    */
   DataRange(double xmin, double xmax);  

   /**
      construct a range for [xmin, xmax] , [ymin, ymax] 
    */
   DataRange(double xmin, double xmax, double ymin, double ymax);  
   /**
      construct a range for [xmin, xmax] , [ymin, ymax] , [zmin, zmax] 
    */
   DataRange(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax); 
   /**
      get range dimension
    */ 
   unsigned int NDim() const { return fRanges.size(); }

   /**
      return range size for coordinate icoord (starts from zero)
      Size == 0 indicates no range is present [-inf, + inf]
   */
   unsigned int Size(unsigned int icoord) const { 
      return icoord <  fRanges.size() ? fRanges[icoord].size() : 0;
   }

   /** 
       return the vector of ranges for the coordinate icoord
   */ 
   const RangeSet & Ranges(unsigned int icoord) const { 
      // return icoord <  fRanges.size() ? fRanges[icoord] : RangeSet(); 
      return fRanges.at(icoord); 
   }

   /** 
       return the first range for the coordinate icoord.
       Useful method when only one range is present for the given coordinate 
   */ 
   std::pair<double, double> operator() (unsigned int icoord) const {
     return Size(icoord) >  0 ? fRanges[icoord].front() : std::make_pair<double,double>(0,0);     
   }  

   /**
      get the first range for given coordinate
    */
   void GetRange(unsigned int icoord, double & xmin, double & xmax) const { 
      if (Size(icoord) == 0) { 
         xmin = 0; 
         xmax = 0; 
         return;
      }
      xmin = fRanges[icoord].front().first; 
      xmax = fRanges[icoord].front().second; 
   }

   /** 
      Destructor (no operations)
   */ 
   ~DataRange ()  {}  



   /**
      add a range [xmin,xmax] for the new coordinate icoord 
    */
   void AddRange(double xmin, double xmax, unsigned  int  icoord = 0 ); 

   /**
      clear all ranges in one coordinate (is now -inf, +inf)
    */
   void Clear (unsigned  int  icoord = 0 );

   /**
      check if a point is inside the range for the given coordinate
    */
   bool IsInside(double x, unsigned int icoord = 0) const; 

protected: 
   /** 
       internal function to remove all the existing ranges between xmin and xmax 
       called when a new range is inserted
   */
   void CleanRangeSet(unsigned int icoord, double xmin, double xmax); 


private: 

   RangeIntervals fRanges;  // list of all ranges


}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_DataRange */
