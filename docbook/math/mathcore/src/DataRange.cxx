// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:05:02 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class DataRange

#include "Fit/DataRange.h"
#include "Math/Error.h"

#include <algorithm>
#include <limits>

namespace ROOT { 

   namespace Fit { 

DataRange::DataRange(double xmin, double xmax) : 
   fRanges( std::vector<RangeSet> (1) )
{
   // construct a range for [xmin, xmax] 
   if (xmin < xmax) { 
      RangeSet rx(1); 
      rx[0] = std::make_pair(xmin, xmax); 
      fRanges[0] = rx; 
   }
}


DataRange::DataRange(double xmin, double xmax, double ymin, double ymax) : 
   fRanges( std::vector<RangeSet> (2) )
{
   // construct a range for [xmin, xmax] , [ymin, ymax] 
   if (xmin < xmax) { 
      RangeSet rx(1); 
      rx[0] = std::make_pair(xmin, xmax); 
      fRanges[0] = rx; 
   }
   
   if (ymin < ymax) { 
      RangeSet ry(1); 
      ry[0] = std::make_pair(ymin, ymax); 
      fRanges[1] = ry; 
   }
}

DataRange::DataRange(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) : 
   fRanges( std::vector<RangeSet> (3) )
{
   // construct a range for [xmin, xmax] , [ymin, ymax] , [zmin, zmax] 
   if (xmin < xmax) { 
      RangeSet rx(1); 
      rx[0] = std::make_pair(xmin, xmax); 
      fRanges[0] = rx; 
   }
   if (ymin < ymax) {    
      RangeSet ry(1); 
      ry[0] = std::make_pair(ymin, ymax); 
      fRanges[1] = ry; 
   }
   if (zmin < zmax) {    
      RangeSet rz(1); 
      rz[0] = std::make_pair(zmin, zmax); 
      fRanges[2] = rz; 
   }
}

bool lessRange( const std::pair<double,double> & r1, const std::pair<double,double> & r2 ) { 
   // compare ranges using max position so in case of included ranges smaller one comes first
   return r1.second <  r2.second; 
}

std::pair<double, double> DataRange::operator() (unsigned int icoord,unsigned int irange) const {
   if ( Size(icoord) >  irange )
      return fRanges[icoord].at(irange);
   else if (irange == 0)  {
      // return [-inf +inf] for the other dimension 
      double xmin = 0; double xmax = 0; 
      GetInfRange(xmin,xmax);
      return std::make_pair<double,double>(xmin,xmax);     
   }                                               
   else { 
      // in case the irange-th does not exist for the given coordinate
      MATH_ERROR_MSG("DataRange::operator()","invalid range number - return (0,0)");
      return std::make_pair<double,double>(0,0);     
   }
}  

void DataRange::AddRange(unsigned  int  icoord , double xmin, double xmax  ) { 
   // add a range [xmin,xmax] for the new coordinate icoord 

   if (xmin >= xmax) return;  // no op in case of bad values

   // case the  coordinate is larger than the current allocated vector size
   if (icoord >= fRanges.size() ) { 
      RangeSet rx(1); 
      rx[0] = std::make_pair(xmin, xmax); 
      fRanges.resize(icoord+1);
      fRanges[icoord] = rx; 
      return;
   } 
   RangeSet & rs = fRanges[icoord]; 
   // case the vector  of the ranges is empty in the given coordinate
   if ( rs.size() == 0) { 
      rs.push_back(std::make_pair(xmin,xmax) ); 
      return;
   } 
   // case of  an already existing range
   // need to establish a policy (use OR or AND )

   CleanRangeSet(icoord,xmin,xmax); 
   // add the new one
   rs.push_back(std::make_pair(xmin,xmax) ); 
   // sort range in increasing values of xmax 
   std::sort( rs.begin(), rs.end() , lessRange);

}

void DataRange::SetRange(unsigned  int  icoord , double xmin, double xmax  ) { 
   // set a new range [xmin,xmax] for the new coordinate icoord 

   if (xmin >= xmax) return;  // no op in case of bad values

   // case the  coordinate is larger than the current allocated vector size
   if (icoord >= fRanges.size() ) { 
      fRanges.resize(icoord+1);
      RangeSet rs(1); 
      rs[0] = std::make_pair(xmin, xmax); 
      fRanges[icoord] = rs; 
      return;
   }
   // add range 
   RangeSet & rs = fRanges[icoord]; 
   // deleting existing ones if (exists)
   if (rs.size() > 1) MATH_WARN_MSG("DataRange::SetRange","remove existing range and keep only the set one");
   rs.resize(1); 
   rs[0] =  std::make_pair(xmin, xmax); 
   return; 
}

bool DataRange::IsInside(double x, unsigned int icoord ) const { 
   // check if a point is in range

   if (Size(icoord) == 0) return true;  // no range existing (is like -inf, +inf)  
   const RangeSet & ranges = fRanges[icoord];
   for (RangeSet::const_iterator itr = ranges.begin(); itr != ranges.end(); ++itr) { 
      if ( x < (*itr).first ) return false; 
      if ( x <= (*itr).second) return true; 
   }
   return false; // point is larger than last xmax
} 

void DataRange::Clear(unsigned int icoord ) { 
   // remove all ranges for coordinate icoord
   if (Size(icoord) == 0) return;  // no op in this case 
   fRanges[icoord].clear(); 
}


void DataRange::CleanRangeSet(unsigned int icoord, double xmin, double xmax) { 
   //  remove all the existing ranges between xmin and xmax 
   //  called when a new range is inserted

   // loop on existing ranges 
   RangeSet & ranges = fRanges[icoord]; 
   for (RangeSet::iterator itr = ranges.begin(); itr != ranges.end(); ++itr) { 
      // delete included ranges
      if ( itr->first >= xmin && itr->second <= xmax) { 
         itr = ranges.erase(itr);
         // itr goes to next element, so go back before adding
         --itr;
      }
   }
   
}

void DataRange::GetInfRange(double &xmin, double &xmax) { 
   // get the full range [-inf, +inf] for xmin and xmax 
   xmin = -std::numeric_limits<double>::infinity(); 
   xmax = std::numeric_limits<double>::infinity(); 
}

   } // end namespace Fit

} // end namespace ROOT

