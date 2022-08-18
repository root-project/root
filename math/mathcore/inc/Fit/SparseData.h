// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline Wed Aug 28 15:23:43 2009

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class SparseData

#ifndef ROOT_Fit_SparseData
#define ROOT_Fit_SparseData

#include "Fit/BinData.h"
#include <vector>
#include <memory>

namespace ROOT {

   namespace Fit {

      // This is a proxy to a std::list<Box>
      class ProxyListBox;

      /**
         SparseData class representing the data of a THNSparse histogram
         The data needs to be converted to a BinData class before fitting using
         the GetBinData functions.

         @ingroup FitData
      */

      class SparseData : public FitData  {
      public:
         /// Constructor with a vector
         SparseData(std::vector<double>& min, std::vector<double>& max);

         /// Constructor with a dimension and two arrays
         SparseData(const unsigned int dim, double min[], double max[]);

         /// Copy constructor
         SparseData(const SparseData & rhs);

         /// Destructor
         ~SparseData() override;

         /// Assignment operator
         SparseData & operator=(const SparseData & rhs);

         /// Returns the number of points stored
         unsigned int NPoints() const;
         /// Returns the dimension of the object (bins)
         unsigned int NDim() const;

         /// Adds a new bin specified by the vectors
         void Add(std::vector<double>& min, std::vector<double>& max,
                  const double content, const double error = 1.0);

         void GetPoint(const unsigned int i,
                       std::vector<double>& min, std::vector<double>&max,
                       double& content, double& error);

         /// Debug method to print the list of bins stored
         void PrintList() const;

         /// Transforms the data into a ROOT::Fit::BinData structure
         void GetBinData(BinData&) const;
         /// Same as before, but returning a BInData with integral format (containing bin edges)
         void GetBinDataIntegral(BinData&) const;
         /// Same as before, but including zero content bins
         void GetBinDataNoZeros(BinData&) const;

      private :
         std::unique_ptr<ProxyListBox> fList;
      };

   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Fit_SparseData */
