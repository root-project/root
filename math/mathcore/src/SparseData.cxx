// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline Wed Aug 28 15:33:03 2009

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class BinData

#include <iostream>
#include <iterator>
#include <algorithm>

#include <vector>
#include <list>

#include <stdexcept>

#include <cmath>
#include <limits>

// #include "TMath.h"
#include "Fit/BinData.h"
#include "Fit/SparseData.h"

using namespace std;

namespace ROOT {

   namespace Fit {

      //This class is a helper. It represents a bin in N
      //dimensions. The change in the name is to avoid name collision.
      class Box
      {
      public:
         // Creates a Box with limits specified by the vectors and
         // content=value and error=error
         Box(const vector<double>& min, const vector<double>& max,
             const double value = 0.0, const double error = 1.0):
            fMin(min), fMax(max), fVal(value), fError(error)
         { }

         // Compares to Boxes to see if they are equal in all its
         // variables. This is to be used by the std::find algorithm
         bool operator==(const Box& b)
         { return (fMin == b.fMin) && (fMax == b.fMax)
               && (fVal == b.fVal) && (fError == b.fError);  }

         // Get the list of minimum coordinates
         const vector<double>& GetMin() const { return fMin; }
         // Get the list of maximum coordinates
         const vector<double>& GetMax() const { return fMax; }
         // Get the value of the Box
         double GetVal() const { return fVal; }
         // Get the rror of the Box
         double GetError() const { return fError; }

         // Add an amount to the content of the Box
         void AddVal(const double value) { fVal += value; }

         friend class BoxContainer;
         friend ostream& operator <<(ostream& os, const Box& b);

      private:
         vector<double> fMin;
         vector<double> fMax;
         double fVal;
         double fError;
      };

      // This class is just a helper to be used in std::for_each to
      // simplify the code later. It's just a definition of a method
      // that will discern whether a Box is included into another one
      class BoxContainer
      {
      private:
         const Box& fBox;
      public:
         //Constructs the BoxContainer object with a Box that is meant
         //to include another one that will be provided later
         BoxContainer(const Box& b): fBox(b) {}

         bool operator() (const Box& b1)
         { return operator()(fBox, b1);  }

         // Looks if b2 is included in b1
         bool operator() (const Box& b1, const Box& b2)
         {
            bool isIn = true;
            vector<double>::const_iterator boxit = b2.fMin.begin();
            vector<double>::const_iterator bigit = b1.fMax.begin();
            while ( isIn && boxit != b2.fMin.end() )
            {
               if ( (*boxit) >= (*bigit) ) isIn = false;
               ++boxit;
               ++bigit;
            }

            boxit = b2.fMax.begin();
            bigit = b1.fMin.begin();
            while ( isIn && boxit != b2.fMax.end() )
            {
               if ( (*boxit) <= (*bigit) ) isIn = false;
               ++boxit;
               ++bigit;
            }

            return isIn;
         }
      };

      // Another helper class to be used in std::for_each to simplify
      // the code later. It implements the operator() to know if a
      // specified Box is big enough to contain any 'space' inside.
      class AreaComparer
      {
      public:
         AreaComparer(vector<double>::iterator iter):
            fThereIsArea(true),
            fIter(iter),
            fLimit(8 * std::numeric_limits<double>::epsilon())
         {};

         void operator() (double value)
         {
            if ( fabs(value- (*fIter)) < fLimit )
//             if ( TMath::AreEqualRel(value, (*fIter), fLimit) )
               fThereIsArea = false;

            ++fIter;
         }

         bool IsThereArea() { return fThereIsArea; }

      private:
         bool fThereIsArea;
         vector<double>::iterator fIter;
         double fLimit;
      };


      // This is the key of the SparseData structure. This method
      // will, by recursion, divide the area passed as an argument in
      // min and max into pieces to insert the Box defined by bmin and
      // bmax. It will do so from the highest dimension until it gets
      // to 1 and create the corresponding boxes to divide the
      // original space.
      void DivideBox( const vector<double>& min, const vector<double>& max,
                      const vector<double>& bmin, const vector<double>& bmax,
                      const unsigned int size, const unsigned int n,
                      list<Box>& l, const double val, const double error)
      {
         vector<double> boxmin(min);
         vector<double> boxmax(max);

         boxmin[n] = min[n];
         boxmax[n] = bmin[n];
         if ( for_each(boxmin.begin(), boxmin.end(), AreaComparer(boxmax.begin())).IsThereArea() )
            l.push_back(Box(boxmin, boxmax));

         boxmin[n] = bmin[n];
         boxmax[n] = bmax[n];
         if ( n == 0 )
         {
            if ( for_each(boxmin.begin(), boxmin.end(), AreaComparer(boxmax.begin())).IsThereArea() )
               l.push_back(Box(boxmin, boxmax, val, error));
         }
         else
            DivideBox(boxmin, boxmax, bmin, bmax, size, n-1, l, val, error);

         boxmin[n] = bmax[n];
         boxmax[n] = max[n];
         if ( for_each(boxmin.begin(), boxmin.end(), AreaComparer(boxmax.begin())).IsThereArea() )
            l.push_back(Box(boxmin, boxmax));
      }

      class ProxyListBox
      {
      public:
         void PushBack(Box& box) { fProxy.push_back(box); }
         list<Box>::iterator Begin() { return fProxy.begin(); }
         list<Box>::iterator End() { return fProxy.end(); }
         void Remove(list<Box>::iterator it) { fProxy.erase(it); }
         list<Box>& GetList() { return fProxy; }
      private:
         list<Box> fProxy;
      };


      SparseData::SparseData(vector<double>& min, vector<double>& max)
      {
         // Creates a SparseData convering the range defined by min
         // and max. For this it will create an empty Box for that
         // range.
         Box originalBox(min, max);
         fList = new ProxyListBox();
         fList->PushBack(originalBox);
      }

      SparseData::SparseData(const unsigned int dim, double min[], double max[])
      {
         // Creates a SparseData convering the range defined by min
         // and max. For this it will create an empty Box for that
         // range.
         vector<double> minv(min,min+dim);
         vector<double> maxv(max,max+dim);
         Box originalBox(minv, maxv);
         fList = new ProxyListBox();
         fList->PushBack(originalBox);
      }

      SparseData::~SparseData()
      { delete fList; }

      unsigned int SparseData::NPoints() const
      {
         // Returns the number of points stored, including the 0 ones.
         return fList->GetList().size();
      }

      unsigned int SparseData::NDim() const
      {
         // Returns the number of dimension of the SparseData object.
         return fList->Begin()->GetMin().size();
      }

      void SparseData::Add(std::vector<double>& min, std::vector<double>& max,
                           const double content, const double error)
      {
         // Add a box to the stored ones. For that, it will look for
         // the box that contains the new data and either replace it
         // or updated it.

         // Little box is the new Bin to be added
         Box littleBox(min, max);
         list<Box>::iterator it;
         // So we look for the Bin already in the list that contains
         // littleBox
         it = std::find_if(fList->Begin(), fList->End(), BoxContainer(littleBox));
         if ( it != fList->End() )
//             cout << "Found: " << *it << endl;
            ;
         else {
            cout << "SparseData::Add -> FAILED! box not found! " << endl;
            cout << littleBox << endl;
            return; // Does not add the box, as it is part of the
                    // underflow/overflow bin
         }
         // If it happens to have a value, then we add the value,
         if ( it->GetVal() )
            it->AddVal( content );
         else
         {
            // otherwise, we divide the container!
            DivideBox(it->GetMin(), it->GetMax(),
                      littleBox.GetMin(), littleBox.GetMax(),
                      it->GetMin().size(), it->GetMin().size() - 1,
                      fList->GetList(), content, error );
            // and remove it from the list
            fList->Remove(it);
         }
      }

      void SparseData::GetPoint(const unsigned int i,
                                std::vector<double>& min, std::vector<double>&max,
                                double& content, double& error)
      {
         // Get the point number i. This is a method to explore the
         // data stored in the class.

         unsigned int counter = 0;
         list<Box>::iterator it = fList->Begin();
         while ( it != fList->End() && counter != i ) {
            ++it;
            ++counter;
         }

         if ( (it == fList->End()) || (counter != i) )
            throw std::out_of_range("SparseData::GetPoint");

         min = it->GetMin();
         max = it->GetMax();
         content = it->GetVal();
         error = it->GetError();
      }

      void SparseData::PrintList() const
      {
         // Debug method to print a list with all the data stored.
         copy(fList->Begin(), fList->End(), ostream_iterator<Box>(cout, "\n------\n"));
      }


      void SparseData::GetBinData(BinData& bd) const
      {
         // Created the corresponding BinData

         list<Box>::iterator it = fList->Begin();
         const unsigned int dim = it->GetMin().size();

         bd.Initialize(fList->GetList().size(), dim);
         // Visit all the stored Boxes
         for ( ; it != fList->End(); ++it )
         {
            vector<double> mid(dim);
            // fill up the vector with the mid point of the Bin
            for ( unsigned int i = 0; i < dim; ++i)
            {
               mid[i] = ((it->GetMax()[i] - it->GetMin()[i]) /2) + it->GetMin()[i];
            }
            // And store it into the BinData structure
            bd.Add(&mid[0], it->GetVal(), it->GetError());
         }
      }

      void SparseData::GetBinDataIntegral(BinData& bd) const
      {
         // Created the corresponding BinData as with the Integral
         // option.

         list<Box>::iterator it = fList->Begin();

         bd.Initialize(fList->GetList().size(), it->GetMin().size());
         // Visit all the stored Boxes
         for ( ; it != fList->End(); ++it )
         {
            //Store the minimum value
            bd.Add(&(it->GetMin()[0]), it->GetVal(), it->GetError());
            //and the maximum
            bd.AddBinUpEdge(&(it->GetMax()[0]));
         }
      }

      void SparseData::GetBinDataNoZeros(BinData& bd) const
      {
         // Created the corresponding BinData, but it does not include
         // all the data with value equal to 0.

         list<Box>::iterator it = fList->Begin();
         const unsigned int dim = it->GetMin().size();

         bd.Initialize(fList->GetList().size(), dim);
         // Visit all the stored Boxes
         for ( ; it != fList->End(); ++it )
         {
            // if the value is zero, jump to the next
            if ( it->GetVal() == 0 ) continue;
            vector<double> mid(dim);
            // fill up the vector with the mid point of the Bin
            for ( unsigned int i = 0; i < dim; ++i)
            {
               mid[i] = ((it->GetMax()[i] - it->GetMin()[i]) /2) + it->GetMin()[i];
            }
            // And store it into the BinData structure
            bd.Add(&mid[0], it->GetVal(), it->GetError());
         }
      }

      // Just for debugging pourposes
      ostream& operator <<(ostream& os, const ROOT::Fit::Box& b)
      {
         os << "min: ";
         copy(b.GetMin().begin(), b.GetMin().end(), ostream_iterator<double>(os, " "));
         os << "max: ";
         copy(b.GetMax().begin(), b.GetMax().end(), ostream_iterator<double>(os, " "));
         os << "val: " << b.GetVal();

         return os;
      }
   } // end namespace Fit

} // end namespace ROOT
