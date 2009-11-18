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
            _min(min), _max(max), _val(value), _error(error)
         { }
         
         // Compares to Boxes to see if they are equal in all its
         // variables. This is to be used by the std::find algorithm
         bool operator==(const Box& b)
         { return (_min == b._min) && (_max == b._max) 
               && (_val == b._val) && (_error == b._error);  }
         
         // Get the list of minimum coordinates
         const vector<double>& getMin() const { return _min; }
         // Get the list of maximum coordinates
         const vector<double>& getMax() const { return _max; }
         // Get the value of the Box
         double getVal() const { return _val; }
         // Get the rror of the Box
         double getError() const { return _error; }
         
         // Add an amount to the content of the Box
         void addVal(const double value) { _val += value; }
         
         friend class BoxContainer;
         friend ostream& operator <<(ostream& os, const Box& b);
         
      private:
         vector<double> _min;
         vector<double> _max;
         double _val;
         double _error;
      };
      
      // This class is just a helper to be used in std::for_each to
      // simplify the code later. It's just a definition of a method
      // that will discern whether a Box is included into another one
      class BoxContainer
      {
      private:
         const Box& _b;
      public:
         //Constructs the BoxContainer object with a Box that is meant
         //to include another one that will be provided later
         BoxContainer(const Box& b): _b(b) {}
         
         bool operator() (const Box& b1)
         { return operator()(_b, b1);  }
         
         // Looks if b2 is included in b1
         bool operator() (const Box& b1, const Box& b2)
         {
            bool isIn = true;
            vector<double>::const_iterator boxit = b2._min.begin();
            vector<double>::const_iterator bigit = b1._max.begin();
            while ( isIn && boxit != b2._min.end() )
            {
               if ( (*boxit) >= (*bigit) ) isIn = false;
               boxit++;
               bigit++;
            }
            
            boxit = b2._max.begin();
            bigit = b1._min.begin();
            while ( isIn && boxit != b2._max.end() )
            {
               if ( (*boxit) <= (*bigit) ) isIn = false;
               boxit++;
               bigit++;
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
         AreaComparer(vector<double>::iterator iter, double cmpLimit = 1e-16): 
            thereIsArea(true), 
            it(iter),
            limit(cmpLimit)
         {};
         
         void operator() (double value)
         {
            if ( fabs(value- (*it)) < limit )
               thereIsArea = false;
            
            it++;
         }
         
         bool isThereArea() { return thereIsArea; }
         
      private:
         bool thereIsArea;
         vector<double>::iterator it;
         double limit;
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
         if ( for_each(boxmin.begin(), boxmin.end(), AreaComparer(boxmax.begin())).isThereArea() )
            l.push_back(Box(boxmin, boxmax));
         
         boxmin[n] = bmin[n];
         boxmax[n] = bmax[n];
         if ( n == 0 ) 
         {
            if ( for_each(boxmin.begin(), boxmin.end(), AreaComparer(boxmax.begin())).isThereArea() )
               l.push_back(Box(boxmin, boxmax, val, error));
         }
         else
            DivideBox(boxmin, boxmax, bmin, bmax, size, n-1, l, val, error);
         
         boxmin[n] = bmax[n];
         boxmax[n] = max[n];
         if ( for_each(boxmin.begin(), boxmin.end(), AreaComparer(boxmax.begin())).isThereArea() )
            l.push_back(Box(boxmin, boxmax));
      }
      
      class ProxyListBox
      {
      public:
         void push_back(Box& box) { l.push_back(box); }
         list<Box>::iterator begin() { return l.begin(); }
         list<Box>::iterator end() { return l.end(); }
         void remove(list<Box>::iterator it) { l.erase(it); }
         list<Box>& getList() { return l; }
      private:
         list<Box> l;
      };


      SparseData::SparseData(vector<double>& min, vector<double>& max)
      {
         Box originalBox(min, max);
         l = new ProxyListBox();
         l->push_back(originalBox);
      }

      SparseData::SparseData(const unsigned int dim, double min[], double max[])
      {
         vector<double> minv(min,min+dim);
         vector<double> maxv(max,max+dim);
         Box originalBox(minv, maxv);
         l = new ProxyListBox();
         l->push_back(originalBox);
      }

      SparseData::~SparseData()
      { delete l; }

      unsigned int SparseData::NPoints() const
      {
         return l->getList().size();
      }
      
      unsigned int SparseData::NDim() const
      {
         return l->begin()->getMin().size();
      }

      void SparseData::Add(std::vector<double>& min, std::vector<double>& max, 
                           const double content, const double error)
      {
         // Little box is the new Bin to be added
         Box littleBox(min, max);
         list<Box>::iterator it;
         // So we look for the Bin already in the list that contains
         // littleBox
         it = std::find_if(l->begin(), l->end(), BoxContainer(littleBox));
         if ( it != l->end() )
//             cout << "Found: " << *it << endl;
            ;
         else {
            cout << "SparseData::Add -> FAILED! box not found! " << endl;
            cout << littleBox << endl;
            return; // Does not add the box, as it is part of the
                    // underflow/overflow bin
         }
         // If it happens to have a value, then we add the value,
         if ( it->getVal() )
            it->addVal( content );
         else
         {
            // otherwise, we divide the container!
            DivideBox(it->getMin(), it->getMax(),
                      littleBox.getMin(), littleBox.getMax(),
                      it->getMin().size(), it->getMin().size() - 1,
                      l->getList(), content, error );
            // and remove it from the list
            l->remove(it);
         }
      }

      void SparseData::GetPoint(const unsigned int i, 
                                std::vector<double>& min, std::vector<double>&max,
                                double& content, double& error)
      {
         unsigned int counter = 0;
         list<Box>::iterator it = l->begin();
         while ( it != l->end() && counter != i ) {
            ++it; 
            ++counter;
         }

         if ( (it == l->end()) || (counter != i) )
            throw std::out_of_range("SparseData::GetPoint");

         min = it->getMin();
         max = it->getMax();
         content = it->getVal();
         error = it->getError();
      }

      void SparseData::PrintList() const
      {
         copy(l->begin(), l->end(), ostream_iterator<Box>(cout, "\n------\n"));
      }


      void SparseData::GetBinData(BinData& bd) const
      {
         list<Box>::iterator it = l->begin();
         const unsigned int dim = it->getMin().size();

         bd.Initialize(l->getList().size(), dim); 
         // Visit all the stored Boxes
         for ( ; it != l->end(); ++it )
         {
            vector<double> mid(dim);
            // fill up the vector with the mid point of the Bin
            for ( unsigned int i = 0; i < dim; ++i)
            {
               mid[i] = ((it->getMax()[i] - it->getMin()[i]) /2) + it->getMin()[i];
            }
            // And store it into the BinData structure
            bd.Add(&mid[0], it->getVal(), it->getError());
         }
      }

      void SparseData::GetBinDataIntegral(BinData& bd) const
      {
         list<Box>::iterator it = l->begin();

         bd.Initialize(l->getList().size(), it->getMin().size()); 
         // Visit all the stored Boxes
         for ( ; it != l->end(); ++it )
         {
            //Store the minimum value
            bd.Add(&(it->getMin()[0]), it->getVal(), it->getError());
            //and the maximum
            bd.AddBinUpEdge(&(it->getMax()[0]));
         }
      }

      void SparseData::GetBinDataNoZeros(BinData& bd) const
      {
         list<Box>::iterator it = l->begin();
         const unsigned int dim = it->getMin().size();

         bd.Initialize(l->getList().size(), dim);
         // Visit all the stored Boxes
         for ( ; it != l->end(); ++it )
         {
            // if the value is zero, jump to the next
            if ( it->getVal() == 0 ) continue;
            vector<double> mid(dim);
            // fill up the vector with the mid point of the Bin
            for ( unsigned int i = 0; i < dim; ++i)
            {
               mid[i] = ((it->getMax()[i] - it->getMin()[i]) /2) + it->getMin()[i];
            }
            // And store it into the BinData structure
            bd.Add(&mid[0], it->getVal(), it->getError());
         }
      }

      // Just for debugging pourposes
      ostream& operator <<(ostream& os, const ROOT::Fit::Box& b)
      {
         os << "min: ";
         copy(b.getMin().begin(), b.getMin().end(), ostream_iterator<double>(os, " "));
         os << "max: ";
         copy(b.getMax().begin(), b.getMax().end(), ostream_iterator<double>(os, " "));
         os << "val: " << b.getVal();
         
         return os;
      }     
   } // end namespace Fit

} // end namespace ROOT
