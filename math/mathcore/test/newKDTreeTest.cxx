// @(#)root/mathcore:$Id$
// Author: C. Gumpert    09/2011

// program to test new KDTree class

#include <time.h>
// STL include(s)
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "assert.h"

// custom include(s)
#include "Math/KDTree.h"
#include "Math/TDataPoint.h"

template<class _DataPoint>
void CreatePseudoData(const unsigned long int nPoints,std::vector<const _DataPoint*>& vDataPoints)
{
   _DataPoint* pData = 0;
   for(unsigned long int i = 0; i < nPoints; ++i)
   {
      pData = new _DataPoint();
      for(unsigned int k = 0; k < _DataPoint::Dimension(); ++k)
      pData->SetCoordinate(k,rand() % 1000);
      pData->SetWeight(rand() % 1000);
      vDataPoints.push_back(pData);
   }
}

template<class _DataPoint>
void DeletePseudoData(std::vector<const _DataPoint*>& vDataPoints)
{
   for(typename std::vector<const _DataPoint*>::iterator it = vDataPoints.begin();
       it != vDataPoints.end(); ++it)
   delete *it;

   vDataPoints.clear();
}

template<class _DataPoint>
ROOT::Math::KDTree<_DataPoint>* BuildTree(const std::vector<const _DataPoint*>& vDataPoints,const unsigned int iBucketSize)
{
   ROOT::Math::KDTree<_DataPoint>* pTree = 0;
   try
   {
      pTree = new ROOT::Math::KDTree<_DataPoint>(iBucketSize);
      //pTree->SetSplitOption(TKDTree<_DataPoint>::kBinContent);

      for(typename std::vector<const _DataPoint*>::const_iterator it = vDataPoints.begin(); it != vDataPoints.end(); ++it)
      pTree->Insert(**it);
   }
   catch (std::exception& e)
   {
      std::cerr << "exception caught: " << e.what() << std::endl;
      if(pTree)
      delete pTree;

      pTree = 0;
   }

   return pTree;
}

template<class _DataPoint>
bool CheckBasicTreeProperties(const ROOT::Math::KDTree<_DataPoint>* pTree,const std::vector<const _DataPoint*>& vDataPoints)
{
   if(pTree->GetEntries() != vDataPoints.size())
   {
      std::cout << "  --> wrong number of data points in tree: " << pTree->GetEntries() << " != " << vDataPoints.size() << std::endl;
      return false;
   }

   double fSumw = 0;
   double fSumw2 = 0;
   for(typename std::vector<const _DataPoint*>::const_iterator it = vDataPoints.begin();
       it != vDataPoints.end(); ++it)
   {
      fSumw += (*it)->GetWeight();
      fSumw2 += pow((*it)->GetWeight(),2);
   }

   if(fabs(pTree->GetTotalSumw2() - fSumw2)/fSumw2 > 1e-4)
   {
      std::cout << "  --> inconsistent Sum weights^2 in tree: " << pTree->GetTotalSumw2() << " != " << fSumw2 << std::endl;
      return false;
   }

   if(fabs(pTree->GetTotalSumw() - fSumw)/fSumw > 1e-4)
   {
      std::cout << "  --> inconsistent Sum weights in tree: " << pTree->GetTotalSumw() << " != " << fSumw << std::endl;
      return false;
   }

   if(fabs(pTree->GetEffectiveEntries() - pow(fSumw,2)/fSumw2)/(pow(fSumw,2)/fSumw2) > 1e-4)
   {
      std::cout << "  --> inconsistent effective entries in tree: " << pTree->GetEffectiveEntries() << " != " << pow(fSumw,2)/fSumw2 << std::endl;
      return false;
   }

   return true;
}

template<class _DataPoint>
bool CheckBinBoundaries(const ROOT::Math::KDTree<_DataPoint>* pTree)
{
   typedef std::pair<typename _DataPoint::value_type,typename _DataPoint::value_type> tBoundary;

   std::cout << "  --> checking " << pTree->GetNBins() << " bins" << std::endl;

   unsigned int iBin = 0;
   for(typename ROOT::Math::KDTree<_DataPoint>::iterator it = pTree->First(); it != pTree->End(); ++it,++iBin)
   {
      const std::vector<const _DataPoint*> vDataPoints = it.TN()->GetPoints();
      assert(vDataPoints.size() == it->GetEntries());

      std::vector<tBoundary> vBoundaries = it->GetBoundaries();
      assert(_DataPoint::Dimension() == vBoundaries.size());

      // check whether all points in this bin are inside the boundaries
      for(typename std::vector<const _DataPoint*>::const_iterator pit = vDataPoints.begin();
          pit != vDataPoints.end(); ++pit)
      {
         for(unsigned int k = 0; k < _DataPoint::Dimension(); ++k)
         {
            if(((*pit)->GetCoordinate(k) < vBoundaries.at(k).first) || ((*pit)->GetCoordinate(k) > vBoundaries.at(k).second))
            {
               std::cout << "  --> boundaries of bin " << iBin << " in " << k << ". dimension are inconsistent with data point in bucket" << std::endl;
               return false;
            }
         }
      }
   }

   return true;
}

template<class _DataPoint>
bool CheckEffectiveBinEntries(const ROOT::Math::KDTree<_DataPoint>* pTree)
{
   for(typename ROOT::Math::KDTree<_DataPoint>::iterator it = pTree->First(); it != pTree->End(); ++it)
   {
      if(it->GetEffectiveEntries() > 2*pTree->GetBucketSize())
      {
         std::cout << "  --> found bin with " << it->GetEffectiveEntries() << " while the bucketsize is " << pTree->GetBucketSize() << std::endl;
         return false;
      }
   }

   return true;
}

template<class _DataPoint>
bool CheckFindBin(const ROOT::Math::KDTree<_DataPoint>* pTree)
{
   typedef std::pair<typename _DataPoint::value_type,typename _DataPoint::value_type> tBoundary;

   _DataPoint test;
   std::cout << "  --> test reference point at (";
   for(unsigned int k = 0; k < _DataPoint::Dimension(); ++k)
   {
      test.SetCoordinate(k,rand() % 1000);
      std::cout << test.GetCoordinate(k);
      if(k < _DataPoint::Dimension()-1)
      std::cout << ",";
   }
   std::cout << ")" << std::endl;

   const typename ROOT::Math::KDTree<_DataPoint>::Bin* bin = pTree->FindBin(test);

   // check whether test point is actually inside the bin boundaries
   // is not necessarily the case if the point as the range of the bin which is NOT determined by a splitting but by the minimum coordinate of points inside the bin
   std::vector<tBoundary> vBoundaries = bin->GetBoundaries();
   assert(_DataPoint::Dimension() == vBoundaries.size());

   for(unsigned int k = 0; k < _DataPoint::Dimension(); ++k)
   {
      if((test.GetCoordinate(k) < vBoundaries.at(k).first) || (test.GetCoordinate(k) > vBoundaries.at(k).second))
      {
         if(pTree->IsFrozen() && bin)
         {
            std::cout << "  --> " << test.GetCoordinate(k) << " is not within (" << vBoundaries.at(k).first << "," << vBoundaries.at(k).second << ")" << std::endl;
            return false;
         }
      }
   }

   return true;
}

template<class _DataPoint>
bool CheckNearestNeighborSearches(const ROOT::Math::KDTree<_DataPoint>* pTree,const std::vector<const _DataPoint*>& vDataPoints)
{
   _DataPoint test;
   std::cout << "  --> test with reference point at (";
   for(unsigned int k = 0; k < _DataPoint::Dimension(); ++k)
   {
      test.SetCoordinate(k,rand() % 1000);
      std::cout << test.GetCoordinate(k);
      if(k < _DataPoint::Dimension()-1)
      std::cout << ",";
   }
   std::cout << ")" << std::endl;

   std::vector<const _DataPoint*> vFoundPoints;
   std::vector<const _DataPoint*> vFoundPointsCheck;

   double fDist = rand() % 500;
   std::cout << "  --> look for points within in  distance of " << fDist << std::endl;
   pTree->GetPointsWithinDist(test,fDist,vFoundPoints);

   // get points by hand
   for(typename std::vector<const _DataPoint*>::const_iterator it = vDataPoints.begin();
       it != vDataPoints.end(); ++it)
   {
      if((*it)->Distance(test) <= fDist)
      {
         vFoundPointsCheck.push_back(*it);
         // check whether this point was also found by the algorithm
         bool bChecked = false;
         for(unsigned int i = 0; i < vFoundPoints.size(); ++i)
         {
            if(vFoundPoints.at(i) == *it)
            {
               bChecked = true;
               break;
            }
         }

         if(!bChecked)
         {
            std::cout << "  --> point (";
            for(unsigned int k = 0; k < _DataPoint::Dimension(); ++k)
            {
               std::cout << (*it)->GetCoordinate(k);
               if(k < _DataPoint::Dimension()-1)
               std::cout << ",";
            }
            std::cout << ") was not found by the algorithm while its distance to the reference point is " << (*it)->Distance(test) << std::endl;

            return false;
         }
      }
   }

   if(vFoundPointsCheck.size() != vFoundPoints.size())
   {
      std::cout << "  --> GetPointsWithinDist returns wrong number of found points (" << vFoundPointsCheck.size() << " expected/ " << vFoundPoints.size() << " found)" << std::endl;
      return false;
   }

   const int nNeighbors = (int)(rand() % 100/1000.0 * pTree->GetEntries() + 1);
   std::cout << "  --> look for " << nNeighbors << " nearest neighbors" << std::endl;

   std::vector<std::pair<const _DataPoint*,double> > vFoundNeighbors;
   std::vector<std::pair<const _DataPoint*,double> > vFoundNeighborsCheck;
   typename std::vector<std::pair<const _DataPoint*,double> >::iterator nit;

   pTree->GetClosestPoints(test,nNeighbors,vFoundNeighbors);
   fDist = vFoundNeighbors.back().second;

   // check closest points manually
   for(typename std::vector<const _DataPoint*>::const_iterator it = vDataPoints.begin();
       it != vDataPoints.end(); ++it)
   {
      if((*it)->Distance(test) <= fDist)
      vFoundNeighborsCheck.push_back(std::make_pair(*it,(*it)->Distance(test)));
   }

   // vFoundNeighborsCheck can have more data points because there might be more points with the same (maximal) distance
   if(vFoundNeighborsCheck.size() < vFoundNeighbors.size())
   {
      std::cout << "  --> GetClosestPoints returns wrong number of found points (" << vFoundNeighborsCheck.size() << " expected/ " << vFoundNeighbors.size() << " found)" << std::endl;
      return false;
   }

   //check whether all points found by the algorithm are also found manually
   bool bChecked = false;
   for(unsigned int i = 0; i < vFoundNeighbors.size(); ++i)
   {
      bChecked = false;
      for(unsigned int j = 0; j < vFoundNeighborsCheck.size(); ++j)
      {
         if(vFoundNeighbors.at(i).first == vFoundNeighborsCheck.at(j).first)
         {
            if(fabs(vFoundNeighbors.at(i).second - vFoundNeighborsCheck.at(j).second)/vFoundNeighbors.at(i).second < 1e-2)
            bChecked = true;

            break;
         }
      }

      if(!bChecked)
      return false;
   }

   return true;
}

template<class _DataPoint>
bool CheckTreeClear(ROOT::Math::KDTree<_DataPoint>* pTree,const std::vector<const _DataPoint*>& vDataPoints)
{
   pTree->Reset();
   if(pTree->GetEntries() != 0)
   {
      std::cout << "  --> tree contains still " << pTree->GetEntries() << " data points after calling Clear()" << std::endl;
      return false;
   }
   if(pTree->GetNBins() != 1)
   {
      std::cout << "  --> tree contains more than one bin after calling Clear()" << std::endl;
      return false;
   }
   if(pTree->GetEffectiveEntries() != 0)
   {
      std::cout << "  --> tree contains still " << pTree->GetEffectiveEntries() << " effective entries after calling Clear()" << std::endl;
      return false;
   }

   // try to fill tree again
   try
   {
      for(typename std::vector<const _DataPoint*>::const_iterator it = vDataPoints.begin(); it != vDataPoints.end(); ++it)
      pTree->Insert(**it);
   }
   catch (std::exception& e)
   {
      std::cout << "  --> unable to fill tree after calling Clear()" << std::endl;
      std::cerr << "exception caught: " << e.what() << std::endl;

      return false;
   }

   return true;
}

int main()
{
   std::cout << "\nunit test for class KDTree" << std::endl;
   std::cout << "==========================\n" << std::endl;

   int iSeed = time(0);
   std::cout << "using random seed: " << iSeed << std::endl;

   srand(iSeed);

   const unsigned long int NPOINTS = 1e5;
   const unsigned int BUCKETSIZE = 1e2;
   const unsigned int DIM = 5;

   typedef ROOT::Math::TDataPoint<DIM> DP;

   std::cout << "using " << NPOINTS << " data points in " << DIM << " dimensions" << std::endl;
   std::cout << "bucket size: " << BUCKETSIZE << std::endl;

   std::vector<const DP*> vDataPoints;
   CreatePseudoData(NPOINTS,vDataPoints);

   ROOT::Math::KDTree<DP>* pTree = BuildTree(vDataPoints,BUCKETSIZE);

   if(CheckBasicTreeProperties(pTree,vDataPoints))
   std::cerr << "basic tree properties...DONE" << std::endl;
   else
   std::cerr << "basic tree properties...FAILED" << std::endl;

   if(CheckBinBoundaries(pTree))
   std::cerr << "consistency check of bin boundaries...DONE" << std::endl;
   else
   std::cerr << "consistency check of bin boundaries...FAILED" << std::endl;

   if(CheckEffectiveBinEntries(pTree))
   std::cerr << "check effective entries per bin...DONE" << std::endl;
   else
   std::cerr << "check effective entries per bin...FAILED" << std::endl;

   if(CheckFindBin(pTree))
   std::cerr << "check FindBin...DONE" << std::endl;
   else
   std::cerr << "check FindBin...FAILED" << std::endl;

   if(CheckNearestNeighborSearches(pTree,vDataPoints))
   std::cerr << "check nearest neighbor searches...DONE" << std::endl;
   else
   std::cerr << "check nearest neighbor searches...FAILED" << std::endl;

   if(CheckTreeClear(pTree,vDataPoints))
   std::cerr << "check KDTree::Clear...DONE" << std::endl;
   else
   std::cerr << "check KDTree:Clear...FAILED" << std::endl;

   //pTree->Print();
   pTree->Freeze();
   //pTree->Print();
   ROOT::Math::KDTree<DP>* pCopy = pTree->GetFrozenCopy();
   //pCopy->Print();

   delete pCopy;
   delete pTree;

   DeletePseudoData(vDataPoints);

   return 0;
}
