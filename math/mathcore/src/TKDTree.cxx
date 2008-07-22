// Authors: Marian Ivanov and Alexandru Bercuci 04/03/2005

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TKDTree.h"
#include "TRandom.h"

#include "TString.h"
#include <string.h>

#ifndef R__ALPHA
templateClassImp(TKDTree)
#endif


//////////////////////////////////////////////////////////////////////////
//
//                      kd-tree and its implementation in TKDTree
//
// Contents:
// 1. What is kd-tree
// 2. How to cosntruct kdtree - Pseudo code
// 3. TKDTree implementation - technical details
//    a. The order of nodes in internal arrays
//    b. Division algorithm
//    c. The order of nodes in boundary related arrays
//
//
//
// 1. What is kdtree ? ( http://en.wikipedia.org/wiki/Kd-tree )
//
// In computer science, a kd-tree (short for k-dimensional tree) is a space-partitioning data structure 
// for organizing points in a k-dimensional space. kd-trees are a useful data structure for several 
// applications, such as searches involving a multidimensional search key (e.g. range searches and 
// nearest neighbour searches). kd-trees are a special case of BSP trees.

// A kd-tree uses only splitting planes that are perpendicular to one of the coordinate system axes. 
// This differs from BSP trees, in which arbitrary splitting planes can be used. 
// In addition, in the typical definition every node of a kd-tree, from the root to the leaves, stores a point.
// This differs from BSP trees, in which leaves are typically the only nodes that contain points 
// (or other geometric primitives). As a consequence, each splitting plane must go through one of
// the points in the kd-tree. kd-trees are a variant that store data only in leaf nodes. 

// 2. Constructing a classical kd-tree ( Pseudo code)

// Since there are many possible ways to choose axis-aligned splitting planes, there are many different ways 
// to construct kd-trees. The canonical method of kd-tree construction has the following constraints:

//     * As one moves down the tree, one cycles through the axes used to select the splitting planes. 
//      (For example, the root would have an x-aligned plane, the root's children would both have y-aligned 
//       planes, the root's grandchildren would all have z-aligned planes, and so on.)
//     * At each step, the point selected to create the splitting plane is the median of the points being 
//       put into the kd-tree, with respect to their coordinates in the axis being used. (Note the assumption
//       that we feed the entire set of points into the algorithm up-front.)

// This method leads to a balanced kd-tree, in which each leaf node is about the same distance from the root.
// However, balanced trees are not necessarily optimal for all applications. 
// The following pseudo-code illustrates this canonical construction procedure:

// function kdtree (list of points pointList, int depth)
// {
//     if pointList is empty
//         return nil;
//     else
//     {
//         // Select axis based on depth so that axis cycles through all valid values
//         var int axis := depth mod k;

//         // Sort point list and choose median as pivot element
//         select median from pointList;

//         // Create node and construct subtrees
//         var tree_node node;
//         node.location := median;
//         node.leftChild := kdtree(points in pointList before median, depth+1);
//         node.rightChild := kdtree(points in pointList after median, depth+1);
//         return node;
//     }
// }

// Our construction method is optimized to save memory, and differs a bit from the constraints above. 
// In particular, the division axis is chosen as the one with the biggest spread, and the point to create the
// splitting plane is chosen so, that one of the two subtrees contains exactly 2^k points and is a 
// perfectly balanced binary tree, and, while at the same time, trying to keep the number of points in the
// 2 subtrees as close as possible. The following section gives more details about our implementation.
//
// 3. TKDtree implementation details. 
//    3a. Order of nodes in the node information arrays:
//
// TKDtree is optimized to minimize memory consumption.
// Nodes of the TKDTree do not store pointers to the left and right children or to the parent node,
// but instead there are several 1-d arrays of size fNnodes with information about the nodes. 
// The order of the nodes information in the arrays is described below. It's important to understand
// it, if one's class needs to store some kind of additional information on the per node basis, for
// example, the fit function parameters.  
//
// Drawback:   Insertion to the TKDtree is not supported.  
// Advantage:  Random access is supported -  simple formulas  
//
// As noted above, the construction of the kd-tree involves choosing the axis and the point on 
// that axis to divide the remaining points approximately in half. The exact algorithm for choosing
// the division point is described in the next section. The sequence of divisions is
// recorded in the following arrays:
// fAxix[fNnodes]  - Division axis (0,1,2,3 ...)
// fValue[fNnodes] - Division value 
//
// Given the index of a node in those arrays, it's easy to find the indices, corresponding to 
// children nodes or the parent node:
// Suppose, the parent node is stored under the index inode. Then:
// Left child index  = inode*2+1
// Right child index =  (inode+1)*2  
// Suppose, that the child node is stored under the index inode. Then:
// Parent index = inode/2
// 
// Number of division nodes and number of terminals :
// fNnodes = (fNPoints/fBucketSize)
// 
// The nodes are filled always from left side to the right side:
// Let inode be the index of a node, and irow - the index of a row 
// The TKDTree looks the following way:
// Ideal case:
// Number of _terminal_ nodes = 2^N,  N=3
//
//            INode
// irow 0     0                                                                   -  1 inode 
// irow 1     1                              2                                    -  2 inodes
// irow 2     3              4               5               6                    -  4 inodes
// irow 3     7       8      9      10       11     12       13      14           -  8 inodes
//
//
// Non ideal case:
// Number of _terminal_ nodes = 2^N+k,  N=3  k=1

//           INode
// irow 0     0                                                                   - 1 inode
// irow 1     1                              2                                    - 2 inodes
// irow 2     3              4               5               6                    - 3 inodes
// irow 3     7       8      9      10       11     12       13      14           - 8 inodes
// irow 4     15  16                                                              - 2 inodes 
//
//
// 3b. The division algorithm:
// 
// As described above, the kd-tree is built by repeatingly dividing the given set of points into
// 2 smaller sets. The cut is made on the axis with the biggest spread, and the value on the axis,
// on which the cut is performed, is chosen based on the following formula: 
// Suppose, we want to divide n nodes into 2 groups, left and right. Then the left and right 
// will have the following number of nodes:
//
// n=2^k+rest
//
// Left  = 2^k-1 +  ((rest>2^k-2) ?  2^k-2      : rest)
// Right = 2^k-1 +  ((rest>2^k-2) ?  rest-2^k-2 : 0)
//
// For example, let n_nodes=67. Then, the closest 2^k=64, 2^k-1=32, 2^k-2=16. 
// Left node gets 32+3=35 sub-nodes, and the right node gets 32 sub-nodes
//
// The division process continues until all the nodes contain not more than a predefined number 
// of points.
//
// 3c. The order of nodes in boundary-related arrays
// 
// Some kd-tree based algorithms need to know the boundaries of each node. This information can
// be computed by calling the TKDTree::MakeBoundaries() function. It fills the following arrays:
//  
// fRange : array containing the boundaries of the domain:
// | 1st dimension (min + max) | 2nd dimension (min + max) | ...
// fBoundaries : nodes boundaries
// | 1st node {1st dim * 2 elements | 2nd dim * 2 elements | ...} | 2nd node {...} | ...
// The nodes are arranged in the order described in section 3a.
//////////////////////////////////////////////////////////////////////////


//_________________________________________________________________
template <typename  Index, typename Value>
TKDTree<Index, Value>::TKDTree() :
   TObject()
   ,fDataOwner(kFALSE)
   ,fNnodes(0)
   ,fNDim(0)
   ,fNDimm(0)
   ,fNPoints(0)
   ,fBucketSize(0)
   ,fAxis(0x0)
   ,fValue(0x0)
   ,fRange(0x0)
   ,fData(0x0)
   ,fBoundaries(0x0)
   ,fIndPoints(0x0)
   ,fRowT0(0)
   ,fCrossNode(0)
   ,fOffset(0)
   ,fkNNdim(0)
   ,fkNN(0x0)
   ,fkNNdist(0x0)
   ,fDistBuffer(0x0)
   ,fIndBuffer(0x0)
{
// Default constructor. Nothing is built
}

template <typename  Index, typename Value>
TKDTree<Index, Value>::TKDTree(Index npoints, Index ndim, UInt_t bsize) :
   TObject()
   ,fDataOwner(0)
   ,fNnodes(0)
   ,fNDim(ndim)
   ,fNDimm(2*ndim)
   ,fNPoints(npoints)
   ,fBucketSize(bsize)
   ,fAxis(0x0)
   ,fValue(0x0)
   ,fRange(0x0)
   ,fData(0x0)
   ,fBoundaries(0x0)
   ,fIndPoints(0x0)
   ,fRowT0(0)
   ,fCrossNode(0)
   ,fOffset(0)
   ,fkNNdim(0)
   ,fkNN(0x0)
   ,fkNNdist(0x0)
   ,fDistBuffer(0x0)
   ,fIndBuffer(0x0)
{
// Create the kd-tree of npoints from ndim-dimensional space. Parameter bsize stands for the
// maximal number of points in the terminal nodes (buckets). 
// Proceed by calling one of the SetData() functions and then the Build() function
// Note, that updating the tree with new data after the Build() function has been called is
// not possible!

}

//_________________________________________________________________
template <typename  Index, typename Value>
TKDTree<Index, Value>::TKDTree(Index npoints, Index ndim, UInt_t bsize, Value **data) :
   TObject()
   ,fDataOwner(0)
   ,fNnodes(0)
   ,fNDim(ndim)
   ,fNDimm(2*ndim)
   ,fNPoints(npoints)
   ,fBucketSize(bsize)
   ,fAxis(0x0)
   ,fValue(0x0)
   ,fRange(0x0)
   ,fData(data) //Columnwise!!!!!
   ,fBoundaries(0x0)
   ,fIndPoints(0x0)
   ,fRowT0(0)
   ,fCrossNode(0)
   ,fOffset(0)
   ,fkNNdim(0)
   ,fkNN(0x0)
   ,fkNNdist(0x0)
   ,fDistBuffer(0x0)
   ,fIndBuffer(0x0)
{
   // Build a kd-tree from the provided data array. 
   // Parameteres: 
   // - npoints - total number of points. Adding points after the tree is built is not supported
   // - ndim    - number of dimensions
   // - bsize   - maximal number of points in the terminal nodes
   // - data    - the data array
   //
   // The data should be placed columnwise (like in a TTree).
   // The columnwise orientation is chosen to simplify the usage together with TTree::GetV1() like functions.
   // An example of filling such an array for a 2d case:
   // Double_t **data = new Double_t*[2];
   // data[0] = new Double_t[npoints];
   // data[1] = new Double_t[npoints];
   // for (Int_t i=0; i<npoints; i++){
   //    data[0][i]=gRandom->Uniform(-1, 1); //fill the x-coordinate
   //    data[1][i]=gRandom->Uniform(-1, 1); //fill the y-coordinate
   // }
   //
   // By default, the kd-tree doesn't own the data. If you want the kd-tree to delete the data array, call
   // kdtree->SetOwner(kTRUE).
   //
   // After the kd-tree is built, the data is not needed anymore and can be deleted??????????????
   //
   // 

   
   Build();
}

//_________________________________________________________________
template <typename  Index, typename Value>
TKDTree<Index, Value>::~TKDTree()
{
   // Destructor
   
   if (fIndBuffer) delete [] fIndBuffer;
   if (fDistBuffer) delete [] fDistBuffer;
   if (fkNNdist) delete [] fkNNdist;
   if (fkNN) delete [] fkNN;
   if (fAxis) delete [] fAxis;
   if (fValue) delete [] fValue;
   if (fIndPoints) delete [] fIndPoints;
   if (fRange) delete [] fRange;
   if (fBoundaries) delete [] fBoundaries;
   if (fData) {
      if (fDataOwner==1){
         //the tree owns all the data
         for(int idim=0; idim<fNDim; idim++) delete [] fData[idim];
      }
      if (fDataOwner>0) {
         //the tree owns the array of pointers
         delete [] fData;
      }
   }
//   if (fDataOwner && fData){
//      for(int idim=0; idim<fNDim; idim++) delete [] fData[idim];
//      delete [] fData;
//   }
}


//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::Build(){
   //
   // Build binning
   //
   // 1. calculate number of nodes
   // 2. calculate first terminal row
   // 3. initialize index array
   // 4. non recursive building of the binary tree
   
   //
   // The tree is divided recursively. See class description, section 3b for the details
   // of the division alogrithm
   //
   //1.
   fNnodes = fNPoints/fBucketSize-1;
   if (fNPoints%fBucketSize) fNnodes++;
   
   //2.
   fRowT0=0;
   for (;(fNnodes+1)>(1<<fRowT0);fRowT0++);
   fRowT0-=1;
   //         2 = 2**0 + 1
   //         3 = 2**1 + 1
   //         4 = 2**1 + 2
   //         5 = 2**2 + 1
   //         6 = 2**2 + 2
   //         7 = 2**2 + 3
   //         8 = 2**2 + 4
   
   //3.
   // allocate space for boundaries
   fRange = new Value[2*fNDim];
   fIndPoints= new Index[fNPoints];
   for (Index i=0; i<fNPoints; i++) fIndPoints[i] = i;
   fAxis  = new UChar_t[fNnodes];
   fValue = new Value[fNnodes];
   //
   fCrossNode = (1<<(fRowT0+1))-1;
   if (fCrossNode<fNnodes) fCrossNode = 2*fCrossNode+1;
   //
   //  fOffset = (((fNnodes+1)-(1<<fRowT0)))*2;
   Int_t   over   = (fNnodes+1)-(1<<fRowT0);
   Int_t   filled = ((1<<fRowT0)-over)*fBucketSize;
   fOffset = fNPoints-filled;
   //
   // 	printf("Row0      %d\n", fRowT0);
   // 	printf("CrossNode %d\n", fCrossNode);
   // 	printf("Offset    %d\n", fOffset);
   //
   //
   //4.
   //    stack for non recursive build - size 128 bytes enough
   Int_t rowStack[128];
   Int_t nodeStack[128];
   Int_t npointStack[128];
   Int_t posStack[128];
   Int_t currentIndex = 0;
   Int_t iter =0;
   rowStack[0]    = 0;
   nodeStack[0]   = 0;
   npointStack[0] = fNPoints;
   posStack[0]   = 0;
   //
   Int_t nbucketsall =0;
   while (currentIndex>=0){
      iter++;
      //
      Int_t npoints  = npointStack[currentIndex];
      if (npoints<=fBucketSize) {
         //printf("terminal node : index %d iter %d\n", currentIndex, iter);
         currentIndex--;
         nbucketsall++;
         continue; // terminal node
      }
      Int_t crow     = rowStack[currentIndex];
      Int_t cpos     = posStack[currentIndex];
      Int_t cnode    = nodeStack[currentIndex];		
      //printf("currentIndex %d npoints %d node %d\n", currentIndex, npoints, cnode);
      //
      // divide points
      Int_t nbuckets0 = npoints/fBucketSize;           //current number of  buckets
      if (npoints%fBucketSize) nbuckets0++;            //
      Int_t restRows = fRowT0-rowStack[currentIndex];  // rest of fully occupied node row
      if (restRows<0) restRows =0;
      for (;nbuckets0>(2<<restRows); restRows++);
      Int_t nfull = 1<<restRows;
      Int_t nrest = nbuckets0-nfull;
      Int_t nleft =0, nright =0;
      //
      if (nrest>(nfull/2)){
         nleft  = nfull*fBucketSize;
         nright = npoints-nleft;
      }else{
         nright = nfull*fBucketSize/2;
         nleft  = npoints-nright;
      }
      
      //
      //find the axis with biggest spread
      Value maxspread=0;
      Value tempspread, min, max;
      Index axspread=0;
      Value *array;
      for (Int_t idim=0; idim<fNDim; idim++){
         array = fData[idim];
         Spread(npoints, array, fIndPoints+cpos, min, max);
         tempspread = max - min;
         if (maxspread < tempspread) {
            maxspread=tempspread;
            axspread = idim;
         }
         if(cnode) continue;
         //printf("set %d %6.3f %6.3f\n", idim, min, max);
         fRange[2*idim] = min; fRange[2*idim+1] = max;
      }
      array = fData[axspread];
      KOrdStat(npoints, array, nleft, fIndPoints+cpos);
      fAxis[cnode]  = axspread;
      fValue[cnode] = array[fIndPoints[cpos+nleft]];
      //printf("Set node %d : ax %d val %f\n", cnode, node->fAxis, node->fValue);
      //
      //
      npointStack[currentIndex] = nleft;
      rowStack[currentIndex]    = crow+1;
      posStack[currentIndex]    = cpos;
      nodeStack[currentIndex]   = cnode*2+1;
      currentIndex++;
      npointStack[currentIndex] = nright;
      rowStack[currentIndex]    = crow+1;
      posStack[currentIndex]    = cpos+nleft;
      nodeStack[currentIndex]   = (cnode*2)+2;
      //
      if (0){
         // consistency check
         Info("Build()", Form("points %d left %d right %d", npoints, nleft, nright));
         if (nleft<nright) Warning("Build", "Problem Left-Right");
         if (nleft<0 || nright<0) Warning("Build()", "Problem Negative number");
      }
   }    
}


//_________________________________________________________________
template <typename  Index, typename Value>
Bool_t	TKDTree<Index, Value>::FindNearestNeighbors(const Value *point, const Int_t k, Index *&in, Value *&d)
{
   // Find "k" nearest neighbors to "point".
   //
   // Return true in case of success and false in case of failure.
   // The indexes of the nearest k points are stored in the array "in" in
   // increasing order of the distance to "point" and the maxim distance
   // in "d".
   //
   // The arrays "in" and "d" are managed internally by the TKDTree.
   
   Index inode = FindNode(point);
   if(inode < fNnodes){
      Error("FindNearestNeighbors()", "Point outside data range.");
      return kFALSE;
   }
   
   UInt_t debug = 0;
   Int_t nCalculations = 0;
   
   // allocate working memory
   if(!fDistBuffer){
      fDistBuffer = new Value[fBucketSize];
      fIndBuffer  = new Index[fBucketSize];
   }
   if(fkNNdim < k){
      if(debug>=1){
         if(fkNN) printf("Reallocate memory %d -> %d \n", fkNNdim, 2*k);
         else printf("Allocate %d memory\n", 2*k);
      }
      fkNNdim  = 2*k;
      if(fkNN){
         delete [] fkNN; 
         delete [] fkNNdist;
      }
      fkNN     = new Index[fkNNdim];
      fkNNdist = new Value[fkNNdim];
   }
   memset(fkNN, -1, k*sizeof(Index));
   for(int i=0; i<k; i++) fkNNdist[i] = 9999.;
   Index itmp, jtmp; Value ftmp, gtmp;
   
   // calculate number of boundaries with the data domain.	
   if(!fBoundaries) MakeBoundaries();
   Value *bounds = &fBoundaries[inode*2*fNDim];
   Index nBounds = 0;
   for(int idim=0; idim<fNDim; idim++){
      if((bounds[2*idim] - fRange[2*idim]) < 1.E-10) nBounds++;
      if((bounds[2*idim+1] - fRange[2*idim+1]) < 1.E-10) nBounds++;
   }
   if(debug>=1) printf("Calculate boundaries [nBounds = %d]\n", nBounds);
   
   
   // traverse tree
   UChar_t ax;
   Value   val, dif;
   Int_t nAllNodes = fNnodes + fNPoints/fBucketSize + ((fNPoints%fBucketSize)?1:0);
   Int_t nodeStack[128], nodeIn[128];
   Index currentIndex = 0;
   nodeStack[0]   = inode;
   nodeIn[0] = inode;
   while(currentIndex>=0){
      Int_t tnode = nodeStack[currentIndex];
      Int_t entry = nodeIn[currentIndex];
      currentIndex--;
      if(debug>=1) printf("Analyse tnode %d entry %d\n", tnode, entry);
      
      // check if node is still eligible
      Int_t inode = (tnode-1)>>1; //tnode/2 + (tnode%2) - 1;
      ax = fAxis[inode];
      val = fValue[inode];
      dif = point[ax] - val;
      if((TMath::Abs(dif) > fkNNdist[k-1]) &&
         ((dif > 0. && (tnode&1)) || (dif < 0. && !(tnode&1)))) {
         if(debug>=1) printf("\tremove %d\n", (tnode-1)>>1);
         
         // mark bound
         nBounds++;
         // end all recursions
         if(nBounds==2 * fNDim) break;
         continue;
      }
      
      if(IsTerminal(tnode)){
         if(debug>=2) printf("\tProcess terminal node %d\n", tnode);
         // Link data to terminal node
         Int_t offset = (tnode >= fCrossNode) ? (tnode-fCrossNode)*fBucketSize : fOffset+(tnode-fNnodes)*fBucketSize;
         Index *indexPoints = &fIndPoints[offset];
         Int_t nbs = (tnode == 2*fNnodes) ? fNPoints%fBucketSize : fBucketSize;
         nbs = nbs ? nbs : fBucketSize;
         nCalculations += nbs;
         
         Int_t npoints = 0;
         for(int idx=0; idx<nbs; idx++){
            // calculate distance in the L1 metric
            fDistBuffer[npoints] = 0.;
            for(int idim=0; idim<fNDim; idim++) fDistBuffer[npoints] += TMath::Abs(point[idim] - fData[idim][indexPoints[idx]]);
            // register candidate neighbor
            if(fDistBuffer[npoints] < fkNNdist[k-1]){
               fIndBuffer[npoints] = indexPoints[idx];
               npoints++;
            }
         }
         for(int ibrowse=0; ibrowse<npoints; ibrowse++){
            if(fDistBuffer[ibrowse] >= fkNNdist[k-1]) continue;
            //insert neighbor
            int iNN=0;
            while(fDistBuffer[ibrowse] >= fkNNdist[iNN]) iNN++;
            if(debug>=2) printf("\t\tinsert data %d @ %d distance %f\n", fIndBuffer[ibrowse], iNN, fDistBuffer[ibrowse]);
            
            itmp = fkNN[iNN]; ftmp = fkNNdist[iNN];
            fkNN[iNN]     = fIndBuffer[ibrowse];
            fkNNdist[iNN] = fDistBuffer[ibrowse];
            for(int ireplace=iNN+1; ireplace<k; ireplace++){
               jtmp = fkNN[ireplace]; gtmp = fkNNdist[ireplace];
               fkNN[ireplace]     = itmp; fkNNdist[ireplace] = ftmp;
               itmp = jtmp; ftmp = gtmp;
               if(ftmp == 9999.) break;
            }
            if(debug>=3){
               for(int i=0; i<k; i++){
                  if(fkNNdist[i] == 9999.) break;
                  printf("\t\tfkNNdist[%d] = %f\n", i, fkNNdist[i]);
               }
            }				
         }
      }
      
      // register parent
      Int_t pn = (tnode-1)>>1;//tnode/2 + (tnode%2) - 1;
      if(pn >= 0 && entry != pn){
         // check if parent node is eligible at all
         if(TMath::Abs(point[fAxis[pn]] - fValue[pn]) > fkNNdist[k-1]){
            // mark bound
            nBounds++;
            // end all recursions
            if(nBounds==2 * fNDim) break;
         }
         currentIndex++;
         nodeStack[currentIndex]=pn;
         nodeIn[currentIndex]=tnode;
         if(debug>=2) printf("\tregister %d\n", nodeStack[currentIndex]);
      }
      if(IsTerminal(tnode)) continue;
      
      // register children nodes
      Int_t cn;
      Bool_t kAllow[] = {kTRUE, kTRUE};
      ax = fAxis[tnode];
      val = fValue[tnode];
      if(TMath::Abs(point[ax] - val) > fkNNdist[k-1]){
         if(point[ax] > val) kAllow[0] = kFALSE;
         else kAllow[1] = kFALSE;
      }
      for(int ic=1; ic<=2; ic++){
         if(!kAllow[ic-1]) continue;
         cn = (tnode<<1)+ic;
         if(cn < nAllNodes && entry != cn){
            currentIndex++;
            nodeStack[currentIndex] = cn;
            nodeIn[currentIndex]=tnode;
            if(debug>=2) printf("\tregister %d\n", nodeStack[currentIndex]);
         }
      }		
   }
   // save results
   in = fkNN;
   d  = fkNNdist;  
   return kTRUE;
}



//_________________________________________________________________
template <typename  Index, typename Value>
Index TKDTree<Index, Value>::FindNode(const Value * point)
{
   // returns the index of the terminal node to which point belongs
   // (index in the fAxis, fValue, etc arrays)
   // returns -1 in case of failure

   Index stackNode[128], inode;
   Int_t currentIndex =0;
   stackNode[0] = 0;
   while (currentIndex>=0){
      inode    = stackNode[currentIndex];
      if (IsTerminal(inode)) return inode;
      
      currentIndex--;
      if (point[fAxis[inode]]<=fValue[inode]){
         currentIndex++;
         stackNode[currentIndex]=(inode<<1)+1;
      }
      if (point[fAxis[inode]]>=fValue[inode]){
         currentIndex++;
         stackNode[currentIndex]=(inode+1)<<1;
      }
   }
   
   return -1;
}



//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::FindPoint(Value * point, Index &index, Int_t &iter){
  //
  // find the index of point
  // works only if we keep fData pointers
  
  Int_t stackNode[128];
  Int_t currentIndex =0;
  stackNode[0] = 0;
  iter =0;
  //
  while (currentIndex>=0){
    iter++;
    Int_t inode    = stackNode[currentIndex];
    currentIndex--;
    if (IsTerminal(inode)){
      // investigate terminal node
      Int_t indexIP  = (inode >= fCrossNode) ? (inode-fCrossNode)*fBucketSize : (inode-fNnodes)*fBucketSize+fOffset;
      printf("terminal %d indexP %d\n", inode, indexIP);
      for (Int_t ibucket=0;ibucket<fBucketSize;ibucket++){
	Bool_t isOK    = kTRUE;
	indexIP+=ibucket;
	printf("ibucket %d index %d\n", ibucket, indexIP);
	if (indexIP>=fNPoints) continue;
	Int_t index0   = fIndPoints[indexIP];
	for (Int_t idim=0;idim<fNDim;idim++) if (fData[idim][index0]!=point[idim]) isOK = kFALSE;
	if (isOK) index = index0;
      }
      continue;
    }
    
    if (point[fAxis[inode]]<=fValue[inode]){
      currentIndex++;
      stackNode[currentIndex]=(inode*2)+1;
    }
    if (point[fAxis[inode]]>=fValue[inode]){
      currentIndex++;
      stackNode[currentIndex]=(inode*2)+2;
    }
  }
  //
  //  printf("Iter\t%d\n",iter);
}

//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::FindInRangeA(Value * point, Value * delta, Index *res , Index &npoints, Index & iter, Int_t bnode)
{
 //
  // Find all points in the range specified by    (point +- range)
  // res     - Resulting indexes are stored in res array 
  // npoints - Number of selected indexes in range
  // NOTICE:
  // For some cases it is better to not keep the original data - because of memory consumption
  // If the data are not kept - only boundary conditions are investigated
  // In that case, if the size of buckets(terminal nodes) is > 1, some of the returned data points can be 
  // outside the range
  // What is guranteed in this mode: All of the points in the range are selected + some fraction of others (but close)

   Index stackNode[128];
   iter=0;
   Index currentIndex = 0;
   stackNode[currentIndex] = bnode; 
   while (currentIndex>=0){
      iter++;
      Int_t inode    = stackNode[currentIndex];
      //
      currentIndex--;
      if (!IsTerminal(inode)){
         // not terminal
         //TKDNode<Index, Value> * node = &(fNodes[inode]);
         if (point[fAxis[inode]] - delta[fAxis[inode]] <  fValue[inode]) {
            currentIndex++; 
            stackNode[currentIndex]= (inode*2)+1;
         }
         if (point[fAxis[inode]] + delta[fAxis[inode]] >= fValue[inode]){
            currentIndex++; 
            stackNode[currentIndex]= (inode*2)+2;
         }
      }else{
         Int_t indexIP  = 
            (inode >= fCrossNode) ? (inode-fCrossNode)*fBucketSize : (inode-fNnodes)*fBucketSize+fOffset;
         for (Int_t ibucket=0;ibucket<fBucketSize;ibucket++){
            if (indexIP+ibucket>=fNPoints) break;
            res[npoints]   = fIndPoints[indexIP+ibucket];
            npoints++;
         }
      }
   }
   if (fData){
      //
      //  compress rest if data still accesible
      //
      Index npoints2 = npoints;
      npoints=0;
      for (Index i=0; i<npoints2;i++){
         Bool_t isOK = kTRUE;
         for (Index idim = 0; idim< fNDim; idim++){
            if (TMath::Abs(fData[idim][res[i]]- point[idim])>delta[idim]) 
               isOK = kFALSE;	
         }
         if (isOK){
            res[npoints] = res[i];
            npoints++;
         }
      }    
   }
}


//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::FindInRangeB(Value * point, Value * delta, Index *res , Index &npoints,Index & iter, Int_t bnode)
{
   //
   Long64_t  goldStatus = (1<<(2*fNDim))-1;  // gold status
   Index stackNode[128];
   Long64_t   stackStatus[128]; 
   iter=0;
   Index currentIndex   = 0;
   stackNode[currentIndex]   = bnode; 
   stackStatus[currentIndex] = 0;
   while (currentIndex>=0){
      Int_t inode     = stackNode[currentIndex];
      Long64_t status = stackStatus[currentIndex];
      currentIndex--;
      iter++;
      if (IsTerminal(inode)){
         Int_t indexIP  = 
            (inode >= fCrossNode) ? (inode-fCrossNode)*fBucketSize : (inode-fNnodes)*fBucketSize+fOffset;
         for (Int_t ibucket=0;ibucket<fBucketSize;ibucket++){
            if (indexIP+ibucket>=fNPoints) break;
            res[npoints]   = fIndPoints[indexIP+ibucket];
            npoints++;
         }
         continue;
      }      
      // not terminal    
      if (status == goldStatus){
         Int_t ileft  = inode;
         Int_t iright = inode;
         for (;ileft<fNnodes; ileft   = (ileft<<1)+1);
         for (;iright<fNnodes; iright = (iright<<1)+2);
         Int_t indexL  = 
            (ileft >= fCrossNode) ? (ileft-fCrossNode)*fBucketSize : (ileft-fNnodes)*fBucketSize+fOffset;
         Int_t indexR  = 
            (iright >= fCrossNode) ? (iright-fCrossNode)*fBucketSize : (iright-fNnodes)*fBucketSize+fOffset;
         if (indexL<=indexR){
            Int_t endpoint = indexR+fBucketSize;
            if (endpoint>fNPoints) endpoint=fNPoints;
            for (Int_t ipoint=indexL;ipoint<endpoint;ipoint++){
               res[npoints]   = fIndPoints[ipoint];
               npoints++;
            }	  
            continue;
         }
      }
      //
      // TKDNode<Index, Value> * node = &(fNodes[inode]);
      if (point[fAxis[inode]] - delta[fAxis[inode]] <  fValue[inode]) {
         currentIndex++; 
         stackNode[currentIndex]= (inode<<1)+1;
         if (point[fAxis[inode]] + delta[fAxis[inode]] > fValue[inode])
            stackStatus[currentIndex]= status | (1<<(2*fAxis[inode]));
      }
      if (point[fAxis[inode]] + delta[fAxis[inode]] >= fValue[inode]){
         currentIndex++; 
         stackNode[currentIndex]= (inode<<1)+2;
         if (point[fAxis[inode]] - delta[fAxis[inode]]<fValue[inode])
            stackStatus[currentIndex]= status | (1<<(2*fAxis[inode]+1));
      }
   }
   if (fData){
      // compress rest
      Index npoints2 = npoints;
      npoints=0;
      for (Index i=0; i<npoints2;i++){
         Bool_t isOK = kTRUE;
         for (Index idim = 0; idim< fNDim; idim++){
            if (TMath::Abs(fData[idim][res[i]]- point[idim])>delta[idim]) 
               isOK = kFALSE;	
         }
         if (isOK){
            res[npoints] = res[i];
            npoints++;
         }
      }    
   }
}

//_________________________________________________________________
template <typename Index, typename Value>
Index*  TKDTree<Index, Value>::GetPointsIndexes(Int_t node) const 
{
   //return the indices of the points in that terminal node

   if(node < fNnodes) return 0x0;
   Int_t offset = (node >= fCrossNode) ? (node-fCrossNode)*fBucketSize : fOffset+(node-fNnodes)*fBucketSize;
   return &fIndPoints[offset];
}

//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::SetData(Index npoints, Index ndim, UInt_t bsize, Value **data)
{
// Set the data array. See the constructor function comments for details

// TO DO
// 
// Check reconstruction/reallocation of memory of data. Maybe it is not
// necessary to delete and realocate space but only to use the same space
   
   Clear();
   
   //Columnwise!!!!
   fData = data;
   fNPoints = npoints;
   fNDim = ndim;
   fBucketSize = bsize;
   
   Build();
}

//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::SetData(Index idim, Value *data)
{
   //Set the coordinate #ndim of all points (the column #ndim of the data matrix)
   //After setting all the data columns, proceed by calling Build() function
   //Note, that calling this function after Build() is not possible
   //Note also, that no checks on the array sizes is performed anywhere

   if (fAxis || fValue) {
      Error("SetData", "The tree has already been built, no updates possible");
      return;
   }

   if (!fData) {
      fData = new Value*[fNDim];
   }
   fData[idim]=data;
   fDataOwner = 2;
}


//_________________________________________________________________
template <typename  Index, typename Value>
void TKDTree<Index, Value>::Spread(Index ntotal, Value *a, Index *index, Value &min, Value &max) const
{
   //Calculate spread of the array a
   
   Index i;
   min = a[index[0]];
   max = a[index[0]];
   for (i=0; i<ntotal; i++){
      if (a[index[i]]<min) min = a[index[i]];
      if (a[index[i]]>max) max = a[index[i]];
   }
}


//_________________________________________________________________
template <typename  Index, typename Value>
Value TKDTree<Index, Value>::KOrdStat(Index ntotal, Value *a, Index k, Index *index) const
{
   //
   //copy of the TMath::KOrdStat because I need an Index work array
   
   Index i, ir, j, l, mid;
   Index arr;
   Index temp;
   
   Index rk = k;
   l=0;
   ir = ntotal-1;
   for(;;) {
      if (ir<=l+1) { //active partition contains 1 or 2 elements
         if (ir == l+1 && a[index[ir]]<a[index[l]])
         {temp = index[l]; index[l]=index[ir]; index[ir]=temp;}
         Value tmp = a[index[rk]];
         return tmp;
      } else {
         mid = (l+ir) >> 1; //choose median of left, center and right
         {temp = index[mid]; index[mid]=index[l+1]; index[l+1]=temp;}//elements as partitioning element arr.
          if (a[index[l]]>a[index[ir]])  //also rearrange so that a[l]<=a[l+1]
          {temp = index[l]; index[l]=index[ir]; index[ir]=temp;}
          
          if (a[index[l+1]]>a[index[ir]])
          {temp=index[l+1]; index[l+1]=index[ir]; index[ir]=temp;}
          
          if (a[index[l]]>a[index[l+1]])
          {temp = index[l]; index[l]=index[l+1]; index[l+1]=temp;}
          
          i=l+1;        //initialize pointers for partitioning
          j=ir;
          arr = index[l+1];
          for (;;) {
             do i++; while (a[index[i]]<a[arr]);
             do j--; while (a[index[j]]>a[arr]);
             if (j<i) break;  //pointers crossed, partitioning complete
             {temp=index[i]; index[i]=index[j]; index[j]=temp;}
          }
          index[l+1]=index[j];
          index[j]=arr;
          if (j>=rk) ir = j-1; //keep active the partition that
          if (j<=rk) l=i;      //contains the k_th element
      }
   }
}

//_________________________________________________________________
template <typename Index, typename Value>
void TKDTree<Index, Value>::MakeBoundaries(Value *range)
{
// Build boundaries for each node.
   
   
   if(range) memcpy(fRange, range, fNDimm*sizeof(Value));
   // total number of nodes including terminal nodes
   Int_t totNodes = fNnodes + fNPoints/fBucketSize + ((fNPoints%fBucketSize)?1:0);
   fBoundaries = new Value[totNodes*fNDimm];
   //Info("MakeBoundaries(Value*)", Form("Allocate boundaries for %d nodes", totNodes));
   
   
   // loop
   Value *tbounds = 0x0, *cbounds = 0x0;
   Int_t cn;
   for(int inode=fNnodes-1; inode>=0; inode--){
      tbounds = &fBoundaries[inode*fNDimm];
      memcpy(tbounds, fRange, fNDimm*sizeof(Value));
      
      // check left child node
      cn = (inode<<1)+1;
      if(IsTerminal(cn)) CookBoundaries(inode, kTRUE);
      cbounds = &fBoundaries[fNDimm*cn];
      for(int idim=0; idim<fNDim; idim++) tbounds[idim<<1] = cbounds[idim<<1];
      
      // check right child node
      cn = (inode+1)<<1;
      if(IsTerminal(cn)) CookBoundaries(inode, kFALSE);
      cbounds = &fBoundaries[fNDimm*cn];
      for(int idim=0; idim<fNDim; idim++) tbounds[(idim<<1)+1] = cbounds[(idim<<1)+1];
   }
}

//_________________________________________________________________
template <typename Index, typename Value>
void TKDTree<Index, Value>::CookBoundaries(const Int_t node, Bool_t LEFT)
{
   // define index of this terminal node
   Int_t index = (node<<1) + (LEFT ? 1 : 2);
   //Info("CookBoundaries()", Form("Node %d", index));
   
   // build and initialize boundaries for this node
   Value *tbounds = &fBoundaries[fNDimm*index];
   memcpy(tbounds, fRange, fNDimm*sizeof(Value));
   Bool_t flag[256];  // cope with up to 128 dimensions 
   memset(flag, kFALSE, fNDimm);
   Int_t nvals = 0;
   
   // recurse parent nodes
   Int_t pn = node;
   while(pn >= 0 && nvals < fNDimm){
      if(LEFT){
         index = (fAxis[pn]<<1)+1;
         if(!flag[index]) {
            tbounds[index] = fValue[pn];
            flag[index] = kTRUE;
            nvals++;
         }
      } else {
         index = fAxis[pn]<<1;
         if(!flag[index]) {
            tbounds[index] = fValue[pn];
            flag[index] = kTRUE;
            nvals++;
         }
      }
      LEFT = pn&1;
      pn =  (pn - 1)>>1;
   }
}



//______________________________________________________________________
TKDTreeIF *TKDTreeTestBuild(const Int_t npoints, const Int_t bsize){  
   //
   // Example function to  
   //
   Float_t *data0 =  new Float_t[npoints*2];
   Float_t *data[2];
   data[0] = &data0[0];
   data[1] = &data0[npoints];
   for (Int_t i=0;i<npoints;i++) {
      data[1][i]= gRandom->Rndm();
      data[0][i]= gRandom->Rndm();
   }
   TKDTree<Int_t, Float_t> *kdtree = new TKDTreeIF(npoints, 2, bsize, data);
   return kdtree;	
}



template class TKDTree<Int_t, Float_t>;
template class TKDTree<Int_t, Double_t>;

