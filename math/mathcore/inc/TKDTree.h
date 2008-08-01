#ifndef ROOT_TKDTree
#define ROOT_TKDTree

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#include "TMath.h"


template <typename Index, typename Value> class TKDTree : public TObject
{
public:
	
   TKDTree();
   TKDTree(Index npoints, Index ndim, UInt_t bsize);
   TKDTree(Index npoints, Index ndim, UInt_t bsize, Value **data);
   ~TKDTree();
   
   void            Build();  // build the tree
   
   // Get indexes of left and right daughter nodes
   Int_t GetLeft(Int_t inode)  const    {return inode*2+1;}
   Int_t GetRight(Int_t inode) const    {return (inode+1)*2;}
   Int_t GetParent(Int_t inode) const  {return inode/2;}
   //  
   // Other getters
   Index*  GetPointsIndexes(Int_t node) const;
   UChar_t GetNodeAxis(Int_t id) const {return (id < 0 || id >= fNNodes) ? 0 : fAxis[id];}
   Value   GetNodeValue(Int_t id) const {return (id < 0 || id >= fNNodes) ? 0 : fValue[id];}
   Int_t   GetNNodes() const {return fNNodes;}
   Value*  GetBoundaries();
   Value*  GetBoundariesExact();
   Value*  GetBoundary(const Int_t node);
   Value*  GetBoundaryExact(const Int_t node);
   Index   GetNPoints() { return fNPoints; }
   Index   GetNDim()    { return fNDim; }
   Index   GetNPointsNode(Int_t node) const;

   //Getters for internal variables.
   Int_t   GetRowT0() {return fRowT0;}      //! smallest terminal row
   Int_t   GetCrossNode() {return fCrossNode;}  //! cross node
   Int_t   GetOffset() {return fOffset;}     //! offset in fIndPoints
   Index*  GetIndPoints() {return fIndPoints;}
   Index   GetBucketSize() {return fBucketSize;}

   Bool_t  FindNearestNeighbors(const Value *point, const Int_t kNN, Index *&i, Value *&d);
   Index   FindNode(const Value * point);
   void    FindPoint(Value * point, Index &index, Int_t &iter);
   void    FindInRangeA(Value * point, Value * delta, Index *res , Index &npoints,Index & iter, Int_t bnode);
   void    FindInRangeB(Value * point, Value * delta, Index *res , Index &npoints,Index & iter, Int_t bnode);
   void    FindBNodeA(Value * point, Value * delta, Int_t &inode);
   Bool_t  IsTerminal(Index inode) const {return (inode>=fNNodes);}
   Int_t   IsOwner() { return fDataOwner; }
   Value   KOrdStat(Index ntotal, Value *a, Index k, Index *index) const;
   void    MakeBoundaries(Value *range = 0x0);
   void    MakeBoundariesExact();
   void    SetData(Index npoints, Index ndim, UInt_t bsize, Value **data);
   Int_t   SetData(Index idim, Value *data);
   void    SetOwner(Int_t owner) { fDataOwner = owner; }
   void    Spread(Index ntotal, Value *a, Index *index, Value &min, Value &max) const;
   
 private:
   TKDTree(const TKDTree &); // not implemented
   TKDTree<Index, Value>& operator=(const TKDTree<Index, Value>&); // not implemented
   void CookBoundaries(const Int_t node, Bool_t left);
   
   
 protected:
   Int_t   fDataOwner;  //! 0 - not owner, 2 - owner of the pointer array, 1 - owner of the whole 2-d array
   Int_t   fNNodes;     // size of node array
   Index   fNDim;       // number of dimensions
   Index   fNDimm;      // dummy 2*fNDim
   Index   fNPoints;    // number of multidimensional points
   Index   fBucketSize; // size of the terminal nodes
   UChar_t *fAxis;      //[fNNodes] nodes cutting axis
   Value   *fValue;     //[fNNodes] nodes cutting value
   //
   Value   *fRange;     //[fNDimm] range of data for each dimension
   Value   **fData;     //! data points
   Value   *fBoundaries;//! nodes boundaries


   Index   *fIndPoints; //! array of points indexes
   Int_t   fRowT0;      //! smallest terminal row - first row that contains terminal nodes
   Int_t   fCrossNode;  //! cross node - node that begins the last row (with terminal nodes only)
   Int_t   fOffset;     //! offset in fIndPoints - if there are 2 rows, that contain terminal nodes
                        //  fOffset returns the index in the fIndPoints array of the first point
                        //  that belongs to the first node on the second row.

   // kNN related data
   Int_t   fkNNdim;     //! current kNN arrays allocated dimension
   Index   *fkNN;       //! k nearest neighbors indexes
   Value   *fkNNdist;   //! k nearest neighbors distances
   Value   *fDistBuffer;//! working space for kNN
   Index   *fIndBuffer; //! working space for kNN

   ClassDef(TKDTree, 1)  // KD tree
};
      
      
typedef TKDTree<Int_t, Double_t> TKDTreeID;
typedef TKDTree<Int_t, Float_t> TKDTreeIF;

// Test functions:
TKDTreeIF *  TKDTreeTestBuild();

#endif

