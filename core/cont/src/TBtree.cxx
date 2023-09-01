// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TBtree
\ingroup Containers
B-tree class. TBtree inherits from the TSeqCollection ABC.

## B-tree Implementation notes

This implements B-trees with several refinements. Most of them can be found
in Knuth Vol 3, but some were developed to adapt to restrictions imposed
by C++. First, a restatement of Knuth's properties that a B-tree must
satisfy, assuming we make the enhancement he suggests in the paragraph
at the bottom of page 476. Instead of storing null pointers to non-existent
nodes (which Knuth calls the leaves) we utilize the space to store keys.
Therefore, what Knuth calls level (l-1) is the bottom of our tree, and
we call the nodes at this level LeafNodes. Other nodes are called InnerNodes.
The other enhancement we have adopted is in the paragraph at the bottom of
page 477: overflow control.

The following are modifications of Knuth's properties on page 478:

  1.  Every InnerNode has at most Order keys, and at most Order+1 sub-trees.
  2.  Every LeafNode has at most 2*(Order+1) keys.
  3.  An InnerNode with k keys has k+1 sub-trees.
  4.  Every InnerNode that is not the root has at least InnerLowWaterMark keys.
  5.  Every LeafNode that is not the root has at least LeafLowWaterMark keys.
  6.  If the root is a LeafNode, it has at least one key.
  7.  If the root is an InnerNode, it has at least one key and two sub-trees.
  8.  All LeafNodes are the same distance from the root as all the other
      LeafNodes.
  9.  For InnerNode n with key n[i].key, then sub-tree n[i-1].tree contains
      all keys < n[i].key, and sub-tree n[i].tree contains all keys
      >= n[i].key.
  10. Order is at least 3.

The values of InnerLowWaterMark and LeafLowWaterMark may actually be set
by the user when the tree is initialized, but currently they are set
automatically to:
~~~ {.cpp}
        InnerLowWaterMark = ceiling(Order/2)
        LeafLowWaterMark  = Order - 1
~~~
If the tree is only filled, then all the nodes will be at least 2/3 full.
They will almost all be exactly 2/3 full if the elements are added to the
tree in order (either increasing or decreasing). [Knuth says McCreight's
experiments showed almost 100% memory utilization. I don't see how that
can be given the algorithms that Knuth gives. McCreight must have used
a different scheme for balancing. [No, he used a different scheme for
splitting: he did a two-way split instead of the three way split as we do
here. Which means that McCreight does better on insertion of ordered data,
but we should do better on insertion of random data.]]

It must also be noted that B-trees were designed for DISK access algorithms,
not necessarily in-memory sorting, as we intend it to be used here. However,
if the order is kept small (< 6?) any inefficiency is negligible for
in-memory sorting. Knuth points out that balanced trees are actually
preferable for memory sorting. I'm not sure that I believe this, but
it's interesting. Also, deleting elements from balanced binary trees, being
beyond the scope of Knuth's book (p. 465), is beyond my scope. B-trees
are good enough.

A B-tree is declared to be of a certain ORDER (3 by default). This number
determines the number of keys contained in any interior node of the tree.
Each interior node will contain ORDER keys, and therefore ORDER+1 pointers
to sub-trees. The keys are numbered and indexed 1 to ORDER while the
pointers are numbered and indexed 0 to ORDER. The 0th ptr points to the
sub-tree of all elements that are less than key[1]. Ptr[1] points to the
sub-tree that contains all the elements greater than key[1] and less than
key[2]. etc. The array of pointers and keys is allocated as ORDER+1
pairs of keys and nodes, meaning that one key field (key[0]) is not used
and therefore wasted.  Given that the number of interior nodes is
small, that this waste allows fewer cases of special code, and that it
is useful in certain of the methods, it was felt to be a worthwhile waste.

The size of the exterior nodes (leaf nodes) does not need to be related to
the size of the interior nodes at all. Since leaf nodes contain only
keys, they may be as large or small as we like independent of the size
of the interior nodes. For no particular reason other than it seems like
a good idea, we will allocate 2*(ORDER+1) keys in each leaf node, and they
will be numbered and indexed from 0 to 2*ORDER+1. It does have the advantage
of keeping the size of the leaf and interior arrays the same, so that if we
find allocation and de-allocation of these arrays expensive, we can modify
their allocation to use a garbage ring, or something.

Both of these numbers will be run-time constants associated with each tree
(each tree at run-time can be of a different order). The variable "order"
is the order of the tree, and the inclusive upper limit on the indices of
the keys in the interior nodes.  The variable "order2" is the inclusive
upper limit on the indices of the leaf nodes, and is designed
~~~ {.cpp}
    (1) to keep the sizes of the two kinds of nodes the same;
    (2) to keep the expressions involving the arrays of keys looking
        somewhat the same:   lower limit        upper limit
          for inner nodes:        1                order
          for leaf  nodes:        0                order2
        Remember that index 0 of the inner nodes is special.
~~~
Currently, order2 = 2*(order+1).
~~~ {.cpp}
 Picture: (also see Knuth Vol 3 pg 478)

           +--+--+--+--+--+--...
           |  |  |  |  |  |
 parent--->|  |     |     |
           |  |     |     |
           +*-+*-+*-+--+--+--...
            |  |  |
       +----+  |  +-----+
       |       +-----+  |
       V             |  V
       +----------+  |  +----------+
       |          |  |  |          |
 this->|          |  |  |          |<--sib
       +----------+  |  +----------+
                     V
                    data
~~~
It is conceptually VERY convenient to think of the data as being the
very first element of the sib node. Any primitive that tells sib to
perform some action on n nodes should include this 'hidden' element.
For InnerNodes, the hidden element has (physical) index 0 in the array,
and in LeafNodes, the hidden element has (virtual) index -1 in the array.
Therefore, there are two 'size' primitives for nodes:
~~~ {.cpp}
Psize       - the physical size: how many elements are contained in the
              array in the node.
Vsize       - the 'virtual' size; if the node is pointed to by
              element 0 of the parent node, then Vsize == Psize;
              otherwise the element in the parent item that points to this
              node 'belongs' to this node, and Vsize == Psize+1;
~~~
Parent nodes are always InnerNodes.

These are the primitive operations on Nodes:
~~~ {.cpp}
Append(elt)     - adds an element to the end of the array of elements in a
                  node.  It must never be called where appending the element
                  would fill the node.
Split()         - divide a node in two, and create two new nodes.
SplitWith(sib)  - create a third node between this node and the sib node,
                  divvying up the elements of their arrays.
PushLeft(n)     - move n elements into the left sibling
PushRight(n)    - move n elements into the right sibling
BalanceWithRight() - even up the number of elements in the two nodes.
BalanceWithLeft()  - ditto
~~~
To allow this implementation of btrees to also be an implementation of
sorted arrays/lists, the overhead is included to allow O(log n) access
of elements by their rank (`give me the 5th largest element').
Therefore, each Item keeps track of the number of keys in and below it
in the tree (remember, each item's tree is all keys to the RIGHT of the
item's own key).
~~~ {.cpp}
[ [ < 0 1 2 3 > 4 < 5 6 7 > 8 < 9 10 11 12 > ] 13 [ < 14 15 16 > 17 < 18 19 20 > ] ]
   4  1 1 1 1   4   1 1 1   5   1  1  1  1      7  3   1  1  1    4    1  1  1
~~~
*/

#include "TBtree.h"
#include "TBuffer.h"
#include "TObject.h"

#include <stdlib.h>


ClassImp(TBtree);

////////////////////////////////////////////////////////////////////////////////
/// Create a B-tree of certain order (by default 3).

TBtree::TBtree(int order)
{
   Init(order);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete B-tree. Objects are not deleted unless the TBtree is the
/// owner (set via SetOwner()).

TBtree::~TBtree()
{
   if (fRoot) {
      Clear();
      SafeDelete(fRoot);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add object to B-tree.

void TBtree::Add(TObject *obj)
{
   if (IsArgNull("Add", obj)) return;
   if (!obj->IsSortable()) {
      Error("Add", "object must be sortable");
      return;
   }
   if (!fRoot) {
      fRoot = new TBtLeafNode(nullptr, obj, this);
      R__CHECK(fRoot != nullptr);
      IncrNofKeys();
   } else {
      TBtNode *loc;
      Int_t idx;
      if (fRoot->Found(obj, &loc, &idx)) {
         // loc and idx are set to either where the object
         // was found, or where it should go in the Btree.
         // Nothing is here now, but later we might give the user
         // the ability to declare a B-tree as `unique elements only',
         // in which case we would handle an exception here.
      }
      loc->Add(obj, idx);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Cannot use this method since B-tree decides order.

TObject *TBtree::After(const TObject *) const
{
   MayNotUse("After");
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// May not use this method since B-tree decides order.

TObject *TBtree::Before(const TObject *) const
{
   MayNotUse("Before");
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from B-tree. Does NOT delete objects unless the TBtree
/// is the owner (set via SetOwner()).

void TBtree::Clear(Option_t *)
{
   if (IsOwner())
      Delete();
   else {
      SafeDelete(fRoot);
      fSize = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from B-tree AND delete all heap based objects.

void TBtree::Delete(Option_t *)
{
   for (Int_t i = 0; i < fSize; i++) {
      TObject *obj = At(i);
      if (obj && obj->IsOnHeap())
         TCollection::GarbageCollect(obj);
   }
   SafeDelete(fRoot);
   fSize = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find object using its name (see object's GetName()). Requires sequential
/// search of complete tree till object is found.

TObject *TBtree::FindObject(const char *name) const
{
   return TCollection::FindObject(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Find object using the objects Compare() member function.

TObject *TBtree::FindObject(const TObject *obj) const
{
   if (!obj->IsSortable()) {
      Error("FindObject", "object must be sortable");
      return nullptr;
   }
   if (!fRoot)
      return nullptr;
   else {
      TBtNode *loc;
      Int_t idx;
      return fRoot->Found(obj, &loc, &idx);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add object and return its index in the tree.

Int_t TBtree::IdxAdd(const TObject &obj)
{
   Int_t r;
   if (!obj.IsSortable()) {
      Error("IdxAdd", "object must be sortable");
      return -1;
   }
   if (!fRoot) {
      fRoot = new TBtLeafNode(nullptr, &obj, this);
      R__ASSERT(fRoot != nullptr);
      IncrNofKeys();
      r = 0;
   } else {
      TBtNode *loc;
      int idx;
      if (fRoot->Found(&obj, &loc, &idx)) {
         // loc and idx are set to either where the object
         // was found, or where it should go in the Btree.
         // Nothing is here now, but later we might give the user
         // the ability to declare a B-tree as `unique elements only',
         // in which case we would handle an exception here.
         // std::cerr << "Multiple entry warning\n";
      } else {
         R__CHECK(loc->fIsLeaf);
      }
      if (loc->fIsLeaf) {
         if (!loc->fParent)
            r = idx;
         else
            r = idx + loc->fParent->FindRankUp(loc);
      } else {
         TBtInnerNode *iloc = (TBtInnerNode*) loc;
         r = iloc->FindRankUp(iloc->GetTree(idx));
      }
      loc->Add(&obj, idx);
   }
   R__CHECK(r == Rank(&obj) || &obj == (*this)[r]);
   return r;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize a B-tree.

void TBtree::Init(Int_t order)
{
   if (order < 3) {
      Warning("Init", "order must be at least 3");
      order = 3;
   }
   fRoot   = nullptr;
   fOrder  = order;
   fOrder2 = 2 * (fOrder+1);
   fLeafMaxIndex  = fOrder2 - 1;     // fItem[0..fOrder2-1]
   fInnerMaxIndex = fOrder;          // fItem[1..fOrder]
   //
   // the low water marks trigger an exploration for balancing
   // or merging nodes.
   // When the size of a node falls below X, then it must be possible to
   // either balance this node with another node, or it must be possible
   // to merge this node with another node.
   // This can be guaranteed only if (this->Size() < (MaxSize()-1)/2).
   //
   //

   // == MaxSize() satisfies the above because we compare
   // lowwatermark with fLast
   fLeafLowWaterMark  = ((fLeafMaxIndex+1)-1) / 2 - 1;
   fInnerLowWaterMark = (fOrder-1) / 2;
}

//______________________________________________________________________________
//void TBtree::PrintOn(std::ostream& out) const
//{
//   // Print a B-tree.
//
//   if (!fRoot)
//      out << "<empty>";
//   else
//      fRoot->PrintOn(out);
//}

////////////////////////////////////////////////////////////////////////////////
/// Returns a B-tree iterator.

TIterator *TBtree::MakeIterator(Bool_t dir) const
{
   return new TBtreeIter(this, dir);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the rank of the object in the tree.

Int_t TBtree::Rank(const TObject *obj) const
{
   if (!obj->IsSortable()) {
      Error("Rank", "object must be sortable");
      return -1;
   }
   if (!fRoot)
      return -1;
   else
      return fRoot->FindRank(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove an object from the tree.

TObject *TBtree::Remove(TObject *obj)
{
   if (!obj->IsSortable()) {
      Error("Remove", "object must be sortable");
      return nullptr;
   }
   if (!fRoot)
      return nullptr;

   TBtNode *loc;
   Int_t idx;
   TObject *ob = fRoot->Found(obj, &loc, &idx);
   if (!ob)
      return nullptr;
   loc->Remove(idx);
   return ob;
}

////////////////////////////////////////////////////////////////////////////////
/// The root of the tree is full. Create an InnerNode that
/// points to it, and then inform the InnerNode that it is full.

void TBtree::RootIsFull()
{
   TBtNode *oldroot = fRoot;
   fRoot = new TBtInnerNode(nullptr, this, oldroot);
   R__ASSERT(fRoot != nullptr);
   oldroot->Split();
}

////////////////////////////////////////////////////////////////////////////////
/// If root is empty clean up its space.

void TBtree::RootIsEmpty()
{
   if (fRoot->fIsLeaf) {
      TBtLeafNode *lroot = (TBtLeafNode*)fRoot;
      R__CHECK(lroot->Psize() == 0);
      delete lroot;
      fRoot = nullptr;
   } else {
      TBtInnerNode *iroot = (TBtInnerNode*)fRoot;
      R__CHECK(iroot->Psize() == 0);
      fRoot = iroot->GetTree(0);
      fRoot->fParent = nullptr;
      delete iroot;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Stream all objects in the btree to or from the I/O buffer.

void TBtree::Streamer(TBuffer &b)
{
   UInt_t R__s, R__c;
   if (b.IsReading()) {
      b.ReadVersion(&R__s, &R__c);   //Version_t v = b.ReadVersion();
      b >> fOrder;
      b >> fOrder2;
      b >> fInnerLowWaterMark;
      b >> fLeafLowWaterMark;
      b >> fInnerMaxIndex;
      b >> fLeafMaxIndex;
      TSeqCollection::Streamer(b);
      b.CheckByteCount(R__s, R__c,TBtree::IsA());
   } else {
      R__c = b.WriteVersion(TBtree::IsA(), kTRUE);
      b << fOrder;
      b << fOrder2;
      b << fInnerLowWaterMark;
      b << fLeafLowWaterMark;
      b << fInnerMaxIndex;
      b << fLeafMaxIndex;
      TSeqCollection::Streamer(b);
      b.SetByteCount(R__c, kTRUE);
   }
}


/** \class TBtItem
Item stored in inner nodes of a TBtree.
*/

////////////////////////////////////////////////////////////////////////////////
/// Create an item to be stored in the tree. An item contains a counter
/// of the number of keys (i.e. objects) in the node. A pointer to the
/// node and a pointer to the object being stored.

TBtItem::TBtItem()
{
   fNofKeysInTree = 0;
   fTree = nullptr;
   fKey  = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create an item to be stored in the tree. An item contains a counter
/// of the number of keys (i.e. objects) in the node. A pointer to the
/// node and a pointer to the object being stored.

TBtItem::TBtItem(TBtNode *n, TObject *obj)
{
   fNofKeysInTree = n->NofKeys()+1;
   fTree = n;
   fKey  = obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Create an item to be stored in the tree. An item contains a counter
/// of the number of keys (i.e. objects) in the node. A pointer to the
/// node and a pointer to the object being stored.

TBtItem::TBtItem(TObject *obj, TBtNode *n)
{
   fNofKeysInTree = n->NofKeys()+1;
   fTree = n;
   fKey  = obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a tree item.

TBtItem::~TBtItem()
{
}

/** \class TBtNode
Abstract base class (ABC) of a TBtree node.
*/

////////////////////////////////////////////////////////////////////////////////
/// Create a B-tree node.

TBtNode::TBtNode(Int_t isleaf, TBtInnerNode *p, TBtree *t)
{
   fLast   = -1;
   fIsLeaf = isleaf;
   fParent = p;
   if (!p) {
      R__CHECK(t != nullptr);
      fTree = t;
   } else
#ifdef cxxbug
//  BUG in the cxx compiler. cxx complains that it cannot access fTree
//  from TBtInnerNode. To reproduce the cxx bug uncomment the following line
//  and delete the line after.
//    fTree = p->fTree;
      fTree = p->GetParentTree();
#else
      fTree = p->fTree;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a B-tree node.

TBtNode::~TBtNode()
{
}

/** \class TBtreeIter
// Iterator of btree.
*/

ClassImp(TBtreeIter);

////////////////////////////////////////////////////////////////////////////////
/// Create a B-tree iterator.

TBtreeIter::TBtreeIter(const TBtree *t, Bool_t dir)
            : fTree(t), fCurCursor(0), fCursor(0), fDirection(dir)
{
   Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TBtreeIter::TBtreeIter(const TBtreeIter &iter) : TIterator(iter)
{
   fTree      = iter.fTree;
   fCursor    = iter.fCursor;
   fCurCursor = iter.fCurCursor;
   fDirection = iter.fDirection;
}

////////////////////////////////////////////////////////////////////////////////
/// Overridden assignment operator.

TIterator &TBtreeIter::operator=(const TIterator &rhs)
{
   if (this != &rhs && rhs.IsA() == TBtreeIter::Class()) {
      const TBtreeIter &rhs1 = (const TBtreeIter &)rhs;
      fTree      = rhs1.fTree;
      fCursor    = rhs1.fCursor;
      fCurCursor = rhs1.fCurCursor;
      fDirection = rhs1.fDirection;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Overloaded assignment operator.

TBtreeIter &TBtreeIter::operator=(const TBtreeIter &rhs)
{
   if (this != &rhs) {
      fTree      = rhs.fTree;
      fCursor    = rhs.fCursor;
      fCurCursor = rhs.fCurCursor;
      fDirection = rhs.fDirection;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the B-tree iterator.

void TBtreeIter::Reset()
{
   if (fDirection == kIterForward)
      fCursor = 0;
   else
      fCursor = fTree->GetSize() - 1;

   fCurCursor = fCursor;
}

////////////////////////////////////////////////////////////////////////////////
/// Get next object from B-tree. Returns 0 when no more objects in tree.

TObject *TBtreeIter::Next()
{
   fCurCursor = fCursor;
   if (fDirection == kIterForward) {
      if (fCursor < fTree->GetSize())
         return (*fTree)[fCursor++];
   } else {
      if (fCursor >= 0)
         return (*fTree)[fCursor--];
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// This operator compares two TIterator objects.

Bool_t TBtreeIter::operator!=(const TIterator &aIter) const
{
   if (aIter.IsA() == TBtreeIter::Class()) {
      const TBtreeIter &iter(dynamic_cast<const TBtreeIter &>(aIter));
      return (fCurCursor != iter.fCurCursor);
   }
   return false; // for base class we don't implement a comparison
}

////////////////////////////////////////////////////////////////////////////////
/// This operator compares two TBtreeIter objects.

Bool_t TBtreeIter::operator!=(const TBtreeIter &aIter) const
{
   return (fCurCursor != aIter.fCurCursor);
}

////////////////////////////////////////////////////////////////////////////////
/// Return current object or nullptr.

TObject* TBtreeIter::operator*() const
{
   return (((fCurCursor >= 0) && (fCurCursor < fTree->GetSize())) ?
           (*fTree)[fCurCursor] : nullptr);
}

/** \class TBtInnerNode
// Inner node of a TBtree.
*/

////////////////////////////////////////////////////////////////////////////////
/// Create a B-tree innernode.

TBtInnerNode::TBtInnerNode(TBtInnerNode *p, TBtree *t) : TBtNode(0,p,t)
{
   const Int_t index = MaxIndex() + 1;
   fItem = new TBtItem[ index ];
   if (!fItem)
      ::Fatal("TBtInnerNode::TBtInnerNode", "no more memory");
}

////////////////////////////////////////////////////////////////////////////////
/// Called only by TBtree to initialize the TBtInnerNode that is
/// about to become the root.

TBtInnerNode::TBtInnerNode(TBtInnerNode *parent, TBtree *tree, TBtNode *oldroot)
                : TBtNode(0, parent, tree)
{
   fItem = new TBtItem[MaxIndex()+1];
   if (!fItem)
      ::Fatal("TBtInnerNode::TBtInnerNode", "no more memory");
   Append(nullptr, oldroot);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBtInnerNode::~TBtInnerNode()
{
   if (fLast > 0)
      delete fItem[0].fTree;
   for (Int_t i = 1; i <= fLast; i++)
      delete fItem[i].fTree;

   delete [] fItem;
}

////////////////////////////////////////////////////////////////////////////////
/// This is called only from TBtree::Add().

void TBtInnerNode::Add(const TObject *obj, Int_t index)
{
   R__ASSERT(index >= 1 && obj->IsSortable());
   TBtLeafNode *ln = GetTree(index-1)->LastLeafNode();
   ln->Add(obj, ln->fLast+1);
}

////////////////////////////////////////////////////////////////////////////////
/// Add one element.

void TBtInnerNode::AddElt(TBtItem &itm, Int_t at)
{
   R__ASSERT(0 <= at && at <= fLast+1);
   R__ASSERT(fLast < MaxIndex());
   for (Int_t i = fLast+1; i > at ; i--)
      GetItem(i) = GetItem(i-1);
   SetItem(at, itm);
   fLast++;
}

////////////////////////////////////////////////////////////////////////////////
/// Add one element.

void TBtInnerNode::AddElt(Int_t at, TObject *k, TBtNode *t)
{
   TBtItem newitem(k, t);
   AddElt(newitem, at);
}

////////////////////////////////////////////////////////////////////////////////
/// Add one element.

void TBtInnerNode::Add(TBtItem &itm, Int_t at)
{
   AddElt(itm, at);
   if (IsFull())
      InformParent();
}

////////////////////////////////////////////////////////////////////////////////
/// Add one element.

void TBtInnerNode::Add(Int_t at, TObject *k, TBtNode *t)
{
   TBtItem newitem(k, t);
   Add(newitem, at);
}

////////////////////////////////////////////////////////////////////////////////
/// This should never create a full node that is, it is not used
/// anywhere where THIS could possibly be near full.

void TBtInnerNode::AppendFrom(TBtInnerNode *src, Int_t start, Int_t stop)
{
   if (start > stop)
      return;
   R__ASSERT(0 <= start && start <= src->fLast);
   R__ASSERT(0 <= stop  && stop  <= src->fLast );
   R__ASSERT(fLast + stop - start + 1 < MaxIndex()); // full-node check
   for (Int_t i = start; i <= stop; i++)
      SetItem(++fLast, src->GetItem(i));
}

////////////////////////////////////////////////////////////////////////////////
/// Never called from anywhere where it might fill up THIS.

void TBtInnerNode::Append(TObject *d, TBtNode *n)
{
   R__ASSERT(fLast < MaxIndex());
   if (d) R__ASSERT(d->IsSortable());
   SetItem(++fLast, d, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Append itm to this tree.

void TBtInnerNode::Append(TBtItem &itm)
{
   R__ASSERT(fLast < MaxIndex());
   SetItem(++fLast, itm);
}

////////////////////////////////////////////////////////////////////////////////
/// THIS has more than LEFTSIB. Move some item from THIS to LEFTSIB.
/// PIDX is the index of the parent item that will change when keys
/// are moved.

void TBtInnerNode::BalanceWithLeft(TBtInnerNode *leftsib, Int_t pidx)
{
   R__ASSERT(Vsize() >= leftsib->Psize());
   R__ASSERT(fParent->GetTree(pidx) == this);
   Int_t newThisSize = (Vsize() + leftsib->Psize())/2;
   Int_t noFromThis = Psize() - newThisSize;
   PushLeft(noFromThis, leftsib, pidx);
}

////////////////////////////////////////////////////////////////////////////////
/// THIS has more than RIGHTSIB. Move some items from THIS to RIGHTSIB.
/// PIDX is the index of the parent item that will change when keys
/// are moved.

void TBtInnerNode::BalanceWithRight(TBtInnerNode *rightsib, Int_t pidx)
{
   R__ASSERT(Psize() >= rightsib->Vsize());
   R__ASSERT(fParent->GetTree(pidx) == rightsib);
   Int_t newThisSize = (Psize() + rightsib->Vsize())/2;
   Int_t noFromThis = Psize() - newThisSize;
   PushRight(noFromThis, rightsib, pidx);
}

////////////////////////////////////////////////////////////////////////////////
/// PINDX is the index of the parent item whose key will change when
/// keys are shifted from one InnerNode to the other.

void TBtInnerNode::BalanceWith(TBtInnerNode *rightsib, Int_t pindx)
{
   if (Psize() < rightsib->Vsize())
      rightsib->BalanceWithLeft(this, pindx);
   else
      BalanceWithRight(rightsib, pindx);
}

////////////////////////////////////////////////////////////////////////////////
/// THAT is a child of THIS that has just shrunk by 1.

void TBtInnerNode::DecrNofKeys(TBtNode *that)
{
   Int_t i = IndexOf(that);
   fItem[i].fNofKeysInTree--;
   if (fParent)
      fParent->DecrNofKeys(this);
   else
      fTree->DecrNofKeys();
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively look for WHAT starting in the current node.

Int_t TBtInnerNode::FindRank(const TObject *what) const
{
   if (((TObject *)what)->Compare(GetKey(1)) < 0)
      return GetTree(0)->FindRank(what);
   Int_t sum = GetNofKeys(0);
   for (Int_t i = 1; i < fLast; i++) {
      if (((TObject*)what)->Compare(GetKey(i)) == 0)
         return sum;
      sum++;
      if (((TObject *)what)->Compare(GetKey(i+1)) < 0)
         return sum + GetTree(i)->FindRank(what);
      sum += GetNofKeys(i);
   }
   if (((TObject*)what)->Compare(GetKey(fLast)) == 0)
      return sum;
   sum++;
   // *what > GetKey(fLast), so recurse on last fItem.fTree
   return sum + GetTree(fLast)->FindRank(what);
}

////////////////////////////////////////////////////////////////////////////////
/// FindRankUp is FindRank in reverse.
/// Whereas FindRank looks for the object and computes the rank
/// along the way while walking DOWN the tree, FindRankUp already
/// knows where the object is and has to walk UP the tree from the
/// object to compute the rank.

Int_t TBtInnerNode::FindRankUp(const TBtNode *that) const
{
   Int_t l   = IndexOf(that);
   Int_t sum = 0;
   for (Int_t i = 0; i < l; i++)
      sum += GetNofKeys(i);
   return sum + l + (!fParent ? 0 : fParent->FindRankUp(this));
}

////////////////////////////////////////////////////////////////////////////////
/// Return the first leaf node.

TBtLeafNode *TBtInnerNode::FirstLeafNode()
{
   return GetTree(0)->FirstLeafNode();
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively look for WHAT starting in the current node.

TObject *TBtInnerNode::Found(const TObject *what, TBtNode **which, Int_t *where)
{
   R__ASSERT(what->IsSortable());
   for (Int_t i = 1 ; i <= fLast; i++) {
      if (GetKey(i)->Compare(what) == 0) {
         // then could go in either fItem[i].fTree or fItem[i-1].fTree
         // should go in one with the most room, but that's kinda
         // hard to calculate, so we'll stick it in fItem[i].fTree
         *which = this;
         *where = i;
         return GetKey(i);
      }
      if (GetKey(i)->Compare(what) > 0)
         return GetTree(i-1)->Found(what, which, where);
   }
   // *what > *(*this)[fLast].fKey, so recurse on last fItem.fTree
   return GetTree(fLast)->Found(what, which, where);
}

////////////////////////////////////////////////////////////////////////////////
/// THAT is a child of THIS that has just grown by 1.

void TBtInnerNode::IncrNofKeys(TBtNode *that)
{
   Int_t i = IndexOf(that);
   fItem[i].fNofKeysInTree++;
   if (fParent)
      fParent->IncrNofKeys(this);
   else
      fTree->IncrNofKeys();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a number in the range 0 to this->fLast
/// 0 is returned if THAT == fTree[0].

Int_t TBtInnerNode::IndexOf(const TBtNode *that) const
{
   for (Int_t i = 0; i <= fLast; i++)
      if (GetTree(i) == that)
         return i;
   R__CHECK(0);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Tell the parent that we are full.

void TBtInnerNode::InformParent()
{
   if (!fParent) {
      // then this is the root of the tree and needs to be split
      // inform the btree.
      R__ASSERT(fTree->fRoot == this);
      fTree->RootIsFull();
   } else
      fParent->IsFull(this);
}

////////////////////////////////////////////////////////////////////////////////
/// The child node THAT is full. We will either redistribute elements
/// or create a new node and then redistribute.
/// In an attempt to minimize the number of splits, we adopt the following
/// strategy:
///  - redistribute if possible
///  - if not possible, then split with a sibling

void TBtInnerNode::IsFull(TBtNode *that)
{
   if (that->fIsLeaf) {
      TBtLeafNode *leaf = (TBtLeafNode *)that;
      TBtLeafNode *left = nullptr;
      TBtLeafNode *right= nullptr;
      // split LEAF only if both sibling nodes are full.
      Int_t leafidx = IndexOf(leaf);
      Int_t hasRightSib = (leafidx < fLast) &&
                           ((right = (TBtLeafNode*)GetTree(leafidx+1)) != nullptr);
      Int_t hasLeftSib  = (leafidx > 0) &&
                           ((left = (TBtLeafNode*)GetTree(leafidx-1)) != nullptr);
      Int_t rightSibFull = (hasRightSib && right->IsAlmostFull());
      Int_t leftSibFull  = (hasLeftSib  && left->IsAlmostFull());
      if (rightSibFull) {
         if (leftSibFull) {
            // both full, so pick one to split with
            left->SplitWith(leaf, leafidx);
         } else if (hasLeftSib) {
            // left sib not full, so balance with it
            leaf->BalanceWithLeft(left, leafidx);
         } else {
            // there is no left sibling, so split with right
            leaf->SplitWith(right, leafidx+1);
         }
      } else if (hasRightSib) {
         // right sib not full, so balance with it
         leaf->BalanceWithRight(right, leafidx+1);
      } else if (leftSibFull) {
         // no right sib, and left sib is full, so split with it
         left->SplitWith(leaf, leafidx);
      } else if (hasLeftSib) {
         // left sib not full so balance with it
         leaf->BalanceWithLeft(left, leafidx);
      } else {
         // neither a left or right sib; should never happen
         R__CHECK(0);
      }
   } else {
      TBtInnerNode *inner = (TBtInnerNode *)that;
      // split INNER only if both sibling nodes are full
      Int_t inneridx = IndexOf(inner);
      TBtInnerNode *left = nullptr;
      TBtInnerNode *right = nullptr;
      Int_t hasRightSib = (inneridx < fLast) &&
                           ((right = (TBtInnerNode*)GetTree(inneridx+1)) != nullptr);
      Int_t hasLeftSib  = (inneridx > 0) &&
                           ((left=(TBtInnerNode*)GetTree(inneridx-1)) != nullptr);
      Int_t rightSibFull = (hasRightSib && right->IsAlmostFull());
      Int_t leftSibFull  = (hasLeftSib  && left->IsAlmostFull());
      if (rightSibFull) {
         if (leftSibFull) {
            left->SplitWith(inner, inneridx);
         } else if (hasLeftSib) {
            inner->BalanceWithLeft(left, inneridx);
         } else {
            // there is no left sibling
            inner->SplitWith(right, inneridx+1);
         }
      } else if (hasRightSib) {
         inner->BalanceWithRight(right, inneridx+1);
      } else if (leftSibFull) {
         left->SplitWith(inner, inneridx);
      } else if (hasLeftSib) {
         inner->BalanceWithLeft(left, inneridx);
      } else {
         R__CHECK(0);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// The child node THAT is <= half full. We will either redistribute
/// elements between children, or THAT will be merged with another child.
/// In an attempt to minimize the number of mergers, we adopt the following
/// strategy:
///  - redistribute if possible
///  - if not possible, then merge with a sibling

void TBtInnerNode::IsLow(TBtNode *that)
{
   if (that->fIsLeaf) {
      TBtLeafNode *leaf = (TBtLeafNode *)that;
      TBtLeafNode *left = nullptr;
      TBtLeafNode *right = nullptr;
      // split LEAF only if both sibling nodes are full.
      Int_t leafidx = IndexOf(leaf);
      Int_t hasRightSib = (leafidx < fLast) &&
                           ((right = (TBtLeafNode*)GetTree(leafidx+1)) != nullptr);
      Int_t hasLeftSib  = (leafidx > 0) &&
                           ((left = (TBtLeafNode*)GetTree(leafidx-1)) != nullptr);
      if (hasRightSib && (leaf->Psize() + right->Vsize()) >= leaf->MaxPsize()) {
         // then cannot merge,
         // and balancing this and rightsib will leave them both
         // more than half full
         leaf->BalanceWith(right, leafidx+1);
      } else if (hasLeftSib && (leaf->Vsize() + left->Psize()) >= leaf->MaxPsize()) {
         // ditto
         left->BalanceWith(leaf, leafidx);
      } else if (hasLeftSib) {
         // then they should be merged
         left->MergeWithRight(leaf, leafidx);
      } else if (hasRightSib) {
         leaf->MergeWithRight(right, leafidx+1);
      } else {
         R__CHECK(0); // should never happen
      }
   } else {
      TBtInnerNode *inner = (TBtInnerNode *)that;

      Int_t inneridx = IndexOf(inner);
      TBtInnerNode *left = nullptr;
      TBtInnerNode *right = nullptr;
      Int_t hasRightSib = (inneridx < fLast) &&
                           ((right = (TBtInnerNode*)GetTree(inneridx+1)) != nullptr);
      Int_t hasLeftSib  = (inneridx > 0) &&
                           ((left = (TBtInnerNode*)GetTree(inneridx-1)) != nullptr);
      if (hasRightSib && (inner->Psize() + right->Vsize()) >= inner->MaxPsize()) {
         // cannot merge
         inner->BalanceWith(right, inneridx+1);
      } else if (hasLeftSib && (inner->Vsize() + left->Psize()) >= inner->MaxPsize()) {
         // cannot merge
         left->BalanceWith(inner, inneridx);
      } else if (hasLeftSib) {
         left->MergeWithRight(inner, inneridx);
      } else if (hasRightSib) {
         inner->MergeWithRight(right, inneridx+1);
      } else {
         R__CHECK(0);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the last leaf node.

TBtLeafNode *TBtInnerNode::LastLeafNode()
{
   return GetTree(fLast)->LastLeafNode();
}

////////////////////////////////////////////////////////////////////////////////
/// Merge the 2 part of the tree.

void TBtInnerNode::MergeWithRight(TBtInnerNode *rightsib, Int_t pidx)
{
   R__ASSERT(Psize() + rightsib->Vsize() < MaxIndex());
   if (rightsib->Psize() > 0)
      rightsib->PushLeft(rightsib->Psize(), this, pidx);
   rightsib->SetKey(0, fParent->GetKey(pidx));
   AppendFrom(rightsib, 0, 0);
   fParent->IncNofKeys(pidx-1, rightsib->GetNofKeys(0)+1);
   fParent->RemoveItem(pidx);
   delete rightsib;
}

////////////////////////////////////////////////////////////////////////////////
/// Number of key.

Int_t TBtInnerNode::NofKeys() const
{
   Int_t sum = 0;
   for (Int_t i = 0; i <= fLast; i++)
      sum += GetNofKeys(i);
   return sum + Psize();
}

////////////////////////////////////////////////////////////////////////////////
/// return an element.

TObject *TBtInnerNode::operator[](Int_t idx) const
{
   for (Int_t j = 0; j <= fLast; j++) {
      Int_t r;
      if (idx < (r = GetNofKeys(j)))
         return (*GetTree(j))[idx];
      if (idx == r) {
         if (j == fLast) {
            ::Error("TBtInnerNode::operator[]", "should not happen, 0 returned");
            return nullptr;
         } else
            return GetKey(j+1);
      }
      idx -= r+1; // +1 because of the key in the node
   }
   ::Error("TBtInnerNode::operator[]", "should not happen, 0 returned");
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// noFromThis==1 => moves the parent item into the leftsib,
/// and the first item in this's array into the parent item.

void TBtInnerNode::PushLeft(Int_t noFromThis, TBtInnerNode *leftsib, Int_t pidx)
{
   R__ASSERT(fParent->GetTree(pidx) == this);
   R__ASSERT(noFromThis > 0 && noFromThis <= Psize());
   R__ASSERT(noFromThis + leftsib->Psize() < MaxPsize());
   SetKey(0, fParent->GetKey(pidx)); // makes AppendFrom's job easier
   leftsib->AppendFrom(this, 0, noFromThis-1);
   ShiftLeft(noFromThis);
   fParent->SetKey(pidx, GetKey(0));
   fParent->SetNofKeys(pidx-1, leftsib->NofKeys());
   fParent->SetNofKeys(pidx, NofKeys());
}

////////////////////////////////////////////////////////////////////////////////
/// The operation is three steps:
///  - Step I.   Make room for the incoming keys in RIGHTSIB.
///  - Step II.  Move the items from THIS into RIGHTSIB.
///  - Step III. Update the length of THIS.

void TBtInnerNode::PushRight(Int_t noFromThis, TBtInnerNode *rightsib, Int_t pidx)
{
   R__ASSERT(noFromThis > 0 && noFromThis <= Psize());
   R__ASSERT(noFromThis + rightsib->Psize() < rightsib->MaxPsize());
   R__ASSERT(fParent->GetTree(pidx) == rightsib);

   //
   // Step I. Make space for noFromThis items
   //
   Int_t start = fLast - noFromThis + 1;
   Int_t tgt, src;
   tgt = rightsib->fLast + noFromThis;
   src = rightsib->fLast;
   rightsib->fLast = tgt;
   rightsib->SetKey(0, fParent->GetKey(pidx));
   IncNofKeys(0);
   while (src >= 0) {
      // do this kind of assignment on TBtInnerNode items only when
      // the parent fields of the moved items do not change, as they
      // don't here.
      // Otherwise, use SetItem so the parents are updated appropriately.
      rightsib->GetItem(tgt--) = rightsib->GetItem(src--);
   }

   // Step II. Move the items from THIS into RIGHTSIB
   for (Int_t i = fLast; i >= start; i-- ) {
      // this is the kind of assignment to use when parents change
      rightsib->SetItem(tgt--, GetItem(i));
   }
   fParent->SetKey(pidx, rightsib->GetKey(0));
   DecNofKeys(0);
   R__CHECK(tgt == -1);

   // Step III.
   fLast -= noFromThis;

   // Step VI.  update NofKeys
   fParent->SetNofKeys(pidx-1, NofKeys());
   fParent->SetNofKeys(pidx, rightsib->NofKeys());
}

////////////////////////////////////////////////////////////////////////////////
/// Remove an element.

void TBtInnerNode::Remove(Int_t index)
{
   R__ASSERT(index >= 1 && index <= fLast);
   TBtLeafNode *lf = GetTree(index)->FirstLeafNode();
   SetKey(index, lf->fItem[0]);
   lf->RemoveItem(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove an item.

void TBtInnerNode::RemoveItem(Int_t index)
{
   R__ASSERT(index >= 1 && index <= fLast);
   for (Int_t to = index; to < fLast; to++)
      fItem[to] = fItem[to+1];
   fLast--;
   if (IsLow()) {
      if (!fParent) {
         // then this is the root; when only one child, make the child the root
         if (Psize() == 0)
            fTree->RootIsEmpty();
      } else
         fParent->IsLow(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Shift to the left.

void TBtInnerNode::ShiftLeft(Int_t cnt)
{
   if (cnt <= 0)
      return;
   for (Int_t i = cnt; i <= fLast; i++)
      GetItem(i-cnt) = GetItem(i);
   fLast -= cnt;
}

////////////////////////////////////////////////////////////////////////////////
/// This function is called only when THIS is the only descendent
/// of the root node, and THIS needs to be split.
/// Assumes that idx of THIS in fParent is 0.

void TBtInnerNode::Split()
{
   TBtInnerNode *newnode = new TBtInnerNode(fParent);
   R__CHECK(newnode != nullptr);
   fParent->Append(GetKey(fLast), newnode);
   newnode->AppendFrom(this, fLast, fLast);
   fLast--;
   fParent->IncNofKeys(1, newnode->GetNofKeys(0));
   fParent->DecNofKeys(0, newnode->GetNofKeys(0));
   BalanceWithRight(newnode, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// THIS and SIB are too full; create a NEWNODE, and balance
/// the number of keys between the three of them.
///
/// picture: (also see Knuth Vol 3 pg 478)
/// ~~~ {.cpp}
///               keyidx keyidx+1
///            +--+--+--+--+--+--...
///            |  |  |  |  |  |
/// fParent--->|  |     |     |
///            |  |     |     |
///            +*-+*-+*-+--+--+--...
///             |  |  |
///        +----+  |  +-----+
///        |       +-----+  |
///        V             |  V
///        +----------+  |  +----------+
///        |          |  |  |          |
///  this->|          |  |  |          |<--sib
///        +----------+  |  +----------+
///                      V
///                    data
/// ~~~
/// keyidx is the index of where the sibling is, and where the
/// newly created node will be recorded (sibling will be moved to
/// keyidx+1)

void TBtInnerNode::SplitWith(TBtInnerNode *rightsib, Int_t keyidx)
{
   R__ASSERT(keyidx > 0 && keyidx <= fParent->fLast);

   rightsib->SetKey(0, fParent->GetKey(keyidx));
   Int_t nofKeys      = Psize() + rightsib->Vsize();
   Int_t newSizeThis  = nofKeys / 3;
   Int_t newSizeNew   = (nofKeys - newSizeThis) / 2;
   Int_t newSizeSib   = (nofKeys - newSizeThis - newSizeNew);
   Int_t noFromThis   = Psize() - newSizeThis;
   Int_t noFromSib    = rightsib->Vsize() - newSizeSib;
   // because of their smaller size, this TBtInnerNode may not have to
   // give up any elements to the new node.  I.e., noFromThis == 0.
   // This will not happen for TBtLeafNodes.
   // We handle this by pulling an item from the rightsib.
   R__CHECK(noFromThis >= 0);
   R__CHECK(noFromSib >= 1);
   TBtInnerNode *newNode = new TBtInnerNode(fParent);
   R__CHECK(newNode != nullptr);
   if (noFromThis > 0) {
      newNode->Append(GetItem(fLast));
      fParent->AddElt(keyidx, GetKey(fLast--), newNode);
      if (noFromThis > 2)
         this->PushRight(noFromThis-1, newNode, keyidx);
      rightsib->PushLeft(noFromSib, newNode, keyidx+1);
   } else {
      // pull an element from the rightsib
      newNode->Append(rightsib->GetItem(0));
      fParent->AddElt(keyidx+1, rightsib->GetKey(1), rightsib);
      rightsib->ShiftLeft(1);
      fParent->SetTree(keyidx, newNode);
      rightsib->PushLeft(noFromSib-1, newNode, keyidx+1);
   }
   fParent->SetNofKeys(keyidx-1, this->NofKeys());
   fParent->SetNofKeys(keyidx, newNode->NofKeys());
   fParent->SetNofKeys(keyidx+1, rightsib->NofKeys());
   if (fParent->IsFull())
      fParent->InformParent();
}

/** \class TBtLeafNode
Leaf node of a TBtree.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBtLeafNode::TBtLeafNode(TBtInnerNode *p, const TObject *obj, TBtree *t): TBtNode(1, p, t)
{
   fItem = new TObject *[MaxIndex()+1];
   memset(fItem, 0, (MaxIndex()+1)*sizeof(TObject*));

   R__ASSERT(fItem != nullptr);
   if (obj)
      fItem[++fLast] = (TObject*)obj;   // cast const away
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TBtLeafNode::~TBtLeafNode()
{
   delete [] fItem;
}

////////////////////////////////////////////////////////////////////////////////
/// Add the object OBJ to the leaf node, inserting it at location INDEX
/// in the fItem array.

void TBtLeafNode::Add(const TObject *obj, Int_t index)
{
   R__ASSERT(obj->IsSortable());
   R__ASSERT(0 <= index && index <= fLast+1);
   R__ASSERT(fLast <= MaxIndex());
   for (Int_t i = fLast+1; i > index ; i--)
      fItem[i] = fItem[i-1];
   fItem[index] = (TObject *)obj;
   fLast++;

   // check for overflow
   if (!fParent)
      fTree->IncrNofKeys();
   else
      fParent->IncrNofKeys(this);

   if (IsFull()) {
      // it's full; tell parent node
      if (!fParent) {
         // this occurs when this leaf is the only node in the
         // btree, and this->fTree->fRoot == this
         R__CHECK(fTree->fRoot == this);
         // in which case we inform the btree, which can be
         // considered the parent of this node
         fTree->RootIsFull();
      } else {
         // the parent is responsible for splitting/balancing subnodes
         fParent->IsFull(this);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// A convenience function, does not worry about the element in
/// the parent, simply moves elements from SRC[start] to SRC[stop]
/// into the current array.
/// This should never create a full node.
/// That is, it is not used anywhere where THIS could possibly be
/// near full.
/// Does NOT handle nofKeys.

void TBtLeafNode::AppendFrom(TBtLeafNode *src, Int_t start, Int_t stop)
{
   if (start > stop)
      return;
   R__ASSERT(0 <= start && start <= src->fLast);
   R__ASSERT(0 <= stop  && stop  <= src->fLast);
   R__ASSERT(fLast + stop - start + 1 < MaxIndex()); // full-node check
   for (Int_t i = start; i <= stop; i++)
      fItem[++fLast] = src->fItem[i];
   R__CHECK(fLast < MaxIndex());
}

////////////////////////////////////////////////////////////////////////////////
/// Never called from anywhere where it might fill up THIS
/// does NOT handle nofKeys.

void TBtLeafNode::Append(TObject *obj)
{
   R__ASSERT(obj->IsSortable());
   fItem[++fLast] = obj;
   R__CHECK(fLast < MaxIndex());
}

////////////////////////////////////////////////////////////////////////////////
/// THIS has more than LEFTSIB;  move some items from THIS to LEFTSIB.

void TBtLeafNode::BalanceWithLeft(TBtLeafNode *leftsib, Int_t pidx)
{
   R__ASSERT(Vsize() >= leftsib->Psize());
   Int_t newThisSize = (Vsize() + leftsib->Psize())/2;
   Int_t noFromThis  = Psize() - newThisSize;
   PushLeft(noFromThis, leftsib, pidx);
}

////////////////////////////////////////////////////////////////////////////////
/// THIS has more than RIGHTSIB;  move some items from THIS to RIGHTSIB.

void TBtLeafNode::BalanceWithRight(TBtLeafNode *rightsib, Int_t pidx)
{
   R__ASSERT(Psize() >= rightsib->Vsize());
   Int_t newThisSize = (Psize() + rightsib->Vsize())/2;
   Int_t noFromThis  = Psize() - newThisSize;
   PushRight(noFromThis, rightsib, pidx);
}

////////////////////////////////////////////////////////////////////////////////
/// PITEM is the parent item whose key will change when keys are shifted
/// from one LeafNode to the other.

void TBtLeafNode::BalanceWith(TBtLeafNode *rightsib, Int_t pidx)
{
   if (Psize() < rightsib->Vsize())
      rightsib->BalanceWithLeft(this, pidx);
   else
      BalanceWithRight(rightsib, pidx);
}

////////////////////////////////////////////////////////////////////////////////
/// WHAT was not in any inner node; it is either here, or it's
/// not in the tree.

Int_t TBtLeafNode::FindRank(const TObject *what) const
{
   for (Int_t i = 0; i <= fLast; i++) {
      if (fItem[i]->Compare(what) == 0)
         return i;
      if (fItem[i]->Compare(what) > 0)
         return -1;
   }
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the first node.

TBtLeafNode *TBtLeafNode::FirstLeafNode()
{
   return this;
}

////////////////////////////////////////////////////////////////////////////////
/// WHAT was not in any inner node; it is either here, or it's
/// not in the tree.

TObject *TBtLeafNode::Found(const TObject *what, TBtNode **which, Int_t *where)
{
   R__ASSERT(what->IsSortable());
   for (Int_t i = 0; i <= fLast; i++) {
      if (fItem[i]->Compare(what) == 0) {
         *which = this;
         *where = i;
         return fItem[i];
      }
      if (fItem[i]->Compare(what) > 0) {
         *which = this;
         *where = i;
         return nullptr;
      }
   }
   *which = this;
   *where = fLast+1;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a number in the range 0 to MaxIndex().

Int_t TBtLeafNode::IndexOf(const TObject *that) const
{
   for (Int_t i = 0; i <= fLast; i++) {
      if (fItem[i] == that)
         return i;
   }
   R__CHECK(0);
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// return the last node.

TBtLeafNode *TBtLeafNode::LastLeafNode()
{
   return this;
}

////////////////////////////////////////////////////////////////////////////////
/// Merge.

void TBtLeafNode::MergeWithRight(TBtLeafNode *rightsib, Int_t pidx)
{
   R__ASSERT(Psize() + rightsib->Vsize() < MaxPsize());
   rightsib->PushLeft(rightsib->Psize(), this, pidx);
   Append(fParent->GetKey(pidx));
   fParent->SetNofKeys(pidx-1, NofKeys());
   fParent->RemoveItem(pidx);
   delete rightsib;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the number of keys.

Int_t TBtLeafNode::NofKeys(Int_t ) const
{
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the number of keys.

Int_t TBtLeafNode::NofKeys() const
{
   return Psize();
}

//______________________________________________________________________________
//void TBtLeafNode::PrintOn(std::ostream& out) const
//{
//    out << " < ";
//    for (Int_t i = 0; i <= fLast; i++)
//        out << *fItem[i] << " " ;
//    out << "> ";
//}

////////////////////////////////////////////////////////////////////////////////
/// noFromThis==1 => moves the parent item into the leftsib,
/// and the first item in this's array into the parent item.

void TBtLeafNode::PushLeft(Int_t noFromThis, TBtLeafNode *leftsib, Int_t pidx)
{
   R__ASSERT(noFromThis > 0 && noFromThis <= Psize());
   R__ASSERT(noFromThis + leftsib->Psize() < MaxPsize());
   R__ASSERT(fParent->GetTree(pidx) == this);
   leftsib->Append(fParent->GetKey(pidx));
   if (noFromThis > 1)
      leftsib->AppendFrom(this, 0, noFromThis-2);
   fParent->SetKey(pidx, fItem[noFromThis-1]);
   ShiftLeft(noFromThis);
   fParent->SetNofKeys(pidx-1, leftsib->NofKeys());
   fParent->SetNofKeys(pidx, NofKeys());
}

////////////////////////////////////////////////////////////////////////////////
/// noFromThis==1 => moves the parent item into the
/// rightsib, and the last item in this's array into the parent
/// item.

void TBtLeafNode::PushRight(Int_t noFromThis, TBtLeafNode *rightsib, Int_t pidx)
{
   R__ASSERT(noFromThis > 0 && noFromThis <= Psize());
   R__ASSERT(noFromThis + rightsib->Psize() < MaxPsize());
   R__ASSERT(fParent->GetTree(pidx) == rightsib);
   // The operation is five steps:
   //  Step I.   Make room for the incoming keys in RIGHTSIB.
   //  Step II.  Move the key in the parent into RIGHTSIB.
   //  Step III. Move the items from THIS into RIGHTSIB.
   //  Step IV.  Move the item from THIS into the parent.
   //  Step V.   Update the length of THIS.
   //
   // Step I.: make space for noFromThis items
   //
   Int_t start = fLast - noFromThis + 1;
   Int_t tgt, src;
   tgt = rightsib->fLast + noFromThis;
   src = rightsib->fLast;
   rightsib->fLast = tgt;
   while (src >= 0)
      rightsib->fItem[tgt--] = rightsib->fItem[src--];

   // Step II. Move the key from the parent into place
   rightsib->fItem[tgt--] = fParent->GetKey(pidx);

   // Step III.Move the items from THIS into RIGHTSIB
   for (Int_t i = fLast; i > start; i--)
      rightsib->fItem[tgt--] = fItem[i];
   R__CHECK(tgt == -1);

   // Step IV.
   fParent->SetKey(pidx, fItem[start]);

   // Step V.
   fLast -= noFromThis;

   // Step VI.  update nofKeys
   fParent->SetNofKeys(pidx-1, NofKeys());
   fParent->SetNofKeys(pidx, rightsib->NofKeys());
}

////////////////////////////////////////////////////////////////////////////////
/// Remove an element.

void TBtLeafNode::Remove(Int_t index)
{
   R__ASSERT(index >= 0 && index <= fLast);
   for (Int_t to = index; to < fLast; to++)
      fItem[to] = fItem[to+1];
   fLast--;
   if (!fParent)
      fTree->DecrNofKeys();
   else
      fParent->DecrNofKeys(this);
   if (IsLow()) {
      if (!fParent) {
         // then this is the root; when no keys left, inform the tree
         if (Psize() == 0)
            fTree->RootIsEmpty();
      } else
         fParent->IsLow(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Shift.

void TBtLeafNode::ShiftLeft(Int_t cnt)
{
   if (cnt <= 0)
      return;
   for (Int_t i = cnt; i <= fLast; i++)
      fItem[i-cnt] = fItem[i];
   fLast -= cnt;
}

////////////////////////////////////////////////////////////////////////////////
/// This function is called only when THIS is the only descendent
/// of the root node, and THIS needs to be split.
/// Assumes that idx of THIS in Parent is 0.

void TBtLeafNode::Split()
{
   TBtLeafNode *newnode = new TBtLeafNode(fParent);
   R__ASSERT(newnode != nullptr);
   fParent->Append(fItem[fLast--], newnode);
   fParent->SetNofKeys(0, fParent->GetTree(0)->NofKeys());
   fParent->SetNofKeys(1, fParent->GetTree(1)->NofKeys());
   BalanceWithRight(newnode, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Split.

void TBtLeafNode::SplitWith(TBtLeafNode *rightsib, Int_t keyidx)
{
   R__ASSERT(fParent == rightsib->fParent);
   R__ASSERT(keyidx > 0 && keyidx <= fParent->fLast);
   Int_t nofKeys      = Psize() + rightsib->Vsize();
   Int_t newSizeThis  = nofKeys / 3;
   Int_t newSizeNew   = (nofKeys - newSizeThis) / 2;
   Int_t newSizeSib   = (nofKeys - newSizeThis - newSizeNew);
   Int_t noFromThis   = Psize() - newSizeThis;
   Int_t noFromSib    = rightsib->Vsize() - newSizeSib;
   R__CHECK(noFromThis >= 0);
   R__CHECK(noFromSib >= 1);
   TBtLeafNode *newNode  = new TBtLeafNode(fParent);
   R__ASSERT(newNode != nullptr);
   fParent->AddElt(keyidx, fItem[fLast--], newNode);
   fParent->SetNofKeys(keyidx, 0);
   fParent->DecNofKeys(keyidx-1);
   this->PushRight(noFromThis-1, newNode, keyidx);
   rightsib->PushLeft(noFromSib, newNode, keyidx+1);
   if (fParent->IsFull())
      fParent->InformParent();
}
