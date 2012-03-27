// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBtree
#define ROOT_TBtree


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBtree                                                               //
//                                                                      //
// Btree class. TBtree inherits from the TSeqCollection ABC.            //
//                                                                      //
// For a more extensive algorithmic description see the TBtree source.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif

#include <iterator>


class TBtNode;
class TBtInnerNode;
class TBtLeafNode;
class TBtreeIter;


class TBtree : public TSeqCollection {

friend class  TBtNode;
friend class  TBtInnerNode;
friend class  TBtLeafNode;

private:
   TBtNode  *fRoot;              //root node of btree

   Int_t     fOrder;             //the order of the tree (should be > 2)
   Int_t     fOrder2;            //order*2+1 (assumes a memory access is
                                 //cheaper than a multiply and increment by one
   Int_t     fInnerLowWaterMark; //inner node low water mark
   Int_t     fLeafLowWaterMark;  //leaf low water mark
   Int_t     fInnerMaxIndex;     //maximum inner node index
   Int_t     fLeafMaxIndex;      //maximum leaf index

   void Init(Int_t i);        //initialize btree
   void RootIsFull();         //called when the root node is full
   void RootIsEmpty();        //called when root is empty

protected:
   void IncrNofKeys() { fSize++; }
   void DecrNofKeys() { fSize--; }

   // add the object to the tree; return the index in the tree at which
   // the object was inserted. NOTE: other insertions and deletions may
   // change this object's index.
   Int_t IdxAdd(const TObject &obj);

public:
   typedef TBtreeIter Iterator_t;

   TBtree(Int_t ordern = 3);  //create a TBtree of order n
   virtual     ~TBtree();
   void        Clear(Option_t *option="");
   void        Delete(Option_t *option="");
   TObject    *FindObject(const char *name) const;
   TObject    *FindObject(const TObject *obj) const;
   TObject   **GetObjectRef(const TObject *) const { return 0; }
   TIterator  *MakeIterator(Bool_t dir = kIterForward) const;

   void        Add(TObject *obj);
   void        AddFirst(TObject *obj) { Add(obj); }
   void        AddLast(TObject *obj) { Add(obj); }
   void        AddAt(TObject *obj, Int_t) { Add(obj); }
   void        AddAfter(const TObject *, TObject *obj) { Add(obj); }
   void        AddBefore(const TObject *, TObject *obj) { Add(obj); }
   TObject    *Remove(TObject *obj);

   TObject    *At(Int_t idx) const;
   TObject    *Before(const TObject *obj) const;
   TObject    *After(const TObject *obj) const;
   TObject    *First() const;
   TObject    *Last() const;

   //void PrintOn(ostream &os) const;

   Int_t       Order() { return fOrder; }
   TObject    *operator[](Int_t i) const;
   Int_t       Rank(const TObject *obj) const;

   ClassDef(TBtree,0)  //A B-tree
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBtNode                                                              //
//                                                                      //
// Abstract base class (ABC) of a TBtree node.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TBtNode {

friend class  TBtree;
friend class  TBtInnerNode;
friend class  TBtLeafNode;

protected:
   Int_t fLast;   // for inner node 1 <= fLast <= fInnerMaxIndex
                  // for leaf node  1 <= fLast <= fLeafMaxIndex
                  // (fLast==0 only temporarily while the tree is being
                  // updated)

   TBtInnerNode *fParent;   // a parent is always an inner node (or 0 for the root)
   TBtree       *fTree;     // the tree of which this node is a part
   Int_t         fIsLeaf;   // run-time type flag

public:
   TBtNode(Int_t isleaf, TBtInnerNode *p, TBtree *t = 0);
   virtual ~TBtNode();

   virtual void Add(const TObject *obj, Int_t index) = 0;
#ifndef __CINT__
   virtual TBtree *GetParentTree() const {return fTree;}
   virtual void Remove(Int_t index) = 0;

   virtual TObject *operator[](Int_t i) const = 0;
   virtual TObject *Found(const TObject *obj, TBtNode **which, Int_t *where) = 0;

   virtual Int_t FindRank(const TObject *obj) const = 0;
   virtual Int_t NofKeys() const = 0; // # keys in or below this node

   virtual TBtLeafNode *FirstLeafNode() = 0;
   virtual TBtLeafNode *LastLeafNode() = 0;

   virtual void Split() = 0;
#endif
   // virtual void PrintOn(ostream &os) const = 0;
   // friend ostream &operator<<(ostream &os, const TBtNode &node);
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBtItem                                                              //
//                                                                      //
// Item stored in inner nodes of a TBtree.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TBtItem {

friend class  TBtInnerNode;

private:
   Int_t      fNofKeysInTree;   // number of keys in TBtree
   TObject   *fKey;             // key
   TBtNode   *fTree;            //! sub-tree

public:
   TBtItem();
   TBtItem(TBtNode *n, TObject *o);
   TBtItem(TObject *o, TBtNode *n);
   ~TBtItem();
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBtInnerNode                                                         //
//                                                                      //
// Inner node of a TBtree.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TBtInnerNode : public TBtNode {

private:
   TBtItem    *fItem;   // actually fItem[MaxIndex()+1] is desired

public:
   TBtInnerNode(TBtInnerNode *parent, TBtree *t = 0);
   TBtInnerNode(TBtInnerNode *parent, TBtree *tree, TBtNode *oldroot);
   ~TBtInnerNode();

#ifndef __CINT__
   void      Add(const TObject *obj, Int_t idx);
   void      Add(TBtItem &i, Int_t idx);
   void      Add(Int_t at, TObject *obj, TBtNode *n);
   void      AddElt(TBtItem &itm, Int_t at);
   void      AddElt(Int_t at, TObject *obj, TBtNode *n);
   void      Remove(Int_t idx);
   void      RemoveItem(Int_t idx);

   TObject  *operator[](Int_t i) const;
   TObject  *Found(const TObject *obj, TBtNode **which, Int_t *where);

   Int_t     NofKeys(Int_t idx) const;
   Int_t     NofKeys() const;
   void      SetTree(Int_t i, TBtNode *node) { fItem[i].fTree = node; node->fParent = this; }
   void      SetKey(Int_t i, TObject *obj) { fItem[i].fKey = obj; }
   void      SetItem(Int_t i, TBtItem &itm) { fItem[i] = itm; itm.fTree->fParent = this; }
   void      SetItem(Int_t i, TObject *obj, TBtNode *node) { SetTree(i, node); SetKey(i, obj); }
   Int_t     GetNofKeys(Int_t i) const;
   void      SetNofKeys(Int_t i, Int_t r);
   Int_t     IncNofKeys(Int_t i, Int_t n=1);
   Int_t     DecNofKeys(Int_t i, Int_t n=1);
   Int_t     FindRank(const TObject *obj) const;
   Int_t     FindRankUp(const TBtNode *n) const;
   TBtNode  *GetTree(Int_t i) const { return fItem[i].fTree; }
   TObject  *GetKey(Int_t i) const { return fItem[i].fKey; }
   TBtItem  &GetItem(Int_t i) const { return fItem[i]; }

   Int_t     IndexOf(const TBtNode *n) const;
   void      IncrNofKeys(TBtNode *np);
   void      DecrNofKeys(TBtNode *np);

   TBtLeafNode *FirstLeafNode();
   TBtLeafNode *LastLeafNode();

   void      InformParent();

   void      Split();
   void      SplitWith(TBtInnerNode *r, Int_t idx);
   void      MergeWithRight(TBtInnerNode *r, Int_t idx);
   void      BalanceWithLeft(TBtInnerNode *l, Int_t idx);
   void      BalanceWithRight(TBtInnerNode *r, Int_t idx);
   void      BalanceWith(TBtInnerNode *n, int idx);
   void      PushLeft(Int_t cnt, TBtInnerNode *leftsib, Int_t parentIdx);
   void      PushRight(Int_t cnt, TBtInnerNode *rightsib, Int_t parentIdx);
   void      AppendFrom(TBtInnerNode *src, Int_t start, Int_t stop);
   void      Append(TObject *obj, TBtNode *n);
   void      Append(TBtItem &itm);
   void      ShiftLeft(Int_t cnt);

   Int_t     Psize() const { return fLast; }
   Int_t     Vsize() const;
   Int_t     MaxIndex() const { return fTree->fInnerMaxIndex; }
   Int_t     MaxPsize() const { return fTree->fInnerMaxIndex; }

   // void      PrintOn(ostream &os) const;

   Int_t     IsFull() const { return fLast == MaxIndex(); }
   void      IsFull(TBtNode *n);
   Int_t     IsAlmostFull() const { return fLast >= MaxIndex() - 1; }
   Int_t     IsLow() const {  return fLast < fTree->fInnerLowWaterMark; }
   void      IsLow(TBtNode *n);
#endif
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBtLeafNode                                                          //
//                                                                      //
// Leaf node of a TBtree.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TBtLeafNode : public TBtNode {

friend class  TBtInnerNode;

private:
   TObject **fItem; // actually TObject *fItem[MaxIndex()+1] is desired

public:
   TBtLeafNode(TBtInnerNode *p, const TObject *obj = 0, TBtree *t = 0);
   ~TBtLeafNode();

#ifndef __CINT__
   void       Add(const TObject *obj, Int_t idx);
   void       Remove(Int_t idx);
   void       RemoveItem(Int_t idx) { Remove(idx); }

   TObject   *operator[](Int_t i) const;
   TObject   *Found(const TObject *obj, TBtNode **which, Int_t *where);

   Int_t      NofKeys(Int_t i) const;
   Int_t      NofKeys() const;
   Int_t      FindRank(const TObject *obj) const;
   TObject   *GetKey(Int_t idx ) { return fItem[idx]; }
   void       SetKey(Int_t idx, TObject *obj) { fItem[idx] = obj; }

   Int_t      IndexOf(const TObject *obj) const;

   TBtLeafNode  *FirstLeafNode();
   TBtLeafNode  *LastLeafNode();

   void       Split();
   void       SplitWith(TBtLeafNode *r, Int_t idx);
   void       MergeWithRight(TBtLeafNode *r, Int_t idx);
   void       BalanceWithLeft(TBtLeafNode *l, Int_t idx);
   void       BalanceWithRight(TBtLeafNode *r, Int_t idx);
   void       BalanceWith(TBtLeafNode *n, Int_t idx);
   void       PushLeft(Int_t cnt, TBtLeafNode *l, Int_t parentIndex);
   void       PushRight(Int_t cnt, TBtLeafNode *r, Int_t parentIndex);
   void       AppendFrom(TBtLeafNode *src, Int_t start, Int_t stop);
   void       Append(TObject *obj);
   void       ShiftLeft(Int_t cnt);

   Int_t      Psize() const { return fLast + 1; }
   Int_t      Vsize() const;
   Int_t      MaxIndex() const { return fTree->fLeafMaxIndex; }
   Int_t      MaxPsize() const { return fTree->fLeafMaxIndex + 1; }

   // void       PrintOn(ostream &os) const;

   Int_t      IsFull() const { return fLast == MaxIndex(); }
   Int_t      IsAlmostFull() const { return fLast >= MaxIndex() - 1; }
   Int_t      IsLow() const { return fLast < fTree->fLeafLowWaterMark; }
#endif
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBtreeIter                                                           //
//                                                                      //
// Iterator of btree.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TBtreeIter : public TIterator,
                   public std::iterator<std::bidirectional_iterator_tag,
                                        TObject*, std::ptrdiff_t,
                                        const TObject**, const TObject*&> {

private:
   const TBtree  *fTree;      //btree being iterated
   Int_t          fCurCursor; //current position in btree
   Int_t          fCursor;    //next position in btree
   Bool_t         fDirection; //iteration direction

   TBtreeIter() : fTree(0), fCurCursor(0), fCursor(0), fDirection(kIterForward) { }

public:
   TBtreeIter(const TBtree *t, Bool_t dir = kIterForward);
   TBtreeIter(const TBtreeIter &iter);
   ~TBtreeIter() { }
   TIterator  &operator=(const TIterator &rhs);
   TBtreeIter &operator=(const TBtreeIter &rhs);

   const TCollection  *GetCollection() const { return fTree; }
   TObject            *Next();
   void                Reset();
   Bool_t              operator!=(const TIterator &aIter) const;
   Bool_t              operator!=(const TBtreeIter &aIter) const;
   TObject            *operator*() const;

   ClassDef(TBtreeIter,0)  //B-tree iterator
};


//----- TBtree inlines ---------------------------------------------------------

inline TObject *TBtree::operator[](Int_t i) const
{
   return (*fRoot)[i];
}

inline TObject *TBtree::At(Int_t i) const
{
   return (*fRoot)[i];
}

inline TObject *TBtree::First() const
{
   return (*fRoot)[0];
}

inline TObject *TBtree::Last() const
{
   return (*fRoot)[fSize-1];
}

//----- TBtInnerNode inlines ---------------------------------------------------

inline Int_t TBtInnerNode::GetNofKeys(Int_t i) const
{
   R__ASSERT(i >= 0 && i <= fLast);
   return fItem[i].fNofKeysInTree;
}

inline Int_t TBtInnerNode::NofKeys(Int_t idx) const
{
   return GetNofKeys(idx);
}

inline void TBtInnerNode::SetNofKeys(Int_t i, Int_t r)
{
   fItem[i].fNofKeysInTree = r;
}

inline Int_t TBtInnerNode::IncNofKeys(Int_t i, Int_t n)
{
   return (fItem[i].fNofKeysInTree += n);
}

inline Int_t TBtInnerNode::DecNofKeys(Int_t i, Int_t n)
{
   return (fItem[i].fNofKeysInTree -= n);
}

inline Int_t TBtInnerNode::Vsize() const
{
   R__ASSERT(fParent != 0 && fParent->GetTree(0) != (TBtNode *)this);
   return Psize()+1;
}


//----- TBtLeafNode inlines ----------------------------------------------------

inline TObject *TBtLeafNode::operator[](Int_t i) const
{
   R__ASSERT(i >= 0 && i <= fLast);
   return fItem[i];
}

inline Int_t TBtLeafNode::Vsize() const
{
   R__ASSERT(fParent != 0 && fParent->GetTree(0) != (TBtNode *)this);
   return Psize()+1;
}

//inline ostream &operator<<(ostream& outputStream, const TBtNode &aNode)
//{
//   aNode.PrintOn(outputStream);
//   return outputStream;
//}

#endif
