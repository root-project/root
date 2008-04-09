// $Id$
#ifndef __SUT_BUCKLIST_H__
#define __SUT_BUCKLIST_H__
/******************************************************************************/
/*                                                                            */
/*                    X r d S u t B u c k L i s t . h h                       */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#ifndef __SUT_BUCKET_H__
#include <XrdSut/XrdSutBucket.hh>
#endif

/******************************************************************************/
/*                                                                            */
/*  Light single-linked list for managing buckets inside the exchanged        */
/*  buffer                                                                    */
/*                                                                            */
/******************************************************************************/

//
// Node definition
//
class XrdSutBuckListNode {
private:
   XrdSutBucket       *buck;
   XrdSutBuckListNode *next;
public:
   XrdSutBuckListNode(XrdSutBucket *b = 0, XrdSutBuckListNode *n = 0)
        { buck = b; next = n;}
   virtual ~XrdSutBuckListNode() { }
   
   XrdSutBucket       *Buck() const { return buck; }

   XrdSutBuckListNode *Next() const { return next; }

   void SetNext(XrdSutBuckListNode *n) { next = n; }
};

class XrdSutBuckList {

private:
   XrdSutBuckListNode *begin;
   XrdSutBuckListNode *current;
   XrdSutBuckListNode *end;
   XrdSutBuckListNode *previous;
   int                 size;

   XrdSutBuckListNode *Find(XrdSutBucket *b);

public:
   XrdSutBuckList(XrdSutBucket *b = 0);
   virtual ~XrdSutBuckList();

   // Access information
   int                 Size() const { return size; }
   XrdSutBucket       *End() const { return end->Buck(); }

   // Modifiers
   void                PutInFront(XrdSutBucket *b);
   void                PushBack(XrdSutBucket *b);
   void                Remove(XrdSutBucket *b);
   
   // Pseudo - iterator functionality
   XrdSutBucket       *Begin();
   XrdSutBucket       *Next();
};

#endif

