// $Id$

const char *XrdSutBuckListCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                     X r d S u t B u c k L i s t . c c                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <XrdSut/XrdSutBuckList.hh>

/******************************************************************************/
/*                                                                            */
/*  Light single-linked list for managing buckets inside the exchanged        */
/*  buffer                                                                    */
/*                                                                            */
/******************************************************************************/

//___________________________________________________________________________
XrdSutBuckList::XrdSutBuckList(XrdSutBucket *b)
{
   // Constructor

   previous = current = begin = end = 0;
   size = 0; 

   if (b) {
      XrdSutBuckListNode *f = new XrdSutBuckListNode(b,0);
      current = begin = end = f;
      size++;
   }
} 

//___________________________________________________________________________
XrdSutBuckList::~XrdSutBuckList()
{
   // Destructor

   XrdSutBuckListNode *n = 0;
   XrdSutBuckListNode *b = begin;
   while (b) {
      n = b->Next();
      delete (b);
      b = n;
   }
}

//___________________________________________________________________________
XrdSutBuckListNode *XrdSutBuckList::Find(XrdSutBucket *b)
{
   // Find node containing bucket b

   XrdSutBuckListNode *nd = begin;
   for (; nd; nd = nd->Next()) {
      if (nd->Buck() == b)
         return nd;
   }
   return (XrdSutBuckListNode *)0;
}

//___________________________________________________________________________
void XrdSutBuckList::PutInFront(XrdSutBucket *b)
{
   // Add at the beginning of the list
   // Check to avoid duplicates

   if (!Find(b)) {
      XrdSutBuckListNode *nb = new XrdSutBuckListNode(b,begin);
      begin = nb;     
      if (!end)
         end = nb;
      size++;
   }
}

//___________________________________________________________________________
void XrdSutBuckList::PushBack(XrdSutBucket *b)
{
   // Add at the end of the list
   // Check to avoid duplicates

   if (!Find(b)) {
      XrdSutBuckListNode *nb = new XrdSutBuckListNode(b,0);
      if (!begin)
         begin = nb;
      if (end)
         end->SetNext(nb);
      end = nb;
      size++;
   }
}

//___________________________________________________________________________
void XrdSutBuckList::Remove(XrdSutBucket *b)
{
   // Remove node containing bucket b

   XrdSutBuckListNode *curr = current;
   XrdSutBuckListNode *prev = previous;

   if (!curr || curr->Buck() != b || (prev && curr != prev->Next())) {
      // We need first to find the address
      curr = begin;
      prev = 0;
      for (; curr; curr = curr->Next()) {
         if (curr->Buck() == b)
            break;
         prev = curr;
      }
   }

   // The bucket is not in the list
   if (!curr)
      return;

   // Now we have all the information to remove
   if (prev) {
      current  = curr->Next();
      prev->SetNext(current);
      previous = curr;
   } else if (curr == begin) {
      // First buffer
      current  = curr->Next();
      begin = current;
      previous = 0;
   }

   // Cleanup and update size
   delete curr;      
   size--;
}

//___________________________________________________________________________
XrdSutBucket *XrdSutBuckList::Begin()
{ 
   // Iterator functionality: init

   previous = 0;
   current = begin;
   if (current)
      return current->Buck();
   return (XrdSutBucket *)0;
}

//___________________________________________________________________________
XrdSutBucket *XrdSutBuckList::Next()
{ 
   // Iterator functionality: get next

   previous = current;
   if (current) {
      current = current->Next();
      if (current)
         return current->Buck();
   }
   return (XrdSutBucket *)0;
}
