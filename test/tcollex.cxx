// @(#)root/test:$Name:  $:$Id: tcollex.cxx,v 1.9 2002/06/30 13:17:33 brun Exp $
// Author: Fons Rademakers   19/08/96

#include <stdlib.h>

#include "Riostream.h"
#include "TString.h"
#include "TObjString.h"
#include "TSortedList.h"
#include "TObjArray.h"
#include "TOrdCollection.h"
#include "THashTable.h"
#include "TBtree.h"
#include "TStopwatch.h"
#include "TObjNum.h"


// To focus on basic collection protocol, this sample program uses
// simple classes inheriting from TObject. One class, TObjString, is a
// collectable string class (a TString wrapped in a TObject) provided
// by the ROOT system. The other class TObjNumI, is an integer wrapped
// in a TObject, just like TObjString.



void Test_TObjArray()
{

   Printf(
   "////////////////////////////////////////////////////////////////\n"
   "// Test of TObjArray                                          //\n"
   "////////////////////////////////////////////////////////////////"
   );

   // Array of capacity 10, Add() will automatically expand the array if necessary.
   TObjArray  a(10);

   Printf("Filling TObjArray");
   a.Add(new TObjNumI(1));           // add at next free slot, pos 0
   a[1] = new TObjNumI(2);           // use operator[], put at pos 1
   TObjNumI *n3 = new TObjNumI(3);
   a.AddAt(n3,2);                    // add at position 2
   a.Add(new TObjNumI(4));           // add at next free slot, pos 3
   a.AddLast(new TObjNumI(10));      // add at pos 4
   TObjNumI n6(6);                   // stack based TObjNumI
   a.AddAt(&n6,5);                   // add at pos 5
   a[6] = new TObjNumI(5);           // add at respective positions
   a[7] = new TObjNumI(8);
   a[8] = new TObjNumI(7);
//   a[10] = &n6;                    // gives out-of-bound error

   Printf("Print array");
   a.Print();                        // invoke Print() of all objects

   Printf("Sort array");
   a.Sort();
   for (int i = 0; i < a.Capacity(); i++)  // typical way of iterating over array
      if (a[i])
         a[i]->Print();      // can also use operator[] to access elements
      else
         Printf("%d empty slot", i);

   Printf("Use binary search to get position of number 6");
   Printf("6 is at position %d", a.BinarySearch(&n6));

   Printf("Find number before 6");
   a.Before(&n6)->Print();

   Printf("Find number after 3");
   a.After(n3)->Print();

   Printf("Remove 3 and print list again");
   a.Remove(n3);
   delete n3;
   a.Print();

   Printf("Iterate forward over list and remove 4 and 7");

   // TIter encapsulates the actual class iterator. The type of iterator
   // used depends on the type of the collection.
   TIter next(&a);

   TObjNumI *obj;
   while ((obj = (TObjNumI*)next()))     // iterator skips empty slots
      if (obj->GetNum() == 4) {
         a.Remove(obj);
         delete obj;
      }

   // Reset the iterator and loop again
   next.Reset();
   while ((obj = (TObjNumI*)next()))
      if (obj->GetNum() == 7) {
         a.Remove(obj);
         delete obj;
      }

   Printf("Iterate backward over list and remove 2");
   TIter next1(&a, kIterBackward);
   while ((obj = (TObjNumI*)next1()))
      if (obj->GetNum() == 2) {
         a.Remove(obj);
         delete obj;
      }

   Printf("Delete remainder of list: 1,5,8,10 (6 not deleted since not on heap)");

   // Delete heap objects and clear list. Attention: do this only when you
   // own all objects stored in the collection. When you stored aliases to
   // the actual objects (i.e. you did not create the objects) use Clear()
   // instead.
   a.Delete();

   Printf("Delete stack based objects (6)");
}

void Test_TOrdCollection()
{
   Printf(
   "////////////////////////////////////////////////////////////////\n"
   "// Test of TOrdCollection                                     //\n"
   "////////////////////////////////////////////////////////////////"
   );

   // Create collection with default size, Add() will automatically expand
   // the collection if necessary.
   TOrdCollection  c;

   Printf("Filling TOrdCollection");
   c.Add(new TObjString("anton"));      // add at next free slot, pos 0
   c.AddFirst(new TObjString("bobo"));  // put at pos 0, bump anton to pos 1
   TObjString *s3 = new TObjString("damon");
   c.AddAt(s3,1);                       // add at position 1, bump anton to pos 2
   c.Add(new TObjString("cassius"));    // add at next free slot, pos 3
   c.AddLast(new TObjString("enigma")); // add at pos 4
   TObjString s6("fons");               // stack based TObjString
   c.AddBefore(s3,&s6);                 // add at pos 1
   c.AddAfter(s3, new TObjString("gaia"));

   Printf("Print collection");
   c.Print();                           // invoke Print() of all objects

   Printf("Sort collection");
   c.Sort();
   c.Print();

   Printf("Use binary search to get position of string damon");
   Printf("damon is at position %d", c.BinarySearch(s3));

   Printf("Find str before fons");
   c.Before(&s6)->Print();

   Printf("Find string after damon");
   c.After(s3)->Print();

   Printf("Remove damon and print list again");
   c.Remove(s3);
   delete s3;
   c.Print();

   Printf("Iterate forward over list and remove cassius");
   TObjString *objs;
   TIter next(&c);
   while ((objs = (TObjString*)next()))     // iterator skips empty slots
      if (objs->String() == "cassius") {
         c.Remove(objs);
         delete objs;
      }

   Printf("Iterate backward over list and remove gaia");
   TIter next1(&c, kIterBackward);
   while ((objs = (TObjString*)next1()))
      if (objs->String() == "gaia") {
         c.Remove(objs);
         delete objs;
      }

   Printf("Delete remainder of list: anton,bobo,enigma (fons not deleted since not on heap)");
   c.Delete();                        // delete heap objects and clear list

   Printf("Delete stack based objects (fons)");
}

void Test_TList()
{
   Printf(
   "////////////////////////////////////////////////////////////////\n"
   "// Test of TList                                              //\n"
   "////////////////////////////////////////////////////////////////"
   );

   // Create a doubly linked list.
   TList l;

   Printf("Filling TList");
   TObjNumI *n3 = new TObjNumI(3);
   l.Add(n3);
   l.AddBefore(n3, new TObjNumI(5));
   l.AddAfter(n3, new TObjNumI(2));
   l.Add(new TObjNumI(1));
   l.AddBefore(n3, new TObjNumI(4));
   TObjNumI n6(6);                     // stack based TObjNumI
   l.AddFirst(&n6);

   Printf("Print list");
   l.Print();

   Printf("Remove 3 and print list again");
   l.Remove(n3);
   delete n3;
   l.Print();

   Printf("Iterate forward over list and remove 4");
   TObjNumI *obj;
   TIter next(&l);
   while ((obj = (TObjNumI*)next()))
      if (obj->GetNum() == 4) l.Remove(obj);

   Printf("Iterate backward over list and remove 2");
   TIter next1(&l, kIterBackward);
   while ((obj = (TObjNumI*)next1()))
      if (obj->GetNum() == 2) {
         l.Remove(obj);
         delete obj;
      }

   Printf("Delete remainder of list: 1, 5 (6 not deleted since not on heap)");
   l.Delete();

   Printf("Delete stack based objects (6)");
}

void Test_TSortedList()
{
   Printf(
   "////////////////////////////////////////////////////////////////\n"
   "// Test of TSortedList                                        //\n"
   "////////////////////////////////////////////////////////////////"
   );

   // Create a sorted doubly linked list.
   TSortedList sl;

   Printf("Filling TSortedList");
   TObjNumI *n3 = new TObjNumI(3);
   sl.Add(n3);
   sl.AddBefore(n3,new TObjNumI(5));
   sl.AddAfter(n3, new TObjNumI(2));
   sl.Add(new TObjNumI(1));
   sl.AddBefore(n3, new TObjNumI(4));
   TObjNumI n6(6);                     // stack based TObjNumI
   sl.AddFirst(&n6);

   Printf("Print list");
   sl.Print();

   Printf("Delete all heap based objects (6 not deleted since not on heap)");
   sl.Delete();

   Printf("Delete stack based objects (6)");
}

void Test_THashTable()
{
   Printf(
   "////////////////////////////////////////////////////////////////\n"
   "// Test of THashTable                                         //\n"
   "////////////////////////////////////////////////////////////////"
   );

   int i;

   // Create a hash table with an initial size of 20 (actually the next prime
   // above 20). No automatic rehashing.
   THashTable ht(20);

   Printf("Filling THashTable");
   Printf("Number of slots before filling: %d", ht.Capacity());
   for (i = 0; i < 1000; i++)
      ht.Add(new TObject);

   Printf("Average collisions: %f", ht.AverageCollisions());

   // rehash the hash table to reduce the collission rate
   ht.Rehash(ht.GetSize());

   Printf("Number of slots after rehash: %d", ht.Capacity());
   Printf("Average collisions after rehash: %f", ht.AverageCollisions());

   ht.Delete();

   // Create a hash table and trigger automatic rehashing when average
   // collision rate becomes larger than 5.
   THashTable ht2(20,5);

   Printf("Filling THashTable with automatic rehash when AverageCollisions>5");
   Printf("Number of slots before filling: %d", ht2.Capacity());
   for (i = 0; i < 1000; i++)
      ht2.Add(new TObject);

   Printf("Number of slots after filling: %d", ht2.Capacity());
   Printf("Average collisions: %f", ht2.AverageCollisions());

   Printf("\nDelete all heap based objects");
   ht2.Delete();
}

void Test_TBtree()
{
   Printf(
   "////////////////////////////////////////////////////////////////\n"
   "// Test of TBtree                                             //\n"
   "////////////////////////////////////////////////////////////////"
   );
   TStopwatch timer;      // create a timer
   TBtree     l;          // btree of order 3

   Printf("Filling TBtree");

   TObjNumI *n3 = new TObjNumI(3);
   l.Add(n3);
   l.AddBefore(n3,new TObjNumI(5));
   l.AddAfter(n3, new TObjNumI(2));
   l.Add(new TObjNumI(1));
   l.AddBefore(n3, new TObjNumI(4));
   TObjNumI n6(6);                     // stack based TObjNumI
   l.AddFirst(&n6);

   timer.Start();
   for (int i = 0; i < 50; i++)
      l.Add(new TObjNumI(i));
   timer.Print();

   Printf("Print TBtree");
   l.Print();

   Printf("Remove 3 and print TBtree again");
   l.Remove(n3);
   l.Print();

   Printf("Iterate forward over TBtree and remove 4 from tree");
   TIter next(&l);
   TObjNumI *obj;
   while ((obj = (TObjNumI*)next()))
      if (obj->GetNum() == 4) l.Remove(obj);

   Printf("Iterate backward over TBtree and remove 2 from tree");
   TIter next1(&l, kIterBackward);
   while ((obj = (TObjNumI*)next1()))
      if (obj->GetNum() == 2) l.Remove(obj);

   Printf("\nDelete all heap based objects");
   l.Delete();

   Printf("Delete stack based objects (6)");
}


int tcollex() {
   Test_TObjArray();
   Test_TOrdCollection();
   Test_TList();
   Test_TSortedList();
   Test_THashTable();
   Test_TBtree();

   return 0;
}

#ifndef __CINT__
int main() {
   return tcollex();
}
#endif
