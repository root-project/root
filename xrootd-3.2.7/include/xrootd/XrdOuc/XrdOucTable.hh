#ifndef __OUC_TABLE__
#define __OUC_TABLE__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c T a b l e . h h                         */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>
#include <string.h>

template<class T>
class XrdOucTable
{
public:

         XrdOucTable(int maxe)
                    {int i;
                     Table = new OucTable[maxe];
                     maxnum = maxe; curnum = 0; avlnum = 0;
                     for (i = 1; i < maxe; i++) Table[i-1].Fnum = i;
                     Table[maxe-1].Fnum = -1;
                    }

        ~XrdOucTable() {delete [] Table;}

// Alloc() returns the next free slot number in the table. A negative value
//         indicates that no free slots are left.
//
int  Alloc() {int i = avlnum;
              if (i >= 0) {avlnum = Table[i].Fnum;
                           if (i >= curnum) curnum = i+1;
                          }
              return i;
             }

// Apply() applies the specified function to every item in the list.
//         An argument may be passed to the function. A null pointer is 
//         returned if the list was completely traversed. Otherwise, the 
//         pointer to the node on which the applied function returned a 
//         non-zero value is returned. An optional starting point may be passed.
//
T       *Apply(int (*func)(T *, void *), void *Arg, int Start=0)
         {int i;
          for (i = Start; i < curnum; i++)
              if (Table[i].Item && (*func)(Table[i].Item, Arg))
                 return Table[i].Item;
          return (T *)0;
         }

// Delete() entry at Tnum and destroy it. The key is destroyed and the slot
// is placed on the free list. The second variation of Remove, deletes by key.
//
void Delete(int Tnum)
           {T *temp;
            if ((temp = Remove(Tnum))) delete temp;
           }

void Delete(const char *key) 
           {T *temp;
            if ((temp = Remove(key))) delete temp;
           }

// Find() finds a table entry matching the specified key. It returns the
//        Item associated with the key or zero if it is not found. If the
//        address of an integer is passed, the associated entry number is
//        also returned (it is unchanged if a null is returned).
//
T       *Find(const char *key, int *Tnum=0)
         {int i;
          for (i = 0; i < curnum; i++)
              if (Table[i].Item && Table[i].Key && !strcmp(Table[i].Key, key))
                 {if (Tnum) *Tnum = i; return Table[i].Item;}
          return 0;
         }

// Insert() inserts the specified node at entry Tnum. If Tnum is negative, a free
//          slot is allocated and the item is inserted there. The slot number is
//          returned. A negative slot number indicates the table is full.
//
int Insert(T *Item, const char *key=0, int Tnum=-1)
          {if ((Tnum < 0 && ((Tnum = Alloc()) < 0)) || Tnum >= maxnum) return -1;
           Table[Tnum].Item = Item; Table[Tnum].Key = strdup(key);
           return Tnum;
          }

// Item() supplies the item value associated with entry Tnum; If the address
//        if ikey is not zero, the associated key value is returned.
//
T  *Item(int Tnum, char **ikey=0) 
        {if (Tnum < 0 || Tnum >= curnum || !Table[Tnum].Item) return (T *)0;
         if (ikey) *ikey = Table[Tnum].Key;
         return Table[Tnum].Item;
        }

// Next() iterates through the table using a cursor. This function is
//        useful for unlocked scanning of the table.
//
int Next(int &Tnum) {int i;
                     for (i = Tnum; i < curnum; i++)
                         if (Table[i].Item) {Tnum = i+1; return i;}
                     return -1;
                    }

// Remove() entry at Tnum and returns it. The key is destroyed and the slot
// is placed on the free list. The second variation of Remove, removes by key.
//
T  *Remove(int Tnum)
          {T *temp;
           if (Tnum < 0 || Tnum >= curnum || !Table[Tnum].Item) return (T *)0;
           if (Table[Tnum].Key) free(Table[Tnum].Key);
           temp = Table[Tnum].Item; Table[Tnum].Item = 0;
           Table[Tnum].Fnum = avlnum;
           avlnum = Tnum;
           if (Tnum == (curnum-1))
              while(curnum && Table[curnum].Item == 0) curnum--;
           return temp;
          }

T  *Remove(const char *key) {int i; 
                             if (Find(key, &i)) return Remove(i);
                             return (T *)0;
                            }

private:
struct OucTable {T           *Item;
                union {char *Key;
                        int   Fnum;};
                 OucTable() {Item = 0; Key = 0;}
                ~OucTable() {if (Key)  free(Key);
                             if (Item) delete Item;
                            }
                };

OucTable *Table;
int       avlnum;
int       maxnum;
int       curnum;
};
#endif
