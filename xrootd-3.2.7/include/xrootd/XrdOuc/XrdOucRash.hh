#ifndef __OUC_RASH__
#define __OUC_RASH__
/******************************************************************************/
/*                                                                            */
/*                         X r d O u c R a s h . h h                          */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

// This templated class implements a radix tree to remap binary quantities using
// a hash-like <key,Value> interface. Define the object as:

// XrdOucRash<key_type, value_type> myobject;

// Where: key_type   is the binary type of the key (short, int, long, long long)
//        value_type is the binary type of the value (one of the types above).

// The binary types may be signed or unsigned. Use the methods defined in
// class XrdOucRash to Add(), Del(), Find(), and Rep() items in the table.
// Use Apply() to scan through all of the items in the table and Purge() to
// remove all items in the table (indices are not removed). Several options
// exist to manage the items (see individual methods and XrdOucRash_Options).

// Warning! This class is not MT-safe and should be protected by an external
//          mutex when used in a multi-threaded environment.
 
#include <sys/types.h>
#include <time.h>

enum XrdOucRash_Options {Rash_default     = 0x0000,
                         Rash_replace     = 0x0002,
                         Rash_count       = 0x0004
                       };

template<typename K, typename V>
class XrdOucRash_Item
{
public:
int                  Count() {return keycount;}

V                   *Data() {return &keydata;}

K                    Key()  {return keyval;}

time_t               Time() {return keytime;}

void                 Update(int newcount, time_t newtime)
                            {keycount = newcount; 
                             if (newtime) keytime = newtime;
                            }

void                 Set(V &keyData, time_t newtime)
                            {keydata = keyData;
                             keytime = newtime;
                            }

     XrdOucRash_Item(K                  &KeyVal,
                     V                  &KeyData,
                     time_t             KeyTime)
          {keyval  = KeyVal;
           keydata = KeyData;
           keytime = KeyTime;
           keycount= 0;
          }

    ~XrdOucRash_Item() {}

private:

K                  keyval;
V                  keydata;
time_t             keytime;
int                keycount;
};

template<typename K, typename V>
class XrdOucRash_Tent
{
public:
XrdOucRash_Tent<K,V> *Table;
XrdOucRash_Item<K,V> *Item;

      XrdOucRash_Tent() {Table = 0; Item = 0;}
     ~XrdOucRash_Tent() {if (Table) delete[] Table;
                         if (Item)  delete(Item);
                         }
};

template<typename K, typename V>
class XrdOucRash
{
public:

// Add() adds a new item to the table. If it exists and repl = 0 then the old
//       entry is returned and the new data is not added. Otherwise the current
//       entry is replaced (see Rep()) and 0 is returned. If we have no memory
//       to add the new entry, an ENOMEM exception is thrown. The
//       LifeTime value is the number of seconds this entry is to be considered
//       valid. When the time has passed, the entry may be deleted. A value
//       of zero, keeps the entry until explicitly deleted. The Hash_count 
//       option keeps track of duplicate key entries for Del. Thus the key must
//       be deleted as many times as it is added before it is physically deleted.
//
V           *Add(K KeyVal, V &KeyData, time_t LifeTime=0,
                 XrdOucRash_Options opt=Rash_default);

// Del() deletes the item from the table. If it doesn't exist, it returns
//       -ENOENT. If it was deleted it returns 0. If it was created with
//       Rash_Count then the count is decremented and count+1 is returned.
//
int          Del(K KeyVal);

// Find() simply looks up an entry in the cache. It can optionally return the
//        lifetime associated with the entry. If the
//
V           *Find(K KeyVal, time_t *KeyTime=0);

// Num() returns the number of items in the table
//
int          Num() {return rashnum;}

// Purge() simply deletes all of the appendages to the table.
//
void         Purge();

// Rep() is simply Add() that allows replacement.
//
V           *Rep(K KeyVal, V &KeyData, const int LifeTime=0,
                 XrdOucRash_Options opt=Rash_default)
                {return Add(KeyVal, KeyData, LifeTime, 
                           (XrdOucRash_Options)(opt | Rash_replace));}

// Apply() applies the specified function to every item in the table. The
//         first argument is the key value, the second is the associated data,
//         the third argument is whatever is the passed in void *variable, The
//         following actions occur for values returned by the applied function:
//         <0 - The table item is deleted.
//         =0 - The next table item is processed.
//         >0 - Processing stops and the address of item is returned.
//
V           *Apply(int (*func)(K, V, void *), void *Arg)
                  {return Apply(rashTable, func, Arg);}

    XrdOucRash() {rashnum = 0;}
   ~XrdOucRash() {Purge();}

private:
V                    *Apply(XrdOucRash_Tent<K,V> *tab,
                            int (*func)(K, V, void *), void *Arg);
XrdOucRash_Item<K,V> *Lookup(K theKey, XrdOucRash_Tent<K,V> **tloc);
void                  Insert(K theKey, XrdOucRash_Item<K,V> *theItem);
unsigned long long    key2ull(K theKey);

XrdOucRash_Tent<K,V> rashTable[16];
int                  rashnum;
};

/******************************************************************************/
/*                 A c t u a l   I m p l e m e n t a t i o n                  */
/******************************************************************************/
  
#include "XrdOuc/XrdOucRash.icc"
#endif
