#ifndef __OOUC_HASH__
#define __OOUC_HASH__
/******************************************************************************/
/*                                                                            */
/*                         X r d O u c H a s h . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <time.h>

/*
Hash_data_is_key - The key and data are the same so when an item is added
                   the data pointer is set to the key address.
Hash_replace     - When adding an item, any existing item is replaced.
Hash_count       - The number of deletion requests must equal the number of
                   additions before the item is actually deleted.
Hash_keep        - When the item is added, the key is not duplicated and
                   when the item is deleted, the key *and* data are not deleted.
Hash_dofree      - When an item is deleted the data is released using free()
                   instead of delete().
Hash_keepdata    - Works like Hash_keep but only applies to the data object.
                   When adding the entry, the key is strdup'd and when deleting
                   an entry, the key is freed.
*/
enum XrdOucHash_Options {Hash_default     = 0x0000,
                        Hash_data_is_key = 0x0001,
                        Hash_replace     = 0x0002,
                        Hash_count       = 0x0004,
                        Hash_keep        = 0x0008,
                        Hash_dofree      = 0x0010,
                        Hash_keepdata    = 0x0020
                       };
  
template<class T>
class XrdOucHash_Item
{
public:
int                 Count() {return keycount;}

T                   *Data() {return keydata;}

      unsigned long  Hash() {return keyhash;}

const char          *Key()  {return keyval;}

XrdOucHash_Item<T>   *Next() {return next;}

time_t               Time() {return keytime;}

void                 Update(int newcount, time_t newtime)
                            {keycount = newcount; 
                             if (newtime) keytime = newtime;
                            }

int                  Same(const unsigned long KeyHash, const char *KeyVal)
                         {return keyhash == KeyHash && !strcmp(keyval, KeyVal);}

void                 SetNext(XrdOucHash_Item<T> *item) {next = item;}

     XrdOucHash_Item(unsigned long      KeyHash,
                    const char        *KeyVal,
                    T                 *KeyData,
                    time_t             KeyTime,
                    XrdOucHash_Item<T> *KeyNext,
                    XrdOucHash_Options  KeyOpts)
          {keyhash = KeyHash; 
           if (KeyOpts & Hash_keep) keyval = KeyVal;
              else keyval  = strdup(KeyVal);
           if (KeyOpts & Hash_data_is_key) keydata = (T *)keyval;
              else keydata = KeyData;
           keytime = KeyTime;
           entopts = KeyOpts;
           next    = KeyNext;
           keycount= 0;
          }

    ~XrdOucHash_Item()
          {if (!(entopts & Hash_keep))
              {if (keydata && keydata != (T *)keyval 
               && !(entopts & Hash_keepdata))
                  {if (entopts & Hash_dofree) free(keydata);
                      else delete keydata;
                  }
               if (keyval)  free((void *)keyval);
              }
           keydata = 0; keyval = 0; keycount = 0;
          }

private:

XrdOucHash_Item<T> *next;
const char        *keyval;
unsigned long      keyhash;
T                 *keydata;
time_t             keytime;
int                keycount;
XrdOucHash_Options  entopts;
};

template<class T>
class XrdOucHash
{
public:

// Add() adds a new item to the hash. If it exists and repl = 0 then the old
//       entry is returned and the new data is not added. Otherwise the current
//       entry is replaced (see Rep()) and 0 is returned. If we have no memory
//       to add the new entry, an ENOMEM exception is thrown. The
//       LifeTime value is the number of seconds this entry is to be considered
//       valid. When the time has passed, the entry may be deleted. A value
//       of zero, keeps the entry until explicitly deleted. A special feature
//       allows the data to be associated with the key to be the actual key
//       using the Hash_data_is_key option. In this case, KeyData is ignored.
//       The Hash_count option keeps track of duplicate key entries for Del.
//
T           *Add(const char *KeyVal, T *KeyData, const int LifeTime=0, 
                 XrdOucHash_Options opt=Hash_default);

// Del() deletes the item from the hash. If it doesn't exist, it returns
//       -ENOENT. Otherwise 0 is returned. If the Hash_count option is specified
//       tyhen the entry is only deleted when the entry count is below 0.
//
int          Del(const char *KeyVal, XrdOucHash_Options opt = Hash_default);

// Find() simply looks up an entry in the cache. It can optionally return the
//        lifetime associated with the entry. If the
//
T           *Find(const char *KeyVal, time_t *KeyTime=0);

// Num() returns the number of items in the hash table
//
int          Num() {return hashnum;}

// Purge() simply deletes all of the appendages to the table.
//
void         Purge();

// Rep() is simply Add() that allows replacement.
//
T           *Rep(const char *KeyVal, T *KeyData, const int LifeTime=0,
                 XrdOucHash_Options opt=Hash_default)
                {return Add(KeyVal, KeyData, LifeTime, 
                            (XrdOucHash_Options)(opt | Hash_replace));}

// Apply() applies the specified function to every item in the hash. The
//         first argument is the key value, the second is the associated data,
//         the third argument is whatever is the passed in void *variable, The
//         following actions occur for values returned by the applied function:
//         <0 - The hash table item is deleted.
//         =0 - The next hash table item is processed.
//         >0 - Processing stops and the hash table item is returned.
//
T           *Apply(int (*func)(const char *, T *, void *), void *Arg);

// When allocateing a new hash, specify the required starting size. Make
// sure that the previous number is the correct Fibonocci antecedent. The
// series is simply n[j] = n[j-1] + n[j-2].
//
    XrdOucHash(int psize = 89, int size=144, int load=80);
   ~XrdOucHash() {if (hashtable) {Purge(); free(hashtable); hashtable = 0;}}

private:
void Remove(int kent, XrdOucHash_Item<T> *hip, XrdOucHash_Item<T> *phip);

XrdOucHash_Item<T> *Search(XrdOucHash_Item<T> *hip, 
                          const unsigned long khash,
                          const char *kval, 
                          XrdOucHash_Item<T> **phip=0);

unsigned long HashVal(const char *KeyVal);

void Expand();

XrdOucHash_Item<T> **hashtable;
int                 prevtablesize;
int                 hashtablesize;
int                 hashnum;
int                 hashmax;
int                 hashload;
};

/******************************************************************************/
/*                 A c t u a l   I m p l e m e n t a t i o n                  */
/******************************************************************************/
  
#include "XrdOuc/XrdOucHash.icc"
#endif
