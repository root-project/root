//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientIdxVector                                                   // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// A vector class optimized for insertions and deletions                //
//   indexed access takes O(1)                                          //
//   insertion takes O(1) plus a very small fraction of O(n)            //
//   deletion takes O(1) plus a very small fraction of O(n)             //
//                                                                      //
// Better suited than XrdClientVector to hold complex objects           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$


#ifndef XRD_CLIIDXVEC_H
#define XRD_CLIIDXVEC_H

#include <stdlib.h>
#include <string.h>

#include "XrdSys/XrdSysHeaders.hh"


#define IDXVEC_MINCAPACITY       128

template<class T>
class XrdClientVector {


private:

    // We keep the corrected size of T
    int sizeof_t;

    char *rawdata; // A raw mem block to hold (casted) T instances

    struct myindex {
	long offs; // Offset to a T inside rawdata
	bool notempty;
    } *index;

    // the number of holes inside rawdata
    // each hole is sizeof_t bytes
    int holecount;

    long size, mincap;
    long capacity, maxsize;

    // Completely packs rawdata
    // Eventually adjusts the sizes in order to contain at least
    // newsize elements
    int BufRealloc(int newsize);

    inline void Init(int cap = -1) {
	if (rawdata) free(rawdata);
	if (index) free(index);

        mincap = (cap > 0) ? cap : IDXVEC_MINCAPACITY;

	rawdata = static_cast<char *>(malloc(mincap * sizeof_t));

	index = static_cast<myindex *>(malloc(mincap * sizeof(myindex)));

	if (!rawdata || !index) {
	  std::cerr << "XrdClientIdxVector::Init .... out of memory. sizeof_t=" << sizeof_t <<
	    " sizeof(myindex)=" << sizeof(myindex) << " capacity=" << mincap << std::endl;
	  abort();
	}

	// and we make every item as empty, i.e. not pointing to anything
	memset(index, 0, mincap * sizeof(myindex));

	holecount = 0;

	size = 0;
	maxsize = capacity = mincap;
    }

    // Makes a null position... not to be exposed
    // Because after this the element pointed to by el becomes invalid
    // Typically el will be moved at the end, at the size+holecount position
    void DestroyElem(myindex *el) {
      reinterpret_cast<T*>(rawdata+el->offs)->~T();
      //      el->notempty = false;
    }

    void put(T& item, long pos) {
	// Puts an element in position pos
	// Hence write at pos in the index array
	// Use a new chunk of rawdata if the item does not point to a chunk
	if (size+holecount >= capacity) {
	  std::cerr << "XrdClientIdxVector::put .... internal error." << std::endl;
	  abort();
	}
	
	T *p;
	long offs = (size+holecount)*sizeof_t;

	if (index[pos].notempty) {
	    offs = index[pos].offs;

	    // did we fill a hole?
	    holecount--;
	}

	p = new(rawdata + offs) T(item);

	if (p) {
	    index[pos].offs = offs;
	    index[pos].notempty = true;
	}
	else {
	    std::cerr << "XrdClientIdxVector::put .... out of memory." << std::endl;
	    abort();
	}

    }

public:

    inline int GetSize() const { return size; }

    void Clear() {
	for (long i = 0; i < size; i++)
	    if (index[i].notempty) DestroyElem(&index[i]);

	Init(mincap);
    }

    XrdClientVector(int cap = -1):
	sizeof_t(0), rawdata(0), index(0)
    {
	// We calculate a size which is aligned on 4-bytes
	sizeof_t = (sizeof(T) + 3) >> 2 << 2;
	Init(cap);
    }

    XrdClientVector(XrdClientVector &v):
	rawdata(0), index(0) {

        sizeof_t = (sizeof(T) + 3) >> 2 << 2;

	Init(v.capacity);
	BufRealloc(v.size);

	for (int i = 0; i < v.size; i++)
	    Push_back( v[i] );
    }

    ~XrdClientVector() {
        for (long i = 0; i < size; i++)
          if (index[i].notempty) DestroyElem(&index[i]);

	if (rawdata) free(rawdata);
	if (index) free(index);
    }

    void Resize(int newsize) {
        long oldsize = size;

        if (newsize > oldsize) {
           BufRealloc(newsize);
           T *item = new T;
           // Add new elements if needed
           for (long i = oldsize; i < newsize; i++) {
              put(*item, size++);
           }
           delete item;
        }
        else {
           for (long i = oldsize; i > newsize; i--)
              Erase(i-1, false);
        }
    }

    void Push_back(T& item) {

	if ( BufRealloc(size+1) )
	    put(item, size++);

    }

//     // Inserts an item in the given position
//     void Insert(T& item, int pos) {
      
// 	if (pos >= size) {
// 	    Push_back(item);
// 	    return;
// 	}

// 	if ( BufRealloc(size+1) ) {

// 	    memmove(&index[pos+1], &index[pos], (size+holecount-pos) * sizeof(myindex));
// 	    index[pos].notempty = false;
// 	    size++;
// 	    put(item, pos);
// 	}

//     }


    // Inserts an item in the given position
    void Insert(T& item, int pos) {
      
        if (pos >= size) {
            Push_back(item);
            return;
        }

        if ( BufRealloc(size+1) ) {

           if (holecount > 0) {
              struct myindex tmpi = index[size];
              memmove(&index[pos+1], &index[pos], (size-pos) * sizeof(myindex));
              index[pos] = tmpi;
           } else {
              memmove(&index[pos+1], &index[pos], (size-pos) * sizeof(myindex));
              index[pos].notempty = false;
           }

           size++;
           put(item, pos);
	}

    }

//     // Removes a single element in position pos
//    void Erase(unsigned int pos, bool dontrealloc=true) {
// 	// We make the position empty, then move the free index to the end
// 	DestroyElem(index + pos);

// 	index[size+holecount] = index[pos];
// 	holecount++;

// 	memmove(&index[pos], &index[pos+1], (size+holecount-pos) * sizeof(myindex));

// 	size--;

//         if (!dontrealloc)
//            BufRealloc(size);

//     }

    // Removes a single element in position pos
   void Erase(unsigned int pos, bool dontrealloc=true) {
	// We make the position empty, then move the free index to the end of the full items
	DestroyElem(index + pos);

	struct myindex tmpi = index[pos];
	holecount++;

	memmove(&index[pos], &index[pos+1], (size-pos-1) * sizeof(myindex));

	size--;
        index[size] = tmpi;
        if (!dontrealloc)
           BufRealloc(size);

    }

    T Pop_back() {
	T r( At(size-1) );

	DestroyElem(index+size-1);

	holecount++;
	size--;
	//BufRealloc(size);

	return (r);
    }

    T Pop_front() {
	T res;

	res = At(0);

	Erase(0);
	return (res);
    }

    // Bounded array like access
    inline T &At(int pos) {
	//        if ( (pos < 0) || (pos >= size) )
	//            abort();

	return *( reinterpret_cast<T*>(rawdata + index[pos].offs));
    }

    inline T &operator[] (int pos) {
	return At(pos);
    }

};


// Completely packs rawdata if needed
// Eventually adjusts the sizes in order to fit newsize elements
template <class T>
int XrdClientVector<T>::BufRealloc(int newsize) {

    // If for some reason we have too many holes, we repack everything
    // this is very heavy!!
    if ((size+holecount >= capacity-2) && (holecount > 4*size))
	while (size+holecount >= capacity-2) {
	    long lastempty = size+holecount-1;  // The first hole to fill

	    // Pack everything in rawdata
	    // Keep the pointers updated

	    // Do the trick

	    // Move the last filled to the first encountered hole
	    memmove(rawdata + index[lastempty].offs, rawdata + index[lastempty].offs + sizeof_t,
		    (size+holecount)*sizeof_t - index[lastempty].offs );

	    // Drop the index
	    index[lastempty].notempty = false;
	    holecount--;

	    // Adjust all the pointers to the subsequent chunks
	    for (long i = 0; i < size+holecount; i++)
		if (index[i].notempty && (index[i].offs > index[lastempty].offs))
		    index[i].offs -= sizeof_t;
	
	}

    if (newsize > maxsize) maxsize = newsize;

    while (newsize+holecount > capacity*2/3) {
	// Too near to the end?
	// double the capacity

	capacity *= 2;

	rawdata = static_cast<char *>(realloc(rawdata, capacity*sizeof_t));
	if (!rawdata) {
	    std::cerr << "XrdClientIdxVector::BufRealloc .... out of memory." << std::endl;
	    abort();
	}

	index = static_cast<myindex *>(realloc(index, capacity*sizeof(myindex)));
	memset(index+capacity/2, 0, capacity*sizeof(myindex)/2);

    }

    while ((newsize+holecount < capacity/3) && (capacity > 2*mincap)) {
	// Too near to the beginning?
	// half the capacity


	capacity /= 2;

	rawdata = static_cast<char *>(realloc(rawdata, capacity*sizeof_t));
	if (!rawdata) {
	    std::cerr << "XrdClientIdxVector::BufRealloc .... out of memory." << std::endl;
	    abort();
	}

	index = static_cast<myindex *>(realloc(index, capacity*sizeof(myindex)));

    }

    return 1;

}


#endif
