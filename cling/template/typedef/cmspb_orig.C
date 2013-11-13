# 1 "/tmp/filesVpDZl_cint.cxx"
# 1 "CARF/PythiaSimEvent/src/PythiaSimEventLinkDef.h" 1
# 1 "CARF/PythiaSimEvent/interface/PythiaSimEvent.h" 1
/* C++ header file: Objectivity/DB DDL version 6.1.0         */




//
//
//   V 0.1  VI 8/10/2000 
//          a SimEvent with Phytia generator infos

# 1 "Utilities/Persistency/interface/Persistency.h" 1


# 1 "Utilities/Persistency/interface/enums.h" 1



# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_string" 1
// string






#pragma include_noerr <string.dll>




# 1 "/home/wmtan/root/cint/lib/prec_stl/string" 1
// lib/prec_stl/string

#pragma ifndef PREC_STL_STRING
#pragma define PREC_STL_STRING
#pragma link off global PREC_STL_STRING;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// string class wrapper , by Masaharu Goto
// Template is on purposely avoided.

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */






/* Any one of these symbols __need_* means that GNU libc
   wants us just to define one data type.  So don't define
   the symbols that indicate this file's entire job has been done.  */





/* snaroff@next.com says the NeXT needs this.  */

/* Irix 5.1 needs this.  */




/* This avoids lossage on SunOS but only if stdtypes.h comes first.
   There's no way to win with the other order!  Sun lossage.  */

/* On 4.3bsd-net2, make sure ansi.h is included, so we have
   one less case to deal with in the following.  */




/* In 4.3bsd-net2, machine/ansi.h defines these symbols, which are
   defined if the corresponding type is *not* defined.
   FreeBSD-2.1 defines _MACHINE_ANSI_H_ instead of _ANSI_H_ */
/* defined(_ANSI_H_) || defined(_MACHINE_ANSI_H_) */
# 72 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"


/* Sequent's header files use _PTRDIFF_T_ in some conflicting way.
   Just ignore it.  */




/* On VxWorks, <type/vxTypesBase.h> may have defined macros like
   _TYPE_size_t which will typedef size_t.  fixincludes patched the
   vxTypesBase.h so that this macro is only defined if _GCC_SIZE_T is
   not defined, and so that defining this macro defines _GCC_SIZE_T.
   If we find that the macros are still defined at this point, we must
   invoke them so that the type is defined as expected.  */













/* In case nobody has defined these types, but we aren't running under
   GCC 2.00, make sure that __PTRDIFF_TYPE__, __SIZE__TYPE__, and
   __WCHAR_TYPE__ have reasonable values.  This can happen if the
   parts of GCC is compiled by an older compiler, that actually
   include gstddef.h, such as collect2.  */

/* Signed type of difference of two pointers.  */

/* Define this type if we are doing the whole job,
   or if we want this type in particular.  */

/* in case <sys/types.h> has defined it. */


















typedef int ptrdiff_t;
/* _GCC_PTRDIFF_T */
/* ___int_ptrdiff_t_h */
/* _BSD_PTRDIFF_T_ */
/* _PTRDIFF_T_ */
/* __PTRDIFF_T */
/* _T_PTRDIFF */
/* _T_PTRDIFF_ */
/* _PTRDIFF_T */

/* If this symbol has done its job, get rid of it.  */


/* _STDDEF_H or __need_ptrdiff_t.  */

/* Unsigned type of `sizeof' something.  */

/* Define this type if we are doing the whole job,
   or if we want this type in particular.  */

/* BeOS */
/* in case <sys/types.h> has defined it. */












/* BeOS */

















typedef unsigned int size_t;
/* __BEOS__ */


/* !(defined (__GNUG__) && defined (size_t)) */
/* __size_t */
/* _SIZET_ */
/* _GCC_SIZE_T */
/* ___int_size_t_h */
/* _SIZE_T_DEFINED */
/* _SIZE_T_DEFINED_ */
/* _BSD_SIZE_T_ */
/* _SIZE_T_ */
/* __SIZE_T */
/* _T_SIZE */
/* _T_SIZE_ */
/* _SYS_SIZE_T_H */
/* _SIZE_T */
/* __size_t__ */

/* _STDDEF_H or __need_size_t.  */


/* Wide character type.
   Locale-writers should change this as necessary to
   be big enough to hold unique values not between 0 and 127,
   and not (wchar_t) -1, for each defined multibyte character.  */

/* Define this type if we are doing the whole job,
   or if we want this type in particular.  */

/* BeOS */












/* BeOS */













/* On BSD/386 1.1, at least, machine/ansi.h defines _BSD_WCHAR_T_
   instead of _WCHAR_T_, and _BSD_RUNE_T_ (which, unlike the other
   symbols in the _FOO_T_ family, stays defined even after its
   corresponding type is defined).  If we define wchar_t, then we
   must undef _WCHAR_T_; for BSD/386 1.1 (and perhaps others), if
   we undef _WCHAR_T_, then we must also define rune_t, since 
   headers like runetype.h assume that if machine/ansi.h is included,
   and _BSD_WCHAR_T_ is not defined, then rune_t is available.
   machine/ansi.h says, "Note that _WCHAR_T_ and _RUNE_T_ must be of
   the same type." */































/* __wchar_t__ */

/* _STDDEF_H or __need_wchar_t.  */








typedef unsigned int  wint_t;




/*  In 4.3bsd-net2, leave these undefined to indicate that size_t, etc.
    are already defined.  */
/*  BSD/OS 3.1 requires the MACHINE_ANSI_H check here.  FreeBSD 2.x apparently
    does not, even though there is a check for MACHINE_ANSI_H above.  */
/* _ANSI_H_ || ( __bsdi__ && _MACHINE_ANSI_H_ ) */
# 328 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"


/* __sys_stdtypes_h */

/* A null pointer constant.  */


/* in case <stdio.h> has defined it. */


/* G++ *//* G++ */







/* NULL not defined and <stddef.h> or need NULL.  */




/* Offset of member MEMBER in a struct of type TYPE.  */



/* _STDDEF_H was defined this time */

/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 17 "/home/wmtan/root/cint/lib/prec_stl/string" 2


//////////////////////////////////////////////////////////////////////////
class string {
 public:
  typedef char value_type;
  typedef char* iterator;
  typedef const char* const_iterator;
  typedef char* pointer;
  typedef const char* const_pointer;
  typedef char& reference;
  typedef const char& const_reference;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;
  typedef int traits_type;

  enum { npos=-1 };
  string() ;
  //string(size_t size,capacity cap) ;
  string(const string& str) ;
  string(const string& str,size_t pos,size_t n) ;
  string(const char* s,size_t n) ;
  string(const char* s) ;
  string(char c,size_t rep);
  //string(const vector<char>& vec);
  ~string() ;
  string& operator=(const string& str);
  string& operator=(const char* s);
  string& operator=(char c);
  string& operator+=(const string& rhs);
  string& operator+=(const char* s);
  string& operator+=(char c);
  //vector<char> operator vector<char>(void) const;
  string& append(const string& str);
  string& append(const string& str,size_t pos,size_t n);
  string& append(const char* s,size_t n);
  string& append(const char* s);
  string& append(char c,size_t rep);
  string& assign(const string& str);
  string& assign(const string& str,size_t pos,size_t n);
  string& assign(const char* s,size_t n);
  string& assign(const char* s);
  string& assign(char c,size_t rep);
  string& insert(size_t pos1,const string& str);
  string& insert(size_t pos1,const string& str,size_t pos2,size_t n);
  string& insert(size_t pos,const char* s,size_t n);
  string& insert(size_t pos,const char* s);
  string& insert(size_t pos,char c,size_t rep);
  //string& remove(size_t pos=0,size_t n=npos);
  string& replace(size_t pos1,size_t n1,const string& str);
  string& replace(size_t pos1,size_t n1,const string& str,size_t pos2,size_t n2);
  string& replace(size_t pos,size_t n1,const char* s,size_t n2);
  string& replace(size_t pos,size_t n1,const char* s);
  string& replace(size_t pos,size_t n,char c,size_t rep);
  //char get_at(size_t pos) const;
  //void put_at(size_t pos,char c);
  char operator[](size_t pos) const;
  const char* c_str(void) const;
  const char* data(void) const;
  size_t length(void) const;
  void resize(size_t n,char c);
  void resize(size_t n);
  int size();
  //size_t reserve(void) const;
  void reserve(size_t res_arg);
  size_t copy(char* s,size_t n,size_t pos=0) /* const */;
  size_t find(const string& str,size_t pos=0) const;
  size_t find(const char* s,size_t pos,size_t n) const;
  size_t find(const char* s,size_t pos=0) const;
  size_t find(char c,size_t pos=0) const;
  size_t rfind(const string& str,size_t pos=npos) const;
  size_t rfind(const char* s,size_t pos,size_t n) const;
  size_t rfind(const char* s,size_t pos=npos) const;
  size_t rfind(char c,size_t pos=npos) const;
  size_t find_first_of(const string& str,size_t pos=0) const;
  size_t find_first_of(const char* s,size_t pos,size_t n) const;
  size_t find_first_of(const char* s,size_t pos=0) const;
  size_t find_first_of(char c,size_t pos=0) const;
  size_t find_last_of(const string& str,size_t pos=npos) const;
  size_t find_last_of(const char* s,size_t pos,size_t n) const;
  size_t find_last_of(const char* s,size_t pos=npos) const;
  size_t find_last_of(char c,size_t pos=npos) const;
  size_t find_first_not_of(const string& str,size_t pos=0) const;
  size_t find_first_not_of(const char* s,size_t pos,size_t n) const;
  size_t find_first_not_of(const char* s,size_t pos=0) const;
  size_t find_first_not_of(char c,size_t pos=0) const;
  size_t find_last_not_of(const string& str,size_t pos=npos) const;
  size_t find_last_not_of(const char* s,size_t pos,size_t n) const;
  size_t find_last_not_of(const char* s,size_t pos=npos) const;
  size_t find_last_not_of(char c,size_t pos=npos) const;
  string substr(size_t pos=0,size_t n=npos) const;
  int compare(const string& str) const;
  //int compare(size_type pos1,size_type n1,const string& str,size_type n2) const;
  int compare(const char* s) const ;
  //int compare(size_type pos1,size_type n1,const char* s,size_type n2=npos)const;
  //operator char*() ;
};

bool operator==(const string& a,const string& b) ;
bool operator!=(const string& a,const string& b) ;
bool operator<(const string& a,const string& b) ;
bool operator>(const string& a,const string& b) ;
bool operator<=(const string& a,const string& b) ;
bool operator>=(const string& a,const string& b) ;
string operator+(const string& a,const string& b) ;

typedef string cstring;

#pragma endif 

# 13 "/home/wmtan/root/cint/stl/_string" 2


// __MAKECINT__// __MAKECINT__
# 63 "/home/wmtan/root/cint/stl/_string"



# 2 "/home/wmtan/root/cint/stl/string" 2

}
# 4 "Utilities/Persistency/interface/enums.h" 2

# 1 "/home/wmtan/root/cint/stl/map" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_map" 1

#pragma include_noerr <map.dll>
#pragma include_noerr <map2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/map" 1
// lib/prec_stl/map

#pragma ifndef PREC_STL_MAP
#pragma define PREC_STL_MAP
#pragma link off global PREC_STL_MAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/map" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class map {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const map::iterator& x ,const map::iterator& y) const;
  friend bool operator!=(const map::iterator& x ,const map::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;
  friend bool operator!=(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.map.cons_ construct/copy/destroy:
  map();






  map(iterator first, iterator last);
  map(reverse_iterator first, reverse_iterator last);

  map(const map& x);
  ~map();
  map& operator=(const map& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.map.access_ element access:
  T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(map&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.map.ops_ map operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const map& x, const map& y);
  friend bool operator< (const map& x, const map& y);
  friend bool operator!=(const map& x, const map& y);
  friend bool operator> (const map& x, const map& y);
  friend bool operator>=(const map& x, const map& y);
  friend bool operator<=(const map& x, const map& y);
  // specialized algorithms:






  // Generic algorithm
  friend map::iterator
    search(map::iterator first1,map::iterator last1,
           map::iterator first2,map::iterator last2);


  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(map::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_map" 2






# 1 "/home/wmtan/root/cint/stl/_multimap" 1

#pragma include_noerr <multimap.dll>
#pragma include_noerr <multimap2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multimap" 1
// lib/prec_stl/multimap

#pragma ifndef PREC_STL_MULTIMAP
#pragma define PREC_STL_MULTIMAP
#pragma link off global PREC_STL_MULTIMAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multimap {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multimap::iterator& x ,const multimap::iterator& y) const;
  friend bool operator!=(const multimap::iterator& x ,const multimap::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;
  friend bool operator!=(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multimap.cons_ construct/copy/destroy:
  multimap();






  multimap(iterator first, iterator last);
  multimap(reverse_iterator first, reverse_iterator last);

  multimap(const multimap& x);
  ~multimap();
  multimap& operator=(const multimap& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.multimap.access_ element access:
  //T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(multimap&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.multimap.ops_ multimap operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const multimap& x, const multimap& y);
  friend bool operator< (const multimap& x, const multimap& y);
  friend bool operator!=(const multimap& x, const multimap& y);
  friend bool operator> (const multimap& x, const multimap& y);
  friend bool operator>=(const multimap& x, const multimap& y);
  friend bool operator<=(const multimap& x, const multimap& y);
  // specialized algorithms:






  // Generic algorithm
  friend multimap::iterator
    search(multimap::iterator first1,multimap::iterator last1,
           multimap::iterator first2,multimap::iterator last2);



  // Generic algorithm
  //friend void reverse(multimap::iterator first,multimap::iterator last);
  //friend void reverse(multimap::reverse_iterator first,multimap::reverse_itetator last);

  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(multimap::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif



# 7 "/home/wmtan/root/cint/stl/_multimap" 2




# 13 "/home/wmtan/root/cint/stl/_map" 2

# 2 "/home/wmtan/root/cint/stl/map" 2

}
# 5 "Utilities/Persistency/interface/enums.h" 2

# 1 "/home/wmtan/root/include/TRef.h" 1
// @(#)root/cont:$Id$
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRef                                                                 //
//                                                                      //
// Persistent Reference link to a TObject                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



# 1 "/home/wmtan/root/include/TObject.h" 1
// @(#)root/base:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObject                                                              //
//                                                                      //
// Mother of all ROOT objects.                                          //
//                                                                      //
// The TObject class provides default behaviour and protocol for all    //
// objects in the ROOT system. It provides protocol for object I/O,     //
// error handling, sorting, inspection, printing, drawing, etc.         //
// Every object which inherits from TObject can be stored in the        //
// ROOT collection classes.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/Rtypes.h" 1
/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rtypes                                                               //
//                                                                      //
// Basic types used by ROOT.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



# 1 "/home/wmtan/root/include/RConfig.h" 1
/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




/*************************************************************************
 *                                                                       *
 * RConfig                                                               *
 *                                                                       *
 * Defines used by ROOT.                                                 *
 *                                                                       *
 *************************************************************************/


# 1 "/home/wmtan/root/include/RVersion.h" 1



/* Version information automatically generated by installer. */

/*
 * These macros can be used in the following way:
 *
 *    #if ROOT_VERSION_CODE >= ROOT_VERSION(2,23,4)
 *       #include <newheader.h>
 *    #else
 *       #include <oldheader.h>
 *    #endif
 *
*/






# 23 "/home/wmtan/root/include/RConfig.h" 2




/*---- new C++ features ------------------------------------------------------*/



/*---- machines --------------------------------------------------------------*/

# 46 "/home/wmtan/root/include/RConfig.h"















# 94 "/home/wmtan/root/include/RConfig.h"


# 118 "/home/wmtan/root/include/RConfig.h"


# 129 "/home/wmtan/root/include/RConfig.h"



# 1 "/usr/include/features.h" 1 3
/* Copyright (C) 1991,92,93,95,96,97,98,99 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */




/* These are defined by the user (or the compiler)
   to specify the desired environment:

   __STRICT_ANSI__      ISO Standard C.
   _ISOC99_SOURCE       Extensions to ISO C 89 from ISO C 99.
   _ISOC9X_SOURCE       Similar, kept for compatibility.
   _POSIX_SOURCE        IEEE Std 1003.1.
   _POSIX_C_SOURCE      If ==1, like _POSIX_SOURCE; if >=2 add IEEE Std 1003.2;
                        if >=199309L, add IEEE Std 1003.1b-1993;
                        if >=199506L, add IEEE Std 1003.1c-1995
   _XOPEN_SOURCE        Includes POSIX and XPG things.  Set to 500 if
                        Single Unix conformance is wanted.
   _XOPEN_SOURCE_EXTENDED XPG things and X/Open Unix extensions.
   _LARGEFILE_SOURCE    Some more functions for correct standard I/O.
   _LARGEFILE64_SOURCE  Additional functionality from LFS for large files.
   _FILE_OFFSET_BITS=N  Select default filesystem interface.
   _BSD_SOURCE          ISO C, POSIX, and 4.3BSD things.
   _SVID_SOURCE         ISO C, POSIX, and SVID things.
   _GNU_SOURCE          All of the above, plus GNU extensions.
   _REENTRANT           Select additionally reentrant object.
   _THREAD_SAFE         Same as _REENTRANT, often used by other systems.

   The `-ansi' switch to the GNU C compiler defines __STRICT_ANSI__.
   If none of these are defined, the default is all but _GNU_SOURCE.
   If more than one of these are defined, they accumulate.
   For example __STRICT_ANSI__, _POSIX_SOURCE and _POSIX_C_SOURCE
   together give you ISO C, 1003.1, and 1003.2, but nothing else.

   These are defined by this file and are used by the
   header files to decide what to declare or define:

   __USE_ISOC9X         Define ISO C 9X things.
   __USE_POSIX          Define IEEE Std 1003.1 things.
   __USE_POSIX2         Define IEEE Std 1003.2 things.
   __USE_POSIX199309    Define IEEE Std 1003.1, and .1b things.
   __USE_POSIX199506    Define IEEE Std 1003.1, .1b, .1c and .1i things.
   __USE_XOPEN          Define XPG things.
   __USE_XOPEN_EXTENDED Define X/Open Unix things.
   __USE_UNIX98         Define Single Unix V2 things.
   __USE_LARGEFILE64    Define LFS things with separate names.
   __USE_FILE_OFFSET64  Define 64bit interface as default.
   __USE_BSD            Define 4.3BSD things.
   __USE_SVID           Define SVID things.
   __USE_MISC           Define things common to BSD and System V Unix.
   __USE_GNU            Define GNU extensions.
   __USE_REENTRANT      Define reentrant/thread-safe *_r functions.
   __FAVOR_BSD          Favor 4.3BSD things in cases of conflict.

   The macros `__GNU_LIBRARY__', `__GLIBC__', and `__GLIBC_MINOR__' are
   defined by this file unconditionally.  `__GNU_LIBRARY__' is provided
   only for compatibility.  All new code should use the other symbols
   to test for features.

   All macros listed above as possibly being defined by this file are
   explicitly undefined if they are not explicitly defined.
   Feature-test macros that are not defined by the user or compiler
   but are implied by the other feature-test macros defined (or by the
   lack of any definitions) are defined by the file.  */


/* Undefine everything, so we get a clean slate.  */



















/* Suppress kernel-name space pollution unless user expressedly asks
   for it.  */




/* Always use ISO C things.  */



/* If _BSD_SOURCE was defined by the user, favor BSD over POSIX.  */







/* If _GNU_SOURCE was defined by the user, turn on all the other features.  */
# 138 "/usr/include/features.h" 3


/* If nothing (other than _GNU_SOURCE) is defined,
   define _BSD_SOURCE and _SVID_SOURCE.  */








/* This is to enable the ISO C 9x extension.  It will go away as soon
   as this standard is officially released.  */




/* If none of the ANSI/POSIX macros are defined, use POSIX.1 and POSIX.2
   (and IEEE Std 1003.1b-1993 unless _XOPEN_SOURCE is defined).  */


























# 196 "/usr/include/features.h" 3


































/* We do support the IEC 559 math functionality, real and complex.  */



/* This macro indicates that the installed library is the GNU C Library.
   For historic reasons the value now is 6 and this will stay from now
   on.  The use of this variable is deprecated.  Use __GLIBC__ and
   __GLIBC_MINOR__ now (see below) when you want to test for a specific
   GNU C library version and use the values in <gnu/lib-names.h> to get
   the sonames of the shared libraries.  */



/* Major and minor version number of the GNU C library package.  Use
   these macros to test for features in specific releases.  */



/* This is here only because every header file already includes this one.  */

# 1 "/home/wmtan/root/cint/include/sys/cdefs.h" 1
/* dummy */
# 250 "/usr/include/features.h" 2 3


/* If we don't have __REDIRECT, prototypes will be missing if
   __USE_FILE_OFFSET64 but not __USE_LARGEFILE[64]. */





/* !ASSEMBLER */

/* Decide whether we can define 'extern inline' functions in headers.  */





/* This is here only because every header file already includes this one.  */

/* Get the definitions of all the appropriate `__stub_FUNCTION' symbols.
   <gnu/stubs.h> contains `#define __stub_FUNCTION' when FUNCTION is a stub
   which will always return failure (and set errno to ENOSYS).

   We avoid including <gnu/stubs.h> when compiling the C library itself to
   avoid a dependency loop.  stubs.h depends on every object file.  If
   this #include were done for the library source code, then every object
   file would depend on stubs.h.  */

# 1 "/usr/include/gnu/stubs.h" 1 3
/* This file is automatically generated.
   It defines a symbol `__stub_FUNCTION' for each function
   in the C library which is a stub, meaning it will fail
   every time called, usually setting errno to ENOSYS.  */



































# 278 "/usr/include/features.h" 2 3



/* features.h  */
# 132 "/home/wmtan/root/include/RConfig.h" 2

















/* turn off if you really want to run on an i386 */









































# 199 "/home/wmtan/root/include/RConfig.h"



























/* MacOS X support, initially following FreeBSD */















/* egcs 1.0.3 */
/* supports overloading of new[] and delete[] */
/* supports overloading placement delete */

/* egcs 1.1.x */
/* ANSI C++ Standard Library conformant */




/* gcc 2.9x (MINOR is 9!) */























# 289 "/home/wmtan/root/include/RConfig.h"



















# 325 "/home/wmtan/root/include/RConfig.h"


# 338 "/home/wmtan/root/include/RConfig.h"












/*--- memory and object statistics -------------------------------------------*/

/* #define R__NOSTATS */


/*--- cpp --------------------------------------------------------------------*/


    /* symbol concatenation operator */




    /* stringizing */


# 374 "/home/wmtan/root/include/RConfig.h"


/* produce an identifier that is almost unique inside a file */




    /* Currently CINT does not really mind to have duplicates and     */
    /* does not work correctly as far as merging tokens is concerned. */



/*---- misc ------------------------------------------------------------------*/









# 24 "/home/wmtan/root/include/Rtypes.h" 2



# 1 "/home/wmtan/root/include/DllImport.h" 1
/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*
  This include file defines DllImport/DllExport macro
  to build DLLs under Windows OS.

  They are defined as dummy for UNIX's
*/




# 33 "/home/wmtan/root/include/DllImport.h"




# 27 "/home/wmtan/root/include/Rtypes.h" 2



# 1 "/home/wmtan/root/include/Rtypeinfo.h" 1
// @(#)root/base:$Id$
// Author: Philippe Canal   23/2/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/








# 30 "/home/wmtan/root/include/Rtypeinfo.h"


# 1 "/home/wmtan/root/cint/include/typeinfo" 1
namespace std {
# 1 "/home/wmtan/root/cint/include/typeinfo.h" 1
/*********************************************************************
* typeinfo.h
*
*  Run time type identification
*
* Memo:
*   typeid(typename) , typeid(expression) is implemented as special 
*  function in the cint body src/G__func.c. 
*
*   As an extention, G__typeid(char *name) is defined in src/G__func.c
*  too for more dynamic use of the typeid.
*
*   type_info is extended to support non-polymorphic type objects.
*
*   In src/G__sizeof.c , G__typeid() is implemented. It relies on
*  specific binary layout of type_info object. If order of type_info
*  member declaration is modified, src/G__sizeof.c must be modified
*  too.
*
*********************************************************************/






# 1 "/home/wmtan/root/cint/include/bool.h" 1
#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#pragma ifdef G__OLDIMPLEMENTATION1604
/* This header file may not be needed any more */

//#undef FALSE
//#undef TRUE



// bool as fundamental type
const bool false=0,true=1;



bool bool() { return false; }

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

#pragma endif

# 27 "/home/wmtan/root/cint/include/typeinfo.h" 2


/*********************************************************************
* Functions embedded in cint core
* Most of those functions are defined in src/sizeof.c
* 
*********************************************************************/
// type_info typeid(expression);
// type_info typeid(char *typename);
// type_info G__typeid(char *expression);
// long G__get_classinfo(char *item,int tagnum);
// long G__get_variableinfo(char *item,long *handle,long *index,long tagnum);
// long G__get_functioninfo(char *item,long *handle,long &index,long tagnum);


/*********************************************************************
* type_info
*
*  Included in ANSI/ISO resolution proposal 1995 spring
* 
*********************************************************************/
class type_info {
 public:
  virtual ~type_info() { }  // type_info is polymorphic
  bool operator==(const type_info&) const;
  bool operator!=(const type_info&) const;
  bool before(const type_info&) const;

  const char* name() const;

 private:
  type_info(const type_info&);
 protected: // original enhancement
  type_info& operator=(const type_info&);

  // implementation dependent representation
 protected:
  long type;      // intrinsic types
  long tagnum;    // class/struct/union
  long typenum;   // typedefs
  long reftype;   // pointing level and reference types
  long size;      // size of the object

 public: // original enhancement
  type_info() { }
};


bool type_info::operator==(const type_info& a) const
{
  if(reftype == a.reftype && tagnum == a.tagnum && type == a.type) 
    return(true);
  else 
    return(false);
}

bool type_info::operator!=(const type_info& a) const
{
  if( *this == a ) return(false);
  else             return(true);
}

bool type_info::before(const type_info& a) const
{
  if(-1!=tagnum) 
    return( tagnum < a.tagnum );
  else if(-1!=a.tagnum) 
    return( -1 < a.tagnum );
  else 
    return( type < a.type );
}

const char* type_info::name() const
{
  static char namestring[100];
  //printf("%d %d %d %d\n",type,tagnum,typenum,reftype);
  strcpy(namestring,G__type2string(type,tagnum,typenum,reftype));
  return(namestring);
}

type_info::type_info(const type_info& a)
{
  type = a.type;
  tagnum = a.tagnum;
  typenum = a.typenum;
  reftype = a.reftype;
  size = a.size;
}

type_info& type_info::operator=(const type_info& a)
{
  type = a.type;
  tagnum = a.tagnum;
  typenum = a.typenum;
  reftype = a.reftype;
  size = a.size;
  return(*this);
}

/**************************************************************************
* original enhancment
**************************************************************************/
type_info::type_info()
{
  type = 0;
  tagnum = typenum = -1;
  reftype = 0;
}


/**************************************************************************
* Further runtime type checking requirement from Fons Rademaker
**************************************************************************/

/*********************************************************************
* G__class_info
*
*********************************************************************/
class G__class_info : public type_info {
 public:
  G__class_info() { init(); }
  G__class_info(type_info& a) { init(a); }
  G__class_info(char *classname) { init(G__typeid(classname)); }
  
  void init() {
    typenum = -1;
    reftype = 0;
    tagnum = G__get_classinfo("next",-1);
    size = G__get_classinfo("size",tagnum);
    type = G__get_classinfo("type",tagnum);
  }

  void init(type_info& a) {
    type_info *p=this;
    *p = a;
  }

  G__class_info& operator=(G__class_info& a) {
    type = a.type;
    tagnum = a.tagnum;
    typenum = a.typenum;
    reftype = a.reftype;
    size = a.size;
  }

  G__class_info& operator=(type_info& a) {
    init(a);
  }

  G__class_info* next() {
    tagnum=G__get_classinfo("next",tagnum);
    if(-1!=tagnum) return(this);
    else {
      size = type = 0;
      return((G__class_info*)__null );
    }
  }

  char *title() {
    return((char*)G__get_classinfo("title",tagnum));
  }

  // char *name() is inherited from type_info

  char *baseclass() {
    return((char*)G__get_classinfo("baseclass",tagnum));
  }


  int isabstract() {
    return((int)G__get_classinfo("isabstract",tagnum));
  }

  // can be implemented
  // int iscompiled();

  int Tagnum() {
    return(tagnum);
  }

};
  

/*********************************************************************
* G__variable_info
*
*********************************************************************/
class G__variable_info {
 public:
  G__variable_info() { init(); }
  G__variable_info(G__class_info& a) { init(a); }
  G__variable_info(char *classname) { init(G__class_info(classname)); }

  void init() {
    G__get_variableinfo("new",&handle,&index,tagnum=-1);
  }

  void init(G__class_info& a) {
    G__get_variableinfo("new",&handle,&index,tagnum=a.Tagnum());
  }

  G__variable_info* next() {
    if(G__get_variableinfo("next",&handle,&index,tagnum)) return(this);
    else  return((G__variable_info*)__null );
  }

  char *title() {
    return((char*)G__get_variableinfo("title",&handle,&index,tagnum));
  }

  char *name() {
    return((char*)G__get_variableinfo("name",&handle,&index,tagnum));
  }

  char *type() {
    return((char*)G__get_variableinfo("type",&handle,&index,tagnum));
  }

  int offset() {
    return((int)G__get_variableinfo("offset",&handle,&index,tagnum));
  }

  // can be implemented
  // char *access(); // return public,protected,private
  // int isstatic();
  // int iscompiled();

 private:
  long handle; // pointer to variable table
  long index;
  long tagnum; // class/struct identity
};  

/*********************************************************************
* G__function_info
*
*********************************************************************/
class G__function_info {
 public:
  G__function_info() { init(); }
  G__function_info(G__class_info& a) { init(a); }
  G__function_info(char *classname) { init(G__class_info(classname)); }

  void init() {
    G__get_functioninfo("new",&handle,&index,tagnum=-1);
  } // initialize for global function

  void init(G__class_info& a) {
    G__get_functioninfo("new",&handle,&index,tagnum=a.Tagnum());
  } // initialize for member function

  G__function_info* next() {
    if(G__get_functioninfo("next",&handle,&index,tagnum)) return(this);
    else return((G__function_info*)__null );
  }

  char *title() {
    return((char*)G__get_functioninfo("title",&handle,&index,tagnum));
  }

  char *name() {
    return((char*)G__get_functioninfo("name",&handle,&index,tagnum));
  }

  char *type() {
    return((char*)G__get_functioninfo("type",&handle,&index,tagnum));
  }

  char *arglist() {
    return((char*)G__get_functioninfo("arglist",&handle,&index,tagnum));
  }

  // can be implemented
  // char *access(); // return public,protected,private
  // int isstatic();
  // int iscompiled();
  // int isvirtual();
  // int ispurevirtual();

 private:
  long handle; // pointer to variable table
  long index;
  long tagnum; // class/struct identity
};  

/*********************************************************************
* G__string_buf
*
*  This struct is used as temporary object for returning title strings.
* Size of buf[] limits maximum length of the title string you can
* describe. You can increase size of it here to increase it.
*
*********************************************************************/
struct G__string_buf {
  char buf[256];
};


/*********************************************************************
* Example code
*
*  Following functions are the examples of how to use the type info
* facilities.
*
*********************************************************************/


void G__list_class(void) {
  G__class_info a;
  do {
    printf("%s:%s =%d '%s'\n",a.name(),a.baseclass(),a.isabstract(),a.title());
  } while(a.next());
}

void G__list_class(char *classname) {
  G__list_memvar(classname);
  G__list_memfunc(classname);
}

void G__list_memvar(char *classname) {
  G__variable_info a=G__variable_info(G__typeid(classname));
  do {
    printf("%s %s; offset=%d '%s'\n",a.type(),a.name(),a.offset(),a.title());
  } while(a.next());
}

void G__list_memfunc(char *classname) {
  G__function_info a=G__function_info(G__typeid(classname));
  do {
    printf("%s %s(%s) '%s'\n",a.type(),a.name(),a.arglist(),a.title());
  } while(a.next());
}


/* of G__TYPEINFO_H */

/* __CINT__ */
# 2 "/home/wmtan/root/cint/include/typeinfo" 2

}
# 32 "/home/wmtan/root/include/Rtypeinfo.h" 2

using std::type_info;




# 30 "/home/wmtan/root/include/Rtypes.h" 2



# 1 "/home/wmtan/root/cint/include/stdio.h" 1





typedef struct fpos_t {
  char dmy[12];
} fpos_t;
#pragma link off class fpos_t;
#pragma link off typedef fpos_t;
typedef unsigned int size_t;

















# 1 "/home/wmtan/root/cint/include/bool.h" 1
#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#pragma ifdef G__OLDIMPLEMENTATION1604
/* This header file may not be needed any more */

//#undef FALSE
//#undef TRUE



// bool as fundamental type
const bool false=0,true=1;



bool bool() { return false; }

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

#pragma endif

# 29 "/home/wmtan/root/cint/include/stdio.h" 2


#pragma include_noerr <stdfunc.dll>

# 33 "/home/wmtan/root/include/Rtypes.h" 2




//---- forward declared class types --------------------------------------------

class TClass;
class TBuffer;
class TMemberInspector;
class TObject;
class TNamed;

//---- types -------------------------------------------------------------------

typedef char           Char_t;      //Signed Character 1 byte (char)
typedef unsigned char  UChar_t;     //Unsigned Character 1 byte (unsigned char)
typedef short          Short_t;     //Signed Short integer 2 bytes (short)
typedef unsigned short UShort_t;    //Unsigned Short integer 2 bytes (unsigned short)




typedef int            Int_t;       //Signed integer 4 bytes (int)
typedef unsigned int   UInt_t;      //Unsigned integer 4 bytes (unsigned int)

// Note: Long_t and ULong_t are currently not portable types




typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 4 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 4 bytes (unsigned long)

typedef float          Float_t;     //Float 4 bytes (float)
typedef double         Double_t;    //Float 8 bytes (double)
typedef char           Text_t;      //General string (char)
typedef bool           Bool_t;      //Boolean (0=false, 1=true) (bool)
typedef unsigned char  Byte_t;      //Byte (8 bits) (unsigned char)
typedef short          Version_t;   //Class version identifier (short)
typedef const char     Option_t;    //Option string (const char)
typedef int            Ssiz_t;      //String size (int)
typedef float          Real_t;      //TVector and TMatrix element type (float)

typedef void         (*Streamer_t)(TBuffer&, void*, Int_t);
typedef void         (*VoidFuncPtr_t)();  //pointer to void function


//---- constants ---------------------------------------------------------------





const Bool_t kTRUE   = 1;
const Bool_t kFALSE  = 0;

const Int_t  kMaxInt      = 2147483647;
const Int_t  kMaxShort    = 32767;
const size_t kBitsPerByte = 8;
const Ssiz_t kNPOS        = ~(Ssiz_t)0;


//--- bit manipulation ---------------------------------------------------------







//---- debug global ------------------------------------------------------------

R__EXTERN Int_t gDebug;


//---- ClassDef macros ---------------------------------------------------------

typedef void (*ShowMembersFunc_t)(void *obj, TMemberInspector &R__insp, char *R__parent);
typedef TClass *(*IsAFunc_t)(const void *obj);

// TBuffer.h declares and implements the following 2 operators
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
template <class Tmpl> TBuffer &operator<<(TBuffer &buf, const Tmpl *obj);

// This might get used if we implement setting a class version.
// template <class RootClass> Short_t GetClassVersion(RootClass *);

namespace ROOT {

   class TGenericClassInfo;
   typedef void *(*NewFunc_t)(void *);
   typedef void *(*NewArrFunc_t)(Long_t size);
   typedef void  (*DelFunc_t)(void *);
   typedef void  (*DelArrFunc_t)(void *);
   typedef void  (*DesFunc_t)(void *);

   template <class RootClass> Short_t SetClassVersion(RootClass *);

   extern TClass *CreateClass(const char *cname, Version_t id,
                              const type_info &info, IsAFunc_t isa,
                              ShowMembersFunc_t show,
                              const char *dfil, const char *ifil,
                              Int_t dl, Int_t il);
   extern void AddClass(const char *cname, Version_t id, const type_info &info,
                        VoidFuncPtr_t dict, Int_t pragmabits);
   extern void RemoveClass(const char *cname);
   extern void ResetClassVersion(TClass*, const char*, Short_t);

   extern TNamed *RegisterClassTemplate(const char *name,
                                        const char *file, Int_t line);


# 154 "/home/wmtan/root/include/Rtypes.h"


   class TInitBehavior {
      // This class defines the interface for the class registration and
      // the TClass creation. To modify the default behavior, one would
      // inherit from this class and overload ROOT::DefineBehavior().
      // See TQObject.h and table/inc/Ttypes.h for examples.
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const = 0;
      virtual void Unregister(const char *classname) const = 0;
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, IsAFunc_t isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const = 0;
   };

   class TDefaultInitBehavior : public TInitBehavior {
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const {
         ROOT::AddClass(cname, id, info, dict, pragmabits);
      }
      virtual void Unregister(const char *classname) const {
         ROOT::RemoveClass(classname);
      }
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, IsAFunc_t isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const {
         return ROOT::CreateClass(cname, id, info, isa, show, dfil, ifil, dl, il);
      }
   };

   const TInitBehavior *DefineBehavior(void * /*parent_type*/,
                                       void * /*actual_type*/);

} // End of namespace ROOT

// The macros below use TGenericClassInfo, so let's ensure it is included

# 1 "/home/wmtan/root/include/TGenericClassInfo.h" 1
// @(#)root/base:$Id$
// Author: Philippe Canal   23/2/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




namespace ROOT {

   class TGenericClassInfo {
      // This class in not inlined because it is used is non time critical
      // section (the dictionaries) and inline would lead to too much
      // repetition of the code (once per class!).

      const TInitBehavior  *fAction;
      TClass              *fClass;
      const char          *fClassName;
      const char          *fDeclFileName;
      Int_t                fDeclFileLine;
      VoidFuncPtr_t        fDictionary;
      const type_info     &fInfo;
      const char          *fImplFileName;
      Int_t                fImplFileLine;
      IsAFunc_t            fIsA;
      void                *fShowMembers;
      Int_t                fVersion;
      NewFunc_t            fNew;
      NewArrFunc_t         fNewArray;
      DelFunc_t            fDelete;
      DelArrFunc_t         fDeleteArray;
      DesFunc_t            fDestructor;
      
   public:
      TGenericClassInfo(const char *fullClassname,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers, VoidFuncPtr_t dictionary,
                       IsAFunc_t isa, Int_t pragmabits);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers,  VoidFuncPtr_t dictionary,
                       IsAFunc_t isa, Int_t pragmabits);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       VoidFuncPtr_t dictionary, 
                       IsAFunc_t isa, Int_t pragmabits);

      void Init(Int_t pragmabits);
      ~TGenericClassInfo();

      const TInitBehavior &GetAction() const;
      TClass              *GetClass();
      const char          *GetClassName() const;
      const char          *GetDeclFileName() const;
      Int_t                GetDeclFileLine() const;
      DelFunc_t            GetDelete() const;
      DelArrFunc_t         GetDeleteArray() const;
      DesFunc_t            GetDestructor() const;
      const char          *GetImplFileName();
      Int_t                GetImplFileLine();
      const type_info     &GetInfo() const;
      IsAFunc_t            GetIsA() const;
      NewFunc_t            GetNew() const;
      NewArrFunc_t         GetNewArray() const;
      void                *GetShowMembers() const;
      Int_t                GetVersion() const;

      TClass              *IsA(const void *obj);

      void                 SetDelete(DelFunc_t deleteFunc);
      void                 SetDeleteArray(DelArrFunc_t deleteArrayFunc);
      void                 SetDestructor(DesFunc_t destructorFunc);
      void                 SetFromTemplate();
      Int_t                SetImplFile(const char *file, Int_t line);
      void                 SetNew(NewFunc_t newFunc);
      void                 SetNewArray(NewArrFunc_t newArrayFunc);
      Short_t              SetVersion(Short_t version);
      
   };

}


# 197 "/home/wmtan/root/include/Rtypes.h" 2



// Common part of ClassDef definition.
// ImplFileLine() is not part of it since CINT uses that as trigger for
// the class comment string.

# 218 "/home/wmtan/root/include/Rtypes.h"









# 235 "/home/wmtan/root/include/Rtypes.h"


# 245 "/home/wmtan/root/include/Rtypes.h"









# 263 "/home/wmtan/root/include/Rtypes.h"


//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp


// This ClassDefT is stricly redundant and is kept only for
// backward compatibility. Using #define ClassDef ClassDefT is confusing
// the CINT parser.






# 289 "/home/wmtan/root/include/Rtypes.h"









# 306 "/home/wmtan/root/include/Rtypes.h"




//---- ClassDefT macros for templates with two template arguments --------------
// ClassDef2T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp2T  corresponds to ClassImpT





//---- ClassDefT macros for templates with three template arguments ------------
// ClassDef3T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp3T  corresponds to ClassImpT





//---- Macro to set the class version of non instrumented classes --------------










# 31 "/home/wmtan/root/include/TObject.h" 2



# 1 "/home/wmtan/root/include/Varargs.h" 1
/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





/* typedef char *va_list; */
# 1 "/home/wmtan/root/cint/include/stdarg.h" 1
/****************************************************************
* stdarg.h
*****************************************************************/



struct va_list {
  void* libp;
  int    ip;
} ;









# 16 "/home/wmtan/root/include/Varargs.h" 2



# 33 "/home/wmtan/root/include/Varargs.h"












# 34 "/home/wmtan/root/include/TObject.h" 2



# 1 "/home/wmtan/root/include/TStorage.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStorage                                                             //
//                                                                      //
// Storage manager.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





typedef void (*FreeHookFun_t)(void*, void *addr, size_t);
typedef void *(*ReAllocFun_t)(void*, size_t);
typedef void *(*ReAllocCFun_t)(void*, size_t, size_t);


class TStorage {

private:
   static ULong_t        fgHeapBegin;      // begin address of heap
   static ULong_t        fgHeapEnd;        // end address of heap
   static size_t         fgMaxBlockSize;   // largest block allocated
   static FreeHookFun_t  fgFreeHook;       // function called on free
   static void          *fgFreeHookData;   // data used by this function
   static ReAllocFun_t   fgReAllocHook;    // custom ReAlloc
   static ReAllocCFun_t  fgReAllocCHook;   // custom ReAlloc with length check
   static Bool_t         fgHasCustomNewDelete; // true if using ROOT's new/delete

public:
   static ULong_t       GetHeapBegin();
   static ULong_t       GetHeapEnd();
   static FreeHookFun_t GetFreeHook();
   static void         *GetFreeHookData();
   static size_t        GetMaxBlockSize();
   static void         *Alloc(size_t size);
   static void          Dealloc(void *ptr);
   static void         *ReAlloc(void *vp, size_t size);
   static void         *ReAlloc(void *vp, size_t size, size_t oldsize);
   static char         *ReAllocChar(char *vp, size_t size, size_t oldsize);
   static Int_t        *ReAllocInt(Int_t *vp, size_t size, size_t oldsize);
   static void         *ObjectAlloc(size_t size);
   static void         *ObjectAlloc(size_t size, void *vp);
   static void          ObjectDealloc(void *vp);
   static void          ObjectDealloc(void *vp, void *ptr);

   static void EnterStat(size_t size, void *p);
   static void RemoveStat(void *p);
   static void PrintStatistics();
   static void SetMaxBlockSize(size_t size);
   static void SetFreeHook(FreeHookFun_t func, void *data);
   static void SetReAllocHooks(ReAllocFun_t func1, ReAllocCFun_t func2);
   static void SetCustomNewDelete();
   static void EnableStatistics(int size= -1, int ix= -1);

   static Bool_t HasCustomNewDelete();

   // only valid after call to a TStorage allocating method
   static void   AddToHeap(ULong_t begin, ULong_t end);
   static Bool_t IsOnHeap(void *p);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TStorage  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TStorage  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TStorage.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 77; }   //Storage manager class
};


inline void TStorage::AddToHeap(ULong_t begin, ULong_t end)
   { if (begin < fgHeapBegin) fgHeapBegin = begin;
     if (end   > fgHeapEnd)   fgHeapEnd   = end; }

inline Bool_t TStorage::IsOnHeap(void *p)
   { return (ULong_t)p >= fgHeapBegin && (ULong_t)p < fgHeapEnd; }

inline size_t TStorage::GetMaxBlockSize() { return fgMaxBlockSize; }

inline void TStorage::SetMaxBlockSize(size_t size) { fgMaxBlockSize = size; }

inline FreeHookFun_t TStorage::GetFreeHook() { return fgFreeHook; }



# 37 "/home/wmtan/root/include/TObject.h" 2



# 1 "/home/wmtan/root/include/Riosfwd.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/












# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {
# 1 "/home/wmtan/root/cint/include/iosfwd.h" 1


# 1 "/home/wmtan/root/cint/include/iostream.h" 1
/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * I/O stream header file iostream.h
 ************************************************************************
 * Description:
 *  CINT iostream header file
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/



/*********************************************************************
* Try initializaing precompiled iostream library
*********************************************************************/
#pragma setstream
#pragma ifdef G__IOSTREAM_H
#pragma ifndef G__KCC
#pragma include <iosenum.h>
#pragma ifndef G__SSTREAM_H
typedef ostrstream ostringstream;
typedef istrstream istringstream;
//typedef strstream stringstream;  // problem, 
#pragma else
typedef ostringstream ostrstream;
typedef istringstream istrstream;
typedef stringstream strstream;
#pragma endif
#pragma endif
#pragma endif

# 1 "/home/wmtan/root/cint/include/bool.h" 1
#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#pragma ifdef G__OLDIMPLEMENTATION1604
/* This header file may not be needed any more */

//#undef FALSE
//#undef TRUE



// bool as fundamental type
const bool false=0,true=1;



bool bool() { return false; }

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

#pragma endif

# 42 "/home/wmtan/root/cint/include/iostream.h" 2


/*********************************************************************
* Use fake iostream only if precompiled version does not exist.
*********************************************************************/
#pragma if !defined(G__IOSTREAM_H) // && !defined(__cplusplus)


#pragma security level0



/*********************************************************************
* ios
*
*********************************************************************/
typedef long streamoff;
typedef long streampos;
//class io_state;
class streambuf;
class fstreambase;
typedef long         SZ_T;       
typedef SZ_T         streamsize;

class ios {
 public:
  typedef int      iostate;
  enum io_state {
    goodbit     = 0x00,   
    badbit      = 0x01,   
    eofbit      = 0x02,  
    failbit     = 0x04  
  };
  typedef int      openmode;
  enum open_mode {
    app         = 0x01,   
    binary      = 0x02,  
    in          = 0x04, 
    out         = 0x08,   
    trunc       = 0x10,                  
    ate         = 0x20 
  };
  typedef int      seekdir;
  enum seek_dir {
    beg         = 0x0,    
    cur         = 0x1,    
    end         = 0x2   
  };        
  typedef int      fmtflags;
  enum fmt_flags {
    boolalpha   = 0x0001,
    dec         = 0x0002,
    fixed       = 0x0004,
    hex         = 0x0008,
    internal    = 0x0010,
    left        = 0x0020,
    oct         = 0x0040,
    right       = 0x0080,
    scientific  = 0x0100,
    showbase    = 0x0200, 
    showpoint   = 0x0400, 
    showpos     = 0x0800, 
    skipws      = 0x1000, 
    unitbuf     = 0x2000, 
    uppercase   = 0x4000, 
    adjustfield = left | right | internal,
    basefield   = dec | oct | hex,
    floatfield  = scientific | fixed
  };
  enum event { 
    erase_event   = 0x0001,
    imbue_event   = 0x0002,
    copyfmt_event = 0x0004
  };
  
  ios() { x_width=0; }
  streamsize width(streamsize wide) { x_width=wide; }
 protected:
  int x_width;
};


/*********************************************************************
* ostream
*
*********************************************************************/

class ostream : /* virtual */ public ios {
        FILE *fout;
      public:
        ostream(FILE *setfout) { fout=setfout; }
        ostream(char *fname) ;
        ~ostream() ;
        void close() { if(fout) fclose(fout); fout= __null ;}
        void flush() { if(fout) fflush(fout); }
        FILE *fp() { return(fout); }
        int rdstate() ;

        ostream& operator <<(char c);
        ostream& operator <<(char *s);
        ostream& operator <<(long i);
        ostream& operator <<(unsigned long i);
        ostream& operator <<(double d);
        ostream& operator <<(void *p);
        ostream& form(char *format ...);
};

ostream::~ostream()
{
  if(fout!=stderr && fout!=stdout && fout!= __null ) {
    fclose(fout);
  }
}

ostream::ostream(char *fname)
{
  fout = fopen(fname,"w");
  if(fout== __null ) {
    fprintf(stderr,"%s can not open\n",fname);
  }
}

ostream& ostream::operator <<(char c)
{
  if(x_width) {
    int init=0;
    if(isprint(c)) init=1;
    for(int i=init;i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%c",c);
  return(*this);
}

ostream& ostream::operator <<(char *s)
{
  if(x_width &&(!s || x_width>strlen(s))) {
    if(s) for(int i=strlen(s);i<x_width;i++) fputc(' ',fout);
    else  for(int i=0;i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%s",s);
  return(*this);
}

ostream& ostream::operator <<(long x)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"%d",x);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%d",x);
  return(*this);
}

ostream& ostream::operator <<(unsigned long x)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"%u",x);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%u",x);
  return(*this);
}

ostream& ostream::operator <<(double d)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"%g",d);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%g",d);
  return(*this);
}

ostream& ostream::operator <<(void *p)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"0x%x",p);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  printf("0x%x",p);
  return(*this);
}

int ostream::rdstate()
{
  if(fout) return(0);
  else   return(1);
}

/* instanciation of cout,cerr */
ostream cout=ostream(stdout);
ostream cerr=ostream(stderr);


/*********************************************************************
* istream
*
*********************************************************************/

class istream : /* virtual */ public ios {
  FILE *fin;
  ostream *tie;
public:
  istream(FILE *setfin) { fin = setfin; tie=(ostream*)__null ; }
  istream(char *fname);
  ~istream();
  void close() { if(fin) fclose(fin); fin= __null ;}
  ostream& tie(ostream& cx); 
  FILE *fp() { return(fin); }
  int rdstate();
  
  istream& operator >>(char& c);
  istream& operator >>(char *s);
  istream& operator >>(short& s);
  istream& operator >>(int& i);
  istream& operator >>(long& i);
  istream& operator >>(unsigned char& c);
  istream& operator >>(unsigned short& s);
  istream& operator >>(unsigned int& i);
  istream& operator >>(unsigned long& i);
  istream& operator >>(double& d);
  istream& operator >>(float& d);
};

istream::~istream()
{
  if(fin!=stdin && fin!= __null ) {
    fclose(fin);
  }
}

istream::istream(char *fname)
{
  fin = fopen(fname,"r");
  if(fin== __null ) {
    fprintf(stderr,"%s can not open\n",fname);
  }
  tie=(ostream*)__null ;
}


ostream& istream::tie(ostream& cx) 
     
{ 
  ostream *tmp; 
  tmp=tie; 
  tie = &cx; 
  return(*tmp);
}

istream& istream::operator >>(char& c)
{
  if(tie) tie->flush();
  c=fgetc(fin);
  return(*this);
}

istream& istream::operator >>(char *s)
{
  if(tie) tie->flush();
  fscanf(fin,"%s",s);
  return(*this);
}

istream& istream::operator >>(short& s)
{
  if(tie) tie->flush();
  fscanf(fin,"%hd",&s);
  return(*this);
}

istream& istream::operator >>(int& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%d",&i);
  return(*this);
}

istream& istream::operator >>(long& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%ld",&i);
  return(*this);
}

istream& istream::operator >>(unsigned char& c)
{
  int i;
  if(tie) tie->flush();
  fscanf(fin,"%u",&i);
  c = i;
  return(*this);
}
istream& istream::operator >>(unsigned short& s)
{
  if(tie) tie->flush();
  fscanf(fin,"%hu",&s);
  return(*this);
}
istream& istream::operator >>(unsigned int& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%u",&i);
  return(*this);
}
istream& istream::operator >>(unsigned long& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%lu",&i);
  return(*this);
}

istream& istream::operator >>(float& f)
{
  if(tie) tie->flush();
  fscanf(fin,"%g",&f);
  return(*this);
}

istream& istream::operator >>(double& d)
{
  if(tie) tie->flush();
  fscanf(fin,"%lg",&d);
  return(*this);
}

int istream::rdstate()
{
  int cx;
  if(!fin) return(1);
  cx = fgetc(fin);
  fseek(fin,-1,(1) );
  if(EOF==cx) return(1);
  return(0);
}

/* instanciation of cin */
istream cin=istream(stdin);

/*********************************************************************
* iostream
*
*********************************************************************/
class iostream : public istream , public ostream {
 public:
  iostream(FILE *setfin) : istream(setfin), ostream(setfin) { }
  iostream(char *fname) : istream(fname), ostream(fname) { }
};


/*********************************************************************
* ofstream, ifstream 
*
*********************************************************************/

class fstream;

class ofstream : public ostream {
 public:
  ofstream(FILE* setfin) : ostream(setfin) { }
  ofstream(char* fname) : ostream(fname) { }
};

class ifstream : public istream {
 public:
  ifstream(FILE* setfin) : istream(setfin) { }
  ifstream(char* fname) : istream(fname) { }
};

class iofstream : public iostream {
 public:
  iofstream(FILE* setfin) : iostream(setfin) { }
  iofstream(char* fname) : iostream(fname) { }
};


ostream& flush(ostream& i) {i.flush(); return(i);}
ostream& endl(ostream& i) {return i << '\n' << flush;}
ostream& ends(ostream& i) {return i << '\0';}
istream& ws(istream& i) {
  fprintf(stderr,"Limitation: ws,WS manipurator not supported\n");
  return(i);
}
istream& WS(istream& i) {
  fprintf(stderr,"Limitation: ws,WS manipurator not supported\n");
  return(i);
}

#pragma endif /* G__IOSTREAM_H */

ostream& ostream::form(char *format ...) {
  char temp[1024];
  return(*this<<G__charformatter(0,temp));
}

/*********************************************************************
* iostream manipurator emulation
*
*  Following description must be deleted when pointer to compiled 
* function is fully supported.
*********************************************************************/
class G__CINT_ENDL { int dmy; } endl;
class G__CINT_ENDS { int dmy; } ends;
class G__CINT_FLUSH { int dmy; } flush;
class G__CINT_ws { int dmy; } ws;
class G__CINT_WS { int dmy; } WS;
class G__CINT_HEX { int dmy; } hex;
class G__CINT_DEC { int dmy; } dec;
class G__CINT_OCT { int dmy; } oct;
class G__CINT_NOSUPPORT { int dmy; } ;


# 1 "/home/wmtan/root/cint/include/_iostream" 1
// include/_iostream

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDL& i) 
        {return(std::endl(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDS& i) 
        {return(std::ends(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_FLUSH& i) 
        {return(std::flush(ostr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_ws& i) 
        {return(std::ws(istr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_WS& i) 
        {return(std::WS(istr));}


std::ostream& operator<<(std::ostream& ostr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::dec);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::hex);
#pragma else
  ostr.unsetf(ios_base::dec);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::hex);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::dec);
  istr.unsetf(ios::oct);
  istr.setf(ios::hex);
#pragma else
  istr.unsetf(ios_base::dec);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::hex);
#pragma endif
  return(istr);
}

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::dec);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::dec);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::oct);
  istr.setf(ios::dec);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::dec);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::dec);
  ostr.setf(ios::oct);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::dec);
  ostr.setf(ios_base::oct);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::dec);
  istr.setf(ios::oct);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::dec);
  istr.setf(ios_base::oct);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(ostr);
}
std::istream& operator<<(std::istream& istr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(istr);
}

// Value evaluation
//template<class T> int G__ateval(const T* x) {return(0);}
template<class T> int G__ateval(const T& x) {return(0);}
int G__ateval(const char* x) {return(0);}
int G__ateval(const void* x) {return(0);}
int G__ateval(const double x) {return(0);}
int G__ateval(const float x) {return(0);}
int G__ateval(const char x) {return(0);}
int G__ateval(const short x) {return(0);}
int G__ateval(const int x) {return(0);}
int G__ateval(const long x) {return(0);}
int G__ateval(const unsigned char x) {return(0);}
int G__ateval(const unsigned short x) {return(0);}
int G__ateval(const unsigned int x) {return(0);}
int G__ateval(const unsigned long x) {return(0);}








# 468 "/home/wmtan/root/cint/include/iostream.h" 2




# 3 "/home/wmtan/root/cint/include/iosfwd.h" 2


# 2 "/home/wmtan/root/cint/include/iosfwd" 2

}
# 23 "/home/wmtan/root/include/Riosfwd.h" 2


using std::istream;
using std::ostream;
using std::fstream;
using std::ifstream;
using std::ofstream;
# 48 "/home/wmtan/root/include/Riosfwd.h"



# 40 "/home/wmtan/root/include/TObject.h" 2







class TList;
class TBrowser;
class TBuffer;
class TObjArray;
class TMethod;
class TTimer;


//----- Global bits (can be set for any object and should not be reused).
//----- Bits 0 - 13 are reserved as global bits. Bits 14 - 23 can be used
//----- in different class hierarchies (make sure there is no overlap in
//----- any given hierarchy).
enum EObjBits {
   kCanDelete        = (1 << ( 0 )) ,   // if object in a list can be deleted
   kMustCleanup      = (1 << ( 3 )) ,   // if object destructor must call RecursiveRemove()
   kObjInCanvas      = (1 << ( 3 )) ,   // for backward compatibility only, use kMustCleanup
   kIsReferenced     = (1 << ( 4 )) ,   // if object is referenced by a TRef or TRefArray
   kHasUUID          = (1 << ( 5 )) ,   // if object has a TUUID (its fUniqueID=UUIDNumber)
   kCannotPick       = (1 << ( 6 )) ,   // if object in a pad cannot be picked
   kNoContextMenu    = (1 << ( 8 )) ,   // if object does not want context menu
   kInvalidObject    = (1 << ( 13 ))    // if object ctor succeeded but object should not be used
};


class TObject {

private:
   UInt_t         fUniqueID;   //object unique identifier
   UInt_t         fBits;       //bit field status word

   static Long_t  fgDtorOnly;    //object for which to call dtor only (i.e. no delete)
   static Bool_t  fgObjectStat;  //if true keep track of objects in TObjectTable

protected:
   void MakeZombie() { fBits |= kZombie; }
   virtual void DoError(int level, const char *location, const char *fmt, va_list va) const;

public:
   //----- Private bits, clients can only test but not change them
   enum {
      kIsOnHeap      = 0x01000000,    // object is on heap
      kNotDeleted    = 0x02000000,    // object has not been deleted
      kZombie        = 0x04000000,    // object ctor failed
      kBitMask       = 0x00ffffff
   };

   //----- Write() options
   enum {
      kSingleKey     = (1 << ( 0 )) ,        // write collection with single key
      kOverwrite     = (1 << ( 1 ))          // overwrite existing object with same name
   };

   TObject();
   TObject(const TObject &object);
   TObject &operator=(const TObject &rhs);
   virtual ~TObject();

   virtual void        AppendPad(Option_t *option="");
   virtual void        Browse(TBrowser *b);
   virtual const char *ClassName() const;
   virtual void        Clear(Option_t * /*option*/ ="") { }
   virtual TObject    *Clone(const char *newname="") const;
   virtual Int_t       Compare(const TObject *obj) const;
   virtual void        Copy(TObject &object) const;
   virtual void        Delete(Option_t *option=""); // *MENU*
   virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
   virtual void        Draw(Option_t *option="");
   virtual void        DrawClass() const; // *MENU*
   virtual TObject    *DrawClone(Option_t *option="") const; // *MENU*
   virtual void        Dump() const; // *MENU*
   virtual void        Execute(const char *method,  const char *params, Int_t *error=0);
   virtual void        Execute(TMethod *method, TObjArray *params, Int_t *error=0);
   virtual void        ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual TObject    *FindObject(const char *name) const;
   virtual TObject    *FindObject(const TObject *obj) const;
   virtual Option_t   *GetDrawOption() const;
   virtual UInt_t      GetUniqueID() const;
   virtual const char *GetName() const;
   virtual const char *GetIconName() const;
   virtual Option_t   *GetOption() const { return ""; }
   virtual char       *GetObjectInfo(Int_t px, Int_t py) const;
   virtual const char *GetTitle() const;
   virtual Bool_t      HandleTimer(TTimer *timer);
   virtual ULong_t     Hash() const;
   virtual Bool_t      InheritsFrom(const char *classname) const;
   virtual Bool_t      InheritsFrom(const TClass *cl) const;
   virtual void        Inspect() const; // *MENU*
   virtual Bool_t      IsFolder() const;
   virtual Bool_t      IsEqual(const TObject *obj) const;
   virtual Bool_t      IsSortable() const { return kFALSE; }
           Bool_t      IsOnHeap() const { return TestBit(kIsOnHeap); }
           Bool_t      IsZombie() const { return TestBit(kZombie); }
   virtual Bool_t      Notify();
   virtual void        ls(Option_t *option="") const;
   virtual void        Paint(Option_t *option="");
   virtual void        Pop();
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       Read(const char *name);
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        SavePrimitive(ofstream &out, Option_t *option);
   virtual void        SetDrawOption(Option_t *option="");  // *MENU*
   virtual void        SetUniqueID(UInt_t uid);
   virtual void        UseCurrentStyle();
   virtual Int_t       Write(const char *name=0, Int_t option=0, Int_t bufsize=0);

   //----- operators
   void    *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
   void    *operator new(size_t sz, void *vp) { return TStorage::ObjectAlloc(sz, vp); }
   void     operator delete(void *ptr);

   void     operator delete(void *ptr, void *vp);


   //----- bit manipulation
   void     SetBit(UInt_t f, Bool_t set);
   void     SetBit(UInt_t f) { fBits |= f & kBitMask; }
   void     ResetBit(UInt_t f) { fBits &= ~(f & kBitMask); }
   Bool_t   TestBit(UInt_t f) const { return (Bool_t) ((fBits & f) != 0); }
   Int_t    TestBits(UInt_t f) const { return (Int_t) (fBits & f); }
   void     InvertBit(UInt_t f) { fBits ^= f & kBitMask; }

   //---- error handling
   virtual void     Info(const char *method, const char *msgfmt, ...) const;
   virtual void     Warning(const char *method, const char *msgfmt, ...) const;
   virtual void     Error(const char *method, const char *msgfmt, ...) const;
   virtual void     SysError(const char *method, const char *msgfmt, ...) const;
   virtual void     Fatal(const char *method, const char *msgfmt, ...) const;

   void     AbstractMethod(const char *method) const;
   void     MayNotUse(const char *method) const;

   //---- static functions
   static Long_t    GetDtorOnly();
   static void      SetDtorOnly(void *obj);
   static Bool_t    GetObjectStat();
   static void      SetObjectStat(Bool_t stat);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TObject  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TObject  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TObject.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 184; }   //Basic ROOT object
};


# 1 "/home/wmtan/root/include/TBuffer.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer                                                              //
//                                                                      //
// Buffer base class used for serializing objects.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





# 1 "/home/wmtan/root/include/Bytes.h" 1
/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Bytes                                                                //
//                                                                      //
// A set of inline byte handling routines.                              //
//                                                                      //
// The set of tobuf() and frombuf() routines take care of packing a     //
// basic type value into a buffer in network byte order (i.e. they      //
// perform byte swapping when needed). The buffer does not have to      //
// start on a machine (long) word boundary.                             //
//                                                                      //
// For __GNUC__ on linux on i486 processors and up                      //
// use the `bswap' opcode provided by the GNU C Library.                //
//                                                                      //
// The set of host2net() and net2host() routines convert a basic type   //
// value from host to network byte order and vice versa. On BIG ENDIAN  //
// machines this is a no op.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////














//______________________________________________________________________________
inline void tobuf(char *&buf, Bool_t x)
{
   UChar_t x1 = x;
   *buf++ = x1;
}

inline void tobuf(char *&buf, UChar_t x)
{
   *buf++ = x;
}

inline void tobuf(char *&buf, UShort_t x)
{


   *((UShort_t *)buf) = Rbswap_16(x);








   buf += sizeof(UShort_t);
}

inline void tobuf(char *&buf, UInt_t x)
{


   *((UInt_t *)buf) = Rbswap_32(x);










   buf += sizeof(UInt_t);
}

inline void tobuf(char *&buf, ULong_t x)
{

   char *sw = (char *)&x;
   if (sizeof(ULong_t) == 8) {
      buf[0] = sw[7];
      buf[1] = sw[6];
      buf[2] = sw[5];
      buf[3] = sw[4];
      buf[4] = sw[3];
      buf[5] = sw[2];
      buf[6] = sw[1];
      buf[7] = sw[0];
   } else {
      buf[0] = 0;
      buf[1] = 0;
      buf[2] = 0;
      buf[3] = 0;
      buf[4] = sw[3];
      buf[5] = sw[2];
      buf[6] = sw[1];
      buf[7] = sw[0];
   }
# 127 "/home/wmtan/root/include/Bytes.h"

   buf += 8;
}

inline void tobuf(char *&buf, Float_t x)
{


   *((UInt_t *)buf) = Rbswap_32(*((UInt_t *)&x));
# 155 "/home/wmtan/root/include/Bytes.h"




   buf += sizeof(Float_t);
}

inline void tobuf(char *&buf, Double_t x)
{

# 184 "/home/wmtan/root/include/Bytes.h"

   char *sw = (char *)&x;
   buf[0] = sw[7];
   buf[1] = sw[6];
   buf[2] = sw[5];
   buf[3] = sw[4];
   buf[4] = sw[3];
   buf[5] = sw[2];
   buf[6] = sw[1];
   buf[7] = sw[0];




   buf += sizeof(Double_t);
}

inline void frombuf(char *&buf, Bool_t *x)
{
   UChar_t x1;
   x1 = *buf++;
   *x = (Bool_t) x1;
}

inline void frombuf(char *&buf, UChar_t *x)
{
   *x = *buf++;
}

inline void frombuf(char *&buf, UShort_t *x)
{


   *x = Rbswap_16(*((UShort_t *)buf));








   buf += sizeof(UShort_t);
}

inline void frombuf(char *&buf, UInt_t *x)
{


   *x = Rbswap_32(*((UInt_t *)buf));










   buf += sizeof(UInt_t);
}

inline void frombuf(char *&buf, ULong_t *x)
{

   char *sw = (char *)x;
   if (sizeof(ULong_t) == 8) {
      sw[0] = buf[7];
      sw[1] = buf[6];
      sw[2] = buf[5];
      sw[3] = buf[4];
      sw[4] = buf[3];
      sw[5] = buf[2];
      sw[6] = buf[1];
      sw[7] = buf[0];
   } else {
      sw[0] = buf[7];
      sw[1] = buf[6];
      sw[2] = buf[5];
      sw[3] = buf[4];
   }







   buf += 8;
}

inline void frombuf(char *&buf, Float_t *x)
{


   *((UInt_t*)x) = Rbswap_32(*((UInt_t *)buf));
# 300 "/home/wmtan/root/include/Bytes.h"




   buf += sizeof(Float_t);
}

inline void frombuf(char *&buf, Double_t *x)
{

# 329 "/home/wmtan/root/include/Bytes.h"

   char *sw = (char *)x;
   sw[0] = buf[7];
   sw[1] = buf[6];
   sw[2] = buf[5];
   sw[3] = buf[4];
   sw[4] = buf[3];
   sw[5] = buf[2];
   sw[6] = buf[1];
   sw[7] = buf[0];




   buf += sizeof(Double_t);
}

inline void tobuf(char *&buf, Char_t x)  { tobuf(buf, (UChar_t) x); }
inline void tobuf(char *&buf, Short_t x) { tobuf(buf, (UShort_t) x); }
inline void tobuf(char *&buf, Int_t x)   { tobuf(buf, (UInt_t) x); }
inline void tobuf(char *&buf, Long_t x)  { tobuf(buf, (ULong_t) x); }

inline void frombuf(char *&buf, Char_t *x)  { frombuf(buf, (UChar_t *) x); }
inline void frombuf(char *&buf, Short_t *x) { frombuf(buf, (UShort_t *) x); }
inline void frombuf(char *&buf, Int_t *x)   { frombuf(buf, (UInt_t *) x); }
inline void frombuf(char *&buf, Long_t *x)  { frombuf(buf, (ULong_t *) x); }


//______________________________________________________________________________

inline UShort_t host2net(UShort_t x)
{

   return Rbswap_16(x);



}

inline UInt_t host2net(UInt_t x)
{

   return Rbswap_32(x);




}

inline ULong_t host2net(ULong_t x)
{
# 398 "/home/wmtan/root/include/Bytes.h"

   return (ULong_t)host2net((UInt_t) x);

}

inline Float_t host2net(Float_t xx)
{

   UInt_t t = Rbswap_32(*((UInt_t *)&xx));
   return *(Float_t *)&t;






}

inline Double_t host2net(Double_t x)
{




   char sw[sizeof(Double_t)];
   *(Double_t *)sw = x;

   char *sb = (char *)&x;
   sb[0] = sw[7];
   sb[1] = sw[6];
   sb[2] = sw[5];
   sb[3] = sw[4];
   sb[4] = sw[3];
   sb[5] = sw[2];
   sb[6] = sw[1];
   sb[7] = sw[0];
   return x;

}
/* R__BYTESWAP */







inline Short_t  host2net(Short_t x) { return host2net((UShort_t)x); }
inline Int_t    host2net(Int_t x)   { return host2net((UInt_t)x); }
inline Long_t   host2net(Long_t x)  { return host2net((ULong_t)x); }

inline UShort_t net2host(UShort_t x) { return host2net(x); }
inline Short_t  net2host(Short_t x)  { return host2net(x); }
inline UInt_t   net2host(UInt_t x)   { return host2net(x); }
inline Int_t    net2host(Int_t x)    { return host2net(x); }
inline ULong_t  net2host(ULong_t x)  { return host2net(x); }
inline Long_t   net2host(Long_t x)   { return host2net(x); }
inline Float_t  net2host(Float_t x)  { return host2net(x); }
inline Double_t net2host(Double_t x) { return host2net(x); }


# 28 "/home/wmtan/root/include/TBuffer.h" 2



class TClass;
class TExMap;

class TBuffer : public TObject {

protected:
   Bool_t    fMode;          //Read or write mode
   Int_t     fVersion;       //Buffer format version
   Int_t     fBufSize;       //Size of buffer
   char     *fBuffer;        //Buffer used to store objects
   char     *fBufCur;        //Current position in buffer
   char     *fBufMax;        //End of buffer
   Int_t     fMapCount;      //Number of objects or classes in map
   Int_t     fMapSize;       //Default size of map
   Int_t     fDisplacement;  //Value to be added to the map offsets
   TExMap   *fMap;           //Map containing object,id pairs for reading/ writing
   TObject  *fParent;        //Pointer to the buffer parent (file) where buffer is read/written

   enum { kIsOwner = (1 << ( 14 ))  };  //If set TBuffer owns fBuffer

   static Int_t fgMapSize; //Default map size for all TBuffer objects

   // Default ctor
   TBuffer() : fMode(0), fBuffer(0) { fMap = 0; fParent = 0;}

   // TBuffer objects cannot be copied or assigned
   TBuffer(const TBuffer &);           // not implemented
   void operator=(const TBuffer &);    // not implemented

   void   CheckCount(UInt_t offset);
   UInt_t CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass = kFALSE);

   void Expand(Int_t newsize);  //Expand buffer to newsize

   Int_t Read(const char *name) { return TObject::Read(name); }
   Int_t Write(const char *name, Int_t opt, Int_t bufs)
                                { return TObject::Write(name, opt, bufs); }

   void     WriteObject(const void *actualObjStart, TClass *actualClass);

public:
   enum EMode { kRead = 0, kWrite = 1 };
   enum { kInitialSize = 1024, kMinimalSize = 128 };
   enum { kMapSize = 503 };

   TBuffer(EMode mode);
   TBuffer(EMode mode, Int_t bufsiz);
   TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE);
   virtual ~TBuffer();

   void     MapObject(const TObject *obj, UInt_t offset = 1);
   void     MapObject(const void *obj, UInt_t offset = 1);
   virtual void Reset() { SetBufferOffset(); ResetMap(); }
   void     InitMap();
   void     ResetMap();
   void     SetReadMode();
   void     SetReadParam(Int_t mapsize);
   void     SetWriteMode();
   void     SetWriteParam(Int_t mapsize);
   void     SetBuffer(void *buf, UInt_t bufsiz = 0, Bool_t adopt = kTRUE);
   void     SetBufferOffset(Int_t offset = 0) { fBufCur = fBuffer+offset; }
   void     SetParent(TObject *parent);
   TObject *GetParent() const;
   char    *Buffer() const { return fBuffer; }
   Int_t    BufferSize() const { return fBufSize; }
   void     DetachBuffer() { fBuffer = 0; }
   Int_t    Length() const { return (Int_t)(fBufCur - fBuffer); }

   Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss);
   void     SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);

   Bool_t   IsReading() const { return (fMode & kWrite) == 0; }
   Bool_t   IsWriting() const { return (fMode & kWrite) != 0; }

   Int_t    ReadBuf(void *buf, Int_t max);
   void     WriteBuf(const void *buf, Int_t max);

   char    *ReadString(char *s, Int_t max);
   void     WriteString(const char *s);

   Version_t ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0);
   UInt_t    WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);

   virtual TClass  *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   virtual void     WriteClass(const TClass *cl);

   virtual TObject *ReadObject(const TClass *cl);
   virtual void     WriteObject(const TObject *obj);

   void    *ReadObjectAny(const TClass* cast);
   Int_t    WriteObjectAny(const void *obj, TClass *ptrClass);

   void     SetBufferDisplacement(Int_t skipped)
            { fDisplacement =  (Int_t)(Length() - skipped); }
   void     SetBufferDisplacement() { fDisplacement = 0; }
   Int_t    GetBufferDisplacement() const { return fDisplacement; }

   Int_t    ReadArray(Bool_t   *&b);
   Int_t    ReadArray(Char_t   *&c);
   Int_t    ReadArray(UChar_t  *&c);
   Int_t    ReadArray(Short_t  *&h);
   Int_t    ReadArray(UShort_t *&h);
   Int_t    ReadArray(Int_t    *&i);
   Int_t    ReadArray(UInt_t   *&i);
   Int_t    ReadArray(Long_t   *&l);
   Int_t    ReadArray(ULong_t  *&l);
   Int_t    ReadArray(Float_t  *&f);
   Int_t    ReadArray(Double_t *&d);

   Int_t    ReadStaticArray(Bool_t   *b);
   Int_t    ReadStaticArray(Char_t   *c);
   Int_t    ReadStaticArray(UChar_t  *c);
   Int_t    ReadStaticArray(Short_t  *h);
   Int_t    ReadStaticArray(UShort_t *h);
   Int_t    ReadStaticArray(Int_t    *i);
   Int_t    ReadStaticArray(UInt_t   *i);
   Int_t    ReadStaticArray(Long_t   *l);
   Int_t    ReadStaticArray(ULong_t  *l);
   Int_t    ReadStaticArray(Float_t  *f);
   Int_t    ReadStaticArray(Double_t *d);

   void     WriteArray(const Bool_t   *b, Int_t n);
   void     WriteArray(const Char_t   *c, Int_t n);
   void     WriteArray(const UChar_t  *c, Int_t n);
   void     WriteArray(const Short_t  *h, Int_t n);
   void     WriteArray(const UShort_t *h, Int_t n);
   void     WriteArray(const Int_t    *i, Int_t n);
   void     WriteArray(const UInt_t   *i, Int_t n);
   void     WriteArray(const Long_t   *l, Int_t n);
   void     WriteArray(const ULong_t  *l, Int_t n);
   void     WriteArray(const Float_t  *f, Int_t n);
   void     WriteArray(const Double_t *d, Int_t n);

   void     ReadFastArray(Bool_t   *b, Int_t n);
   void     ReadFastArray(Char_t   *c, Int_t n);
   void     ReadFastArray(UChar_t  *c, Int_t n);
   void     ReadFastArray(Short_t  *h, Int_t n);
   void     ReadFastArray(UShort_t *h, Int_t n);
   void     ReadFastArray(Int_t    *i, Int_t n);
   void     ReadFastArray(UInt_t   *i, Int_t n);
   void     ReadFastArray(Long_t   *l, Int_t n);
   void     ReadFastArray(ULong_t  *l, Int_t n);
   void     ReadFastArray(Float_t  *f, Int_t n);
   void     ReadFastArray(Double_t *d, Int_t n);

   void     StreamObject(void *obj, const type_info &typeinfo);
   void     StreamObject(void *obj, const char *className);
   void     StreamObject(void *obj, TClass *cl);

   void     WriteFastArray(const Bool_t   *b, Int_t n);
   void     WriteFastArray(const Char_t   *c, Int_t n);
   void     WriteFastArray(const UChar_t  *c, Int_t n);
   void     WriteFastArray(const Short_t  *h, Int_t n);
   void     WriteFastArray(const UShort_t *h, Int_t n);
   void     WriteFastArray(const Int_t    *i, Int_t n);
   void     WriteFastArray(const UInt_t   *i, Int_t n);
   void     WriteFastArray(const Long_t   *l, Int_t n);
   void     WriteFastArray(const ULong_t  *l, Int_t n);
   void     WriteFastArray(const Float_t  *f, Int_t n);
   void     WriteFastArray(const Double_t *d, Int_t n);

   TBuffer  &operator>>(Bool_t   &b);
   TBuffer  &operator>>(Char_t   &c);
   TBuffer  &operator>>(UChar_t  &c);
   TBuffer  &operator>>(Short_t  &h);
   TBuffer  &operator>>(UShort_t &h);
   TBuffer  &operator>>(Int_t    &i);
   TBuffer  &operator>>(UInt_t   &i);
   TBuffer  &operator>>(Long_t   &l);
   TBuffer  &operator>>(ULong_t  &l);
   TBuffer  &operator>>(Float_t  &f);
   TBuffer  &operator>>(Double_t &d);
   TBuffer  &operator>>(Char_t   *c);

   TBuffer  &operator<<(Bool_t   b);
   TBuffer  &operator<<(Char_t   c);
   TBuffer  &operator<<(UChar_t  c);
   TBuffer  &operator<<(Short_t  h);
   TBuffer  &operator<<(UShort_t h);
   TBuffer  &operator<<(Int_t    i);
   TBuffer  &operator<<(UInt_t   i);
   TBuffer  &operator<<(Long_t   l);
   TBuffer  &operator<<(ULong_t  l);
   TBuffer  &operator<<(Float_t  f);
   TBuffer  &operator<<(Double_t d);
   TBuffer  &operator<<(const Char_t  *c);

   //friend TBuffer  &operator>>(TBuffer &b, TObject *&obj);
   //friend TBuffer  &operator>>(TBuffer &b, const TObject *&obj);
   //friend TBuffer  &operator<<(TBuffer &b, const TObject *obj);

   static void    SetGlobalReadParam(Int_t mapsize);
   static void    SetGlobalWriteParam(Int_t mapsize);
   static Int_t   GetGlobalReadParam();
   static Int_t   GetGlobalWriteParam();
   static TClass *GetClass(const type_info &typeinfo);
   static TClass *GetClass(const char *className);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TBuffer  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TBuffer  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TBuffer.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 229; }   //Buffer base class used for serializing objects
};

//---------------------- TBuffer default external operators --------------------


# 263 "/home/wmtan/root/include/TBuffer.h"

template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
template <class Tmpl> TBuffer &operator<<(TBuffer &buf, Tmpl *&obj);




//---------------------- TBuffer inlines ---------------------------------------

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Bool_t b)
{
   if (fBufCur + sizeof(UChar_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, b);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Char_t c)
{
   if (fBufCur + sizeof(Char_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Short_t h)
{
   if (fBufCur + sizeof(Short_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, h);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Int_t i)
{
   if (fBufCur + sizeof(Int_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, i);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Long_t l)
{
   if (fBufCur + sizeof(Long_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, l);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Float_t f)
{
   if (fBufCur + sizeof(Float_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, f);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Double_t d)
{
   if (fBufCur + sizeof(Double_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, d);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(const Char_t *c)
{
   WriteString(c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Bool_t &b)
{
   frombuf(fBufCur, &b);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Char_t &c)
{
   frombuf(fBufCur, &c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Short_t &h)
{
   frombuf(fBufCur, &h);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Int_t &i)
{
   frombuf(fBufCur, &i);
   return *this;
}

//______________________________________________________________________________
//inline TBuffer &TBuffer::operator>>(Long_t &l)
//{
//   frombuf(fBufCur, &l);
//   return *this;
//}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Float_t &f)
{
   frombuf(fBufCur, &f);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Double_t &d)
{
   frombuf(fBufCur, &d);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Char_t *c)
{
   ReadString(c, -1);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UChar_t c)
   { return TBuffer::operator<<((Char_t)c); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UShort_t h)
   { return TBuffer::operator<<((Short_t)h); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UInt_t i)
   { return TBuffer::operator<<((Int_t)i); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(ULong_t l)
   { return TBuffer::operator<<((Long_t)l); }

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UChar_t &c)
   { return TBuffer::operator>>((Char_t&)c); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UShort_t &h)
   { return TBuffer::operator>>((Short_t&)h); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UInt_t &i)
   { return TBuffer::operator>>((Int_t&)i); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(ULong_t &l)
   { return TBuffer::operator>>((Long_t&)l); }

//______________________________________________________________________________



inline TBuffer &operator<<(TBuffer &buf, const TObject *obj)
   { buf.WriteObjectAny(obj,TObject::Class());
     return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, const TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }

//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UChar_t *&c)
   { return TBuffer::ReadArray((Char_t *&)c); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UShort_t *&h)
   { return TBuffer::ReadArray((Short_t *&)h); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UInt_t *&i)
   { return TBuffer::ReadArray((Int_t *&)i); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(ULong_t *&l)
   { return TBuffer::ReadArray((Long_t *&)l); }

//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UChar_t *c)
   { return TBuffer::ReadStaticArray((Char_t *)c); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UShort_t *h)
   { return TBuffer::ReadStaticArray((Short_t *)h); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UInt_t *i)
   { return TBuffer::ReadStaticArray((Int_t *)i); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(ULong_t *l)
   { return TBuffer::ReadStaticArray((Long_t *)l); }

//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UChar_t *c, Int_t n)
   { TBuffer::ReadFastArray((Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UShort_t *h, Int_t n)
   { TBuffer::ReadFastArray((Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UInt_t *i, Int_t n)
   { TBuffer::ReadFastArray((Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(ULong_t *l, Int_t n)
   { TBuffer::ReadFastArray((Long_t *)l, n); }

//______________________________________________________________________________
inline void TBuffer::WriteArray(const UChar_t *c, Int_t n)
   { TBuffer::WriteArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const UShort_t *h, Int_t n)
   { TBuffer::WriteArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const UInt_t *i, Int_t n)
   { TBuffer::WriteArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const ULong_t *l, Int_t n)
   { TBuffer::WriteArray((const Long_t *)l, n); }

//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UChar_t *c, Int_t n)
   { TBuffer::WriteFastArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UShort_t *h, Int_t n)
   { TBuffer::WriteFastArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UInt_t *i, Int_t n)
   { TBuffer::WriteFastArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const ULong_t *l, Int_t n)
   { TBuffer::WriteFastArray((const Long_t *)l, n); }


# 188 "/home/wmtan/root/include/TObject.h" 2




# 26 "/home/wmtan/root/include/TRef.h" 2



class TProcessID;
class TFile;
class TExec;
class TObjArray;

class TRef : public TObject {

protected:
   TProcessID       *fPID;     //!Pointer to ProcessID when TRef was written

   static TObjArray  *fgExecs;  //List of execs
   static TObject    *fgObject; //Pointer to object (set in Action on Demand)
      
public:
   //status bits
   enum { kNotComputed = (1 << ( 12 )) };

   TRef() {fPID = 0;}
   TRef(TObject *obj);
   TRef(const TRef &ref);
   void operator=(TObject *obj);
   TRef& operator=(const TRef &ref);
   virtual ~TRef() {;}
   static Int_t       AddExec(const char *name);
          TObject    *GetObject() const;
   static TObjArray  *GetListOfExecs();
   TProcessID        *GetPID() const {return fPID;}
   Bool_t             IsValid() const {return GetUniqueID()!=0 ? kTRUE : kFALSE;}virtual void       SetAction(const char *name);
   virtual void       SetAction(TObject *parent);
   static  void       SetObject(TObject *obj);
   
   friend Bool_t operator==(const TRef &r1, const TRef &r2);
   friend Bool_t operator!=(const TRef &r1, const TRef &r2);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TRef  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TRef  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TRef.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 63; }   //Persistent Reference link to a TObject
};


# 6 "Utilities/Persistency/interface/enums.h" 2


enum ooMode {
        oocNoOpen,
        oocRead,
        oocUpdate,
        oocNoMROW,
        oocMROW
};

enum ooStatus {
        oocError,
        oocSuccess
};

enum ooErrorLevel {
        oocNoError,
        oocWarning,
        oocUserError,
        oocSystemError,
        oocFatalError
};



# 3 "Utilities/Persistency/interface/Persistency.h" 2



# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 6 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "/home/wmtan/root/cint/stl/map" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_map" 1

#pragma include_noerr <map.dll>
#pragma include_noerr <map2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/map" 1
// lib/prec_stl/map

#pragma ifndef PREC_STL_MAP
#pragma define PREC_STL_MAP
#pragma link off global PREC_STL_MAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/map" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class map {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const map::iterator& x ,const map::iterator& y) const;
  friend bool operator!=(const map::iterator& x ,const map::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;
  friend bool operator!=(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.map.cons_ construct/copy/destroy:
  map();






  map(iterator first, iterator last);
  map(reverse_iterator first, reverse_iterator last);

  map(const map& x);
  ~map();
  map& operator=(const map& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.map.access_ element access:
  T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(map&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.map.ops_ map operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const map& x, const map& y);
  friend bool operator< (const map& x, const map& y);
  friend bool operator!=(const map& x, const map& y);
  friend bool operator> (const map& x, const map& y);
  friend bool operator>=(const map& x, const map& y);
  friend bool operator<=(const map& x, const map& y);
  // specialized algorithms:






  // Generic algorithm
  friend map::iterator
    search(map::iterator first1,map::iterator last1,
           map::iterator first2,map::iterator last2);


  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(map::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_map" 2






# 1 "/home/wmtan/root/cint/stl/_multimap" 1

#pragma include_noerr <multimap.dll>
#pragma include_noerr <multimap2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multimap" 1
// lib/prec_stl/multimap

#pragma ifndef PREC_STL_MULTIMAP
#pragma define PREC_STL_MULTIMAP
#pragma link off global PREC_STL_MULTIMAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multimap {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multimap::iterator& x ,const multimap::iterator& y) const;
  friend bool operator!=(const multimap::iterator& x ,const multimap::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;
  friend bool operator!=(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multimap.cons_ construct/copy/destroy:
  multimap();






  multimap(iterator first, iterator last);
  multimap(reverse_iterator first, reverse_iterator last);

  multimap(const multimap& x);
  ~multimap();
  multimap& operator=(const multimap& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.multimap.access_ element access:
  //T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(multimap&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.multimap.ops_ multimap operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const multimap& x, const multimap& y);
  friend bool operator< (const multimap& x, const multimap& y);
  friend bool operator!=(const multimap& x, const multimap& y);
  friend bool operator> (const multimap& x, const multimap& y);
  friend bool operator>=(const multimap& x, const multimap& y);
  friend bool operator<=(const multimap& x, const multimap& y);
  // specialized algorithms:






  // Generic algorithm
  friend multimap::iterator
    search(multimap::iterator first1,multimap::iterator last1,
           multimap::iterator first2,multimap::iterator last2);



  // Generic algorithm
  //friend void reverse(multimap::iterator first,multimap::iterator last);
  //friend void reverse(multimap::reverse_iterator first,multimap::reverse_itetator last);

  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(multimap::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif



# 7 "/home/wmtan/root/cint/stl/_multimap" 2




# 13 "/home/wmtan/root/cint/stl/_map" 2

# 2 "/home/wmtan/root/cint/stl/map" 2

}
# 7 "Utilities/Persistency/interface/Persistency.h" 2


# 1 "Utilities/Configuration/interface/FixedSizeTypes.h" 1


//
//  Fixed Size Types
//
// V1.0 14/12/99 by VI
//
// V1.1 18/10/02 by WMT
//


typedef char     int8;
typedef unsigned char    uint8;
typedef short     int16;
typedef unsigned short   uint16;

typedef int    int32;
typedef unsigned int   uint32;
//typedef long long    int64;
//typedef unsigned long long  uint64;


typedef float  float32;
typedef double  float64;

typedef int8     c_int8;
typedef uint8    c_uint8;
typedef int16    c_int16;
typedef uint16   c_uint16;

typedef int32    c_int32;
typedef uint32   c_uint32;
//typedef int64    c_int64;
//typedef uint64   c_uint64;


typedef float32  c_float32;
typedef float64  c_float64;


typedef unsigned char  c_bool;





// FixedSizeTypes_H
# 9 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooRunObj.h" 1


# 1 "Utilities/Persistency/interface/ooObj.h" 1


# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 3 "Utilities/Persistency/interface/ooObj.h" 2

# 1 "/home/wmtan/root/cint/stl/map" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_map" 1

#pragma include_noerr <map.dll>
#pragma include_noerr <map2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/map" 1
// lib/prec_stl/map

#pragma ifndef PREC_STL_MAP
#pragma define PREC_STL_MAP
#pragma link off global PREC_STL_MAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/map" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class map {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const map::iterator& x ,const map::iterator& y) const;
  friend bool operator!=(const map::iterator& x ,const map::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;
  friend bool operator!=(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.map.cons_ construct/copy/destroy:
  map();






  map(iterator first, iterator last);
  map(reverse_iterator first, reverse_iterator last);

  map(const map& x);
  ~map();
  map& operator=(const map& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.map.access_ element access:
  T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(map&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.map.ops_ map operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const map& x, const map& y);
  friend bool operator< (const map& x, const map& y);
  friend bool operator!=(const map& x, const map& y);
  friend bool operator> (const map& x, const map& y);
  friend bool operator>=(const map& x, const map& y);
  friend bool operator<=(const map& x, const map& y);
  // specialized algorithms:






  // Generic algorithm
  friend map::iterator
    search(map::iterator first1,map::iterator last1,
           map::iterator first2,map::iterator last2);


  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(map::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_map" 2






# 1 "/home/wmtan/root/cint/stl/_multimap" 1

#pragma include_noerr <multimap.dll>
#pragma include_noerr <multimap2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multimap" 1
// lib/prec_stl/multimap

#pragma ifndef PREC_STL_MULTIMAP
#pragma define PREC_STL_MULTIMAP
#pragma link off global PREC_STL_MULTIMAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multimap {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multimap::iterator& x ,const multimap::iterator& y) const;
  friend bool operator!=(const multimap::iterator& x ,const multimap::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;
  friend bool operator!=(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multimap.cons_ construct/copy/destroy:
  multimap();






  multimap(iterator first, iterator last);
  multimap(reverse_iterator first, reverse_iterator last);

  multimap(const multimap& x);
  ~multimap();
  multimap& operator=(const multimap& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.multimap.access_ element access:
  //T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(multimap&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.multimap.ops_ multimap operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const multimap& x, const multimap& y);
  friend bool operator< (const multimap& x, const multimap& y);
  friend bool operator!=(const multimap& x, const multimap& y);
  friend bool operator> (const multimap& x, const multimap& y);
  friend bool operator>=(const multimap& x, const multimap& y);
  friend bool operator<=(const multimap& x, const multimap& y);
  // specialized algorithms:






  // Generic algorithm
  friend multimap::iterator
    search(multimap::iterator first1,multimap::iterator last1,
           multimap::iterator first2,multimap::iterator last2);



  // Generic algorithm
  //friend void reverse(multimap::iterator first,multimap::iterator last);
  //friend void reverse(multimap::reverse_iterator first,multimap::reverse_itetator last);

  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(multimap::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif



# 7 "/home/wmtan/root/cint/stl/_multimap" 2




# 13 "/home/wmtan/root/cint/stl/_map" 2

# 2 "/home/wmtan/root/cint/stl/map" 2

}
# 4 "Utilities/Persistency/interface/ooObj.h" 2

# 1 "/home/wmtan/root/cint/stl/vector" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_vector" 1

#pragma include_noerr <vector.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/vector" 1
// lib/prec_stl/vector

#pragma ifndef PREC_STL_VECTOR
#pragma define PREC_STL_VECTOR
#pragma link off global PREC_STL_VECTOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/vector" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/vector" 2

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/vector" 2





template<class T,class Allocator=alloc>





class vector {
 public:
  typedef T value_type;


  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;









# 102 "/home/wmtan/root/cint/lib/prec_stl/vector"

  typedef T* iterator;
  typedef const T* const_iterator;







// G__BORLANDCC5








  class reverse_iterator 



        : public std::random_access_iterator<T,difference_type>

        {
   public:
    reverse_iterator(const reverse_iterator& x) ;

    reverse_iterator& operator=(const reverse_iterator& x) ;


    T* base() ;



    T& operator*() const ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
    reverse_iterator operator+(long n);
    reverse_iterator operator-(long n);
    reverse_iterator& operator+=(long n);
    reverse_iterator& operator-=(long n);
    T& operator[](long n) ;
   private:
  };

// G__BORLANDCC5
  friend bool operator==(const vector::reverse_iterator& x
                        ,const vector::reverse_iterator& y) const;
  friend bool operator!=(const vector::reverse_iterator& x
                        ,const vector::reverse_iterator& y) const;


  typedef const reverse_iterator const_reverse_iterator;
















  iterator begin(void) ;
  iterator end(void) ;
  reverse_iterator rbegin(void) ;
  reverse_iterator rend(void) ;






  size_type size(void) const ;
  size_type max_size(void) const ;
  size_type capacity(void) const ;
  bool empty(void) const ;
  T& operator[](size_type n) ;
  vector(void) ;
  vector(size_type n,const T& value=T()) ;
  vector(const vector& x) ;
  vector(const_iterator first,const_iterator last) ;
  ~vector(void) ;
  vector& operator=(const vector& x);
  void reserve(size_type n) ;
  T& front(void) ;
  T& back(void) ;
  void push_back(const T& x) ;
  void swap(vector& x);
  iterator insert(iterator position,const T& x);
  void insert(iterator position,const_iterator first,const_iterator last);
  void insert(iterator position,size_type n,const T& x);
  void pop_back(void) ;
  void erase(iterator position) ;
  void erase(iterator first,iterator last) ;
  void clear() ;

# 217 "/home/wmtan/root/cint/lib/prec_stl/vector"

  // specialized algorithms:








  // Generic algorithm


  // input iter
  friend vector::iterator 
    find(vector::iterator first,vector::iterator last,const T& value);
  // forward iter
  friend vector::iterator 
    find_end(vector::iterator first1,vector::iterator last1,
             vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    find_first_of(vector::iterator first1,vector::iterator last1,
                  vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    adjacent_find(vector::iterator first,vector::iterator last);
  // input iter

  friend vector::difference_type
    count(vector::iterator first,vector::iterator last,const T& value);






  friend bool
    equal(vector::iterator first1,vector::iterator last1,
          vector::iterator first2);
  // forward iter
  friend vector::iterator
    search(vector::iterator first1,vector::iterator last1,
           vector::iterator first2,vector::iterator last2);
  friend vector::iterator
    search_n(vector::iterator first,vector::iterator last
             ,vector::size_type count,const T& value);
  // input and output iter -> forward iter
  friend vector::iterator
    copy(vector::iterator first,vector::iterator last,
         vector::iterator result);
  // bidirectional iter
  friend vector::iterator
    copy_backward(vector::iterator first,vector::iterator last,
                  vector::iterator result);
  // just value_type
  friend void swap(T& a,T& b);
  // forward iter
  friend vector::iterator
    swap_ranges(vector::iterator first1,vector::iterator last1,
                vector::iterator first2);
  friend void iter_swap(vector::iterator a,vector::iterator b);
  friend void replace(vector::iterator first,vector::iterator last,
                      const T& old_value,const T& new_value);
  // input, output iter -> forward iter
  friend vector::iterator 
    replace_copy(vector::iterator first,vector::iterator last,
                 vector::iterator result,
                 const T& old_value,const T& new_value);
  // forward iter
  friend void
    fill(vector::iterator first,vector::iterator last,const T& value);




  friend vector::iterator
    remove(vector::iterator first,vector::iterator last,const T& value);
  // input,output iter -> forward iter
  friend vector::iterator
    remove_copy(vector::iterator first,vector::iterator last,
                vector::iterator result,const T& value);
  friend vector::iterator
    unique(vector::iterator first,vector::iterator last);
  friend vector::iterator 
    unique_copy(vector::iterator first,vector::iterator last,
                vector::iterator result);
  friend void reverse(vector::iterator first,vector::iterator last);
  friend vector::iterator
     reverse_copy(vector::iterator first,vector::iterator last,
                  vector::iterator result);
  // forward iter




  // forward iter
  friend vector::iterator 
    rotate_copy(vector::iterator first,vector::iterator mid,
                vector::iterator last,vector::iterator result);
  // randomaccess iter
  friend void random_shuffle(vector::iterator first,vector::iterator last);
  // randomaccess iter
  friend void sort(vector::iterator first,vector::iterator last);
  friend void stable_sort(vector::iterator first,vector::iterator last);
  friend void partial_sort(vector::iterator first,vector::iterator mid,
                           vector::iterator last);
  friend vector::iterator
    partial_sort_copy(vector::iterator first,vector::iterator last,
                      vector::iterator result_first,
                      vector::iterator result_last);
  friend void nth_element(vector::iterator first,vector::iterator nth,
                          vector::iterator last);
  // forward iter
  friend vector::iterator 
    lower_bound(vector::iterator first,vector::iterator last,const T& value);
  friend vector::iterator 
    upper_bound(vector::iterator first,vector::iterator last,const T& value);




  friend bool binary_search(vector::iterator first,vector::iterator last,
                            const T& value);
  friend vector::iterator merge(vector::iterator first1,vector::iterator last1,
                                vector::iterator first2,vector::iterator last2,
                                vector::iterator result);
  friend void inplace_merge(vector::iterator first,vector::iterator middle,
                            vector::iterator last);
  friend bool includes(vector::iterator first1,vector::iterator last1,
                       vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    set_union(vector::iterator first1,vector::iterator last1,
              vector::iterator first2,vector::iterator last2,
              vector::iterator result);
  friend vector::iterator 
    set_intersection(vector::iterator first1,vector::iterator last1,
                     vector::iterator first2,vector::iterator last2,
                     vector::iterator result);
  friend vector::iterator 
    set_difference(vector::iterator first1,vector::iterator last1,
                   vector::iterator first2,vector::iterator last2,
                   vector::iterator result);
  friend vector::iterator 
    set_symmetric_difference(vector::iterator first1,vector::iterator last1,
                             vector::iterator first2,vector::iterator last2,
                             vector::iterator result);
  // random access
  friend void push_heap(vector::iterator first,vector::iterator last);
  friend void pop_heap(vector::iterator first,vector::iterator last);
  friend void make_heap(vector::iterator first,vector::iterator last);
  friend void sort_heap(vector::iterator first,vector::iterator last);
  // min,max, just value_type
  friend const T& min(const T& a,const T& b);
  friend const T& max(const T& a,const T& b);
  // forward iter
  friend vector::iterator 
    min_element(vector::iterator first,vector::iterator last);
  friend vector::iterator 
    max_element(vector::iterator first,vector::iterator last);
  // input iter
  friend bool
    lexicographical_compare(vector::iterator first1,vector::iterator last1,
                            vector::iterator first2,vector::iterator last2);
  // bidirectional iter
  friend bool next_permutation(vector::iterator first,vector::iterator last);
  friend bool prev_permutation(vector::iterator first,vector::iterator last);

// G__VISUAL,G__GNUC,G__BORLAND
# 406 "/home/wmtan/root/cint/lib/prec_stl/vector"


// G__NOALGORITHM

  // iterator_category resolution
  //friend random_access_iterator_tag iterator_category(vector::iterator x);

};

// G__defined("std::vector<bool>")
# 423 "/home/wmtan/root/cint/lib/prec_stl/vector"




#pragma endif
# 6 "/home/wmtan/root/cint/stl/_vector" 2




# 2 "/home/wmtan/root/cint/stl/vector" 2

}
# 5 "Utilities/Persistency/interface/ooObj.h" 2



# 1 "Utilities/Persistency/interface/ooRefBase.h" 1


# 1 "/home/wmtan/root/cint/include/assert.h" 1
/****************************************************************
* assert.h
*****************************************************************/




# 22 "/home/wmtan/root/cint/include/assert.h"



# 3 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 4 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/cint/stl/vector" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_vector" 1

#pragma include_noerr <vector.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/vector" 1
// lib/prec_stl/vector

#pragma ifndef PREC_STL_VECTOR
#pragma define PREC_STL_VECTOR
#pragma link off global PREC_STL_VECTOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/vector" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/vector" 2

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/vector" 2





template<class T,class Allocator=alloc>





class vector {
 public:
  typedef T value_type;


  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;









# 102 "/home/wmtan/root/cint/lib/prec_stl/vector"

  typedef T* iterator;
  typedef const T* const_iterator;







// G__BORLANDCC5








  class reverse_iterator 



        : public std::random_access_iterator<T,difference_type>

        {
   public:
    reverse_iterator(const reverse_iterator& x) ;

    reverse_iterator& operator=(const reverse_iterator& x) ;


    T* base() ;



    T& operator*() const ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
    reverse_iterator operator+(long n);
    reverse_iterator operator-(long n);
    reverse_iterator& operator+=(long n);
    reverse_iterator& operator-=(long n);
    T& operator[](long n) ;
   private:
  };

// G__BORLANDCC5
  friend bool operator==(const vector::reverse_iterator& x
                        ,const vector::reverse_iterator& y) const;
  friend bool operator!=(const vector::reverse_iterator& x
                        ,const vector::reverse_iterator& y) const;


  typedef const reverse_iterator const_reverse_iterator;
















  iterator begin(void) ;
  iterator end(void) ;
  reverse_iterator rbegin(void) ;
  reverse_iterator rend(void) ;






  size_type size(void) const ;
  size_type max_size(void) const ;
  size_type capacity(void) const ;
  bool empty(void) const ;
  T& operator[](size_type n) ;
  vector(void) ;
  vector(size_type n,const T& value=T()) ;
  vector(const vector& x) ;
  vector(const_iterator first,const_iterator last) ;
  ~vector(void) ;
  vector& operator=(const vector& x);
  void reserve(size_type n) ;
  T& front(void) ;
  T& back(void) ;
  void push_back(const T& x) ;
  void swap(vector& x);
  iterator insert(iterator position,const T& x);
  void insert(iterator position,const_iterator first,const_iterator last);
  void insert(iterator position,size_type n,const T& x);
  void pop_back(void) ;
  void erase(iterator position) ;
  void erase(iterator first,iterator last) ;
  void clear() ;

# 217 "/home/wmtan/root/cint/lib/prec_stl/vector"

  // specialized algorithms:








  // Generic algorithm


  // input iter
  friend vector::iterator 
    find(vector::iterator first,vector::iterator last,const T& value);
  // forward iter
  friend vector::iterator 
    find_end(vector::iterator first1,vector::iterator last1,
             vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    find_first_of(vector::iterator first1,vector::iterator last1,
                  vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    adjacent_find(vector::iterator first,vector::iterator last);
  // input iter

  friend vector::difference_type
    count(vector::iterator first,vector::iterator last,const T& value);






  friend bool
    equal(vector::iterator first1,vector::iterator last1,
          vector::iterator first2);
  // forward iter
  friend vector::iterator
    search(vector::iterator first1,vector::iterator last1,
           vector::iterator first2,vector::iterator last2);
  friend vector::iterator
    search_n(vector::iterator first,vector::iterator last
             ,vector::size_type count,const T& value);
  // input and output iter -> forward iter
  friend vector::iterator
    copy(vector::iterator first,vector::iterator last,
         vector::iterator result);
  // bidirectional iter
  friend vector::iterator
    copy_backward(vector::iterator first,vector::iterator last,
                  vector::iterator result);
  // just value_type
  friend void swap(T& a,T& b);
  // forward iter
  friend vector::iterator
    swap_ranges(vector::iterator first1,vector::iterator last1,
                vector::iterator first2);
  friend void iter_swap(vector::iterator a,vector::iterator b);
  friend void replace(vector::iterator first,vector::iterator last,
                      const T& old_value,const T& new_value);
  // input, output iter -> forward iter
  friend vector::iterator 
    replace_copy(vector::iterator first,vector::iterator last,
                 vector::iterator result,
                 const T& old_value,const T& new_value);
  // forward iter
  friend void
    fill(vector::iterator first,vector::iterator last,const T& value);




  friend vector::iterator
    remove(vector::iterator first,vector::iterator last,const T& value);
  // input,output iter -> forward iter
  friend vector::iterator
    remove_copy(vector::iterator first,vector::iterator last,
                vector::iterator result,const T& value);
  friend vector::iterator
    unique(vector::iterator first,vector::iterator last);
  friend vector::iterator 
    unique_copy(vector::iterator first,vector::iterator last,
                vector::iterator result);
  friend void reverse(vector::iterator first,vector::iterator last);
  friend vector::iterator
     reverse_copy(vector::iterator first,vector::iterator last,
                  vector::iterator result);
  // forward iter




  // forward iter
  friend vector::iterator 
    rotate_copy(vector::iterator first,vector::iterator mid,
                vector::iterator last,vector::iterator result);
  // randomaccess iter
  friend void random_shuffle(vector::iterator first,vector::iterator last);
  // randomaccess iter
  friend void sort(vector::iterator first,vector::iterator last);
  friend void stable_sort(vector::iterator first,vector::iterator last);
  friend void partial_sort(vector::iterator first,vector::iterator mid,
                           vector::iterator last);
  friend vector::iterator
    partial_sort_copy(vector::iterator first,vector::iterator last,
                      vector::iterator result_first,
                      vector::iterator result_last);
  friend void nth_element(vector::iterator first,vector::iterator nth,
                          vector::iterator last);
  // forward iter
  friend vector::iterator 
    lower_bound(vector::iterator first,vector::iterator last,const T& value);
  friend vector::iterator 
    upper_bound(vector::iterator first,vector::iterator last,const T& value);




  friend bool binary_search(vector::iterator first,vector::iterator last,
                            const T& value);
  friend vector::iterator merge(vector::iterator first1,vector::iterator last1,
                                vector::iterator first2,vector::iterator last2,
                                vector::iterator result);
  friend void inplace_merge(vector::iterator first,vector::iterator middle,
                            vector::iterator last);
  friend bool includes(vector::iterator first1,vector::iterator last1,
                       vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    set_union(vector::iterator first1,vector::iterator last1,
              vector::iterator first2,vector::iterator last2,
              vector::iterator result);
  friend vector::iterator 
    set_intersection(vector::iterator first1,vector::iterator last1,
                     vector::iterator first2,vector::iterator last2,
                     vector::iterator result);
  friend vector::iterator 
    set_difference(vector::iterator first1,vector::iterator last1,
                   vector::iterator first2,vector::iterator last2,
                   vector::iterator result);
  friend vector::iterator 
    set_symmetric_difference(vector::iterator first1,vector::iterator last1,
                             vector::iterator first2,vector::iterator last2,
                             vector::iterator result);
  // random access
  friend void push_heap(vector::iterator first,vector::iterator last);
  friend void pop_heap(vector::iterator first,vector::iterator last);
  friend void make_heap(vector::iterator first,vector::iterator last);
  friend void sort_heap(vector::iterator first,vector::iterator last);
  // min,max, just value_type
  friend const T& min(const T& a,const T& b);
  friend const T& max(const T& a,const T& b);
  // forward iter
  friend vector::iterator 
    min_element(vector::iterator first,vector::iterator last);
  friend vector::iterator 
    max_element(vector::iterator first,vector::iterator last);
  // input iter
  friend bool
    lexicographical_compare(vector::iterator first1,vector::iterator last1,
                            vector::iterator first2,vector::iterator last2);
  // bidirectional iter
  friend bool next_permutation(vector::iterator first,vector::iterator last);
  friend bool prev_permutation(vector::iterator first,vector::iterator last);

// G__VISUAL,G__GNUC,G__BORLAND
# 406 "/home/wmtan/root/cint/lib/prec_stl/vector"


// G__NOALGORITHM

  // iterator_category resolution
  //friend random_access_iterator_tag iterator_category(vector::iterator x);

};

// G__defined("std::vector<bool>")
# 423 "/home/wmtan/root/cint/lib/prec_stl/vector"




#pragma endif
# 6 "/home/wmtan/root/cint/stl/_vector" 2




# 2 "/home/wmtan/root/cint/stl/vector" 2

}
# 5 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/cint/stl/set" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_set" 1


#pragma include_noerr <set.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/set" 1
// lib/prec_stl/set

#pragma ifndef PREC_STL_SET
#pragma define PREC_STL_SET
#pragma link off global PREC_STL_SET;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/set" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/set" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/set" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/set" 2






template<class Key,class Compare=std::less<Key>
        ,class Allocator=alloc>







class set {
 public:
  // types:
  typedef Key                                   key_type;
  typedef Key                                   value_type;
  typedef Compare                               key_compare;
  typedef Compare                               value_compare;
  typedef Allocator                             allocator_type;

  typedef Key*                                  pointer;
  typedef const Key*                            const_pointer;
  typedef Key&                                  reference;
  typedef const Key&                            const_reference;
  typedef size_t                                size_type;
  typedef ptrdiff_t                             difference_type;









  class iterator





        : public bidirectional_iterator<Key,difference_type>

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;


    value_type operator*() const;



    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const set::iterator& x ,const set::iterator& y) const;
  friend bool operator!=(const set::iterator& x ,const set::iterator& y) const;





  class reverse_iterator





        : public bidirectional_iterator<Key,difference_type>

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;


    value_type operator*() const;



    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const set::reverse_iterator& x
                        ,const set::reverse_iterator& y) const;
  friend bool operator!=(const set::reverse_iterator& x
                        ,const set::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.set.cons_ construct/copy/destroy:
  //set(const Compare& comp=Compare(), const Allocator&=Allocator());
  set();





  set(const set& x);
  ~set();
  set& operator= (const set& x);
  //allocator_type get_allocator() const;
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();







  // capacity:
  bool          empty() const;
  size_type     size() const;
  size_type     max_size() const;
  // modifiers:
  //pair<iterator,bool> insert(const Key& x);
  iterator            insert(iterator position, const Key& x);




  void      erase(iterator position);



  void      erase(iterator first, iterator last);
  void swap(set<Key,Compare,Allocator>&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // set operations:



  iterator  find(const Key& x) const;

  size_type count(const Key& x) const;




  iterator  lower_bound(const Key& x) const;
  iterator  upper_bound(const Key& x) const;

  //pair<iterator,iterator> equal_range(const Key& x) const;

  friend bool operator==(const set& x, const set& y);
  friend bool operator< (const set& x, const set& y);
  friend bool operator!=(const set& x, const set& y);
  friend bool operator> (const set& x, const set& y);
  friend bool operator>=(const set& x, const set& y);
  friend bool operator<=(const set& x, const set& y);
  // specialized algorithms:
  //friend void swap(set& x, set& y);



  // Generic algorithm


  friend set::iterator 
    find(set::iterator first,set::iterator last,const Key& value);
  friend set::iterator
    search(set::iterator first1,set::iterator last1,
           set::iterator first2,set::iterator last2);

// G__GNUC || G__BORLAND
# 230 "/home/wmtan/root/cint/lib/prec_stl/set"

// G__NOALGORITHM

};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_set" 2





# 1 "/home/wmtan/root/cint/stl/_multiset" 1

#pragma include_noerr <multiset.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multiset" 1
// lib/prec_stl/multiset

#pragma ifndef PREC_STL_MULTISET
#pragma define PREC_STL_MULTISET
#pragma link off global PREC_STL_MULTISET;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multiset" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multiset" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multiset" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multiset" 2






template<class Key,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multiset {
 public:
  // types:
  typedef Key                                   key_type;
  typedef Key                                   value_type;
  typedef Compare                               key_compare;
  typedef Compare                               value_compare;
  typedef Allocator                             allocator_type;

  typedef Key*                                  pointer;
  typedef const Key*                            const_pointer;
  typedef Key&                                  reference;
  typedef const Key&                            const_reference;
  typedef size_t                                size_type;
  typedef ptrdiff_t                             difference_type;









  class iterator





        : public bidirectional_iterator<Key,difference_type>

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;


    value_type operator*() const;



    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multiset::iterator& x ,const multiset::iterator& y) const;
  friend bool operator!=(const multiset::iterator& x ,const multiset::iterator& y) const;





  class reverse_iterator





        : public bidirectional_iterator<Key,difference_type>

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;


    value_type operator*() const;



    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multiset::reverse_iterator& x
                        ,const multiset::reverse_iterator& y) const;
  friend bool operator!=(const multiset::reverse_iterator& x
                        ,const multiset::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multiset.cons_ construct/copy/destroy:
  //multiset(const Compare& comp=Compare(), const Allocator&=Allocator());
  multiset();





  multiset(const multiset& x);
  ~multiset();
  multiset& operator= (const multiset& x);
  //allocator_type get_allocator() const;
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();







  // capacity:
  bool          empty() const;
  size_type     size() const;
  size_type     max_size() const;
  // modifiers:
  //pair<iterator,bool> insert(const Key& x);
  iterator            insert(iterator position, const Key& x);




  void      erase(iterator position);



  void      erase(iterator first, iterator last);
  void swap(multiset<Key,Compare,Allocator>&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // multiset operations:



  iterator  find(const Key& x) const;

  size_type count(const Key& x) const;




  iterator  lower_bound(const Key& x) const;
  iterator  upper_bound(const Key& x) const;

  //pair<iterator,iterator> equal_range(const Key& x) const;

  friend bool operator==(const multiset& x, const multiset& y);
  friend bool operator< (const multiset& x, const multiset& y);
  friend bool operator!=(const multiset& x, const multiset& y);
  friend bool operator> (const multiset& x, const multiset& y);
  friend bool operator>=(const multiset& x, const multiset& y);
  friend bool operator<=(const multiset& x, const multiset& y);
  // specialized algorithms:
  //friend void swap(multiset& x, multiset& y);


  // Generic algorithm


  friend multiset::iterator 
    find(multiset::iterator first,multiset::iterator last,const Key& value);
  friend multiset::iterator
    search(multiset::iterator first1,multiset::iterator last1,
           multiset::iterator first2,multiset::iterator last2);

// G__GNUC || G__BORLAND
# 238 "/home/wmtan/root/cint/lib/prec_stl/multiset"

// G__NOALGORITHM

};

#pragma endif



# 6 "/home/wmtan/root/cint/stl/_multiset" 2




# 12 "/home/wmtan/root/cint/stl/_set" 2


# 2 "/home/wmtan/root/cint/stl/set" 2

}
# 6 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/cint/stl/map" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_map" 1

#pragma include_noerr <map.dll>
#pragma include_noerr <map2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/map" 1
// lib/prec_stl/map

#pragma ifndef PREC_STL_MAP
#pragma define PREC_STL_MAP
#pragma link off global PREC_STL_MAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/map" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class map {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const map::iterator& x ,const map::iterator& y) const;
  friend bool operator!=(const map::iterator& x ,const map::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;
  friend bool operator!=(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.map.cons_ construct/copy/destroy:
  map();






  map(iterator first, iterator last);
  map(reverse_iterator first, reverse_iterator last);

  map(const map& x);
  ~map();
  map& operator=(const map& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.map.access_ element access:
  T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(map&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.map.ops_ map operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const map& x, const map& y);
  friend bool operator< (const map& x, const map& y);
  friend bool operator!=(const map& x, const map& y);
  friend bool operator> (const map& x, const map& y);
  friend bool operator>=(const map& x, const map& y);
  friend bool operator<=(const map& x, const map& y);
  // specialized algorithms:






  // Generic algorithm
  friend map::iterator
    search(map::iterator first1,map::iterator last1,
           map::iterator first2,map::iterator last2);


  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(map::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_map" 2






# 1 "/home/wmtan/root/cint/stl/_multimap" 1

#pragma include_noerr <multimap.dll>
#pragma include_noerr <multimap2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multimap" 1
// lib/prec_stl/multimap

#pragma ifndef PREC_STL_MULTIMAP
#pragma define PREC_STL_MULTIMAP
#pragma link off global PREC_STL_MULTIMAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multimap {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multimap::iterator& x ,const multimap::iterator& y) const;
  friend bool operator!=(const multimap::iterator& x ,const multimap::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;
  friend bool operator!=(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multimap.cons_ construct/copy/destroy:
  multimap();






  multimap(iterator first, iterator last);
  multimap(reverse_iterator first, reverse_iterator last);

  multimap(const multimap& x);
  ~multimap();
  multimap& operator=(const multimap& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.multimap.access_ element access:
  //T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(multimap&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.multimap.ops_ multimap operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const multimap& x, const multimap& y);
  friend bool operator< (const multimap& x, const multimap& y);
  friend bool operator!=(const multimap& x, const multimap& y);
  friend bool operator> (const multimap& x, const multimap& y);
  friend bool operator>=(const multimap& x, const multimap& y);
  friend bool operator<=(const multimap& x, const multimap& y);
  // specialized algorithms:






  // Generic algorithm
  friend multimap::iterator
    search(multimap::iterator first1,multimap::iterator last1,
           multimap::iterator first2,multimap::iterator last2);



  // Generic algorithm
  //friend void reverse(multimap::iterator first,multimap::iterator last);
  //friend void reverse(multimap::reverse_iterator first,multimap::reverse_itetator last);

  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(multimap::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif



# 7 "/home/wmtan/root/cint/stl/_multimap" 2




# 13 "/home/wmtan/root/cint/stl/_map" 2

# 2 "/home/wmtan/root/cint/stl/map" 2

}
# 7 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/cint/include/iostream" 1

namespace std {

}
# 1 "/home/wmtan/root/cint/include/_iostream" 1
// include/_iostream

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDL& i) 
        {return(std::endl(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDS& i) 
        {return(std::ends(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_FLUSH& i) 
        {return(std::flush(ostr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_ws& i) 
        {return(std::ws(istr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_WS& i) 
        {return(std::WS(istr));}


std::ostream& operator<<(std::ostream& ostr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::dec);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::hex);
#pragma else
  ostr.unsetf(ios_base::dec);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::hex);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::dec);
  istr.unsetf(ios::oct);
  istr.setf(ios::hex);
#pragma else
  istr.unsetf(ios_base::dec);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::hex);
#pragma endif
  return(istr);
}

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::dec);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::dec);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::oct);
  istr.setf(ios::dec);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::dec);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::dec);
  ostr.setf(ios::oct);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::dec);
  ostr.setf(ios_base::oct);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::dec);
  istr.setf(ios::oct);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::dec);
  istr.setf(ios_base::oct);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(ostr);
}
std::istream& operator<<(std::istream& istr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(istr);
}

// Value evaluation
//template<class T> int G__ateval(const T* x) {return(0);}
template<class T> int G__ateval(const T& x) {return(0);}
int G__ateval(const char* x) {return(0);}
int G__ateval(const void* x) {return(0);}
int G__ateval(const double x) {return(0);}
int G__ateval(const float x) {return(0);}
int G__ateval(const char x) {return(0);}
int G__ateval(const short x) {return(0);}
int G__ateval(const int x) {return(0);}
int G__ateval(const long x) {return(0);}
int G__ateval(const unsigned char x) {return(0);}
int G__ateval(const unsigned short x) {return(0);}
int G__ateval(const unsigned int x) {return(0);}
int G__ateval(const unsigned long x) {return(0);}








# 5 "/home/wmtan/root/cint/include/iostream" 2

# 8 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/cint/include/fstream" 1
namespace std {
# 1 "/home/wmtan/root/cint/include/fstream.h" 1
/*********************************************************************
* fstream.h
*
*********************************************************************/







# 2 "/home/wmtan/root/cint/include/fstream" 2

}
# 9 "Utilities/Persistency/interface/ooRefBase.h" 2




class opiRefBase;


# 1 "/home/wmtan/root/include/TFile.h" 1
// @(#)root/base:$Id$
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFile                                                                //
//                                                                      //
// ROOT file.                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/TDirectory.h" 1
// @(#)root/base:$Id$
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDirectory                                                           //
//                                                                      //
// Describe directory structure in memory.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/TNamed.h" 1
// @(#)root/base:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNamed                                                               //
//                                                                      //
// The basis for a named object (name, title).                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






# 1 "/home/wmtan/root/include/TList.h" 1
// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TList                                                                //
//                                                                      //
// A doubly linked list. All classes inheriting from TObject can be     //
// inserted in a TList.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/TSeqCollection.h" 1
// @(#)root/cont:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSeqCollection                                                       //
//                                                                      //
// Sequenceable collection abstract base class. TSeqCollection's have   //
// an ordering relation, i.e. there is a first and last element.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/TCollection.h" 1
// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCollection                                                          //
//                                                                      //
// Collection abstract base class. This class inherits from TObject     //
// because we want to be able to have collections of collections.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






# 1 "/home/wmtan/root/include/TIterator.h" 1
// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIterator                                                            //
//                                                                      //
// Iterator abstract base class. This base class provides the interface //
// for collection iterators.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





class TCollection;
class TObject;

class TIterator {

protected:
   TIterator() { }
   TIterator(const TIterator &) { }

public:
   virtual TIterator &operator=(const TIterator &) { return *this; }
   virtual ~TIterator() { }
   virtual const TCollection *GetCollection() const = 0;
   virtual Option_t *GetOption() const { return ""; }
   virtual TObject *Next() = 0;
   virtual void Reset() = 0;
   TObject *operator()() { return Next(); }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TIterator  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TIterator  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TIterator.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 47; }   //Iterator abstract base class
};


# 30 "/home/wmtan/root/include/TCollection.h" 2




# 1 "/home/wmtan/root/include/TString.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TString                                                              //
//                                                                      //
// Basic string class.                                                  //
//                                                                      //
// Cannot be stored in a TCollection... use TObjString instead.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






# 1 "/home/wmtan/root/include/TMath.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMath                                                                //
//                                                                      //
// Encapsulate math routines. For the time being avoid templates.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





class TMath {

private:
   static Double_t GamCf(Double_t a,Double_t x);
   static Double_t GamSer(Double_t a,Double_t x);

public:

   // Fundamental constants
   static Double_t Pi()       { return 3.14159265358979323846; }
   static Double_t TwoPi()    { return 2.0 * Pi(); }
   static Double_t PiOver2()  { return Pi() / 2.0; }
   static Double_t PiOver4()  { return Pi() / 4.0; }
   static Double_t InvPi()    { return 1.0 / Pi(); }
   static Double_t RadToDeg() { return 180.0 / Pi(); }
   static Double_t DegToRad() { return Pi() / 180.0; }

   // e (base of natural log)
   static Double_t E()        { return 2.71828182845904523536; }

   // natural log of 10 (to convert log to ln)
   static Double_t Ln10()     { return 2.30258509299404568402; }

   // base-10 log of e  (to convert ln to log)
   static Double_t LogE()     { return 0.43429448190325182765; }

   // velocity of light
   static Double_t C()        { return 2.99792458e8; }        // m s^-1
   static Double_t Ccgs()     { return 100.0 * C(); }         // cm s^-1
   static Double_t CUncertainty() { return 0.0; }             // exact

   // gravitational constant
   static Double_t G()        { return 6.673e-11; }           // m^3 kg^-1 s^-2
   static Double_t Gcgs()     { return G() / 1000.0; }        // cm^3 g^-1 s^-2
   static Double_t GUncertainty() { return 0.010e-11; }

   // G over h-bar C
   static Double_t GhbarC()   { return 6.707e-39; }           // (GeV/c^2)^-2
   static Double_t GhbarCUncertainty() { return 0.010e-39; }

   // standard acceleration of gravity
   static Double_t Gn()       { return 9.80665; }             // m s^-2
   static Double_t GnUncertainty() { return 0.0; }            // exact

   // Planck's constant
   static Double_t H()        { return 6.62606876e-34; }      // J s
   static Double_t Hcgs()     { return 1.0e7 * H(); }         // erg s
   static Double_t HUncertainty() { return 0.00000052e-34; }

   // h-bar (h over 2 pi)
   static Double_t Hbar()     { return 1.054571596e-34; }     // J s
   static Double_t Hbarcgs()  { return 1.0e7 * Hbar(); }      // erg s
   static Double_t HbarUncertainty() { return 0.000000082e-34; }

   // hc (h * c)
   static Double_t HC()       { return H() * C(); }           // J m
   static Double_t HCcgs()    { return Hcgs() * Ccgs(); }     // erg cm

   // Boltzmann's constant
   static Double_t K()        { return 1.3806503e-23; }       // J K^-1
   static Double_t Kcgs()     { return 1.0e7 * K(); }         // erg K^-1
   static Double_t KUncertainty() { return 0.0000024e-23; }

   // Stefan-Boltzmann constant
   static Double_t Sigma()    { return 5.6704e-8; }           // W m^-2 K^-4
   static Double_t SigmaUncertainty() { return 0.000040e-8; }

   // Avogadro constant (Avogadro's Number)
   static Double_t Na()       { return 6.02214199e+23; }      // mol^-1
   static Double_t NaUncertainty() { return 0.00000047e+23; }

   // universal gas constant (Na * K)
   // http://scienceworld.wolfram.com/physics/UniversalGasConstant.html
   static Double_t R()        { return K() * Na(); }          // J K^-1 mol^-1
   static Double_t RUncertainty() { return R()*((KUncertainty()/K()) + (NaUncertainty()/Na())); }

   // Molecular weight of dry air
   // 1976 US Standard Atmosphere,
   // also see http://atmos.nmsu.edu/jsdap/encyclopediawork.html
   static Double_t MWair()    { return 28.9644; }             // kg kmol^-1 (or gm mol^-1)

   // Dry Air Gas Constant (R / MWair)
   // http://atmos.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm
   static Double_t Rgair()    { return (1000.0 * R()) / MWair(); }  // J kg^-1 K^-1

   // Elementary charge
   static Double_t Qe()       { return 1.602176462e-19; }     // C
   static Double_t QeUncertainty() { return 0.000000063e-19; }

   // Trigo
   static Double_t Sin(Double_t);
   static Double_t Cos(Double_t);
   static Double_t Tan(Double_t);
   static Double_t SinH(Double_t);
   static Double_t CosH(Double_t);
   static Double_t TanH(Double_t);
   static Double_t ASin(Double_t);
   static Double_t ACos(Double_t);
   static Double_t ATan(Double_t);
   static Double_t ATan2(Double_t, Double_t);
   static Double_t ASinH(Double_t);
   static Double_t ACosH(Double_t);
   static Double_t ATanH(Double_t);
   static Double_t Hypot(Double_t x, Double_t y);

   // Misc
   static Double_t Sqrt(Double_t x);
   static Double_t Ceil(Double_t x);
   static Double_t Floor(Double_t x);
   static Double_t Exp(Double_t);
   static Double_t Factorial(Int_t);
   static Double_t Power(Double_t x, Double_t y);
   static Double_t Log(Double_t x);
   static Double_t Log2(Double_t x);
   static Double_t Log10(Double_t x);
   static Int_t    Nint(Float_t x);
   static Int_t    Nint(Double_t x);
   static Int_t    Finite(Double_t x);
   static Int_t    IsNaN(Double_t x);

   // Some integer math
   static Long_t   NextPrime(Long_t x);   // Least prime number greater than x
   static Long_t   Sqrt(Long_t x);
   static Long_t   Hypot(Long_t x, Long_t y);     // sqrt(px*px + py*py)

   // Abs
   static Short_t  Abs(Short_t d);
   static Int_t    Abs(Int_t d);
   static Long_t   Abs(Long_t d);
   static Float_t  Abs(Float_t d);
   static Double_t Abs(Double_t d);

   // Even/Odd
   static Bool_t Even(Long_t a);
   static Bool_t Odd(Long_t a);

   // Sign
   static Short_t  Sign(Short_t a, Short_t b);
   static Int_t    Sign(Int_t a, Int_t b);
   static Long_t   Sign(Long_t a, Long_t b);
   static Float_t  Sign(Float_t a, Float_t b);
   static Double_t Sign(Double_t a, Double_t b);

   // Min
   static Short_t  Min(Short_t a, Short_t b);
   static UShort_t Min(UShort_t a, UShort_t b);
   static Int_t    Min(Int_t a, Int_t b);
   static UInt_t   Min(UInt_t a, UInt_t b);
   static Long_t   Min(Long_t a, Long_t b);
   static ULong_t  Min(ULong_t a, ULong_t b);
   static Float_t  Min(Float_t a, Float_t b);
   static Double_t Min(Double_t a, Double_t b);

   // Max
   static Short_t  Max(Short_t a, Short_t b);
   static UShort_t Max(UShort_t a, UShort_t b);
   static Int_t    Max(Int_t a, Int_t b);
   static UInt_t   Max(UInt_t a, UInt_t b);
   static Long_t   Max(Long_t a, Long_t b);
   static ULong_t  Max(ULong_t a, ULong_t b);
   static Float_t  Max(Float_t a, Float_t b);
   static Double_t Max(Double_t a, Double_t b);

   // Locate Min, Max
   static Int_t  LocMin(Int_t n, const Short_t *a);
   static Int_t  LocMin(Int_t n, const Int_t *a);
   static Int_t  LocMin(Int_t n, const Float_t *a);
   static Int_t  LocMin(Int_t n, const Double_t *a);
   static Int_t  LocMin(Int_t n, const Long_t *a);
   static Int_t  LocMax(Int_t n, const Short_t *a);
   static Int_t  LocMax(Int_t n, const Int_t *a);
   static Int_t  LocMax(Int_t n, const Float_t *a);
   static Int_t  LocMax(Int_t n, const Double_t *a);
   static Int_t  LocMax(Int_t n, const Long_t *a);

   // Range
   static Short_t  Range(Short_t lb, Short_t ub, Short_t x);
   static Int_t    Range(Int_t lb, Int_t ub, Int_t x);
   static Long_t   Range(Long_t lb, Long_t ub, Long_t x);
   static ULong_t  Range(ULong_t lb, ULong_t ub, ULong_t x);
   static Double_t Range(Double_t lb, Double_t ub, Double_t x);

   // Binary search
   static Int_t BinarySearch(Int_t n, const Short_t *array, Short_t value);
   static Int_t BinarySearch(Int_t n, const Short_t **array, Short_t value);
   static Int_t BinarySearch(Int_t n, const Int_t *array, Int_t value);
   static Int_t BinarySearch(Int_t n, const Int_t **array, Int_t value);
   static Int_t BinarySearch(Int_t n, const Float_t *array, Float_t value);
   static Int_t BinarySearch(Int_t n, const Float_t **array, Float_t value);
   static Int_t BinarySearch(Int_t n, const Double_t *array, Double_t value);
   static Int_t BinarySearch(Int_t n, const Double_t **array, Double_t value);
   static Int_t BinarySearch(Int_t n, const Long_t *array, Long_t value);
   static Int_t BinarySearch(Int_t n, const Long_t **array, Long_t value);

   // Hashing
   static ULong_t Hash(const void *txt, Int_t ntxt);
   static ULong_t Hash(const char *str);

   // IsInside
   static Bool_t IsInside(Double_t xp, Double_t yp, Int_t np, Double_t *x, Double_t *y);
   static Bool_t IsInside(Float_t xp, Float_t yp, Int_t np, Float_t *x, Float_t *y);
   static Bool_t IsInside(Int_t xp, Int_t yp, Int_t np, Int_t *x, Int_t *y);

   // Sorting
   static void Sort(Int_t n, const Short_t *a,  Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Int_t *a,    Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Float_t *a,  Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Double_t *a, Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Long_t *a,   Int_t *index, Bool_t down=kTRUE);
   static void BubbleHigh(Int_t Narr, Double_t *arr1, Int_t *arr2);
   static void BubbleLow (Int_t Narr, Double_t *arr1, Int_t *arr2);

   // Advanced
   static Float_t *Cross(Float_t v1[3],Float_t v2[3],Float_t out[3]);     // Calculate the Cross Product of two vectors
   static Float_t  Normalize(Float_t v[3]);                               // Normalize a vector
   static Float_t  NormCross(Float_t v1[3],Float_t v2[3],Float_t out[3]); // Calculate the Normalized Cross Product of two vectors
   static Float_t *Normal2Plane(Float_t v1[3],Float_t v2[3],Float_t v3[3], Float_t normal[3]); // Calcualte a normal vector of a plane

   static Double_t *Cross(Double_t v1[3],Double_t v2[3],Double_t out[3]);// Calculate the Cross Product of two vectors
   static Double_t  Erf(Double_t x);
   static Double_t  Erfc(Double_t x);
   static Double_t  Freq(Double_t x);
   static Double_t  Gamma(Double_t z);
   static Double_t  Gamma(Double_t a,Double_t x);
   static Double_t  BreitWigner(Double_t x, Double_t mean=0, Double_t gamma=1);
   static Double_t  Gaus(Double_t x, Double_t mean=0, Double_t sigma=1);
   static Double_t  Landau(Double_t x, Double_t mean=0, Double_t sigma=1);
   static Double_t  LnGamma(Double_t z);
   static Double_t  Normalize(Double_t v[3]);                             // Normalize a vector
   static Double_t  NormCross(Double_t v1[3],Double_t v2[3],Double_t out[3]); // Calculate the Normalized Cross Product of two vectors
   static Double_t *Normal2Plane(Double_t v1[3],Double_t v2[3],Double_t v3[3], Double_t normal[3]); // Calcualte a normal vector of a plane
   static Double_t  Prob(Double_t chi2,Int_t ndf);
   static Double_t  KolmogorovProb(Double_t z);
   static Double_t  Voigt(Double_t x, Double_t sigma, Double_t lg, Int_t R = 4);

   // Bessel functions
   static Double_t BesselI(Int_t n,Double_t x);      // integer order modified Bessel function I_n(x)
   static Double_t BesselK(Int_t n,Double_t x);      // integer order modified Bessel function K_n(x)
   static Double_t BesselI0(Double_t x);             // modified Bessel function I_0(x)
   static Double_t BesselK0(Double_t x);             // modified Bessel function K_0(x)
   static Double_t BesselI1(Double_t x);             // modified Bessel function I_1(x)
   static Double_t BesselK1(Double_t x);             // modified Bessel function K_1(x)
   static Double_t BesselJ0(Double_t x);             // Bessel function J0(x) for any real x
   static Double_t BesselJ1(Double_t x);             // Bessel function J1(x) for any real x
   static Double_t BesselY0(Double_t x);             // Bessel function Y0(x) for positive x
   static Double_t BesselY1(Double_t x);             // Bessel function Y1(x) for positive x
   static Double_t Struve(Int_t n, Double_t x);      // Struve functions of order 0 and 1

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TMath  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TMath  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TMath.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 276; }   //Interface to math routines
};


//---- Even/odd ----------------------------------------------------------------

inline Bool_t TMath::Even(Long_t a)
   { return ! (a & 1); }

inline Bool_t TMath::Odd(Long_t a)
   { return (a & 1); }

//---- Abs ---------------------------------------------------------------------

inline Short_t TMath::Abs(Short_t d)
   { return (d > 0) ? d : -d; }

inline Int_t TMath::Abs(Int_t d)
   { return (d > 0) ? d : -d; }

inline Long_t TMath::Abs(Long_t d)
   { return (d > 0) ? d : -d; }

inline Float_t TMath::Abs(Float_t d)
   { return (d > 0) ? d : -d; }

inline Double_t TMath::Abs(Double_t d)
   { return (d > 0) ? d : -d; }

//---- Sign --------------------------------------------------------------------

inline Short_t TMath::Sign(Short_t a, Short_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Int_t TMath::Sign(Int_t a, Int_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Long_t TMath::Sign(Long_t a, Long_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Float_t TMath::Sign(Float_t a, Float_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Double_t TMath::Sign(Double_t a, Double_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

//---- Min ---------------------------------------------------------------------

inline Short_t TMath::Min(Short_t a, Short_t b)
   { return a <= b ? a : b; }

inline UShort_t TMath::Min(UShort_t a, UShort_t b)
   { return a <= b ? a : b; }

inline Int_t TMath::Min(Int_t a, Int_t b)
   { return a <= b ? a : b; }

inline UInt_t TMath::Min(UInt_t a, UInt_t b)
   { return a <= b ? a : b; }

inline Long_t TMath::Min(Long_t a, Long_t b)
   { return a <= b ? a : b; }

inline ULong_t TMath::Min(ULong_t a, ULong_t b)
   { return a <= b ? a : b; }

inline Float_t TMath::Min(Float_t a, Float_t b)
   { return a <= b ? a : b; }

inline Double_t TMath::Min(Double_t a, Double_t b)
   { return a <= b ? a : b; }

//---- Max ---------------------------------------------------------------------

inline Short_t TMath::Max(Short_t a, Short_t b)
   { return a >= b ? a : b; }

inline UShort_t TMath::Max(UShort_t a, UShort_t b)
   { return a >= b ? a : b; }

inline Int_t TMath::Max(Int_t a, Int_t b)
   { return a >= b ? a : b; }

inline UInt_t TMath::Max(UInt_t a, UInt_t b)
   { return a >= b ? a : b; }

inline Long_t TMath::Max(Long_t a, Long_t b)
   { return a >= b ? a : b; }

inline ULong_t TMath::Max(ULong_t a, ULong_t b)
   { return a >= b ? a : b; }

inline Float_t TMath::Max(Float_t a, Float_t b)
   { return a >= b ? a : b; }

inline Double_t TMath::Max(Double_t a, Double_t b)
   { return a >= b ? a : b; }

//---- Range -------------------------------------------------------------------

inline Short_t TMath::Range(Short_t lb, Short_t ub, Short_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Int_t TMath::Range(Int_t lb, Int_t ub, Int_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Long_t TMath::Range(Long_t lb, Long_t ub, Long_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline ULong_t TMath::Range(ULong_t lb, ULong_t ub, ULong_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Double_t TMath::Range(Double_t lb, Double_t ub, Double_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

//---- Trig and other functions ------------------------------------------------


# 1 "/home/wmtan/root/cint/include/float.h" 1
































# 394 "/home/wmtan/root/include/TMath.h" 2










// math functions are defined inline so we have to include them here
# 1 "/home/wmtan/root/cint/include/math.h" 1





#pragma include_noerr <stdfunc.dll>

# 405 "/home/wmtan/root/include/TMath.h" 2












# 444 "/home/wmtan/root/include/TMath.h"


inline Double_t TMath::Sin(Double_t x)
   { return sin(x); }

inline Double_t TMath::Cos(Double_t x)
   { return cos(x); }

inline Double_t TMath::Tan(Double_t x)
   { return tan(x); }

inline Double_t TMath::SinH(Double_t x)
   { return sinh(x); }

inline Double_t TMath::CosH(Double_t x)
   { return cosh(x); }

inline Double_t TMath::TanH(Double_t x)
   { return tanh(x); }

inline Double_t TMath::ASin(Double_t x)
   { return asin(x); }

inline Double_t TMath::ACos(Double_t x)
   { return acos(x); }

inline Double_t TMath::ATan(Double_t x)
   { return atan(x); }

inline Double_t TMath::ATan2(Double_t y, Double_t x)
   { return x != 0 ? atan2(y, x) : (y > 0 ? Pi()/2 : -Pi()/2); }

inline Double_t TMath::Sqrt(Double_t x)
   { return sqrt(x); }

inline Double_t TMath::Exp(Double_t x)
   { return exp(x); }

inline Double_t TMath::Power(Double_t x, Double_t y)
   { return pow(x, y); }

inline Double_t TMath::Log(Double_t x)
   { return log(x); }

inline Double_t TMath::Log10(Double_t x)
   { return log10(x); }

inline Int_t TMath::Finite(Double_t x)



   { return finite(x); }


inline Int_t TMath::IsNaN(Double_t x)
   { return isnan(x); }

//-------- Advanced -------------

inline Float_t TMath::NormCross(Float_t v1[3],Float_t v2[3],Float_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}

inline Double_t TMath::NormCross(Double_t v1[3],Double_t v2[3],Double_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}


# 31 "/home/wmtan/root/include/TString.h" 2




# 1 "/home/wmtan/root/include/TRefCnt.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TRefCnt                                                             //
//                                                                      //
//  Base class for reference counted objects.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






class TRefCnt {

protected:
   UInt_t  fRefs;      // (1 less than) number of references

public:
   enum EReferenceFlag { kStaticInit };

   TRefCnt(Int_t initRef = 0) : fRefs((UInt_t)initRef-1) { }
   TRefCnt(EReferenceFlag) { }  // leave fRefs alone
   UInt_t   References() const      { return fRefs+1; }
   void     SetRefCount(UInt_t r)   { fRefs = r-1; }
   void     AddReference()          { fRefs++; }
   UInt_t   RemoveReference()       { return fRefs--; }
};


# 35 "/home/wmtan/root/include/TString.h" 2













class TRegexp;
class TString;
class TSubString;

TString operator+(const TString& s1, const TString& s2);
TString operator+(const TString& s,  const char *cs);
TString operator+(const char *cs, const TString& s);
TString operator+(const TString& s, char c);
TString operator+(const TString& s, Long_t i);
TString operator+(const TString& s, ULong_t i);
TString operator+(char c, const TString& s);
TString operator+(Long_t i, const TString& s);
TString operator+(ULong_t i, const TString& s);
Bool_t  operator==(const TString& s1, const TString& s2);
Bool_t  operator==(const TString& s1, const char *s2);
Bool_t  operator==(const TSubString& s1, const TSubString& s2);
Bool_t  operator==(const TSubString& s1, const TString& s2);
Bool_t  operator==(const TSubString& s1, const char *s2);


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TStringRef                                                          //
//                                                                      //
//  This is the dynamically allocated part of a TString.                //
//  It maintains a reference count. It contains no public member        //
//  functions.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TStringRef : public TRefCnt {

friend class TString;
friend class TStringLong;
friend class TSubString;

private:
   Ssiz_t       fCapacity;      // Max string length (excluding null)
   Ssiz_t       fNchars;        // String length (excluding null)

   void         UnLink(); // disconnect from a TStringRef, maybe delete it

   Ssiz_t       Length() const   { return fNchars; }
   Ssiz_t       Capacity() const { return fCapacity; }
   char        *Data() const     { return (char*)(this+1); }

   char&        operator[](Ssiz_t i)       { return ((char*)(this+1))[i]; }
   char         operator[](Ssiz_t i) const { return ((char*)(this+1))[i]; }

   Ssiz_t       First(char c) const;
   Ssiz_t       First(const char *s) const;
   unsigned     Hash() const;
   unsigned     HashFoldCase() const;
   Ssiz_t       Last(char) const;

   static TStringRef *GetRep(Ssiz_t capac, Ssiz_t nchar);
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TSubString                                                          //
//                                                                      //
//  The TSubString class allows selected elements to be addressed.      //
//  There are no public constructors.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TSubString {

friend class TStringLong;
friend class TString;

friend Bool_t operator==(const TSubString& s1, const TSubString& s2);
friend Bool_t operator==(const TSubString& s1, const TString& s2);
friend Bool_t operator==(const TSubString& s1, const char *s2);

private:
   TString      *fStr;           // Referenced string
   Ssiz_t        fBegin;         // Index of starting character
   Ssiz_t        fExtent;        // Length of TSubString

   // NB: the only constructor is private
   TSubString(const TString& s, Ssiz_t start, Ssiz_t len);

protected:
   void          SubStringError(Ssiz_t, Ssiz_t, Ssiz_t) const;
   void          AssertElement(Ssiz_t i) const;  // Verifies i is valid index

public:
   TSubString(const TSubString& s)
     : fStr(s.fStr), fBegin(s.fBegin), fExtent(s.fExtent) { }

   TSubString&   operator=(const char *s);       // Assignment to char*
   TSubString&   operator=(const TString& s);    // Assignment to TString
   char&         operator()(Ssiz_t i);           // Index with optional bounds checking
   char&         operator[](Ssiz_t i);           // Index with bounds checking
   char          operator()(Ssiz_t i) const;     // Index with optional bounds checking
   char          operator[](Ssiz_t i) const;     // Index with bounds checking

   const char   *Data() const;
   Ssiz_t        Length() const          { return fExtent; }
   Ssiz_t        Start() const           { return fBegin; }
   void          ToLower();              // Convert self to lower-case
   void          ToUpper();              // Convert self to upper-case

   // For detecting null substrings
   Bool_t        IsNull() const          { return fBegin == kNPOS; }
   int           operator!() const       { return fBegin == kNPOS; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TString                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TString {

friend class TSubString;
friend class TStringRef;

friend TString operator+(const TString& s1, const TString& s2);
friend TString operator+(const TString& s,  const char *cs);
friend TString operator+(const char *cs, const TString& s);
friend TString operator+(const TString& s, char c);
friend TString operator+(const TString& s, Long_t i);
friend TString operator+(const TString& s, ULong_t i);
friend TString operator+(char c, const TString& s);
friend TString operator+(Long_t i, const TString& s);
friend TString operator+(ULong_t i, const TString& s);
friend Bool_t  operator==(const TString& s1, const TString& s2);
friend Bool_t  operator==(const TString& s1, const char *s2);

private:
   static Ssiz_t  fgInitialCapac;   // Initial allocation Capacity
   static Ssiz_t  fgResizeInc;      // Resizing increment
   static Ssiz_t  fgFreeboard;      // Max empty space before reclaim

   void           Clone();          // Make self a distinct copy
   void           Clone(Ssiz_t nc); // Make self a distinct copy w. capacity nc

protected:
   char          *fData;          // ref. counted data (TStringRef is in front)

   // Special concatenation constructor
   TString(const char *a1, Ssiz_t n1, const char *a2, Ssiz_t n2);
   TStringRef    *Pref() const { return (((TStringRef*) fData) - 1); }
   void           AssertElement(Ssiz_t nc) const; // Index in range
   void           Clobber(Ssiz_t nc);             // Remove old contents
   void           Cow();                          // Do copy on write as needed
   void           Cow(Ssiz_t nc);                 // Do copy on write as needed
   static Ssiz_t  AdjustCapacity(Ssiz_t nc);
   void           InitChar(char c);               // Initialize from char

public:
   enum EStripType   { kLeading = 0x1, kTrailing = 0x2, kBoth = 0x3 };
   enum ECaseCompare { kExact, kIgnoreCase };

   TString();                       // Null string
   TString(Ssiz_t ic);              // Suggested capacity
   TString(const TString& s)        // Copy constructor
      { fData = s.fData; Pref()->AddReference(); }

   TString(const char *s);              // Copy to embedded null
   TString(const char *s, Ssiz_t n);    // Copy past any embedded nulls
   TString(char c) { InitChar(c); }

   TString(char c, Ssiz_t s);

   TString(const TSubString& sub);

   virtual ~TString();

   // ROOT I/O interface
   virtual void     FillBuffer(char *&buffer);
   virtual void     ReadBuffer(char *&buffer);
   virtual Int_t    Sizeof() const;

   static TString  *ReadString(TBuffer &b, const TClass *clReq);
   static void      WriteString(TBuffer &b, const TString *a);

   friend TBuffer &operator<<(TBuffer &b, const TString *obj);

   // Type conversion
   operator const char*() const { return fData; }

   // Assignment
   TString&    operator=(char s);                // Replace string
   TString&    operator=(const char *s);
   TString&    operator=(const TString& s);
   TString&    operator=(const TSubString& s);
   TString&    operator+=(const char *s);        // Append string
   TString&    operator+=(const TString& s);
   TString&    operator+=(char c);
   TString&    operator+=(Short_t i);
   TString&    operator+=(UShort_t i);
   TString&    operator+=(Int_t i);
   TString&    operator+=(UInt_t i);
   TString&    operator+=(Long_t i);
   TString&    operator+=(ULong_t i);
   TString&    operator+=(Float_t f);
   TString&    operator+=(Double_t f);

   // Indexing operators
   char&         operator[](Ssiz_t i);         // Indexing with bounds checking
   char&         operator()(Ssiz_t i);         // Indexing with optional bounds checking
   TSubString    operator()(Ssiz_t start, Ssiz_t len);   // Sub-string operator
   TSubString    operator()(const TRegexp& re);          // Match the RE
   TSubString    operator()(const TRegexp& re, Ssiz_t start);
   TSubString    SubString(const char *pat, Ssiz_t start = 0,
                           ECaseCompare cmp = kExact);
   char          operator[](Ssiz_t i) const;
   char          operator()(Ssiz_t i) const;
   TSubString    operator()(Ssiz_t start, Ssiz_t len) const;
   TSubString    operator()(const TRegexp& re) const;   // Match the RE
   TSubString    operator()(const TRegexp& re, Ssiz_t start) const;
   TSubString    SubString(const char *pat, Ssiz_t start = 0,
                           ECaseCompare cmp = kExact) const;

   // Non-static member functions
   TString&     Append(const char *cs);
   TString&     Append(const char *cs, Ssiz_t n);
   TString&     Append(const TString& s);
   TString&     Append(const TString& s, Ssiz_t n);
   TString&     Append(char c, Ssiz_t rep = 1);   // Append c rep times
   Bool_t       BeginsWith(const char *s,      ECaseCompare cmp = kExact) const;
   Bool_t       BeginsWith(const TString& pat, ECaseCompare cmp = kExact) const;
   Ssiz_t       Capacity() const         { return Pref()->Capacity(); }
   Ssiz_t       Capacity(Ssiz_t n);
   TString&     Chop();
   int          CompareTo(const char *cs,    ECaseCompare cmp = kExact) const;
   int          CompareTo(const TString& st, ECaseCompare cmp = kExact) const;
   Bool_t       Contains(const char *pat,    ECaseCompare cmp = kExact) const;
   Bool_t       Contains(const TString& pat, ECaseCompare cmp = kExact) const;
   Bool_t       Contains(const TRegexp& pat) const;
   Int_t        CountChar(Int_t c) const;
   TString      Copy() const;
   const char  *Data() const                 { return fData; }
   Bool_t       EndsWith(const char *pat,    ECaseCompare cmp = kExact) const;
   Ssiz_t       First(char c) const          { return Pref()->First(c); }
   Ssiz_t       First(const char *cs) const  { return Pref()->First(cs); }
   unsigned     Hash(ECaseCompare cmp = kExact) const;
   Ssiz_t       Index(const char *pat, Ssiz_t i = 0,
                      ECaseCompare cmp = kExact) const;
   Ssiz_t       Index(const TString& s, Ssiz_t i = 0,
                      ECaseCompare cmp = kExact) const;
   Ssiz_t       Index(const char *pat, Ssiz_t patlen, Ssiz_t i,
                      ECaseCompare cmp) const;
   Ssiz_t       Index(const TString& s, Ssiz_t patlen, Ssiz_t i,
                      ECaseCompare cmp) const;
   Ssiz_t       Index(const TRegexp& pat, Ssiz_t i = 0) const;
   Ssiz_t       Index(const TRegexp& pat, Ssiz_t *ext, Ssiz_t i = 0) const;
   TString&     Insert(Ssiz_t pos, const char *s);
   TString&     Insert(Ssiz_t pos, const char *s, Ssiz_t extent);
   TString&     Insert(Ssiz_t pos, const TString& s);
   TString&     Insert(Ssiz_t pos, const TString& s, Ssiz_t extent);
   Bool_t       IsAscii() const;
   Bool_t       IsNull() const              { return Pref()->fNchars == 0; }
   Ssiz_t       Last(char c) const          { return Pref()->Last(c); }
   Ssiz_t       Length() const              { return Pref()->fNchars; }
   Bool_t       MaybeRegexp();
   TString&     Prepend(const char *cs);     // Prepend a character string
   TString&     Prepend(const char *cs, Ssiz_t n);
   TString&     Prepend(const TString& s);
   TString&     Prepend(const TString& s, Ssiz_t n);
   TString&     Prepend(char c, Ssiz_t rep = 1);  // Prepend c rep times
   istream&     ReadFile(istream& str);      // Read to EOF or null character
   istream&     ReadLine(istream& str,
                         Bool_t skipWhite = kTRUE);   // Read to EOF or newline
   istream&     ReadString(istream& str);             // Read to EOF or null character
   istream&     ReadToDelim(istream& str, char delim = '\n'); // Read to EOF or delimitor
   istream&     ReadToken(istream& str);                // Read separated by white space
   TString&     Remove(Ssiz_t pos);                     // Remove pos to end of string
   TString&     Remove(Ssiz_t pos, Ssiz_t n);           // Remove n chars starting at pos
   TString&     Replace(Ssiz_t pos, Ssiz_t n, const char *s);
   TString&     Replace(Ssiz_t pos, Ssiz_t n, const char *s, Ssiz_t ns);
   TString&     Replace(Ssiz_t pos, Ssiz_t n, const TString& s);
   TString&     Replace(Ssiz_t pos, Ssiz_t n1, const TString& s, Ssiz_t n2);
   TString&     ReplaceAll(const TString& s1, const TString& s2); // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const TString& s1, const char *s2);    // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const    char *s1, const TString& s2); // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const char *s1, const char *s2);       // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const char *s1, Ssiz_t ls1, const char *s2, Ssiz_t ls2);  // Find&Replace all s1 with s2 if any
   void         Resize(Ssiz_t n);                       // Truncate or add blanks as necessary
   TSubString   Strip(EStripType s = kTrailing, char c = ' ');
   TSubString   Strip(EStripType s = kTrailing, char c = ' ') const;
   void         ToLower();                              // Change self to lower-case
   void         ToUpper();                              // Change self to upper-case

   // Static member functions
   static Ssiz_t  InitialCapacity(Ssiz_t ic = 15);      // Initial allocation capacity
   static Ssiz_t  MaxWaste(Ssiz_t mw = 15);             // Max empty space before reclaim
   static Ssiz_t  ResizeIncrement(Ssiz_t ri = 16);      // Resizing increment
   static Ssiz_t  GetInitialCapacity();
   static Ssiz_t  GetResizeIncrement();
   static Ssiz_t  GetMaxWaste();

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TString  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TString  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TString.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 344; }   //Basic string class
};

// Related global functions
istream&  operator>>(istream& str,       TString&   s);
ostream&  operator<<(ostream& str, const TString&   s);
TBuffer&  operator>>(TBuffer& buf,       TString&   s);
TBuffer&  operator<<(TBuffer& buf, const TString&   s);



TBuffer&  operator>>(TBuffer& buf,       TString*& sp);

TString ToLower(const TString&);    // Return lower-case version of argument
TString ToUpper(const TString&);    // Return upper-case version of argument
inline  unsigned Hash(const TString& s) { return s.Hash(); }
inline  unsigned Hash(const TString *s) { return s->Hash(); }
        unsigned Hash(const char *s);

extern char *Form(const char *fmt, ...);     // format in circular buffer
extern void  Printf(const char *fmt, ...);   // format and print
extern char *Strip(const char *str, char c = ' '); // strip c off str, free with delete []
extern char *StrDup(const char *str);        // duplicate str, free with delete []
extern char *Compress(const char *str);      // remove blanks from string, free with delele []
extern int   EscChar(const char *src, char *dst, int dstlen, char *specchars,
                     char escchar);          // copy from src to dst escaping specchars by escchar
extern int   UnEscChar(const char *src, char *dst, int dstlen, char *specchars,
                     char escchar);          // copy from src to dst removing escchar from specchars







//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Inlines                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

inline void TStringRef::UnLink()
{ if (RemoveReference() == 0) delete [] (char*)this; }

inline void TString::Cow()
{ if (Pref()->References() > 1) Clone(); }

inline void TString::Cow(Ssiz_t nc)
{ if (Pref()->References() > 1  || Capacity() < nc) Clone(nc); }

inline TString& TString::Append(const char *cs)
{ return Replace(Length(), 0, cs, strlen(cs)); }

inline TString& TString::Append(const char* cs, Ssiz_t n)
{ return Replace(Length(), 0, cs, n); }

inline TString& TString::Append(const TString& s)
{ return Replace(Length(), 0, s.Data(), s.Length()); }

inline TString& TString::Append(const TString& s, Ssiz_t n)
{ return Replace(Length(), 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::operator+=(const char* cs)
{ return Append(cs, strlen(cs)); }

inline TString& TString::operator+=(const TString& s)
{ return Append(s.Data(), s.Length()); }

inline TString& TString::operator+=(char c)
{ return Append(c); }

inline TString& TString::operator+=(Long_t i)
{ return operator+=(Form("%ld", i)); }

inline TString& TString::operator+=(ULong_t i)
{ return operator+=(Form("%lu", i)); }

inline TString& TString::operator+=(Short_t i)
{ return operator+=((Long_t) i); }

inline TString& TString::operator+=(UShort_t i)
{ return operator+=((ULong_t) i); }

inline TString& TString::operator+=(Int_t i)
{ return operator+=((Long_t) i); }

inline TString& TString::operator+=(UInt_t i)
{ return operator+=((ULong_t) i); }

inline TString& TString::operator+=(Double_t f)
{ return operator+=(Form("%9.9g", f)); }

inline TString& TString::operator+=(Float_t f)
{ return operator+=((Double_t) f); }

inline Bool_t TString::BeginsWith(const char* s, ECaseCompare cmp) const
{ return Index(s, strlen(s), (Ssiz_t)0, cmp) == 0; }

inline Bool_t TString::BeginsWith(const TString& pat, ECaseCompare cmp) const
{ return Index(pat.Data(), pat.Length(), (Ssiz_t)0, cmp) == 0; }

inline Bool_t TString::Contains(const TString& pat, ECaseCompare cmp) const
{ return Index(pat.Data(), pat.Length(), (Ssiz_t)0, cmp) != kNPOS; }

inline Bool_t TString::Contains(const char* s, ECaseCompare cmp) const
{ return Index(s, strlen(s), (Ssiz_t)0, cmp) != kNPOS; }

inline Bool_t TString::Contains(const TRegexp& pat) const
{ return Index(pat, (Ssiz_t)0) != kNPOS; }

inline Ssiz_t TString::Index(const char* s, Ssiz_t i, ECaseCompare cmp) const
{ return Index(s, strlen(s), i, cmp); }

inline Ssiz_t TString::Index(const TString& s, Ssiz_t i, ECaseCompare cmp) const
{ return Index(s.Data(), s.Length(), i, cmp); }

inline Ssiz_t TString::Index(const TString& pat, Ssiz_t patlen, Ssiz_t i,
                             ECaseCompare cmp) const
{ return Index(pat.Data(), patlen, i, cmp); }

inline TString& TString::Insert(Ssiz_t pos, const char* cs)
{ return Replace(pos, 0, cs, strlen(cs)); }

inline TString& TString::Insert(Ssiz_t pos, const char* cs, Ssiz_t n)
{ return Replace(pos, 0, cs, n); }

inline TString& TString::Insert(Ssiz_t pos, const TString& s)
{ return Replace(pos, 0, s.Data(), s.Length()); }

inline TString& TString::Insert(Ssiz_t pos, const TString& s, Ssiz_t n)
{ return Replace(pos, 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::Prepend(const char* cs)
{ return Replace(0, 0, cs, strlen(cs)); }

inline TString& TString::Prepend(const char* cs, Ssiz_t n)
{ return Replace(0, 0, cs, n); }

inline TString& TString::Prepend(const TString& s)
{ return Replace(0, 0, s.Data(), s.Length()); }

inline TString& TString::Prepend(const TString& s, Ssiz_t n)
{ return Replace(0, 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::Remove(Ssiz_t pos)
{ return Replace(pos, TMath::Max(0, Length()-pos), 0, 0); }

inline TString& TString::Remove(Ssiz_t pos, Ssiz_t n)
{ return Replace(pos, n, 0, 0); }

inline TString& TString::Chop()
{ return Remove(TMath::Max(0,Length()-1)); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n, const char* cs)
{ return Replace(pos, n, cs, strlen(cs)); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n, const TString& s)
{ return Replace(pos, n, s.Data(), s.Length()); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n1, const TString& s,
                                 Ssiz_t n2)
{ return Replace(pos, n1, s.Data(), TMath::Min(s.Length(), n2)); }

inline TString&  TString::ReplaceAll(const TString& s1,const TString& s2)
{ return ReplaceAll( s1.Data(), s1.Length(), s2.Data(), s2.Length()) ; }

inline TString&  TString::ReplaceAll(const TString& s1,const char *s2)
{ return ReplaceAll( s1.Data(), s1.Length(), s2, s2 ? strlen(s2):0) ; }

inline TString&  TString::ReplaceAll(const char *s1,const TString& s2)
{ return ReplaceAll( s1, s1 ? strlen(s1): 0, s2.Data(), s2.Length()) ; }

inline TString&  TString::ReplaceAll(const char *s1,const char *s2)
{ return ReplaceAll( s1, s1?strlen(s1):0, s2, s2?strlen(s2):0) ; }

inline char& TString::operator()(Ssiz_t i)
{ Cow(); return fData[i]; }

inline char TString::operator[](Ssiz_t i) const
{ AssertElement(i); return fData[i]; }

inline char TString::operator()(Ssiz_t i) const
{ return fData[i]; }

inline const char* TSubString::Data() const
{ return fStr->Data() + fBegin; }

// Access to elements of sub-string with bounds checking
inline char TSubString::operator[](Ssiz_t i) const
{ AssertElement(i); return fStr->fData[fBegin+i]; }

inline char TSubString::operator()(Ssiz_t i) const
{ return fStr->fData[fBegin+i]; }

// String Logical operators

inline Bool_t     operator==(const TString& s1, const TString& s2)
{
   return ((s1.Length() == s2.Length()) &&
            !memcmp(s1.Data(), s2.Data(), s1.Length()));
}


inline Bool_t     operator!=(const TString& s1, const TString& s2)
{ return !(s1 == s2); }

inline Bool_t     operator< (const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)< 0; }

inline Bool_t     operator> (const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)> 0; }

inline Bool_t     operator<=(const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)<=0; }

inline Bool_t     operator>=(const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)>=0; }

//     Bool_t     operator==(const TString& s1, const char* s2);
inline Bool_t     operator!=(const TString& s1, const char* s2)
{ return !(s1 == s2); }

inline Bool_t     operator< (const TString& s1, const char* s2)
{ return s1.CompareTo(s2)< 0; }

inline Bool_t     operator> (const TString& s1, const char* s2)
{ return s1.CompareTo(s2)> 0; }

inline Bool_t     operator<=(const TString& s1, const char* s2)
{ return s1.CompareTo(s2)<=0; }

inline Bool_t     operator>=(const TString& s1, const char* s2)
{ return s1.CompareTo(s2)>=0; }

inline Bool_t     operator==(const char* s1, const TString& s2)
{ return (s2 == s1); }

inline Bool_t     operator!=(const char* s1, const TString& s2)
{ return !(s2 == s1); }

inline Bool_t     operator< (const char* s1, const TString& s2)
{ return s2.CompareTo(s1)> 0; }

inline Bool_t     operator> (const char* s1, const TString& s2)
{ return s2.CompareTo(s1)< 0; }

inline Bool_t     operator<=(const char* s1, const TString& s2)
{ return s2.CompareTo(s1)>=0; }

inline Bool_t     operator>=(const char* s1, const TString& s2)
{ return s2.CompareTo(s1)<=0; }

// SubString Logical operators
//     Bool_t     operator==(const TSubString& s1, const TSubString& s2);
//     Bool_t     operator==(const TSubString& s1, const char* s2);
//     Bool_t     operator==(const TSubString& s1, const TString& s2);
inline Bool_t     operator==(const TString& s1,    const TSubString& s2)
{ return (s2 == s1); }

inline Bool_t     operator==(const char* s1, const TSubString& s2)
{ return (s2 == s1); }

inline Bool_t     operator!=(const TSubString& s1, const char* s2)
{ return !(s1 == s2); }

inline Bool_t     operator!=(const TSubString& s1, const TString& s2)
{ return !(s1 == s2); }

inline Bool_t     operator!=(const TSubString& s1, const TSubString& s2)
{ return !(s1 == s2); }

inline Bool_t     operator!=(const TString& s1,   const TSubString& s2)
{ return !(s2 == s1); }

inline Bool_t     operator!=(const char* s1,       const TSubString& s2)
{ return !(s2 == s1); }



# 34 "/home/wmtan/root/include/TCollection.h" 2



class TClass;
class TObjectTable;


const Bool_t kIterForward  = kTRUE;
const Bool_t kIterBackward = !kIterForward;


class TCollection : public TObject {

private:
   static TCollection  *fgCurrentCollection;  //used by macro ForEach
   static TObjectTable *fgGarbageCollection;  //used by garbage collector
   static Bool_t        fgEmptyingGarbage;    //used by garbage collector
   static Int_t         fgGarbageStack;       //used by garbage collector

   TCollection(const TCollection &);    // private and not-implemented, collections
   void operator=(const TCollection &); // are too sensitive to be automatically copied

protected:
   enum { kIsOwner = (1 << ( 14 ))  };

   TString   fName;               //name of the collection
   Int_t     fSize;               //number of elements in collection

   TCollection() : fSize(0) { }

public:
   enum { kInitCapacity = 16, kInitHashTableCapacity = 17 };

   virtual            ~TCollection() { }
   virtual void       Add(TObject *obj) = 0;
   void               AddVector(TObject *obj1, ...);
   virtual void       AddAll(TCollection *col);
   Bool_t             AssertClass(TClass *cl) const;
   void               Browse(TBrowser *b);
   Int_t              Capacity() const { return fSize; }
   virtual void       Clear(Option_t *option="") = 0;
   Bool_t             Contains(const char *name) const { return FindObject(name) != 0; }
   Bool_t             Contains(const TObject *obj) const { return FindObject(obj) != 0; }
   virtual void       Delete(Option_t *option="") = 0;
   virtual void       Draw(Option_t *option="");
   virtual void       Dump() const ;
   virtual TObject   *FindObject(const char *name) const;
   TObject           *operator()(const char *name) const;
   virtual TObject   *FindObject(const TObject *obj) const;
   virtual const char *GetName() const;
   virtual TObject  **GetObjectRef(const TObject *obj) const = 0;
   virtual Int_t      GetSize() const { return fSize; }
   virtual Int_t      GrowBy(Int_t delta) const;
   Bool_t             IsArgNull(const char *where, const TObject *obj) const;
   virtual Bool_t     IsEmpty() const { return GetSize() <= 0; }
   virtual Bool_t     IsFolder() const { return kTRUE; }
   Bool_t             IsOwner() const { return TestBit(kIsOwner); }
   virtual void       ls(Option_t *option="") const ;
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const = 0;
   virtual TIterator *MakeReverseIterator() const { return MakeIterator(kIterBackward); }
   virtual void       Paint(Option_t *option="");
   virtual void       Print(Option_t *option="") const;
   virtual void       RecursiveRemove(TObject *obj);
   virtual TObject   *Remove(TObject *obj) = 0;
   virtual void       RemoveAll(TCollection *col);
   void               RemoveAll() { Clear(); }
   void               SetCurrentCollection();
   void               SetName(const char *name) { fName = name; }
   void               SetOwner(Bool_t enable = kTRUE) { enable ? SetBit(kIsOwner) : ResetBit(kIsOwner); }
   virtual Int_t      Write(const char *name=0, Int_t option=0, Int_t bufsize=0);

   static TCollection  *GetCurrentCollection();
   static void          StartGarbageCollection();
   static void          GarbageCollect(TObject *obj);
   static void          EmptyGarbageCollection();

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   3  ; } static void Dictionary(); virtual TClass *IsA() const { return   TCollection  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TCollection  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TCollection.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 110; }   //Collection abstract base class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIter                                                                //
//                                                                      //
// Iterator wrapper. Type of iterator used depends on type of           //
// collection.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TIter {

private:
   TIterator    *fIterator;         //collection iterator

protected:
   TIter() : fIterator(0) { }

public:
   TIter(const TCollection *col, Bool_t dir = kIterForward)
        : fIterator(col ? col->MakeIterator(dir) : 0) { }
   TIter(TIterator *it) : fIterator(it) { }
   TIter(const TIter &iter);
   TIter &operator=(const TIter &rhs);
   virtual            ~TIter() { { if ( fIterator ) { delete  fIterator ;  fIterator  = 0; } }  }
   TObject           *operator()() { return fIterator ? fIterator->Next() : 0; }
   TObject           *Next() { return fIterator ? fIterator->Next() : 0; }
   const TCollection *GetCollection() const { return fIterator ? fIterator->GetCollection() : 0; }
   Option_t          *GetOption() const { return fIterator ? fIterator->GetOption() : ""; }
   void               Reset() { if (fIterator) fIterator->Reset(); }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TIter  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TIter  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TCollection.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 144; }   //Iterator wrapper
};


//---- ForEach macro -----------------------------------------------------------

// Macro to loop over all elements of a list of type "type" while executing
// procedure "proc" on each element









# 26 "/home/wmtan/root/include/TSeqCollection.h" 2




class TSeqCollection : public TCollection {

protected:
   Bool_t            fSorted;    // true if collection has been sorted

   TSeqCollection() { }
   virtual void      Changed() { fSorted = kFALSE; }

public:
   virtual           ~TSeqCollection() { }
   virtual void      Add(TObject *obj) { AddLast(obj); }
   virtual void      AddFirst(TObject *obj) = 0;
   virtual void      AddLast(TObject *obj) = 0;
   virtual void      AddAt(TObject *obj, Int_t idx) = 0;
   virtual void      AddAfter(TObject *after, TObject *obj) = 0;
   virtual void      AddBefore(TObject *before, TObject *obj) = 0;
   virtual void      RemoveFirst() { Remove(First()); }
   virtual void      RemoveLast() { Remove(Last()); }
   virtual TObject  *RemoveAt(Int_t idx) { return Remove(At(idx)); }
   virtual void      RemoveAfter(TObject *after) { Remove(After(after)); }
   virtual void      RemoveBefore(TObject *before) { Remove(Before(before)); }

   virtual TObject  *At(Int_t idx) const = 0;
   virtual TObject  *Before(TObject *obj) const = 0;
   virtual TObject  *After(TObject *obj) const = 0;
   virtual TObject  *First() const = 0;
   virtual TObject  *Last() const = 0;
   Int_t             LastIndex() const { return GetSize() - 1; }
   virtual Int_t     IndexOf(const TObject *obj) const;
   virtual Bool_t    IsSorted() const { return fSorted; }
   void              UnSort() { fSorted = kFALSE; }

   static Int_t      ObjCompare(TObject *a, TObject *b);
   static void       QSort(TObject **a, Int_t first, Int_t last);
   static void       QSort(TObject **a, TObject **b, Int_t first, Int_t last);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TSeqCollection  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TSeqCollection  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TSeqCollection.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 66; }   //Sequenceable collection ABC
};


# 26 "/home/wmtan/root/include/TList.h" 2






const Bool_t kSortAscending  = kTRUE;
const Bool_t kSortDescending = !kSortAscending;

class TObjLink;
class TListIter;


class TList : public TSeqCollection {

friend  class TListIter;

protected:
   TObjLink  *fFirst;     //! pointer to first entry in linked list
   TObjLink  *fLast;      //! pointer to last entry in linked list
   TObjLink  *fCache;     //! cache to speedup sequential calling of Before() and After() functions
   Bool_t     fAscending; //! sorting order (when calling Sort() or for TSortedList)

   TObjLink          *LinkAt(Int_t idx) const;
   TObjLink          *FindLink(const TObject *obj, Int_t &idx) const;
   TObjLink         **DoSort(TObjLink **head, Int_t n);
   Bool_t             LnkCompare(TObjLink *l1, TObjLink *l2);
   virtual TObjLink  *NewLink(TObject *obj, TObjLink *prev = 0);
   virtual TObjLink  *NewOptLink(TObject *obj, Option_t *opt, TObjLink *prev = 0);
   virtual void       DeleteLink(TObjLink *lnk);

public:
   TList() { fFirst = fLast = fCache = 0; }
   TList(TObject *) { fFirst = fLast = fCache = 0; } // for backward compatibility, don't use
   virtual           ~TList();
   virtual void      Clear(Option_t *option="");
   virtual void      Delete(Option_t *option="");
   virtual TObject  *FindObject(const char *name) const;
   virtual TObject  *FindObject(const TObject *obj) const;
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const;

   virtual void      Add(TObject *obj) { AddLast(obj); }
   virtual void      Add(TObject *obj, Option_t *opt) { AddLast(obj, opt); }
   virtual void      AddFirst(TObject *obj);
   virtual void      AddFirst(TObject *obj, Option_t *opt);
   virtual void      AddLast(TObject *obj);
   virtual void      AddLast(TObject *obj, Option_t *opt);
   virtual void      AddAt(TObject *obj, Int_t idx);
   virtual void      AddAfter(TObject *after, TObject *obj);
   virtual void      AddAfter(TObjLink *after, TObject *obj);
   virtual void      AddBefore(TObject *before, TObject *obj);
   virtual void      AddBefore(TObjLink *before, TObject *obj);
   virtual TObject  *Remove(TObject *obj);
   virtual TObject  *Remove(TObjLink *lnk);

   virtual TObject  *At(Int_t idx) const;
   virtual TObject  *After(TObject *obj) const;
   virtual TObject  *Before(TObject *obj) const;
   virtual TObject  *First() const;
   virtual TObjLink *FirstLink() const { return fFirst; }
   virtual TObject **GetObjectRef(const TObject *obj) const;
   virtual TObject  *Last() const;
   virtual TObjLink *LastLink() const { return fLast; }

   virtual void      Sort(Bool_t order = kSortAscending);
   Bool_t            IsAscending() { return fAscending; }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   4  ; } static void Dictionary(); virtual TClass *IsA() const { return   TList  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TList  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TList.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 93; }   //Doubly linked list
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjLink                                                             //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TObjLink {

friend  class TList;

private:
   TObjLink   *fNext;
   TObjLink   *fPrev;
   TObject    *fObject;

protected:
   TObjLink() { fNext = fPrev = this; fObject = 0; }

public:
   TObjLink(TObject *obj) : fNext(0), fPrev(0), fObject(obj) { }
   TObjLink(TObject *obj, TObjLink *lnk);
   virtual ~TObjLink() { }

   TObject                *GetObject() const { return fObject; }
   TObject               **GetObjectRef() { return &fObject; }
   void                    SetObject(TObject *obj) { fObject = obj; }
   virtual Option_t       *GetAddOption() const { return ""; }
   virtual Option_t       *GetOption() const { return fObject->GetOption(); }
   virtual void            SetOption(Option_t *) { }
   TObjLink               *Next() { return fNext; }
   TObjLink               *Prev() { return fPrev; }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjOptLink                                                          //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList including    //
// an option string.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TObjOptLink : public TObjLink {

private:
   TString   fOption;

public:
   TObjOptLink(TObject *obj, Option_t *opt) : TObjLink(obj), fOption(opt) { }
   TObjOptLink(TObject *obj, TObjLink *lnk, Option_t *opt) : TObjLink(obj, lnk), fOption(opt) { }
   ~TObjOptLink() { }
   Option_t        *GetAddOption() const { return fOption.Data(); }
   Option_t        *GetOption() const { return fOption.Data(); }
   void             SetOption(Option_t *option) { fOption = option; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListIter                                                            //
//                                                                      //
// Iterator of linked list.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TListIter : public TIterator {

protected:
   const TList       *fList;         //list being iterated
   TObjLink          *fCurCursor;    //current position in list
   TObjLink          *fCursor;       //next position in list
   Bool_t             fDirection;    //iteration direction
   Bool_t             fStarted;      //iteration started

   TListIter() : fList(0), fCursor(0), fStarted(kFALSE) { }

public:
   TListIter(const TList *l, Bool_t dir = kIterForward);
   TListIter(const TListIter &iter);
   ~TListIter() { }
   TIterator &operator=(const TIterator &rhs);
   TListIter &operator=(const TListIter &rhs);

   const TCollection *GetCollection() const { return fList; }
   Option_t          *GetOption() const;
   void               SetOption(Option_t *option);
   TObject           *Next();
   void               Reset() { fStarted = kFALSE; }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TListIter  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TListIter  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TList.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 186; }   //Linked list iterator
};


# 29 "/home/wmtan/root/include/TNamed.h" 2







class TNamed : public TObject {

protected:
   TString   fName;            //object identifier
   TString   fTitle;           //object title

public:
   TNamed() { }
   TNamed(const char *name, const char *title) : fName(name), fTitle(title) { }
   TNamed(const TString &name, const TString &title) : fName(name), fTitle(title) { }
   TNamed(const TNamed &named);
   TNamed& operator=(const TNamed& rhs);
   virtual ~TNamed() { }
   virtual void     Clear(Option_t *option ="");
   virtual TObject *Clone(const char *newname="") const;
   virtual Int_t    Compare(const TObject *obj) const;
   virtual void     Copy(TObject &named) const;
   virtual void     FillBuffer(char *&buffer);
   virtual const char  *GetName() const {return fName.Data();}
   virtual const char  *GetTitle() const {return fTitle.Data();}
   virtual ULong_t  Hash() const { return fName.Hash(); }
   virtual Bool_t   IsSortable() const { return kTRUE; }
   virtual void     SetName(const char *name); // *MENU*
   virtual void     SetNameTitle(const char *name, const char *title);
   virtual void     SetTitle(const char *title=""); // *MENU*
   virtual void     ls(Option_t *option="") const;
   virtual void     Print(Option_t *option="") const;
   virtual Int_t    Sizeof() const;

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TNamed  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TNamed  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TNamed.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 65; }   //The basis for a named object (name, title)
};


# 25 "/home/wmtan/root/include/TDirectory.h" 2






# 1 "/home/wmtan/root/include/TDatime.h" 1
// @(#)root/base:$Id$
// Author: Rene Brun   05/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDatime                                                              //
//                                                                      //
// Data and time 950130 124559.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/Htypes.h" 1
/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Htypes                                                               //
//                                                                      //
// Types used by the histogramming classes.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






typedef double         Axis_t;      //Axis values type (double)
typedef double         Stat_t;      //Statistics type (double)



# 25 "/home/wmtan/root/include/TDatime.h" 2




class TDatime {

protected:
   UInt_t     fDatime;            //Date (relative to 1995) + time

public:
   TDatime();
   TDatime(const TDatime &d) { fDatime = d.fDatime; }
   TDatime(UInt_t time) { fDatime = time; }
   TDatime(Int_t date, Int_t time);
   TDatime(Int_t year, Int_t month, Int_t day,
           Int_t hour, Int_t min, Int_t sec);
   TDatime(const char *sqlDateTime);

   TDatime operator=(const TDatime &d);

   const char  *AsString() const;
   const char  *AsSQLString() const;
   UInt_t       Convert(Bool_t toGMT = kFALSE) const;
   void         Copy(TDatime &datime) const;
   UInt_t       Get() const { return fDatime; }
   Int_t        GetDate() const;
   Int_t        GetTime() const;
   Int_t        GetYear() const { return (fDatime>>26) + 1995; }
   Int_t        GetMonth() const { return (fDatime<<6)>>28; }
   Int_t        GetDay() const { return (fDatime<<10)>>27; }
   Int_t        GetHour() const { return (fDatime<<15)>>27; }
   Int_t        GetMinute() const { return (fDatime<<20)>>26; }
   Int_t        GetSecond() const { return (fDatime<<26)>>26; }
   void         FillBuffer(char *&buffer);
   void         Print(Option_t *option="") const;
   void         ReadBuffer(char *&buffer);
   void         Set();
   void         Set(UInt_t tloc);
   void         Set(Int_t date, Int_t time);
   void         Set(Int_t year, Int_t month, Int_t day,
                    Int_t hour, Int_t min, Int_t sec);
   Int_t        Sizeof() const {return sizeof(UInt_t);}

   friend Bool_t operator==(const TDatime &d1, const TDatime &d2);
   friend Bool_t operator!=(const TDatime &d1, const TDatime &d2);
   friend Bool_t operator< (const TDatime &d1, const TDatime &d2);
   friend Bool_t operator<=(const TDatime &d1, const TDatime &d2);
   friend Bool_t operator> (const TDatime &d1, const TDatime &d2);
   friend Bool_t operator>=(const TDatime &d1, const TDatime &d2);

   static void GetDateTime(UInt_t datetime, Int_t &date, Int_t &time);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TDatime  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TDatime  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TDatime.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 77; }   //Date and time 950130 124559
};


inline TDatime TDatime::operator=(const TDatime &d)
   { fDatime = d.fDatime; return *this; }

inline Bool_t operator==(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime == d2.fDatime; }
inline Bool_t operator!=(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime != d2.fDatime; }
inline Bool_t operator< (const TDatime &d1, const TDatime &d2)
   { return d1.fDatime < d2.fDatime; }
inline Bool_t operator<=(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime <= d2.fDatime; }
inline Bool_t operator> (const TDatime &d1, const TDatime &d2)
   { return d1.fDatime > d2.fDatime; }
inline Bool_t operator>=(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime >= d2.fDatime; }


# 31 "/home/wmtan/root/include/TDirectory.h" 2



# 1 "/home/wmtan/root/include/TUUID.h" 1
// @(#)root/base:$Id$
// Author: Fons Rademakers   30/9/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUUID                                                                //
//                                                                      //
// This class defines a UUID (Universally Unique IDentifier), also      //
// known as GUIDs (Globally Unique IDentifier). A UUID is 128 bits      //
// long, and if generated according to this algorithm, is either        //
// guaranteed to be different from all other UUIDs/GUIDs generated      //
// until 3400 A.D. or extremely likely to be different. UUIDs were      //
// originally used in the Network Computing System (NCS) and            //
// later in the Open Software Foundation's (OSF) Distributed Computing  //
// Environment (DCE).                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





# 1 "/home/wmtan/root/include/TInetAddress.h" 1
// @(#)root/net:$Id$
// Author: Fons Rademakers   16/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInetAddress                                                         //
//                                                                      //
// This class represents an Internet Protocol (IP) address.             //
// Objects of this class can not be created directly, but only via      //
// the TSystem GetHostByName(), GetSockName(), and GetPeerName()        //
// members and via members of the TServerSocket and TSocket classes.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////









class TInetAddress : public TObject {

friend class TSystem;
friend class TUnixSystem;
friend class TWinNTSystem;
friend class TVmsSystem;
friend class TMacSystem;
friend class TUUID;
friend class TSocket;
friend class TServerSocket;

private:
   TString fHostname;    // fully qualified hostname
   UInt_t  fAddress;     // IP address in host byte order
   Int_t   fFamily;      // address family
   Int_t   fPort;        // port through which we are connected

   TInetAddress(const char *host, UInt_t addr, Int_t family, Int_t port = -1);

public:
   TInetAddress();
   TInetAddress(const TInetAddress &adr);
   TInetAddress &operator=(const TInetAddress &rhs);
   virtual ~TInetAddress() { }

   UInt_t      GetAddress() const { return fAddress; }
   UChar_t    *GetAddressBytes() const;
   const char *GetHostAddress() const;
   const char *GetHostName() const { return (const char *) fHostname; }
   Int_t       GetFamily() const { return fFamily; }
   Int_t       GetPort() const { return fPort; }
   Bool_t      IsValid() const { return fFamily == -1 ? kFALSE : kTRUE; }
   void        Print(Option_t *option="") const;

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TInetAddress  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TInetAddress  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TInetAddress.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 69; }   //Represents an Internet Protocol (IP) address
};


# 34 "/home/wmtan/root/include/TUUID.h" 2






// forward declaration
class TBuffer;
class TFile;
class TDirectory;


class TUUID {

friend class TFile;
friend class TDirectory;

protected:
   UInt_t    fUUIDIndex;             //!index in the list of UUIDs in TProcessUUID
   UInt_t    fTimeLow;               // 60 bit time, lower 32 bits
   UShort_t  fTimeMid;               // middle 16 time bits
   UShort_t  fTimeHiAndVersion;      // high 12 time bits + 4 UUID version bits
   UChar_t   fClockSeqHiAndReserved; // high 6 clock bits + 2 bits reserved
   UChar_t   fClockSeqLow;           // low 8 clock bits
   UChar_t   fNode[6];               // 6 node id bytes

   struct uuid_time_t {
      UInt_t high;
      UInt_t low;
   };

   Int_t CmpTime(uuid_time_t *t1, uuid_time_t *t2);
   void  Format(UShort_t clockseq, uuid_time_t ts);
   void  GetNodeIdentifier();
   void  GetCurrentTime(uuid_time_t *timestamp);
   void  GetSystemTime(uuid_time_t *timestamp);
   void  GetRandomInfo(UChar_t seed[16]);
   void  SetFromString(const char *uuid_str);

   void  StreamerV1(TBuffer &b);
   void         FillBuffer(char *&buffer);
   void         ReadBuffer(char *&buffer);
   Int_t        Sizeof() const { return 18; }

public:
   TUUID();
   TUUID(const char *uuid_str);
   virtual ~TUUID();

   const char  *AsString() const;
   Int_t        Compare(const TUUID &u) const;
   UShort_t     Hash() const;
   void         Print() const;
   TInetAddress GetHostAddress() const;
   TDatime      GetTime() const;
   void         GetUUID(UChar_t uuid[16]) const;
   void         SetUUID(const char *uuid_str);
   UInt_t       GetUUIDNumber() const { return fUUIDIndex; }
   void         SetUUIDNumber(UInt_t index) { fUUIDIndex = index; }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   TUUID  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TUUID  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TUUID.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 94; }   // Universally Unique IDentifier
};


inline TBuffer &operator>>(TBuffer &buf, TUUID &uuid)
{ uuid.Streamer(buf); return buf; }

inline TBuffer &operator<<(TBuffer &buf, const TUUID &uuid)
{ ((TUUID&)uuid).Streamer(buf); return buf; }

inline Bool_t operator==(const TUUID &u1, const TUUID &u2)
{ return (!u1.Compare(u2)) ? kTRUE : kFALSE; }

inline Bool_t operator!=(const TUUID &u1, const TUUID &u2)
{ return !(u1 == u2); }



# 34 "/home/wmtan/root/include/TDirectory.h" 2



class TBrowser;
class TKey;
class TFile;

class TDirectory : public TNamed {

protected:
   Bool_t      fModified;        //true if directory has been modified
   Bool_t      fWritable;        //true if directory is writable
   TDatime     fDatimeC;         //Date and time when directory is created
   TDatime     fDatimeM;         //Date and time of last modification
   Int_t       fNbytesKeys;      //Number of bytes for the keys
   Int_t       fNbytesName;      //Number of bytes in TNamed at creation time
   Seek_t      fSeekDir;         //Location of directory on file
   Seek_t      fSeekParent;      //Location of parent directory on file
   Seek_t      fSeekKeys;        //Location of Keys record on file
   TFile      *fFile;            //pointer to current file in memory
   TObject    *fMother;          //pointer to mother of the directory
   TList      *fList;            //Pointer to objects list in memory
   TList      *fKeys;            //Pointer to keys list in memory
   TUUID       fUUID;            //Unique identifier

          Bool_t cd1(const char *path);
   static Bool_t Cd1(const char *path);

private:
   TDirectory(const TDirectory &directory);  //Directories cannot be copied
   void operator=(const TDirectory &);

public:
   // TDirectory status bits
   enum { kCloseDirectory = (1 << ( 7 ))  };

   TDirectory();
   TDirectory(const char *name, const char *title, Option_t *option="");
   virtual ~TDirectory();
   virtual void        Append(TObject *obj);
           void        Add(TObject *obj) { Append(obj); }
           Int_t       AppendKey(TKey *key);
   virtual void        Browse(TBrowser *b);
           void        Build();
   virtual void        Clear(Option_t *option="");
   virtual void        Close(Option_t *option="");
   virtual void        Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual Bool_t      cd(const char *path = 0);
   virtual void        DeleteAll(Option_t *option="");
   virtual void        Delete(const char *namecycle="");
   virtual void        Draw(Option_t *option="");
   virtual void        FillBuffer(char *&buffer);
   virtual TKey       *FindKey(const char *keyname) const;
   virtual TKey       *FindKeyAny(const char *keyname) const;
   virtual TObject    *FindObject(const char *name) const;
   virtual TObject    *FindObject(const TObject *obj) const;
   virtual TObject    *FindObjectAny(const char *name) const;
   virtual TObject    *Get(const char *namecycle);
   TDatime            &GetCreationDate() {return fDatimeC;}
   virtual TFile      *GetFile() const {return fFile;}
   virtual TKey       *GetKey(const char *name, Short_t cycle=9999) const;
   TList              *GetList() const { return fList; }
   TList              *GetListOfKeys() const { return fKeys; }
   TDatime            &GetModificationDate() {return fDatimeM;}
   TObject            *GetMother() const { return fMother; }
   virtual Int_t       GetNkeys() const {return fKeys->GetSize();}
   virtual Seek_t      GetSeekDir() const { return fSeekDir; }
   virtual Seek_t      GetSeekParent() const { return fSeekParent; }
   virtual Seek_t      GetSeekKeys() const { return fSeekKeys; }
   virtual const char *GetPath() const;
   TUUID               GetUUID() const {return fUUID;}
   Bool_t              IsFolder() const { return kTRUE; }
   Bool_t              IsModified() const { return fModified; }
   Bool_t              IsWritable() const { return fWritable; }
   virtual void        ls(Option_t *option="") const;
   virtual TDirectory *mkdir(const char *name, const char *title="");
   virtual void        Paint(Option_t *option="");
   virtual void        Print(Option_t *option="") const;
   virtual void        Purge(Short_t nkeep=1);
   virtual void        pwd() const;
   virtual void        ReadAll(Option_t *option="");
   virtual Int_t       ReadKeys();
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        Save();
   virtual void        SaveSelf(Bool_t force = kFALSE);
   void                SetWritable(Bool_t writable=kTRUE);
   void                SetModified() {fModified = kTRUE;}
   void                SetMother(const TObject *mother) {fMother = (TObject*)mother;}
   virtual Int_t       Sizeof() const;
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0);
   virtual void        WriteDirHeader();
   virtual void        WriteKeys();

   static Bool_t       Cd(const char *path);
   static void         DecodeNameCycle(const char *namecycle, char *name, Short_t &cycle);
   static void         EncodeNameCycle(char *buffer, const char *name, Short_t cycle);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   3  ; } static void Dictionary(); virtual TClass *IsA() const { return   TDirectory  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TDirectory  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TDirectory.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 131; }   //Describe directory structure in memory
};

R__EXTERN TDirectory   *gDirectory;



# 25 "/home/wmtan/root/include/TFile.h" 2



# 1 "/home/wmtan/root/include/TCache.h" 1
// @(#)root/net:$Id$
// Author: Fons Rademakers   13/01/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCache                                                               //
//                                                                      //
// A caching system to speed up network I/O, i.e. when there is         //
// no operating system caching support (like the buffer cache for       //
// local disk I/O). The cache makes sure that every I/O is done with    //
// a (large) fixed length buffer thereby avoiding many small I/O's.     //
// The default page size is 512KB. The cache size is not very important //
// when writing sequentially a file, since the pages will not be        //
// reused. In that case use a small cache containing 10 to 20 pages.    //
// In case a file is used for random-access the cache size should be    //
// taken much larger to avoid re-reading pages over the network.        //
// Notice that the TTree's have their own caching mechanism (see        //
// TTree::SetMaxVirtualSize()), so when using mainly TTree's with large //
// basket buffers the cache can be kept quite small.                    //
// Currently the TCache system is used by the classes TNetFile,         //
// TRFIOFile and TWebFile.                                              //
//                                                                      //
// Extra improvement would be to run the Free() process in a separate   //
// thread. Possible flush parameters:                                   //
// nfract  25   fraction of dirty buffers above which the flush process //
//              is activated                                            //
// ndirty  500  maximum number of buffer block which may be written     //
//              during a flush                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/home/wmtan/root/include/THashList.h" 1
// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THashList                                                            //
//                                                                      //
// THashList implements a hybrid collection class consisting of a       //
// hash table and a list to store TObject's. The hash table is used for //
// quick access and lookup of objects while the list allows the objects //
// to be ordered. The hash value is calculated using the value returned //
// by the TObject's Hash() function. Each class inheriting from TObject //
// can override Hash() as it sees fit.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





class THashTable;


class THashList : public TList {

protected:
   THashTable   *fTable;    //Hashtable used for quick lookup of objects

public:
   THashList(Int_t capacity=TCollection::kInitHashTableCapacity, Int_t rehash=0);
   THashList(TObject *parent, Int_t capacity=TCollection::kInitHashTableCapacity, Int_t rehash=0);
   virtual    ~THashList();
   Float_t    AverageCollisions() const;
   void       Clear(Option_t *option="");
   void       Delete(Option_t *option="");

   TObject   *FindObject(const char *name) const;
   TObject   *FindObject(const TObject *obj) const;

   void       AddFirst(TObject *obj);
   void       AddFirst(TObject *obj, Option_t *opt);
   void       AddLast(TObject *obj);
   void       AddLast(TObject *obj, Option_t *opt);
   void       AddAt(TObject *obj, Int_t idx);
   void       AddAfter(TObject *after, TObject *obj);
   void       AddAfter(TObjLink *after, TObject *obj);
   void       AddBefore(TObject *before, TObject *obj);
   void       AddBefore(TObjLink *before, TObject *obj);
   void       RecursiveRemove(TObject *obj);
   void       Rehash(Int_t newCapacity);
   TObject   *Remove(TObject *obj);
   TObject   *Remove(TObjLink *lnk);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   THashList  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   THashList  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/THashList.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 66; }   //Doubly linked list with hashtable for lookup
};


# 45 "/home/wmtan/root/include/TCache.h" 2



class TSortedList;
class TFile;


class TCache : public TObject {

friend class TFile;

private:
   // The TPage class describes a cache page
   class TPage : public TObject {
   friend class TCache;
   private:
      Seek_t    fOffset; // offset of page in file
      char     *fData;   // pointer to page data
      Int_t     fSize;   // size of page
   public:
      enum { kDirty = (1 << ( 14 )) , kLocked = (1 << ( 15 ))  };
      TPage(Seek_t offset, char *page, Int_t size)
         { fOffset = offset; fData = page; fSize = size; }
      ~TPage() { delete [] fData; }
      ULong_t Hash() const { return fOffset; }
      Bool_t  IsEqual(const TObject *obj) const
         { return fOffset == ((const TPage*)obj)->fOffset; }
      Bool_t  IsSortable() const { return kTRUE; }
      Int_t   Compare(const TObject *obj) const
         { return fOffset > ((const TPage*)obj)->fOffset ? 1 :
                  fOffset < ((const TPage*)obj)->fOffset ? -1 : 0; }
      Seek_t  Offset() const { return fOffset; }
      char   *Data() const { return fData; }
      Int_t   Size() const { return fSize; }
   };

   class TCacheList : public THashList {
   public:
      TCacheList(Int_t capacity = 1000) : THashList(capacity, 3) { }
      void PageUsed(TObject *page) { TList::Remove(page); TList::AddLast(page); }
   };

   TCacheList  *fCache;         // hash list containing cached pages
   TSortedList *fNew;           // list constaining new pages that have to be written to disk
   TList       *fFree;          // list containing unused pages
   TFile       *fFile;          // file for which pages are being cached
   Seek_t       fEOF;           // end of file
   ULong_t      fHighWater;     // high water mark (i.e. maximum cache size in bytes)
   ULong_t      fLowWater;      // low water mark (free pages till low water mark is reached)
   Int_t        fPageSize;      // size of cached pages
   Int_t        fLowLevel;      // low water mark is at low level percent of high
   Int_t        fDiv;           // page size divider
   Bool_t       fRecursive;     // true to prevent recusively calling ReadBuffer()

   void   SetPageSize(Int_t size);
   TPage *ReadPage(Seek_t offset);
   Int_t  WritePage(TPage *page);
   Int_t  FlushList(TList *list);
   Int_t  FlushNew();
   Int_t  Free(ULong_t upto);

public:
   enum {
      kDfltPageSize = 0x80000,    // 512KB
      kDfltLowLevel = 70          // 70% of fHighWater
   };

   TCache(Int_t maxCacheSize, TFile *file, Int_t pageSize = kDfltPageSize);
   virtual ~TCache();

   Int_t GetMaxCacheSize() const { return Int_t(fHighWater / 1024 / 1024); }
   Int_t GetActiveCacheSize() const;
   Int_t GetPageSize() const { return fPageSize; }
   Int_t GetLowLevel() const { return fLowLevel; }
   Int_t Resize(Int_t maxCacheSize);
   void  SetLowLevel(Int_t percentOfHigh);

   Int_t ReadBuffer(Seek_t offset, char *buf, Int_t len);
   Int_t WriteBuffer(Seek_t offset, const char *buf, Int_t len);
   Int_t Flush();

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   0  ; } static void Dictionary(); virtual TClass *IsA() const { return   TCache  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TCache  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TCache.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 126; }   // Page cache used for remote I/O
};


# 28 "/home/wmtan/root/include/TFile.h" 2



class TFree;
class TArrayC;


class TFile : public TDirectory {

protected:
   Double_t    fSumBuffer;        //Sum of buffer sizes of objects written so far
   Double_t    fSum2Buffer;       //Sum of squares of buffer sizes of objects written so far
   Double_t    fBytesWrite;       //Number of bytes written to this file
   Double_t    fBytesRead;        //Number of bytes read from this file
   Seek_t      fBEGIN;            //First used byte in file
   Seek_t      fEND;              //Last used byte in file
   Seek_t      fSeekFree;         //Location on disk of free segments structure
   Seek_t      fSeekInfo;         //Location on disk of StreamerInfo record
   Int_t       fD;                //File descriptor
   Int_t       fVersion;          //File format version
   Int_t       fCompress;         //Compression level from 0(not compressed) to 9 (max compression)
   Int_t       fNbytesFree;       //Number of bytes for free segments structure
   Int_t       fNbytesInfo;       //Number of bytes for StreamerInfo record
   Int_t       fWritten;          //Number of objects written so far
   Int_t       fNProcessIDs;      //Number of TProcessID written to this file
   TString     fOption;           //File options
   Char_t      fUnits;            //Number of bytes for file pointers
   TList      *fFree;             //Free segments linked list table
   TArrayC    *fClassIndex;       //!Index of TStreamerInfo classes written to this file
   TCache     *fCache;            //!Page cache used to reduce number of small I/O's
   TObjArray  *fProcessIDs;       //!Array of pointers to TProcessIDs

   static Double_t fgBytesWrite;    //Number of bytes written by all TFile objects
   static Double_t fgBytesRead;     //Number of bytes read by all TFile objects

   enum { kBegin = 64, kUnits = 4 };

   void Init(Bool_t create);

   // Interface to basic system I/O routines
   virtual Int_t  SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   virtual Int_t  SysClose(Int_t fd);
   virtual Int_t  SysRead(Int_t fd, void *buf, Int_t len);
   virtual Int_t  SysWrite(Int_t fd, const void *buf, Int_t len);
   virtual Seek_t SysSeek(Int_t fd, Seek_t offset, Int_t whence);
   virtual Int_t  SysStat(Int_t fd, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);
   virtual Int_t  SysSync(Int_t fd);

private:
   TFile(const TFile &);            //Files cannot be copied
   void operator=(const TFile &);

public:
   // TFile status bits
   enum {
      kRecovered       = (1 << ( 10 )) ,
      kHasReferences   = (1 << ( 11 )) ,
      kDevNull         = (1 << ( 12 )) ,
      kWriteError      = (1 << ( 14 ))  // BIT(13) is taken up by TObject
   };
   enum ERelativeTo { kBeg = 0, kCur = 1, kEnd = 2 };

   TFile();
   TFile(const char *fname, Option_t *option="", const char *ftitle="", Int_t compress=1);
   virtual ~TFile();
   virtual void      Close(Option_t *option=""); // *MENU*
   virtual void      Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual void      Delete(const char *namecycle="");
   virtual void      Draw(Option_t *option="");
   virtual void      FillBuffer(char *&buffer);
   virtual void      Flush();
   Int_t             GetBestBuffer() const;
   TArrayC          *GetClassIndex() const { return fClassIndex; }
   Int_t             GetCompressionLevel() const { return fCompress; }
   Float_t           GetCompressionFactor();
   virtual Seek_t    GetEND() const { return fEND; }
   virtual Int_t     GetErrno() const;
   virtual void      ResetErrno() const;
   Int_t             GetFd() const { return fD; }
   TObjArray        *GetListOfProcessIDs() const {return fProcessIDs;}
   TList            *GetListOfFree() const { return fFree; }
   virtual Int_t     GetNfree() const { return fFree->GetSize(); }
   virtual Int_t     GetNProcessIDs() const { return fNProcessIDs; }
   Option_t         *GetOption() const { return fOption.Data(); }
   Double_t          GetBytesRead() const { return fBytesRead; }
   Double_t          GetBytesWritten() const { return fBytesWrite; }
   Int_t             GetVersion() const { return fVersion; }
   Int_t             GetRecordHeader(char *buf, Seek_t first, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen);
   virtual Seek_t    GetSize() const;
   TList            *GetStreamerInfoList();
   virtual void      IncrementProcessIDs() {fNProcessIDs++;}
   virtual Bool_t    IsOpen() const;
   virtual void      ls(Option_t *option="") const;
   virtual void      MakeFree(Seek_t first, Seek_t last);
   virtual void      MakeProject(const char *dirname, const char *classes="*", Option_t *option="new"); // *MENU*
   virtual void      Map(); // *MENU*
   virtual void      Paint(Option_t *option="");
   virtual void      Print(Option_t *option="") const;
   virtual Bool_t    ReadBuffer(char *buf, Int_t len);
   virtual void      ReadFree();
   virtual void      ReadStreamerInfo();
   virtual Int_t     Recover();
   virtual Int_t     ReOpen(Option_t *mode);
   virtual void      Seek(Seek_t offset, ERelativeTo pos = kBeg);
   virtual void      SetCompressionLevel(Int_t level=1);
   virtual void      SetEND(Seek_t last) { fEND = last; }
   virtual void      SetOption(Option_t *option=">") { fOption = option; }
   virtual void      ShowStreamerInfo();
   virtual Int_t     Sizeof() const;
   void              SumBuffer(Int_t bufsize);
   virtual void      UseCache(Int_t maxCacheSize = 10, Int_t pageSize = TCache::kDfltPageSize);
   virtual Bool_t    WriteBuffer(const char *buf, Int_t len);
   virtual Int_t     Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0);
   virtual void      WriteFree();
   virtual void      WriteHeader();
   virtual void      WriteStreamerInfo();

   static TFile     *Open(const char *name, Option_t *option = "",
                          const char *ftitle = "", Int_t compress = 1,
                          Int_t netopt = 0);

   static Double_t   GetFileBytesRead();
   static Double_t   GetFileBytesWritten();

   static void       SetFileBytesRead(Double_t bytes=0);
   static void       SetFileBytesWritten(Double_t bytes=0);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   3  ; } static void Dictionary(); virtual TClass *IsA() const { return   TFile  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TFile  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TFile.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 155; }   //ROOT file
};

R__EXTERN TFile   *gFile;


# 16 "Utilities/Persistency/interface/ooRefBase.h" 2

# 1 "/home/wmtan/root/include/TKey.h" 1
// @(#)root/base:$Id$
// Author: Rene Brun   28/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TKey                                                                 //
//                                                                      //
// Header description of a logical record on file.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////











class TClass;
class TBrowser;

class TKey : public TNamed {

protected:
    Int_t       fVersion;        //Key version identifier
    Int_t       fNbytes;         //Number of bytes for the object on file
    Int_t       fObjlen;         //Length of uncompressed object in bytes
    TDatime     fDatime;         //Date/Time of insertion in file
    Short_t     fKeylen;         //Number of bytes for the key itself
    Short_t     fCycle;          //Cycle number
    Seek_t      fSeekKey;        //Location of object on file
    Seek_t      fSeekPdir;       //Location of parent directory on file
    TString     fClassName;      //Object Class name
    Int_t       fLeft;           //Number of bytes left in current segment
    char        *fBuffer;        //Object buffer
    TBuffer     *fBufferRef;     //Pointer to the TBuffer object

    virtual void     Create(Int_t nbytes);
    virtual Int_t    Read(const char *name) { return TObject::Read(name); }

public:
    TKey();
    TKey(const char *name, const char *title, TClass *cl, Int_t nbytes);
    TKey(const TString &name, const TString &title, TClass *cl, Int_t nbytes);
    TKey(TObject *obj, const char *name, Int_t bufsize);
    TKey(Seek_t pointer, Int_t nbytes);
    virtual ~TKey();
    virtual void      Browse(TBrowser *b);
    virtual void      Delete(Option_t *option="");
    virtual void      DeleteBuffer();
    virtual void      FillBuffer(char *&buffer);
    virtual const char *GetClassName() const {return fClassName.Data();}
    virtual char     *GetBuffer() const {return fBuffer+fKeylen;}
         TBuffer     *GetBufferRef() const {return fBufferRef;}
         Short_t      GetCycle() const ;
         Short_t      GetKeep() const;
           Int_t      GetKeylen() const  {return fKeylen;}
           Int_t      GetNbytes() const  {return fNbytes;}
           Int_t      GetObjlen() const  {return fObjlen;}
           Int_t      GetVersion() const {return fVersion;}
    virtual Seek_t    GetSeekKey() const  {return fSeekKey;}
    virtual Seek_t    GetSeekPdir() const {return fSeekPdir;}
    virtual ULong_t   Hash() const;
    Bool_t            IsFolder() const;
    virtual void      Keep();
    virtual void      ls(Option_t *option="") const;
    virtual void      Print(Option_t *option="") const;
    virtual Int_t     Read(TObject *obj);
    virtual TObject  *ReadObj();
    virtual void      ReadBuffer(char *&buffer);
    virtual void      ReadFile();
    virtual void      SetBuffer() { fBuffer = new char[fNbytes];}
    virtual void      SetParent(TObject *parent);
    virtual Int_t     Sizeof() const;
    virtual Int_t     WriteFile(Int_t cycle=1);

    private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   2  ; } static void Dictionary(); virtual TClass *IsA() const { return   TKey  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   TKey  ::Streamer(b); } static const char *DeclFileName() { return "/home/wmtan/root/include/TKey.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 92; }   //Header description of a logical record on file
};


# 17 "Utilities/Persistency/interface/ooRefBase.h" 2


class ooObj;
class ooRunObj;

class opiRefBase {
friend class ooObj;
friend ostream& operator<< (ostream &, const opiRefBase &);
public:
opiRefBase();

opiRefBase(const string& str);

opiRefBase(const TObject *o, const TDirectory *d, const TFile *f);

opiRefBase(const TObject *o);

virtual ~opiRefBase();

operator TObject* () const;
TObject * operator->() const;

TObject & operator*() const;

ooRunObj & operator()() const;

bool operator==(const opiRefBase& pref ) const;
bool operator!=(const opiRefBase& pref ) const;
bool operator==(int zero) const;
bool operator!=(int zero) const;

opiRefBase & operator=(const TObject *o);

bool operator<(const opiRefBase& pref ) const;

const char *sprint() const;
void print() const;
const char *fileName() const;
const char *typeName() const;
const string typeString() const;
const string typeN() const;
const char *name() const;

bool isValid() const;

bool isValid(ooMode openMode) const;

bool isDirectory() const;

bool isDirectoryNonFile() const;

bool isFile() const;

TDirectory *saveDir(const char * objname = 0) const;

TDirectory* oContainedIn() const;

TFile *fileContainedIn() const;

TDirectory *saveDir(const string & objname) const { return saveDir(objname.c_str());}

ooStatus nameObj(const opiRefBase& scope, const char *name) const;

ooStatus lookupObj(const opiRefBase& scope, const char *name, ooMode openMode=oocRead);

const char *getObjName(const opiRefBase& scope) const;

string get_DB() const;

//Opens an object given a reference.
ooStatus open(ooMode openMode=oocRead) const;

TObject * openIt(ooMode openMode=oocRead) const;

ooStatus close () const;

ooStatus update() const;

set<TObject *>  GetSet() const;

bool  isDirEmpty() const;

static map<int, TDirectory *> rootdir;

static map<int, bool> existing;

static map<int, const char *> newname;

protected:

TFile * openAFile(const char *dbname, ooMode openMode, const char *hostName=0, const char *pathName=0) const;

TDirectory * openADir(TDirectory *fptr, const char *dirname) const;

TDirectory * mkADir(TDirectory *fptr, const char *dirname) const;

TFile * openFile(ooMode openMode) const;

TDirectory * openDir(TDirectory * fptr) const;

TObject * openObj(TDirectory * dptr, ooMode& openMode) const;

void SetObjectContainingNameScopes() const;

static const string FileClass;

static const string DirectoryClass;

protected:
TRef objRef;
TRef dirRef;
TRef fileRef;
string oName;
string dName;
string fName;

mutable ooRunObj *pObjContainingNameScopes; //! Transient
private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   opiRefBase  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   opiRefBase  ::Streamer(b); } static const char *DeclFileName() { return "Utilities/Persistency/interface/ooRefBase.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 134; } 
};

class DataBase {
public:
        DataBase() : fptr(0) {}
        DataBase(TFile *fp) : fptr(fp) {}
        virtual ~DataBase() {}

        void Commit() const {Commit(fptr);}
        TFile *getFptr() const {return fptr;}
        static void Commit(TFile *fp);
        static void writeDataBases();
        static map<string, DataBase *> databases;
protected:
        TFile *fptr;
};

ostream & operator<<( ostream& o, const opiRefBase& pref);

# 8 "Utilities/Persistency/interface/ooObj.h" 2




class opiRefBase;

typedef string NumOID;

class ooObj : public TNamed {
friend class opiRefBase;
public:

ooObj(); 

ooObj(const char *name, const char *title); 

virtual ~ooObj();

virtual void objDelete() {}

TDirectory *saveDir(const char *objname = 0);

TDirectory *GetDirectory() const;

TFile *GetFile() const;

NumOID GetNumOID() const {return NumOID(GetName());}

void ooUpdate();

static void clearObjects();

static void writeObjects(uint32 eventInJob);

void SetTheName();

void SetTheName(char *info);

void SetTheName(int number);

private:
void SetNumOID();

ooObj(const ooObj & obj); 

protected:
static multimap<uint32, ooObj *> readObjects;
static vector<opiRefBase *> objects;
static vector<TFile *> filesToClose;

bool doUpdate;          //! Transient
TDirectory *tDirectory; //! Transient
private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   ooObj  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   ooObj  ::Streamer(b); } static const char *DeclFileName() { return "Utilities/Persistency/interface/ooObj.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 60; } 
};


# 3 "Utilities/Persistency/interface/ooRunObj.h" 2

# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 4 "Utilities/Persistency/interface/ooRunObj.h" 2

# 1 "/home/wmtan/root/cint/stl/map" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_map" 1

#pragma include_noerr <map.dll>
#pragma include_noerr <map2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/map" 1
// lib/prec_stl/map

#pragma ifndef PREC_STL_MAP
#pragma define PREC_STL_MAP
#pragma link off global PREC_STL_MAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/map" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class map {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const map::iterator& x ,const map::iterator& y) const;
  friend bool operator!=(const map::iterator& x ,const map::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;
  friend bool operator!=(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.map.cons_ construct/copy/destroy:
  map();






  map(iterator first, iterator last);
  map(reverse_iterator first, reverse_iterator last);

  map(const map& x);
  ~map();
  map& operator=(const map& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.map.access_ element access:
  T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(map&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.map.ops_ map operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const map& x, const map& y);
  friend bool operator< (const map& x, const map& y);
  friend bool operator!=(const map& x, const map& y);
  friend bool operator> (const map& x, const map& y);
  friend bool operator>=(const map& x, const map& y);
  friend bool operator<=(const map& x, const map& y);
  // specialized algorithms:






  // Generic algorithm
  friend map::iterator
    search(map::iterator first1,map::iterator last1,
           map::iterator first2,map::iterator last2);


  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(map::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_map" 2






# 1 "/home/wmtan/root/cint/stl/_multimap" 1

#pragma include_noerr <multimap.dll>
#pragma include_noerr <multimap2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multimap" 1
// lib/prec_stl/multimap

#pragma ifndef PREC_STL_MULTIMAP
#pragma define PREC_STL_MULTIMAP
#pragma link off global PREC_STL_MULTIMAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multimap {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multimap::iterator& x ,const multimap::iterator& y) const;
  friend bool operator!=(const multimap::iterator& x ,const multimap::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;
  friend bool operator!=(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multimap.cons_ construct/copy/destroy:
  multimap();






  multimap(iterator first, iterator last);
  multimap(reverse_iterator first, reverse_iterator last);

  multimap(const multimap& x);
  ~multimap();
  multimap& operator=(const multimap& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.multimap.access_ element access:
  //T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(multimap&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.multimap.ops_ multimap operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const multimap& x, const multimap& y);
  friend bool operator< (const multimap& x, const multimap& y);
  friend bool operator!=(const multimap& x, const multimap& y);
  friend bool operator> (const multimap& x, const multimap& y);
  friend bool operator>=(const multimap& x, const multimap& y);
  friend bool operator<=(const multimap& x, const multimap& y);
  // specialized algorithms:






  // Generic algorithm
  friend multimap::iterator
    search(multimap::iterator first1,multimap::iterator last1,
           multimap::iterator first2,multimap::iterator last2);



  // Generic algorithm
  //friend void reverse(multimap::iterator first,multimap::iterator last);
  //friend void reverse(multimap::reverse_iterator first,multimap::reverse_itetator last);

  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(multimap::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif



# 7 "/home/wmtan/root/cint/stl/_multimap" 2




# 13 "/home/wmtan/root/cint/stl/_map" 2

# 2 "/home/wmtan/root/cint/stl/map" 2

}
# 5 "Utilities/Persistency/interface/ooRunObj.h" 2


class opiRefBase;

class ooRunObj : public ooObj {
friend class opiRefBase;
friend ostream & operator<< (ostream &, const ooRunObj &);
public:

ooRunObj(); 

ooRunObj(const char *name, const char *title); 

ooRunObj(const ooRunObj& obj); 

virtual ~ooRunObj();

virtual void objDelete() {}

map<string, opiRefBase> & GetNamesInScope();

multimap<string, opiRefBase> & GetScopesContainingName();

protected:
map<string, opiRefBase> namesInScope;

multimap<string, opiRefBase> scopesContainingName;

private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   ooRunObj  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   ooRunObj  ::Streamer(b); } static const char *DeclFileName() { return "Utilities/Persistency/interface/ooRunObj.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 33; } 
};
ostream & operator<<( ostream& o, const ooRunObj& pref);
ostream & operator<<( ostream& o, const pair<string,opiRefBase>& pref);


# 10 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooEvObj.h" 1




class opiRefBase;

class ooEvObj : public ooObj {
friend class opiRefBase;
public:

ooEvObj(); 

ooEvObj(const ooEvObj& obj); 

virtual ~ooEvObj();

virtual void objDelete();

static uint32 eventCount;

protected:
private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   ooEvObj  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   ooEvObj  ::Streamer(b); } static const char *DeclFileName() { return "Utilities/Persistency/interface/ooEvObj.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 22; } 
};


# 11 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooCollObj.h" 1



# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 4 "Utilities/Persistency/interface/ooCollObj.h" 2

# 1 "/home/wmtan/root/cint/stl/map" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_map" 1

#pragma include_noerr <map.dll>
#pragma include_noerr <map2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/map" 1
// lib/prec_stl/map

#pragma ifndef PREC_STL_MAP
#pragma define PREC_STL_MAP
#pragma link off global PREC_STL_MAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/map" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/map" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class map {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const map::iterator& x ,const map::iterator& y) const;
  friend bool operator!=(const map::iterator& x ,const map::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;
  friend bool operator!=(const map::reverse_iterator& x
                        ,const map::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.map.cons_ construct/copy/destroy:
  map();






  map(iterator first, iterator last);
  map(reverse_iterator first, reverse_iterator last);

  map(const map& x);
  ~map();
  map& operator=(const map& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.map.access_ element access:
  T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(map&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.map.ops_ map operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const map& x, const map& y);
  friend bool operator< (const map& x, const map& y);
  friend bool operator!=(const map& x, const map& y);
  friend bool operator> (const map& x, const map& y);
  friend bool operator>=(const map& x, const map& y);
  friend bool operator<=(const map& x, const map& y);
  // specialized algorithms:






  // Generic algorithm
  friend map::iterator
    search(map::iterator first1,map::iterator last1,
           map::iterator first2,map::iterator last2);


  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(map::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_map" 2






# 1 "/home/wmtan/root/cint/stl/_multimap" 1

#pragma include_noerr <multimap.dll>
#pragma include_noerr <multimap2.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/multimap" 1
// lib/prec_stl/multimap

#pragma ifndef PREC_STL_MULTIMAP
#pragma define PREC_STL_MULTIMAP
#pragma link off global PREC_STL_MULTIMAP;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2

# 1 "/home/wmtan/root/cint/stl/_functional" 1

#pragma include_noerr <functional.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/functional" 1
// lib/prec_stl/functional

#pragma ifndef PREC_STL_FUNCTIONAL
#pragma define PREC_STL_FUNCTIONAL
#pragma link off global PREC_STL_FUNCTIONAL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab

// clause _lib.base_, base:
template <class Arg, class Result>
struct unary_function
{
  typedef Arg    argument_type;
  typedef Result result_type;
};


template <class Arg1, class Arg2, class Result>
struct binary_function
{
  typedef Arg1   first_argument_type;
  typedef Arg2   second_argument_type;
  typedef Result result_type;
};


// clause _lib.arithmetic.operations_, arithmetic operations:
template <class T> struct plus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct minus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct multiplies : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};


template <class T> struct divides : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct modulus : binary_function<T,T,T> {
  T operator()(const T& x, const T& y) const;
};

template <class T> struct negate : unary_function<T,T> {
  T operator()(const T& x) const;
};


// clause _lib.comparisons_, comparisons:
template <class T> struct equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct not_equal_to : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct greater_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct less_equal : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};


// clause _lib.logical.operations_, logical operations:
template <class T> struct logical_and : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_or : binary_function<T,T,bool> {
  bool operator()(const T& x, const T& y) const;
};

template <class T> struct logical_not : unary_function<T,bool> {
  bool operator()(const T& x) const;
};


// clause _lib.negators_, negators:
template <class Predicate>
class unary_negate
  : public unary_function<Predicate::argument_type, bool>
{
public:
  explicit unary_negate(const Predicate& pred);
  bool operator()(const argument_type& x) const;
};






template <class Predicate>
class binary_negate
  : public binary_function<Predicate::first_argument_type,
                           Predicate::second_argument_type, bool>
{
public:
  explicit binary_negate(const Predicate& pred);
  bool operator()(const first_argument_type&  x,
                  const second_argument_type& y) const;
};

// operations omitted (cint can't handle template forward decls...)







// clause _lib.binders_, binders:
template <class Operation> 
class binder1st
  : public unary_function<Operation::second_argument_type,
                          Operation::result_type>
{
protected:
  Operation                      op;
  Operation::first_argument_type value;
public:
  binder1st(const Operation& x, const Operation::first_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)





template <class Operation>
class binder2nd
  : public unary_function<Operation::first_argument_type,
                          Operation::result_type>
{
protected:
  Operation                       op;
  Operation::second_argument_type value;
public:
  binder2nd(const Operation& x, const Operation::second_argument_type& y);
  result_type operator()(const argument_type& x) const;
};


// operations omitted (cint can't handle template forward decls...)






// clause _lib.function.pointer.adaptors_, adaptors:
template <class Arg, class Result>
class pointer_to_unary_function
  : public unary_function<Arg, Result>
{
public:
  explicit pointer_to_unary_function(Result (*f)(Arg));
  Result operator()(Arg x) const;
};

// operations omitted (cint can't handle template forward decls...)





template <class Arg1, class Arg2, class Result>
class pointer_to_binary_function
  : public binary_function<Arg1,Arg2,Result>
{
public:
  explicit pointer_to_binary_function(Result (*f)(Arg1, Arg2));
  Result operator()(Arg1 x, Arg2 y) const;
};

// operations omitted (cint can't handle template forward decls...)






// omit these for now.
# 221 "/home/wmtan/root/cint/lib/prec_stl/functional"


#pragma endif
# 6 "/home/wmtan/root/cint/stl/_functional" 2




# 22 "/home/wmtan/root/cint/lib/prec_stl/multimap" 2


//////////////////////////////////////////////////////////////////////////




template<class Key,class T,class Compare=std::less<Key>
        ,class Allocator=alloc>







class multimap {
 public:
  typedef Key                                       key_type;
  typedef T                                         mapped_type;
  typedef pair<Key,T>                               value_type;
  //typedef pair<const Key,T>                         value_type;
  typedef Compare                                   key_compare;
  typedef Allocator                                 allocator_type;

  //typedef Key*                                     pointer;
  //typedef const Key*                               const_pointer;
  //typedef Key&                                     reference;
  //typedef const Key&                               const_reference;
  typedef size_t                                   size_type;
  typedef ptrdiff_t                                difference_type;









  class iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    iterator() ;
    iterator(const iterator& x) ;

    iterator& operator=(const iterator& x) ;

    value_type& operator*() ;
    iterator& operator++();
    iterator operator++(int a);
    iterator& operator--();
    iterator operator--(int a);







  };

  friend bool operator==(const multimap::iterator& x ,const multimap::iterator& y) const;
  friend bool operator!=(const multimap::iterator& x ,const multimap::iterator& y) const;





  class reverse_iterator 





        : public bidirectional_iterator<T,difference_type> 

        {
   public:
    reverse_iterator(const reverse_iterator& x);

    reverse_iterator& operator=(const reverse_iterator& x) ;

    value_type& operator*() ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
   private:
  };
  friend bool operator==(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;
  friend bool operator!=(const multimap::reverse_iterator& x
                        ,const multimap::reverse_iterator& y) const;

  typedef const iterator const_iterator;
  typedef const reverse_iterator const_reverse_iterator;

  // _lib.multimap.cons_ construct/copy/destroy:
  multimap();






  multimap(iterator first, iterator last);
  multimap(reverse_iterator first, reverse_iterator last);

  multimap(const multimap& x);
  ~multimap();
  multimap& operator=(const multimap& x);
  // iterators:
  iterator               begin();
  iterator               end();
  reverse_iterator       rbegin();
  reverse_iterator       rend();






  // capacity:
  bool      empty() const;
  size_type size() const;
  size_type max_size() const;
  // _lib.multimap.access_ element access:
  //T& operator[](const Key& x);
  // modifiers:
  //pair<iterator, bool> insert(const value_type& x);
  iterator             insert(iterator position, const value_type& x);





  void insert(iterator first, iterator last);
  void insert(reverse_iterator first, reverse_iterator last);

  void      erase(iterator position);
  size_type erase(const Key& x);
  void      erase(iterator first, iterator last);
  void swap(multimap&);
  void clear();
  // observers:
  //key_compare   key_comp() const;
  //value_compare value_comp() const;
  // _lib.multimap.ops_ multimap operations:
  iterator       find(const Key& x);
  //const_iterator find(const Key& x) const;
  size_type      count(const Key& x) const;
  iterator       lower_bound(const Key& x);
  //const_iterator lower_bound(const Key& x) const;
  iterator       upper_bound(const Key& x);
  //const_iterator upper_bound(const Key& x) const;





  friend bool operator==(const multimap& x, const multimap& y);
  friend bool operator< (const multimap& x, const multimap& y);
  friend bool operator!=(const multimap& x, const multimap& y);
  friend bool operator> (const multimap& x, const multimap& y);
  friend bool operator>=(const multimap& x, const multimap& y);
  friend bool operator<=(const multimap& x, const multimap& y);
  // specialized algorithms:






  // Generic algorithm
  friend multimap::iterator
    search(multimap::iterator first1,multimap::iterator last1,
           multimap::iterator first2,multimap::iterator last2);



  // Generic algorithm
  //friend void reverse(multimap::iterator first,multimap::iterator last);
  //friend void reverse(multimap::reverse_iterator first,multimap::reverse_itetator last);

  // iterator_category resolution
  //friend bidirectional_iterator_tag iterator_category(multimap::iterator x);

};

//////////////////////////////////////////////////////////////////////////

#pragma endif



# 7 "/home/wmtan/root/cint/stl/_multimap" 2




# 13 "/home/wmtan/root/cint/stl/_map" 2

# 2 "/home/wmtan/root/cint/stl/map" 2

}
# 5 "Utilities/Persistency/interface/ooCollObj.h" 2


class opiRefBase;

class ooCollObj : public ooObj {
friend class opiRefBase;
public:

ooCollObj(); 

ooCollObj(const ooCollObj & obj); 

virtual ~ooCollObj();

virtual void objDelete() {}

map<string, opiRefBase> & GetNamesInScope();

multimap<string, opiRefBase> & GetScopesContainingName();

protected:

ooRunObj *pNameScope; //! Transient

private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   ooCollObj  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   ooCollObj  ::Streamer(b); } static const char *DeclFileName() { return "Utilities/Persistency/interface/ooCollObj.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 29; } 
};


# 12 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooRef.h" 1



# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 4 "Utilities/Persistency/interface/ooRef.h" 2

# 1 "/home/wmtan/root/cint/include/iostream" 1

namespace std {

}
# 1 "/home/wmtan/root/cint/include/_iostream" 1
// include/_iostream

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDL& i) 
        {return(std::endl(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDS& i) 
        {return(std::ends(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_FLUSH& i) 
        {return(std::flush(ostr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_ws& i) 
        {return(std::ws(istr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_WS& i) 
        {return(std::WS(istr));}


std::ostream& operator<<(std::ostream& ostr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::dec);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::hex);
#pragma else
  ostr.unsetf(ios_base::dec);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::hex);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::dec);
  istr.unsetf(ios::oct);
  istr.setf(ios::hex);
#pragma else
  istr.unsetf(ios_base::dec);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::hex);
#pragma endif
  return(istr);
}

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::dec);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::dec);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::oct);
  istr.setf(ios::dec);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::dec);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::dec);
  ostr.setf(ios::oct);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::dec);
  ostr.setf(ios_base::oct);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::dec);
  istr.setf(ios::oct);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::dec);
  istr.setf(ios_base::oct);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(ostr);
}
std::istream& operator<<(std::istream& istr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(istr);
}

// Value evaluation
//template<class T> int G__ateval(const T* x) {return(0);}
template<class T> int G__ateval(const T& x) {return(0);}
int G__ateval(const char* x) {return(0);}
int G__ateval(const void* x) {return(0);}
int G__ateval(const double x) {return(0);}
int G__ateval(const float x) {return(0);}
int G__ateval(const char x) {return(0);}
int G__ateval(const short x) {return(0);}
int G__ateval(const int x) {return(0);}
int G__ateval(const long x) {return(0);}
int G__ateval(const unsigned char x) {return(0);}
int G__ateval(const unsigned short x) {return(0);}
int G__ateval(const unsigned int x) {return(0);}
int G__ateval(const unsigned long x) {return(0);}








# 5 "/home/wmtan/root/cint/include/iostream" 2

# 5 "Utilities/Persistency/interface/ooRef.h" 2

# 1 "/home/wmtan/root/cint/include/fstream" 1
namespace std {

}
# 6 "Utilities/Persistency/interface/ooRef.h" 2












template <class T>
class opiRef : public opiRefBase {
public:
opiRef() : opiRefBase() {}
opiRef(const T *p) : opiRefBase(p, (p ? p->GetDirectory() : static_cast<const TDirectory *>(0)), (p ? p->GetFile() : static_cast<const TFile *>(0)))  {}
virtual ~opiRef() {}

opiRef(const string & name) : opiRefBase(name) {}









operator T* () const {return dynamic_cast<T *>(openIt());}

T * operator->() const {return dynamic_cast<T *>(openIt());}

T & operator*() const {return *dynamic_cast<T *>(openIt());}

bool operator<(const opiRef<T>& pref) const {return this->GetNumOID() < pref->GetNumOID();}

NumOID GetNumOID() const {return this->GetNumOID();}

// Functions for TDirectory and/or TFile only.  Cause compilation error if otherwise instantiated.
void mkDir(TDirectory *f, const char *dirname) {this_mkDir_function_is_for_TDirectory_only(f, dName);}
void fopen(const char *dbname, ooMode openMode, const char *hostName=0, const char *pathName=0) {This_open_function_is_for_TFile_only(dbname, openMode);}
bool exist(TFile *f, const char *contName) {This_exist_function_is_for_TDirectory_only(f, contName); return(true);}
bool exist(const char *dbName, ooMode openMode = oocNoOpen) {This_exist_function_is_for_TFile_only(dbName, openMode); return(true);}
ooStatus dopen(TFile *f, const char *contName, ooMode openMode=oocRead) {This_open_function_is_for_TDirectory_Only(f, contName); return oocError;}
opiRef<TFile> containedIn() const {This_containdIn_function_is_for_TDirectory_only(); return gFile;}
opiRef<TFile>& containedIn(opiRef<TFile>& xRef) const {This_containdIn_function_is_for_TDirectory_only(xRef); return xRef;}
const char *hostName() const {This_hostName_function_is_for_TFile_only(); return 0;}
const char *pathName() const {This_pathName_function_is_for_TFile_only(); return 0;}
opiRef<TDirectory> getDefaultContObj() const {This_getDefaultContObj_function_is_for_TFile_only(); return gDirectory;}
void set_DB(const string dbname) {This_set_DB_function_is_for_TFile_only(dbname);}



//

//--- for TDirectory only.
ooStatus refreshOpen(ooMode openMode, bool *isUpdated, bool = false)  const {*isUpdated = false; return open(openMode);}

//--- for federation only 
ooStatus dumpCatalog(void *outputFile = 0) const {return oocSuccess;}   // stubbed

// --  undocumented
int size() const {return (*this)->size();}

protected:
};

# 1 "Utilities/Persistency/interface/ooRef.icc" 1
// TDirectory/TFile constructor specializations

template <> opiRef<TDirectory>::opiRef(const TDirectory *p) : opiRefBase(p, p, p->GetFile()) {}

template <> opiRef<TFile>::opiRef(const TFile *p) : opiRefBase(p, p, p) {}



// TDirectory specializations

//Opens a directory (if it exists), given its name.
template <> bool opiRef<TDirectory>::exist(TFile *f, const char *contName) {
        TDirectory *p = openADir(f, contName);
        if (p) {
                (*this) = p;
                return true;
        }
        return false;
}

//Opens a directory (if it exists), given its name.
template <> ooStatus opiRef<TDirectory>::dopen(TFile *f,  const char* contName, ooMode openMode) {
        if (openMode == oocNoOpen) return(oocSuccess); // Just to suppress unused argument warning message.
        TDirectory *p = openADir(f, contName);
        if (p) {
                (*this) = p;
                return (oocSuccess);
        }
        return oocError;
}

//Creates or opens a directory given its name and the parent directory pointer.
template <> void opiRef<TDirectory>::mkDir(TDirectory *f, const char *dirname) {
        const char *dname = (dirname && dirname[0] ? dirname : "NULL");
        TDirectory *p = mkADir(f, dname);
        (*this) = p;
}

template <> opiRef<TFile> opiRef<TDirectory>::containedIn() const {
        return (*this)->GetFile();
}

template <> opiRef<TFile>& opiRef<TDirectory>::containedIn(opiRef<TFile>& xRef) const {
        xRef = (*this)->GetFile();
        return xRef;
}

// TFile specializations

template <> void opiRef<TFile>::fopen(const char *dbname, ooMode openMode, const char *hostName, const char *pathName) {
        TFile *fptr = openAFile(dbname, openMode, hostName, pathName);
        (*this) = fptr;
}

template <> bool opiRef<TFile>::exist(const char *dbName, ooMode openMode) {
        ifstream inFile(dbName);
        if (!inFile) return false;
        inFile.close();
        if (openMode != oocNoOpen) fopen(dbName, openMode);
        return true;
}

# 1 "Utilities/GenUtil/interface/Hostname.h" 1
//
// Simple static class to return the hostname painlessly
//
//  V 1.0  TW 4/10/01
//




# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 10 "Utilities/GenUtil/interface/Hostname.h" 2

/**Simple static class to return the hostname painlessly
 */
class Hostname {
public:
  static const std::string & name();
private:
  Hostname();
  static const int bufsize;
  std::string hostname_;
};

// _HOSTNAME_H_
# 63 "Utilities/Persistency/interface/ooRef.icc" 2

//Opens or creates a file given the file name.
template <> const char * opiRef<TFile>::hostName() const {return Hostname::name().c_str();}     // FIX

# 1 "/home/wmtan/root/cint/include/unistd.h" 1
/****************************************************************
* unistd.h
*****************************************************************/



#pragma include_noerr <systypes.h>



/* NOTE: posix.dl is not generated by default. 
 * Goto $CINTSYSDIR/lib/posix directory and do 'sh setup' if you use UNIX. */

#pragma include_noerr "posix.dll"








/* G__TESTMAIN */


# 67 "Utilities/Persistency/interface/ooRef.icc" 2

template <> const char * opiRef<TFile>::pathName() const {
  static string loc;
  char * buf=0; 
  char * ndir = getcwd(buf,0);
  if (ndir!=0) {
    loc = ndir;
    ::free(ndir);
  } else loc = "";
  return loc.c_str();;
}       // FIX

template <> opiRef<TDirectory> opiRef<TFile>::getDefaultContObj() const {
        return openADir(dynamic_cast<TFile *>(openIt()), "default");
}

template <> void opiRef<TFile>::set_DB(const string dbname) {fopen(dbname.c_str(), oocUpdate, 0, 0);}

# 74 "Utilities/Persistency/interface/ooRef.h" 2


# 13 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooFDObj.h" 1





class ooFDObj : public ooRunObj {
public:
ooFDObj() {}
virtual ~ooFDObj() {}
};


# 14 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooError.h" 1


class ooError {
public:
        unsigned int errorN;
        char *message;
};

typedef ooStatus (*ooErrorHandlerPtr) (
        ooErrorLevel errorLevel,
        ooError &errorID,
        opiRef<TObject> *contextobj,
        char *errormsg);


# 15 "Utilities/Persistency/interface/Persistency.h" 2

# 1 "Utilities/Persistency/interface/ooTrans.h" 1


class ooTrans {
public:
  ooTrans() {}
  virtual ~ooTrans() {}
  bool isActive() {return(false);}
};

# 16 "Utilities/Persistency/interface/Persistency.h" 2


typedef unsigned int ooTypeNumber;


// #define ooRefThis(_class) (*(new opiRef<_class>(this)))







inline ooStatus ooDelete(opiRefBase) {return oocSuccess;}
inline void ooRunStatus() {}



# 11 "CARF/PythiaSimEvent/interface/PythiaSimEvent.h" 2


# 1 "CARF/SimEvent/interface/SimEventWithGen.h" 1
/* C++ header file: Objectivity/DB DDL version 6.1.3         */




//
//
//   V 0.1  VI 8/10/2000 
//          a SimEvent with generator infos



# 1 "CARF/SimEvent/interface/SimEvent.h" 1
/* C++ header file: Objectivity/DB DDL version 6.1.3         */




//
//  the persistent part of SimEvent 

//  version     0.1   99/06/03  (VI)
//                    by value.....
//
//  version     0.2   99/12/15  (VI)
//             simpler now
//  version     0.3   00/01/16  (VI)
//             tread safe
//  version     0.4   00/02/14  (VI)
//             tracks moved in Body
//  version     0.5   00/07/19  (VI)
//             deep copy added
//  version     0.6   01/11/12  (VI)
//             sim body moved in its own ddl



// objy does not understand this
# 1 "Utilities/Configuration/interface/Objy_ddlx.h" 1



# 26 "CARF/SimEvent/interface/SimEvent.h" 2



# 1 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 1


//
//  an Abstract Class for a generic Simulated event
//  version     0.1   98/10/03
//                    pure transient for G3carf'98
//              0.2   99/02/09 (St.Wynhoff)
//                    add MC tracks
//  version     0.3   99/06/02  VI
//                    new interface and implementation (now really abstract...)
//  version     0.4   99/07/27  VI
//          explicit event id added
//
//  version     0.5   99/12/15  VI
//                       Name changed
//  version     0.6   01/11/18  VI
//                       core not here
//  version     0.7   02/08/14  VI
//                    use refcounts

# 1 "CARF/BaseSimEvent/interface/CoreSimEvent.h" 1


//
//  an Abstract Class for a generic Simulated event
//  version     0.1   98/10/03
//                    pure transient for G3carf'98
//              0.2   99/02/09 (St.Wynhoff)
//                    add MC tracks
//  version     0.3   99/06/02  VI
//                    new interface and implementation (now really abstract...)
//  version     0.4   99/07/27  VI
//          explicit event id added
//
//  version     0.5   99/12/15  VI
//                       Name changed
//  version     0.6   01/11/18  VI
//                       only core part here




# 1 "CARF/BaseMD/interface/EventId.h" 1


//
//  standard event id
//
//  version     1.0   99/12/15  VI
//                  


# 1 "/home/wmtan/root/cint/stl/string" 1
namespace std {

}
# 10 "CARF/BaseMD/interface/EventId.h" 2


class EventId  {
  
public:
  /// construct header for dummy event
  EventId() : runNumber_(0), eventInRun_(0){}
  /// constructor from run number and event number
  EventId(int r, int er) :
    runNumber_(r), eventInRun_(er) {}
  /// late constructor
  inline void init(int r, int er) {
    runNumber_=r; eventInRun_=er;
  }
  /// run
  inline int runNumber() const { return runNumber_;}
  /// event in run
  inline int eventInRun() const { return eventInRun_;}


public:
  
  void from(const string & ev);

private:
  int runNumber_;
  int eventInRun_;
};


# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 40 "CARF/BaseMD/interface/EventId.h" 2

ostream & operator<<(ostream & o, const EventId& id);

inline bool operator<(const  EventId& lh, const  EventId& rh) {
  return (lh.runNumber()<rh.runNumber()) ||
    ( (!(rh.runNumber()<lh.runNumber()))&&(lh.eventInRun()<rh.eventInRun()) );
}

inline bool operator==(const  EventId& lh, const  EventId& rh) {
  return (lh.runNumber()==rh.runNumber()) &&
    (lh.eventInRun()==rh.eventInRun());
}

// EVENTId_H
# 22 "CARF/BaseSimEvent/interface/CoreSimEvent.h" 2



// forward declaration;
class TSimEvent;

/** an base Class for a generic Simulated event.
    Encapsulate the identification of the original
    event....
*/
class CoreSimEvent{

public:
  /// Event Identifier
  typedef EventId Id; 
  
public:
  /// default constructor
  CoreSimEvent() : weight_(0) {}
  
  // constructor
  explicit CoreSimEvent(const Id & iid, const TSimEvent& tev);
  
  /// return the Id
  const Id & id() const { return orcaId_;}
  
  
  /// return event weight
  float weight() const { return weight_;}
  
  /// return the generator level id
  const Id & originalId() const { return originalId_;}
  
private:
  
  Id orcaId_;
  
  float  weight_;
  
  Id originalId_;
  
};


# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 66 "CARF/BaseSimEvent/interface/CoreSimEvent.h" 2

ostream& operator <<(ostream& o , const CoreSimEvent& se); 


// CoreSIMEVENT_H
# 21 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 2


# 1 "Utilities/DBI/interface/stdVectorConstInterface.h" 1


//
// a "consistent" inteface to T & P  vectors....
//
//  Version 0.1 VI 21/5/99
//

# 1 "/home/wmtan/root/cint/stl/vector" 1
namespace std {
# 1 "/home/wmtan/root/cint/stl/_vector" 1

#pragma include_noerr <vector.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/vector" 1
// lib/prec_stl/vector

#pragma ifndef PREC_STL_VECTOR
#pragma define PREC_STL_VECTOR
#pragma link off global PREC_STL_VECTOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from ANSI/ISO C++ 1997/Nov draft 
// Got some ideas from Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/home/wmtan/root/cint/stl/_iterator" 1


#pragma include_noerr <iterator.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/iterator" 1
// lib/prec_stl/iterator

#pragma ifndef PREC_STL_ITERATOR
#pragma define PREC_STL_ITERATOR
#pragma link off global PREC_STL_ITERATOR;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;



#pragma mask_newdelete 0x10;


// Imported from STL HP implementation 1994
// Imported from STL SGI implementation 1997 
// Imported from ANSI/ISO C++ draft Nov 1997
// Modified by Masaharu Goto
// May need to improve for the latest standard


////////////////////////////////////////////////////////////////////////
// iterator_tag
////////////////////////////////////////////////////////////////////////
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

////////////////////////////////////////////////////////////////////////
// iterator template
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};




////////////////////////////////////////////////////////////////////////
// iterator_category overloaded function
////////////////////////////////////////////////////////////////////////
template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag 
iterator_category(const T*) {
    return random_access_iterator_tag();
}


// iterator_traits, iterator and reverse_iterator template may not be
// needed for precompiled library interface 

////////////////////////////////////////////////////////////////////////
// iterator_traits
////////////////////////////////////////////////////////////////////////

template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};

// template partial specialization, implement in cint5.15.14 1587
template <class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};

// incomplete implementation in cint5.15.14 1587, need some fix
// iterator_traits<const int*> is changed as iterator_traits<const int* const>
// or something, but cint5.15.14 can not handle this well
template <class T>
struct iterator_traits<const T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef const T*                   pointer;
  typedef const T&                   reference;
};

////////////////////////////////////////////////////////////////////////
// iterator
////////////////////////////////////////////////////////////////////////
template<class Category, class T, class Distance = ptrdiff_t,
         class Pointer = T*, class Reference = T&>
struct iterator {
  typedef T         value_type;
  typedef Distance  difference_type;
  typedef Pointer   pointer;
  typedef Reference reference;
  typedef Category  iterator_category;
};


////////////////////////////////////////////////////////////////////////
// reverse_iterator
////////////////////////////////////////////////////////////////////////
template <class Iterator>
class reverse_iterator 







{




 public:






   typedef Iterator::pointer   pointer;
   typedef Iterator::reference reference;
   typedef ptrdiff_t difference_type;


   reverse_iterator();
   //reverse_iterator(Iterator x);



   Iterator base() const;      // explicit
   reference operator*() const;
   pointer   operator->() const;
   reverse_iterator& operator++();
   reverse_iterator  operator++(int);
   reverse_iterator& operator--();
   reverse_iterator  operator--(int);
   reverse_iterator  operator+ (difference_type n) const;
   reverse_iterator& operator+=(difference_type n);
   reverse_iterator  operator- (difference_type n) const;
   reverse_iterator& operator-=(difference_type n);
   reference operator[](difference_type n) const;
}; 

# 207 "/home/wmtan/root/cint/lib/prec_stl/iterator"


# 269 "/home/wmtan/root/cint/lib/prec_stl/iterator"


// G__GNUC>=3
# 575 "/home/wmtan/root/cint/lib/prec_stl/iterator"


#pragma endif
# 7 "/home/wmtan/root/cint/stl/_iterator" 2




# 19 "/home/wmtan/root/cint/lib/prec_stl/vector" 2

# 1 "/home/wmtan/root/cint/stl/_memory" 1


#pragma include_noerr <memory.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/memory" 1
// lib/prec_stl/memory

#pragma ifndef PREC_STL_MEMORY
#pragma define PREC_STL_MEMORY
#pragma link off global PREC_STL_MEMORY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

// Implemented by Scott Snyder, Fermi-lab
// Modified by Masaharu Goto
// SGI KCC porting by Philippe Canal, Fermi-lab

# 1 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h" 1
/*
 * This stddef.h file is used to replace the gnu provided one when
 * ooddlx is run. 
 * It fixes a bug in the GNU version which defines NULL as ((void *)0)
 * even for __cplusplus parsers
 *
 * NOTE: this file is compiler dependent and has been taken from 
 * egcs-2.91.60 19981201 (egcs-1.1.1 release) 
 *  
 * Dirk.Duellmann@cern.ch
 */
/* !_STDDEF_H && !_STDDEF_H_ && !_ANSI_STDDEF_H && !__STDDEF_H__
          || __need_XXX was not defined before */
# 358 "/local/stage1/wmtan/ofc/COBRA_7_0_0/src/Porting/Linux2-EGCS/wrappers/stddef.h"

# 13 "/home/wmtan/root/cint/lib/prec_stl/memory" 2







//////////////////////////////////////////////////////////////////////
# 74 "/home/wmtan/root/cint/lib/prec_stl/memory"


template <int inst>
class __malloc_alloc_template {
 public:
  static void * allocate(size_t n);
  static void deallocate(void *p, size_t /* n */);
  static void * reallocate(void *p, size_t /* old_sz */, size_t new_sz);

  static void (* __set_malloc_handler(void (*f)()))();



};

typedef __malloc_alloc_template<0> malloc_alloc;
typedef malloc_alloc alloc;

//////////////////////////////////////////////////////////////////////
// non gcc, non HPUX compiler// G__GNUC
# 165 "/home/wmtan/root/cint/lib/prec_stl/memory"

//////////////////////////////////////////////////////////////////////

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  







  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
# 227 "/home/wmtan/root/cint/lib/prec_stl/memory"


  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#pragma endif
# 7 "/home/wmtan/root/cint/stl/_memory" 2




# 20 "/home/wmtan/root/cint/lib/prec_stl/vector" 2

# 1 "/home/wmtan/root/cint/stl/_utility" 1

#pragma include_noerr <utility.dll>



# 1 "/home/wmtan/root/cint/lib/prec_stl/utility" 1
// lib/prec_stl/utility

#pragma ifndef PREC_STL_UTILITY
#pragma define PREC_STL_UTILITY
#pragma link off global PREC_STL_UTILITY;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

template <class T1, class T2>
struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) {}
    pair(const T1& a, const T2& b) : first(a), second(b) {}
};

#pragma endif
# 6 "/home/wmtan/root/cint/stl/_utility" 2




# 21 "/home/wmtan/root/cint/lib/prec_stl/vector" 2





template<class T,class Allocator=alloc>





class vector {
 public:
  typedef T value_type;


  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;









# 102 "/home/wmtan/root/cint/lib/prec_stl/vector"

  typedef T* iterator;
  typedef const T* const_iterator;







// G__BORLANDCC5








  class reverse_iterator 



        : public std::random_access_iterator<T,difference_type>

        {
   public:
    reverse_iterator(const reverse_iterator& x) ;

    reverse_iterator& operator=(const reverse_iterator& x) ;


    T* base() ;



    T& operator*() const ;
    reverse_iterator& operator++();
    reverse_iterator operator++(int a);
    reverse_iterator& operator--();
    reverse_iterator operator--(int a);
    reverse_iterator operator+(long n);
    reverse_iterator operator-(long n);
    reverse_iterator& operator+=(long n);
    reverse_iterator& operator-=(long n);
    T& operator[](long n) ;
   private:
  };

// G__BORLANDCC5
  friend bool operator==(const vector::reverse_iterator& x
                        ,const vector::reverse_iterator& y) const;
  friend bool operator!=(const vector::reverse_iterator& x
                        ,const vector::reverse_iterator& y) const;


  typedef const reverse_iterator const_reverse_iterator;
















  iterator begin(void) ;
  iterator end(void) ;
  reverse_iterator rbegin(void) ;
  reverse_iterator rend(void) ;






  size_type size(void) const ;
  size_type max_size(void) const ;
  size_type capacity(void) const ;
  bool empty(void) const ;
  T& operator[](size_type n) ;
  vector(void) ;
  vector(size_type n,const T& value=T()) ;
  vector(const vector& x) ;
  vector(const_iterator first,const_iterator last) ;
  ~vector(void) ;
  vector& operator=(const vector& x);
  void reserve(size_type n) ;
  T& front(void) ;
  T& back(void) ;
  void push_back(const T& x) ;
  void swap(vector& x);
  iterator insert(iterator position,const T& x);
  void insert(iterator position,const_iterator first,const_iterator last);
  void insert(iterator position,size_type n,const T& x);
  void pop_back(void) ;
  void erase(iterator position) ;
  void erase(iterator first,iterator last) ;
  void clear() ;

# 217 "/home/wmtan/root/cint/lib/prec_stl/vector"

  // specialized algorithms:








  // Generic algorithm


  // input iter
  friend vector::iterator 
    find(vector::iterator first,vector::iterator last,const T& value);
  // forward iter
  friend vector::iterator 
    find_end(vector::iterator first1,vector::iterator last1,
             vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    find_first_of(vector::iterator first1,vector::iterator last1,
                  vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    adjacent_find(vector::iterator first,vector::iterator last);
  // input iter

  friend vector::difference_type
    count(vector::iterator first,vector::iterator last,const T& value);






  friend bool
    equal(vector::iterator first1,vector::iterator last1,
          vector::iterator first2);
  // forward iter
  friend vector::iterator
    search(vector::iterator first1,vector::iterator last1,
           vector::iterator first2,vector::iterator last2);
  friend vector::iterator
    search_n(vector::iterator first,vector::iterator last
             ,vector::size_type count,const T& value);
  // input and output iter -> forward iter
  friend vector::iterator
    copy(vector::iterator first,vector::iterator last,
         vector::iterator result);
  // bidirectional iter
  friend vector::iterator
    copy_backward(vector::iterator first,vector::iterator last,
                  vector::iterator result);
  // just value_type
  friend void swap(T& a,T& b);
  // forward iter
  friend vector::iterator
    swap_ranges(vector::iterator first1,vector::iterator last1,
                vector::iterator first2);
  friend void iter_swap(vector::iterator a,vector::iterator b);
  friend void replace(vector::iterator first,vector::iterator last,
                      const T& old_value,const T& new_value);
  // input, output iter -> forward iter
  friend vector::iterator 
    replace_copy(vector::iterator first,vector::iterator last,
                 vector::iterator result,
                 const T& old_value,const T& new_value);
  // forward iter
  friend void
    fill(vector::iterator first,vector::iterator last,const T& value);




  friend vector::iterator
    remove(vector::iterator first,vector::iterator last,const T& value);
  // input,output iter -> forward iter
  friend vector::iterator
    remove_copy(vector::iterator first,vector::iterator last,
                vector::iterator result,const T& value);
  friend vector::iterator
    unique(vector::iterator first,vector::iterator last);
  friend vector::iterator 
    unique_copy(vector::iterator first,vector::iterator last,
                vector::iterator result);
  friend void reverse(vector::iterator first,vector::iterator last);
  friend vector::iterator
     reverse_copy(vector::iterator first,vector::iterator last,
                  vector::iterator result);
  // forward iter




  // forward iter
  friend vector::iterator 
    rotate_copy(vector::iterator first,vector::iterator mid,
                vector::iterator last,vector::iterator result);
  // randomaccess iter
  friend void random_shuffle(vector::iterator first,vector::iterator last);
  // randomaccess iter
  friend void sort(vector::iterator first,vector::iterator last);
  friend void stable_sort(vector::iterator first,vector::iterator last);
  friend void partial_sort(vector::iterator first,vector::iterator mid,
                           vector::iterator last);
  friend vector::iterator
    partial_sort_copy(vector::iterator first,vector::iterator last,
                      vector::iterator result_first,
                      vector::iterator result_last);
  friend void nth_element(vector::iterator first,vector::iterator nth,
                          vector::iterator last);
  // forward iter
  friend vector::iterator 
    lower_bound(vector::iterator first,vector::iterator last,const T& value);
  friend vector::iterator 
    upper_bound(vector::iterator first,vector::iterator last,const T& value);




  friend bool binary_search(vector::iterator first,vector::iterator last,
                            const T& value);
  friend vector::iterator merge(vector::iterator first1,vector::iterator last1,
                                vector::iterator first2,vector::iterator last2,
                                vector::iterator result);
  friend void inplace_merge(vector::iterator first,vector::iterator middle,
                            vector::iterator last);
  friend bool includes(vector::iterator first1,vector::iterator last1,
                       vector::iterator first2,vector::iterator last2);
  friend vector::iterator 
    set_union(vector::iterator first1,vector::iterator last1,
              vector::iterator first2,vector::iterator last2,
              vector::iterator result);
  friend vector::iterator 
    set_intersection(vector::iterator first1,vector::iterator last1,
                     vector::iterator first2,vector::iterator last2,
                     vector::iterator result);
  friend vector::iterator 
    set_difference(vector::iterator first1,vector::iterator last1,
                   vector::iterator first2,vector::iterator last2,
                   vector::iterator result);
  friend vector::iterator 
    set_symmetric_difference(vector::iterator first1,vector::iterator last1,
                             vector::iterator first2,vector::iterator last2,
                             vector::iterator result);
  // random access
  friend void push_heap(vector::iterator first,vector::iterator last);
  friend void pop_heap(vector::iterator first,vector::iterator last);
  friend void make_heap(vector::iterator first,vector::iterator last);
  friend void sort_heap(vector::iterator first,vector::iterator last);
  // min,max, just value_type
  friend const T& min(const T& a,const T& b);
  friend const T& max(const T& a,const T& b);
  // forward iter
  friend vector::iterator 
    min_element(vector::iterator first,vector::iterator last);
  friend vector::iterator 
    max_element(vector::iterator first,vector::iterator last);
  // input iter
  friend bool
    lexicographical_compare(vector::iterator first1,vector::iterator last1,
                            vector::iterator first2,vector::iterator last2);
  // bidirectional iter
  friend bool next_permutation(vector::iterator first,vector::iterator last);
  friend bool prev_permutation(vector::iterator first,vector::iterator last);

// G__VISUAL,G__GNUC,G__BORLAND
# 406 "/home/wmtan/root/cint/lib/prec_stl/vector"


// G__NOALGORITHM

  // iterator_category resolution
  //friend random_access_iterator_tag iterator_category(vector::iterator x);

};

// G__defined("std::vector<bool>")
# 423 "/home/wmtan/root/cint/lib/prec_stl/vector"




#pragma endif
# 6 "/home/wmtan/root/cint/stl/_vector" 2




# 2 "/home/wmtan/root/cint/stl/vector" 2

}
# 9 "Utilities/DBI/interface/stdVectorConstInterface.h" 2

# 1 "Utilities/DBI/interface/IHandleAny.h" 1


//
//
//  29/03/2002 VI
//    added equality
//
class IHandleAny {
public :

  ///
  virtual ~IHandleAny() {}

  ///
  virtual IHandleAny * clone() const =0;

  // X& operator*() const =0;
  ///
  // virtual void * operator->() const =0;
  ///
  virtual void * get() const =0;

  virtual bool open(bool /* readOnly */ =true) = 0;

  virtual bool close() = 0;

  inline operator void * () { return get();} 

  virtual bool isValid() const =0;

  virtual bool equal(const IHandleAny & rh) const =0;

};

inline bool operator==(const IHandleAny & lh, const IHandleAny & rh) {
  return lh.equal(rh);
}


# 10 "Utilities/DBI/interface/stdVectorConstInterface.h" 2

# 1 "Utilities/DBI/interface/ZeroHandleAny.h" 1





class ZeroHandleAny : public IHandleAny {
public :
  ///
  ZeroHandleAny() {}
  ///
  virtual ~ZeroHandleAny() {}

  ///
  virtual IHandleAny * clone() const { return new ZeroHandleAny();}

  // X& operator*() const =0;
  ///
  // virtual void * operator->() const =0;
  ///
  virtual void * get() const { return 0;}

  virtual bool open(bool /* readOnly */ =true) { return false;}

  virtual bool close() { return false;}

  inline operator void * () { return get();} 

  virtual bool isValid() const { return false;}

  virtual bool equal(const IHandleAny & rh) const {
    const ZeroHandleAny * orh = dynamic_cast< const ZeroHandleAny *>(&rh);
    return orh!=0;
  }
};


# 11 "Utilities/DBI/interface/stdVectorConstInterface.h" 2



/** a "consistent" inteface to Transient & Persistent  vectors....
 */
template<class T>
class  stdVectorConstInterface {

public:

  typedef stdVectorConstInterface<T> self;
  typedef T value_type;
  typedef const value_type* const_pointer;
  typedef const value_type* const_iterator;
  typedef const value_type& const_reference;
  typedef size_t size_type;

public:

  ///
  stdVectorConstInterface(): begin_(0), size_(0), owner_(new ZeroHandleAny()){}
  ///

  stdVectorConstInterface(const_iterator b, const_iterator e, const IHandleAny & io= ZeroHandleAny() ) : 
    begin_(b), size_(e-b), owner_(io.clone()) {
    (*owner_).open(true); 
  }
  
  stdVectorConstInterface(const_iterator b,  size_type s, const IHandleAny & io=ZeroHandleAny()  ) : 
    begin_(b), size_(s), owner_(io.clone()) {
    (*owner_).open(true); 
  }

  stdVectorConstInterface(const vector<T>& v ) : 
    begin_(&*v.begin()), size_(v.size()) , owner_(new ZeroHandleAny()) {}

  ~stdVectorConstInterface() {
    (*owner_).close();
    delete owner_;
  }

  ///
  self& init(const vector<T>& v) { 
    (*owner_).close();
    delete owner_; owner_= new ZeroHandleAny();
    begin_=&*v.begin(); size_=v.size(); 
    return *this;
  }

  ///
  self& init(const_iterator b, const_iterator e, const IHandleAny & io=ZeroHandleAny()) {     (*owner_).close();
    delete owner_;
    begin_=b; 
    size_=e-b; 
    owner_=io.clone();
    (*owner_).open(true); 
    return *this;
  }
 
  self& init(const_iterator  b, size_type s, const IHandleAny & io=ZeroHandleAny()) { 
    (*owner_).close();
    delete owner_;
    begin_=b; 
    size_=s; 
    owner_=io.clone();
    (*owner_).open(true); 
    return *this;
  }


  /// copy constr
  stdVectorConstInterface(const self& rh) :
    begin_(rh.begin_), size_(rh.size_), owner_((*rh.owner_).clone()) { 
    (*owner_).open(true); 
  }

  ///
  self& operator=(const self& rh) {
    if ( &rh == this) return *this;
    (*owner_).close();
    delete owner_; 
    owner_=(*rh.owner_).clone(); 
    (*owner_).open(true); 
    begin_=rh.begin(); size_=rh.size(); 
    return *this;
  }
  
  inline void clear() { 
    begin_=0; size_=0; 
    (*owner_).close();
    delete owner_; 
    owner_=new ZeroHandleAny(); 
  }

  inline void setOwner( const IHandleAny & io) { 
    (*owner_).close();
    delete owner_; 
    owner_=io.clone(); 
    (*owner_).open(true);
  }

  /// data begin
  inline const_iterator begin() const {return begin_; }
  /// data end
  inline const_iterator end() const {return begin()+size();}
  /// data size
  inline size_type size() const {return size_;}
  /// empty...
  inline bool empty() const { return size_==0;}


  /// return first element
  inline const_reference front() const { return *begin(); }
  
  /// return last element 
  inline const_reference back() const { return *(end() - 1); }
  
  /// return element n
  inline const_reference operator[](size_type n) const { return *(begin() + n); }

  const  IHandleAny & owner() const { return *owner_;}

protected:

  const_iterator begin_;
  size_type size_;

  IHandleAny * owner_;

};


// stdVectorConstInterface_H
# 23 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 2

# 1 "Utilities/GenUtil/interface/refc_ptr.h" 1


//  a pointer which reference count the pointed object
//  similar to std auto_ptr but does allow standard copy constructor


template <class X, bool Intr=false> class refc_ptr {};


/** a pointer which reference count the pointed object (non intrusive)
 */
template <class X> class refc_ptr<X, false> {
public:
  typedef refc_ptr<X,false> self;
  typedef X element_type;
  typedef X *  pointer;
  typedef X & reference;
 
private:

  X* ptr;
  int * counts_;

public:
 
  ///
  explicit refc_ptr() : ptr(0), counts_(0) {}
  explicit refc_ptr(X* p) : ptr(p), counts_(new int(1)) {}

  ///
  refc_ptr(const self& a) : ptr(a.get()),counts_(a.counts()) { add();}


  ///
  template <class T, bool I> refc_ptr(const refc_ptr<T, I>& a) 
    : ptr(a.get()),counts_(a.counts()) {add();}


  ///
  self& operator=(const self& a)  {
    if (a.get() != ptr) {
      remove();
      ptr = a.get();
      counts_ = a.counts();
      add();
    }
    return *this;
  }


  ///
  template <class T, bool I> self& operator=(const refc_ptr<T, I>& a)  {
    if (a.get() != ptr) {
      remove();
      ptr = a.get();
      counts_ = a.counts();
      add();
    }
    return *this;
  }

  template <class T, bool I> void dyncast(const refc_ptr<T, I>& a) {
    pointer p = dynamic_cast<pointer>(a.get());
    if (p != ptr) {
      remove();
      ptr = p;
      counts_ = a.counts();
      add();
    }
  }


  ///
  ~refc_ptr() {
    remove();
  }
  ///
  X& operator*() const  { return *ptr; }
  ///
  X* operator->() const  { return ptr; }
  ///
  X* get() const  { return ptr; }

  int* counts() const { 
    return const_cast<refc_ptr<X>*>(this)->counts_;
  }
 

  void add(){if (counts_) (*counts_)++;}

  void remove(){ 
    if (ptr==0||counts_==0) return; 
    (*counts_)--;
    if ((*counts_)==0) {
      if (!intrusive()) { delete counts_;}  
      delete ptr; 
    }
    ptr=0;counts_=0;
  }

  bool intrusive() {
    return (!( int(counts_)<int(ptr) )) &&
            (char*)(counts_)-(char*)(ptr) < int(sizeof(*ptr)); 
  }

};

/** a pointer which reference count the pointed object (non intrusive)
 */
template <class X> class refc_ptr<X,true> {
public:
  typedef refc_ptr<X,true> self;
  typedef X element_type;
  typedef X * pointer;
  typedef X & reference;

private:

  pointer ptr;

public:
  ///
  refc_ptr() : ptr(0)  {}
  ///
  explicit refc_ptr(pointer p) : ptr(p) { add();}

  ///
  refc_ptr(const  self& a) : 
    ptr(a.get())  { add();}


  ///
  template <class T, bool I> refc_ptr(const refc_ptr<T, I>& a) : ptr(a.get()) {
    if ( counts() != a.counts()) {/* messs*/ }
    add();
  }

  ///
  self& operator=(const self& a)  {
    if (a.get() != ptr) {
      remove();
      ptr = a.get();
      add();
    }
    return *this;
  }




  ///
  template <class T, bool I> self& operator=(const refc_ptr<T,I>& a)  {
    if (a.get() != ptr) {
      remove();
      ptr = a.get();
      if ( counts() != a.counts()) {/* messs*/ }
      add();
    }
    return *this;
  }

  template <class T, bool I> void dyncast(const refc_ptr<T, I>& a) {
    pointer p = dynamic_cast<pointer>(a.get());
    if (p != ptr) {
      remove();
      ptr = p;
      if ( counts() != a.counts()) {/* messs*/ }
      add();
    }
  }



  ///
  ~refc_ptr() {
    remove();
  }
  ///
  reference operator*() const  { return *ptr; }
  ///
  pointer operator->() const  { return ptr; }
  ///
  pointer get() const  { return ptr; }

  int* counts() const {
    if(ptr==0) return 0;
    return &(const_cast<self*>(this)->ptr->counts_);
  }
 

  void add(){ if (ptr) (*ptr).counts_++;}

  void remove(){ 
    if (ptr==0) return; 
    (*ptr).counts_--;
    if ((*ptr).counts_==0) {delete ptr;}
    ptr=0;
  }


};


//  refc_ptr_H


# 24 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 2


# 1 "CARF/BaseSimEvent/interface/EmbdSimVertex.h" 1


//
//
// Persistent Vertex
//
//  version     0.1   01/11/18
//                    just embedded (requires contex: BaseSimEvent)



# 1 "CARF/BaseSimEvent/interface/CoreSimVertex.h" 1


//
//  a simulated vertex
//  version     0.1   98/10/03
//                    pure transient for G3carf'98
//  version     0.2   00/02/14
//                    moved to int16
// version      0.3   01/11/18
//                      just core part...

# 1 "/home/wmtan/root/cint/include/cmath" 1
namespace std {

#pragma include_noerr <stdcxxfunc.dll>
}

# 12 "CARF/BaseSimEvent/interface/CoreSimVertex.h" 2

# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.h" 1
// -*- C++ -*-
// CLASSDOC OFF
// $Id$
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
// HepLorentzVector is a Lorentz vector consisting of Hep3Vector and
// double components. Lorentz transformations (rotations and boosts)
// of these vectors are perfomed by multiplying with objects of
// the HepLorenzRotation class.
//
// .SS See Also
// ThreeVector.h, Rotation.h, LorentzRotation.h
//
// .SS Authors
// Leif Lonnblad and Anders Nilsson. Modified by Evgueni Tcherniaev, Mark Fischler
//








# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/CLHEP.h" 1
// -*- C++ -*-
// $Id$
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
// This file contains definitions of some usefull utilities and macros.
//



# 1 "/home/wmtan/root/cint/include/stdlib.h" 1



#pragma setstdstruct

typedef unsigned int size_t;





typedef unsigned int wchar_t;
#pragma include_noerr <stdfunc.dll>

# 12 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/CLHEP.h" 2

# 1 "/home/wmtan/root/cint/include/limits.h" 1














const unsigned int      UINT_MAX =(4294967295);
const unsigned long     ULONG_MAX =(4294967295);


# 13 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/CLHEP.h" 2






# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/CLHEP-default.h" 1
/* config/CLHEP-i686-unknown-linux-g++.h.  Generated automatically by configure.  */
// -*- C++ -*-
// CLASSDOC OFF
// $Id$
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
// This file should define some platform dependent features necessary for
// the CLHEP class library. Go through it and change the definition of the
// macros to suit you platform.
//



// Define if your FORTRAN compiler post-pends an underscore on all
// routine names. This is done automatically by the configure script.


// Define if your C++ compiler has STL classes defined in the std namespace.
// This is done automatically by the configure script.


// Define if your C++ compiler has I/O streams defined in the std namespace.
// This is done automatically by the configure script.


// Define if your C++ compiler has <iosfwd>.
// This is done automatically by the configure script.


// Define if your C++ compiler has <sstream>.
// This is done automatically by the configure script.
/* #undef HEP_HAVE_SSTREAM */

// Define if your C++ compiler uses the ANSI compliant std::ios_base
// instead of the old std::ios.
// This is done automatically by the configure script.
/* #undef HEP_USE_IOS_BASE */

// Define if your C++ compiler requires the "sub" function (see the
// Matrix/ module) without const. Such a bug was noticed for some
// versions of DEC CXX, SGI CC and HP aCC.
// This is done automatically by the configure script.


/* _CLHEP_COMPILER_H_ */
# 19 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/CLHEP.h" 2



// CLASSDOC OFF
// **** You should probably not touch anything below this line: ****

typedef double HepDouble;
typedef int    HepInt;
typedef float  HepFloat;
typedef bool   HepBoolean;









































// Default to generate random matrix
//




// Default to have assigment from three vector and rotation to matrix
//




// GNU g++ compiler can optimize when returning an object.
// However g++ on HP cannot deal with this.
//


// All the stuff needed by std::ios versus std::ios_base
# 98 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/CLHEP.h"












/* _CLHEP_H_ */
# 28 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.h" 2

# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ThreeVector.h" 1
// -*- C++ -*-
// CLASSDOC OFF
// $Id$
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
// Hep3Vector is a general 3-vector class defining vectors in three
// dimension using double components. Rotations of these vectors are
// performed by multiplying with an object of the HepRotation class.
//
// .SS See Also
// LorentzVector.h, Rotation.h, LorentzRotation.h 
//
// .SS Authors
// Leif Lonnblad and Anders Nilsson; ZOOM additions by Mark Fischler
//









# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/iostream.h" 1
// -*- C++ -*-
// CLASSDOC OFF
// $Id$
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
// Work around compiler annoyance with iostream[.h]
//









# 1 "/home/wmtan/root/cint/include/iostream" 1

namespace std {

}
# 1 "/home/wmtan/root/cint/include/_iostream" 1
// include/_iostream

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDL& i) 
        {return(std::endl(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDS& i) 
        {return(std::ends(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_FLUSH& i) 
        {return(std::flush(ostr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_ws& i) 
        {return(std::ws(istr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_WS& i) 
        {return(std::WS(istr));}


std::ostream& operator<<(std::ostream& ostr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::dec);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::hex);
#pragma else
  ostr.unsetf(ios_base::dec);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::hex);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_HEX& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::dec);
  istr.unsetf(ios::oct);
  istr.setf(ios::hex);
#pragma else
  istr.unsetf(ios_base::dec);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::hex);
#pragma endif
  return(istr);
}

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::dec);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::oct);
  ostr.setf(ios_base::dec);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_DEC& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::oct);
  istr.setf(ios::dec);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::oct);
  istr.setf(ios_base::dec);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::dec);
  ostr.setf(ios::oct);
#pragma else
  ostr.unsetf(ios_base::hex);
  ostr.unsetf(ios_base::dec);
  ostr.setf(ios_base::oct);
#pragma endif
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_OCT& i) {
#pragma ifndef G__TMPLTIOS
  istr.unsetf(ios::hex);
  istr.unsetf(ios::dec);
  istr.setf(ios::oct);
#pragma else
  istr.unsetf(ios_base::hex);
  istr.unsetf(ios_base::dec);
  istr.setf(ios_base::oct);
#pragma endif
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(ostr);
}
std::istream& operator<<(std::istream& istr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(istr);
}

// Value evaluation
//template<class T> int G__ateval(const T* x) {return(0);}
template<class T> int G__ateval(const T& x) {return(0);}
int G__ateval(const char* x) {return(0);}
int G__ateval(const void* x) {return(0);}
int G__ateval(const double x) {return(0);}
int G__ateval(const float x) {return(0);}
int G__ateval(const char x) {return(0);}
int G__ateval(const short x) {return(0);}
int G__ateval(const int x) {return(0);}
int G__ateval(const long x) {return(0);}
int G__ateval(const unsigned char x) {return(0);}
int G__ateval(const unsigned short x) {return(0);}
int G__ateval(const unsigned int x) {return(0);}
int G__ateval(const unsigned long x) {return(0);}








# 5 "/home/wmtan/root/cint/include/iostream" 2

# 20 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/config/iostream.h" 2



/* HEP_IOSTREAM_H */
# 28 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ThreeVector.h" 2






class HepRotation;
class HepEulerAngles;
class HepAxisAngle;

class Hep3Vector {

public:

// Basic properties and operations on 3-vectors:  

  enum { X=0, Y=1, Z=2, NUM_COORDINATES=3, SIZE=NUM_COORDINATES };
  // Safe indexing of the coordinates when using with matrices, arrays, etc.
  // (BaBar)

  inline Hep3Vector(double x = 0.0, double y = 0.0, double z = 0.0);
  // The constructor.  

  inline Hep3Vector(const Hep3Vector &);
  // The copy constructor.

  inline ~Hep3Vector();
  // The destructor.  Not virtual - inheritance from this class is dangerous.

  double operator () (int) const;
  // Get components by index -- 0-based (Geant4) 

  inline double operator [] (int) const;
  // Get components by index -- 0-based (Geant4) 

  double & operator () (int);
  // Set components by index.  0-based.

  inline double & operator [] (int);
  // Set components by index.  0-based.

  inline double x() const;
  inline double y() const;
  inline double z() const;
  // The components in cartesian coordinate system.  Same as getX() etc.

  inline void setX(double);
  inline void setY(double);
  inline void setZ(double);
  // Set the components in cartesian coordinate system.

  inline void set( double x, double y, double z); 
  // Set all three components in cartesian coordinate system.

  inline double phi() const;
  // The azimuth angle.

  inline double theta() const;
  // The polar angle.

  inline double cosTheta() const;
  // Cosine of the polar angle.

  inline double cos2Theta() const;
  // Cosine squared of the polar angle - faster than cosTheta(). (ZOOM)

  inline double mag2() const;
  // The magnitude squared (r^2 in spherical coordinate system).

  inline double mag() const;
  // The magnitude (r in spherical coordinate system).

  inline void setPhi(double);
  // Set phi keeping mag and theta constant (BaBar).

  inline void setTheta(double);
  // Set theta keeping mag and phi constant (BaBar).

         void setMag(double);
  // Set magnitude keeping theta and phi constant (BaBar).

  inline double perp2() const;
  // The transverse component squared (rho^2 in cylindrical coordinate system).

  inline double perp() const;
  // The transverse component (rho in cylindrical coordinate system).

  inline void setPerp(double);
  // Set the transverse component keeping phi and z constant.

  void setCylTheta(double);
  // Set theta while keeping transvers component and phi fixed 

  inline double perp2(const Hep3Vector &) const;
  // The transverse component w.r.t. given axis squared.

  inline double perp(const Hep3Vector &) const;
  // The transverse component w.r.t. given axis.

  inline Hep3Vector & operator = (const Hep3Vector &);
  // Assignment.

  inline bool operator == (const Hep3Vector &) const;
  inline bool operator != (const Hep3Vector &) const;
  // Comparisons (Geant4). 

  bool isNear (const Hep3Vector &, double epsilon=tolerance) const;
  // Check for equality within RELATIVE tolerance (default 2.2E-14). (ZOOM)
  // |v1 - v2|**2 <= epsilon**2 * |v1.dot(v2)| 

  double howNear(const Hep3Vector & v ) const;
  // sqrt ( |v1-v2|**2 / v1.dot(v2) ) with a maximum of 1.
  // If v1.dot(v2) is negative, will return 1.

  double deltaR(const Hep3Vector & v) const;
  // sqrt( pseudorapity_difference**2 + deltaPhi **2 )

  inline Hep3Vector & operator += (const Hep3Vector &);
  // Addition.

  inline Hep3Vector & operator -= (const Hep3Vector &);
  // Subtraction.

  inline Hep3Vector operator - () const;
  // Unary minus.

  inline Hep3Vector & operator *= (double);
  // Scaling with real numbers.

         Hep3Vector & operator /= (double);
  // Division by (non-zero) real number.

  inline Hep3Vector unit() const;
  // Vector parallel to this, but of length 1.

  inline Hep3Vector orthogonal() const;
  // Vector orthogonal to this (Geant4).

  inline double dot(const Hep3Vector &) const;
  // double product.

  inline Hep3Vector cross(const Hep3Vector &) const;
  // Cross product.

  double angle(const Hep3Vector &) const;
  // The angle w.r.t. another 3-vector.

  double pseudoRapidity() const;
  // Returns the pseudo-rapidity, i.e. -ln(tan(theta/2))

  void setEta  ( double p );
  // Set pseudo-rapidity, keeping magnitude and phi fixed.  (ZOOM)

  void setCylEta  ( double p );
  // Set pseudo-rapidity, keeping transverse component and phi fixed.  (ZOOM)

  Hep3Vector & rotateX(double);
  // Rotates the Hep3Vector around the x-axis.

  Hep3Vector & rotateY(double);
  // Rotates the Hep3Vector around the y-axis.

  Hep3Vector & rotateZ(double);
  // Rotates the Hep3Vector around the z-axis.

  Hep3Vector & rotateUz(const Hep3Vector&);
  // Rotates reference frame from Uz to newUz (unit vector) (Geant4).

    Hep3Vector & rotate(double, const Hep3Vector &);
  // Rotates around the axis specified by another Hep3Vector.
  // (Uses methods of HepRotation, forcing linking in of Rotation.cc.)

  Hep3Vector & operator *= (const HepRotation &);
  Hep3Vector & transform(const HepRotation &);
  // Transformation with a Rotation matrix.


// = = = = = = = = = = = = = = = = = = = = = = = =
//
// Esoteric properties and operations on 3-vectors:  
//
// 1 - Set vectors in various coordinate systems
// 2 - Synonyms for accessing coordinates and properties
// 3 - Comparisions (dictionary, near-ness, and geometric)
// 4 - Intrinsic properties 
// 5 - Properties releative to z axis and arbitrary directions
// 6 - Polar and azimuthal angle decomposition and deltaPhi
// 7 - Rotations 
//
// = = = = = = = = = = = = = = = = = = = = = = = =

// 1 - Set vectors in various coordinate systems

  inline void setRThetaPhi  (double r, double theta, double phi);
  // Set in spherical coordinates:  Angles are measured in RADIANS

  inline void setREtaPhi  ( double r, double eta,  double phi );
  // Set in spherical coordinates, but specify peudorapidiy to determine theta.

  inline void setRhoPhiZ   (double rho, double phi, double z);
  // Set in cylindrical coordinates:  Phi angle is measured in RADIANS

  void setRhoPhiTheta ( double rho, double phi, double theta);
  // Set in cylindrical coordinates, but specify theta to determine z.

  void setRhoPhiEta ( double rho, double phi, double eta);
  // Set in cylindrical coordinates, but specify pseudorapidity to determine z.

// 2 - Synonyms for accessing coordinates and properties

  inline double getX() const; 
  inline double getY() const;
  inline double getZ() const; 
  // x(), y(), and z()

  inline double getR    () const;
  inline double getTheta() const;
  inline double getPhi  () const;
  // mag(), theta(), and phi()

  inline double r       () const;
  // mag()

  inline double rho     () const;
  inline double getRho  () const;
  // perp()

  double eta     () const;
  double getEta  () const;
  // pseudoRapidity() 

  inline void setR ( double s );
  // setMag()

  inline void setRho ( double s );
  // setPerp()

// 3 - Comparisions (dictionary, near-ness, and geometric)

  int compare (const Hep3Vector & v) const;
  bool operator > (const Hep3Vector & v) const;
  bool operator < (const Hep3Vector & v) const;
  bool operator>= (const Hep3Vector & v) const;
  bool operator<= (const Hep3Vector & v) const;
  // dictionary ordering according to z, then y, then x component

  inline double diff2 (const Hep3Vector & v) const;
  // |v1-v2|**2

  static double setTolerance (double tol);
  static inline double getTolerance ();
  // Set the tolerance used in isNear() for Hep3Vectors 

  bool isParallel (const Hep3Vector & v, double epsilon=tolerance) const;
  // Are the vectors parallel, within the given tolerance?

  bool isOrthogonal (const Hep3Vector & v, double epsilon=tolerance) const;
  // Are the vectors orthogonal, within the given tolerance?

  double howParallel   (const Hep3Vector & v) const;
  // | v1.cross(v2) / v1.dot(v2) |, to a maximum of 1.

  double howOrthogonal (const Hep3Vector & v) const;
  // | v1.dot(v2) / v1.cross(v2) |, to a maximum of 1.

  enum { ToleranceTicks = 100 };

// 4 - Intrinsic properties 

  double beta    () const;
  // relativistic beta (considering v as a velocity vector with c=1)
  // Same as mag() but will object if >= 1

  double gamma() const;
  // relativistic gamma (considering v as a velocity vector with c=1)

  double coLinearRapidity() const;
  // inverse tanh (beta)

// 5 - Properties relative to Z axis and to an arbitrary direction

          // Note that the non-esoteric CLHEP provides 
          // theta(), cosTheta(), cos2Theta, and angle(const Hep3Vector&)

  inline double angle() const;
  // angle against the Z axis -- synonym for theta()

  inline double theta(const Hep3Vector & v2) const;  
  // synonym for angle(v2)

  double cosTheta (const Hep3Vector & v2) const;
  double cos2Theta(const Hep3Vector & v2) const;
  // cos and cos^2 of the angle between two vectors

  inline Hep3Vector project () const;
         Hep3Vector project (const Hep3Vector & v2) const;
  // projection of a vector along a direction.  

  inline Hep3Vector perpPart() const;
  inline Hep3Vector perpPart (const Hep3Vector & v2) const;
  // vector minus its projection along a direction.

  double rapidity () const;
  // inverse tanh(v.z())

  double rapidity (const Hep3Vector & v2) const;
  // rapidity with respect to specified direction:  
  // inverse tanh (v.dot(u)) where u is a unit in the direction of v2

  double eta(const Hep3Vector & v2) const;
  // - ln tan of the angle beween the vector and the ref direction.

// 6 - Polar and azimuthal angle decomposition and deltaPhi

  // Decomposition of an angle within reference defined by a direction:

  double polarAngle (const Hep3Vector & v2) const;
  // The reference direction is Z: the polarAngle is abs(v.theta()-v2.theta()).

  double deltaPhi (const Hep3Vector & v2) const;
  // v.phi()-v2.phi(), brought into the range (-PI,PI]

  double azimAngle  (const Hep3Vector & v2) const;
  // The reference direction is Z: the azimAngle is the same as deltaPhi

  double polarAngle (const Hep3Vector & v2, 
                                        const Hep3Vector & ref) const;
  // For arbitrary reference direction, 
  //    polarAngle is abs(v.angle(ref) - v2.angle(ref)).

  double azimAngle  (const Hep3Vector & v2, 
                                        const Hep3Vector & ref) const;
  // To compute azimangle, project v and v2 into the plane normal to
  // the reference direction.  Then in that plane take the angle going
  // clockwise around the direction from projection of v to that of v2.

// 7 - Rotations 

// These mehtods **DO NOT** use anything in the HepRotation class.
// Thus, use of v.rotate(axis,delta) does not force linking in Rotation.cc.

  Hep3Vector & rotate  (const Hep3Vector & axis, double delta);
  // Synonym for rotate (delta, axis)

  Hep3Vector & rotate  (const HepAxisAngle & ax);
  // HepAxisAngle is a struct holding an axis direction and an angle.

  Hep3Vector & rotate (const HepEulerAngles & e);
  Hep3Vector & rotate (double phi,
                        double theta,
                        double psi);
  // Rotate via Euler Angles. Our Euler Angles conventions are 
  // those of Goldstein Classical Mechanics page 107.

protected:
  void setSpherical (double r, double theta, double phi);
  void setCylindrical (double r, double phi, double z);
  double negativeInfinity() const;

protected:

  double dx;
  double dy;
  double dz;
  // The components.

  static double tolerance;
  // default tolerance criterion for isNear() to return true.
};

// Global Methods

Hep3Vector rotationXOf (const Hep3Vector & vec, double delta);
Hep3Vector rotationYOf (const Hep3Vector & vec, double delta);
Hep3Vector rotationZOf (const Hep3Vector & vec, double delta);

Hep3Vector rotationOf (const Hep3Vector & vec, 
                                const Hep3Vector & axis, double delta);
Hep3Vector rotationOf (const Hep3Vector & vec, const HepAxisAngle & ax);

Hep3Vector rotationOf (const Hep3Vector & vec, 
                                double phi, double theta, double psi);
Hep3Vector rotationOf (const Hep3Vector & vec, const HepEulerAngles & e);
// Return a new vector based on a rotation of the supplied vector

std ::ostream & operator << (std ::ostream &, const Hep3Vector &);
// Output to a stream.

std ::istream & operator >> (std ::istream &, Hep3Vector &);
// Input from a stream.

extern const Hep3Vector HepXHat, HepYHat, HepZHat;









typedef Hep3Vector HepThreeVectorD;
typedef Hep3Vector HepThreeVectorF;

Hep3Vector operator / (const Hep3Vector &, double a);
// Division of 3-vectors by non-zero real number





# 461 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ThreeVector.h"

# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ThreeVector.icc" 1
// -*- C++ -*-
// $Id$
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
// 
// This is the definitions of the inline member functions of the
// Hep3Vector class.
//





// ------------------
// Access to elements
// ------------------

// x, y, z

double & Hep3Vector::operator[] (int i)       { return operator()(i); }
double   Hep3Vector::operator[] (int i) const { return operator()(i); }

inline double Hep3Vector::x() const { return dx; }
inline double Hep3Vector::y() const { return dy; }
inline double Hep3Vector::z() const { return dz; }

inline double Hep3Vector::getX() const { return dx; }
inline double Hep3Vector::getY() const { return dy; }
inline double Hep3Vector::getZ() const { return dz; }

inline void Hep3Vector::setX(double x) { dx = x; }
inline void Hep3Vector::setY(double y) { dy = y; }
inline void Hep3Vector::setZ(double z) { dz = z; }

inline void Hep3Vector::set(double x, double y, double z) { 
  dx = x; 
  dy = y; 
  dz = z; 
}

// --------------
// Global methods
// --------------

inline Hep3Vector operator + (const Hep3Vector & a, const Hep3Vector & b) {
  return Hep3Vector(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

inline Hep3Vector operator - (const Hep3Vector & a, const Hep3Vector & b) {
  return Hep3Vector(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

inline Hep3Vector operator * (const Hep3Vector & p, double a) {
  return Hep3Vector(a*p.x(), a*p.y(), a*p.z());
}

inline Hep3Vector operator * (double a, const Hep3Vector & p) {
  return Hep3Vector(a*p.x(), a*p.y(), a*p.z());
}

inline double operator * (const Hep3Vector & a, const Hep3Vector & b) {
  return a.dot(b);
}

// --------------------------
// Set in various coordinates
// --------------------------

inline void Hep3Vector::setRThetaPhi
                  ( double r, double theta, double phi ) {
  setSpherical (r, theta, phi); 
}

inline void Hep3Vector::setREtaPhi
                  ( double r, double eta,  double phi ) {
  setSpherical (r, 2*atan(exp(-eta)), phi); 
}

inline void Hep3Vector::setRhoPhiZ
                  ( double rho, double phi, double z) {
  setCylindrical (rho, phi, z); 
}

// ------------
// Constructors
// ------------

inline Hep3Vector::Hep3Vector(double x, double y, double z)
  : dx(x), dy(y), dz(z) {}

inline Hep3Vector::Hep3Vector(const Hep3Vector & p)
: dx(p.dx), dy(p.dy), dz(p.dz) {}

inline Hep3Vector::~Hep3Vector() {}

inline Hep3Vector & Hep3Vector::operator = (const Hep3Vector & p) {
  dx = p.dx;
  dy = p.dy;
  dz = p.dz;
  return *this;
}

// ------------------
// Access to elements
// ------------------

// r, theta, phi

inline double Hep3Vector::mag2() const { return dx*dx + dy*dy + dz*dz; }
inline double Hep3Vector::mag()  const { return sqrt(mag2()); }
inline double Hep3Vector::r()    const { return mag(); }

inline double Hep3Vector::theta()       const {
  return dx == 0.0 && dy == 0.0 && dz == 0.0 ? 0.0 : atan2(perp(),dz);
}
inline double Hep3Vector::phi() const {
  return dx == 0.0 && dy == 0.0 ? 0.0 : atan2(dy,dx);
}

inline double Hep3Vector::getR()     const { return mag();   }
inline double Hep3Vector::getTheta() const { return theta(); }
inline double Hep3Vector::getPhi()   const { return phi();   }
inline double Hep3Vector::angle()    const { return theta(); }

inline double Hep3Vector::cosTheta() const {
  double ptot = mag();
  return ptot == 0.0 ? 1.0 : dz/ptot;
}

inline double Hep3Vector::cos2Theta() const {
  double ptot2 = mag2();
  return ptot2 == 0.0 ? 1.0 : dz*dz/ptot2;
}

inline void Hep3Vector::setR(double r) { setMag(r); }

inline void Hep3Vector::setTheta(double th) {
  double ma   = mag();
  double ph   = phi();
  setX(ma*sin(th)*cos(ph));
  setY(ma*sin(th)*sin(ph));
  setZ(ma*cos(th));
}

inline void Hep3Vector::setPhi(double ph) {
  double xy   = perp();
  setX(xy*cos(ph));
  setY(xy*sin(ph));
}

// perp, eta, 

inline double Hep3Vector::perp2()  const { return dx*dx + dy*dy; }
inline double Hep3Vector::perp()   const { return sqrt(perp2()); }
inline double Hep3Vector::rho()    const { return perp();  }
inline double Hep3Vector::eta()    const { return pseudoRapidity();}

inline double Hep3Vector::getRho() const { return perp();  }
inline double Hep3Vector::getEta() const { return pseudoRapidity();}

inline void Hep3Vector::setPerp(double r) {
  double p = perp();
  if (p != 0.0) {
    dx *= r/p;
    dy *= r/p;
  }
}
inline void Hep3Vector::setRho(double rho) { setPerp (rho); }

// ----------
// Comparison
// ----------

inline bool Hep3Vector::operator == (const Hep3Vector& v) const {
  return (v.x()==x() && v.y()==y() && v.z()==z()) ? true : false;
}

inline bool Hep3Vector::operator != (const Hep3Vector& v) const {
  return (v.x()!=x() || v.y()!=y() || v.z()!=z()) ? true : false;
}

inline double Hep3Vector::getTolerance () {
  return tolerance;
}

// ----------
// Arithmetic
// ----------

inline Hep3Vector& Hep3Vector::operator += (const Hep3Vector & p) {
  dx += p.x();
  dy += p.y();
  dz += p.z();
  return *this;
}

inline Hep3Vector& Hep3Vector::operator -= (const Hep3Vector & p) {
  dx -= p.x();
  dy -= p.y();
  dz -= p.z();
  return *this;
}

inline Hep3Vector Hep3Vector::operator - () const {
  return Hep3Vector(-dx, -dy, -dz);
}

inline Hep3Vector& Hep3Vector::operator *= (double a) {
  dx *= a;
  dy *= a;
  dz *= a;
  return *this;
}

// -------------------
// Combine two Vectors
// -------------------

inline double Hep3Vector::diff2(const Hep3Vector & p) const {
  return (*this-p).mag2();
}

inline double Hep3Vector::dot(const Hep3Vector & p) const {
  return dx*p.x() + dy*p.y() + dz*p.z();
}

inline Hep3Vector Hep3Vector::cross(const Hep3Vector & p) const {
  return Hep3Vector(dy*p.z()-p.y()*dz, dz*p.x()-p.z()*dx, dx*p.y()-p.x()*dy);
}

inline double Hep3Vector::perp2(const Hep3Vector & p)  const {
  double tot = p.mag2();
  double ss  = dot(p);
  return tot > 0.0 ? mag2()-ss*ss/tot : mag2();
}

inline double Hep3Vector::perp(const Hep3Vector & p) const {
  return sqrt(perp2(p));
}

inline Hep3Vector Hep3Vector::perpPart () const {
  return Hep3Vector (dx, dy, 0);
}
inline Hep3Vector Hep3Vector::project () const {
  return Hep3Vector (0, 0, dz);
}

inline Hep3Vector Hep3Vector::perpPart (const Hep3Vector & v2) const {
  return ( *this - project(v2) );
}

inline double Hep3Vector::angle(const Hep3Vector & q) const {
  return acos(cosTheta(q));
}

inline double Hep3Vector::theta(const Hep3Vector & q) const { 
  return angle(q); 
}

inline double Hep3Vector::azimAngle(const Hep3Vector & v2) const { 
  return deltaPhi(v2); 
}

// ----------
// Properties
// ----------

inline Hep3Vector Hep3Vector::unit() const {
  double  tot = mag2();
  Hep3Vector p(x(),y(),z());
  return tot > 0.0 ? p *= (1.0/sqrt(tot)) : p;
}

inline Hep3Vector Hep3Vector::orthogonal() const {
  double x = dx < 0.0 ? -dx : dx;
  double y = dy < 0.0 ? -dy : dy;
  double z = dz < 0.0 ? -dz : dz;
  if (x < y) {
    return x < z ? Hep3Vector(0,dz,-dy) : Hep3Vector(dy,-dx,0);
  }else{
    return y < z ? Hep3Vector(-dz,0,dx) : Hep3Vector(dy,-dx,0);
  }
}





# 462 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ThreeVector.h" 2



/* HEP_THREEVECTOR_H */
# 29 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.h" 2







// Declarations of classes and global methods
class HepLorentzVector;
class HepLorentzRotation;
class HepRotation;
class HepAxisAngle;
class HepEulerAngles;
class Tcomponent;
HepLorentzVector rotationXOf( const HepLorentzVector & vec, double delta );
HepLorentzVector rotationYOf( const HepLorentzVector & vec, double delta );
HepLorentzVector rotationZOf( const HepLorentzVector & vec, double delta );
HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, const Hep3Vector & axis, double delta );
HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, const HepAxisAngle & ax );
HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, const HepEulerAngles & e );
HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, double phi,
                                    double theta,
                                    double psi );
inline 
HepLorentzVector  boostXOf( const HepLorentzVector & vec, double beta );
inline 
HepLorentzVector  boostYOf( const HepLorentzVector & vec, double beta );
inline 
HepLorentzVector  boostZOf( const HepLorentzVector & vec, double beta );
inline HepLorentzVector  boostOf
    ( const HepLorentzVector & vec, const Hep3Vector & betaVector );
inline HepLorentzVector  boostOf
    ( const HepLorentzVector & vec, const Hep3Vector & axis,  double beta );

enum ZMpvMetric_t { TimePositive, TimeNegative };


class HepLorentzVector {

public:

  enum { X=0, Y=1, Z=2, T=3, NUM_COORDINATES=4, SIZE=NUM_COORDINATES };
  // Safe indexing of the coordinates when using with matrices, arrays, etc.
  // (BaBar)

  inline HepLorentzVector(double x, double y,
                          double z, double t);
  // Constructor giving the components x, y, z, t.

  inline HepLorentzVector(double x, double y, double z);
  // Constructor giving the components x, y, z with t-component set to 0.0.

  inline HepLorentzVector(double t);
  // Constructor giving the t-component with x, y and z set to 0.0.

  inline HepLorentzVector();
  // Default constructor with x, y, z and t set to 0.0.

  inline HepLorentzVector(const Hep3Vector & p, double e);
  inline HepLorentzVector(double e, const Hep3Vector & p);
  // Constructor giving a 3-Vector and a time component.

  inline HepLorentzVector(const HepLorentzVector &);
  // Copy constructor.

  inline ~HepLorentzVector();
  // The destructor.

  inline operator const Hep3Vector & () const;
  inline operator Hep3Vector & ();
  // Conversion (cast) to Hep3Vector.

  inline double x() const;
  inline double y() const;
  inline double z() const;
  inline double t() const;
  // Get position and time.

  inline void setX(double);
  inline void setY(double);
  inline void setZ(double);
  inline void setT(double);
  // Set position and time.

  inline double px() const;
  inline double py() const;
  inline double pz() const;
  inline double e() const;
  // Get momentum and energy.

  inline void setPx(double);
  inline void setPy(double);
  inline void setPz(double);
  inline void setE(double);
  // Set momentum and energy.

  inline Hep3Vector vect() const;
  // Get spatial component. 

  inline void setVect(const Hep3Vector &);
  // Set spatial component. 

  inline double theta() const;
  inline double cosTheta() const;
  inline double phi() const;
  inline double rho() const;
  // Get spatial vector components in spherical coordinate system.

  inline void setTheta(double);
  inline void setPhi(double);
  inline void setRho(double);
  // Set spatial vector components in spherical coordinate system.

  double operator () (int) const;
  inline double operator [] (int) const;
  // Get components by index.

  double & operator () (int);
  inline double & operator [] (int);
  // Set components by index.

  inline HepLorentzVector & operator = (const HepLorentzVector &);
  // Assignment. 

  inline HepLorentzVector   operator +  (const HepLorentzVector &) const;
  inline HepLorentzVector & operator += (const HepLorentzVector &);
  // Additions.

  inline HepLorentzVector   operator -  (const HepLorentzVector &) const;
  inline HepLorentzVector & operator -= (const HepLorentzVector &);
  // Subtractions.

  inline HepLorentzVector operator - () const;
  // Unary minus.

  inline HepLorentzVector & operator *= (double);
         HepLorentzVector & operator /= (double);
  // Scaling with real numbers.

  inline bool operator == (const HepLorentzVector &) const;
  inline bool operator != (const HepLorentzVector &) const;
  // Comparisons.

  inline double perp2() const;
  // Transverse component of the spatial vector squared.

  inline double perp() const;
  // Transverse component of the spatial vector (R in cylindrical system).

  inline void setPerp(double);
  // Set the transverse component of the spatial vector.

  inline double perp2(const Hep3Vector &) const;
  // Transverse component of the spatial vector w.r.t. given axis squared.

  inline double perp(const Hep3Vector &) const;
  // Transverse component of the spatial vector w.r.t. given axis.

  inline double angle(const Hep3Vector &) const;
  // Angle wrt. another vector.

  inline double mag2() const;
  // Dot product of 4-vector with itself. 
  // By default the metric is TimePositive, and mag2() is the same as m2().

  inline double m2() const;
  // Invariant mass squared.

  inline double mag() const;
  inline double m() const;
  // Invariant mass. If m2() is negative then -sqrt(-m2()) is returned.

  inline double mt2() const;
  // Transverse mass squared.

  inline double mt() const;
  // Transverse mass.

  inline double et2() const;
  // Transverse energy squared.

  inline double et() const;
  // Transverse energy.

  inline double dot(const HepLorentzVector &) const;
  inline double operator * (const HepLorentzVector &) const;
  // Scalar product.

  inline double invariantMass2( const HepLorentzVector & w ) const;
  // Invariant mass squared of pair of 4-vectors 

  double invariantMass ( const HepLorentzVector & w ) const;
  // Invariant mass of pair of 4-vectors 

  inline void setVectMag(const Hep3Vector & spatial, double magnitude);
  inline void setVectM(const Hep3Vector & spatial, double mass);
  // Copy spatial coordinates, and set energy = sqrt(mass^2 + spatial^2)

  inline double plus() const;
  inline double minus() const;
  // Returns the positive/negative light-cone component t +/- z.

  Hep3Vector boostVector() const;
  // Boost needed from rest4Vector in rest frame to form this 4-vector
  // Returns the spatial components divided by the time component.

  HepLorentzVector & boost(double, double, double);
  inline HepLorentzVector & boost(const Hep3Vector &);
  // Lorentz boost.

  HepLorentzVector & boostX( double beta );
  HepLorentzVector & boostY( double beta );
  HepLorentzVector & boostZ( double beta );
  // Boost along an axis, by magnitue beta (fraction of speed of light)

  double rapidity() const;
  // Returns the rapidity, i.e. 0.5*ln((E+pz)/(E-pz))

  inline double pseudoRapidity() const;
  // Returns the pseudo-rapidity, i.e. -ln(tan(theta/2))

  inline bool isTimelike() const;
  // Test if the 4-vector is timelike

  inline bool isSpacelike() const;
  // Test if the 4-vector is spacelike

  inline bool isLightlike(double epsilon=tolerance) const;
  // Test for lightlike is within tolerance epsilon

  HepLorentzVector &  rotateX(double);
  // Rotate the spatial component around the x-axis.

  HepLorentzVector &  rotateY(double);
  // Rotate the spatial component around the y-axis.

  HepLorentzVector &  rotateZ(double);
  // Rotate the spatial component around the z-axis.

  HepLorentzVector &  rotateUz(const Hep3Vector &);
  // Rotates the reference frame from Uz to newUz (unit vector).

  HepLorentzVector & rotate(double, const Hep3Vector &);
  // Rotate the spatial component around specified axis.

  inline HepLorentzVector & operator *= (const HepRotation &);
  inline HepLorentzVector & transform(const HepRotation &);
  // Transformation with HepRotation.

  HepLorentzVector & operator *= (const HepLorentzRotation &);
  HepLorentzVector & transform(const HepLorentzRotation &);
  // Transformation with HepLorenzRotation.

// = = = = = = = = = = = = = = = = = = = = = = = =
//
// Esoteric properties and operations on 4-vectors:  
//
// 0 - Flexible metric convention and axial unit 4-vectors
// 1 - Construct and set 4-vectors in various ways 
// 2 - Synonyms for accessing coordinates and properties
// 2a - Setting space coordinates in different ways 
// 3 - Comparisions (dictionary, near-ness, and geometric)
// 4 - Intrinsic properties 
// 4a - Releativistic kinematic properties 
// 4b - Methods combining two 4-vectors
// 5 - Properties releative to z axis and to arbitrary directions
// 7 - Rotations and Boosts
//
// = = = = = = = = = = = = = = = = = = = = = = = =

// 0 - Flexible metric convention 

  static ZMpvMetric_t setMetric( ZMpvMetric_t m );
  static ZMpvMetric_t getMetric();

// 1 - Construct and set 4-vectors in various ways 

  inline void set        (double x, double y, double z, double  t);
  inline void set        (double x, double y, double z, Tcomponent t);
  inline HepLorentzVector(double x, double y, double z, Tcomponent t);
  // Form 4-vector by supplying cartesian coordinate components

  inline void set        (Tcomponent t, double x, double y, double z);
  inline HepLorentzVector(Tcomponent t, double x, double y, double z);
  // Deprecated because the 4-doubles form uses x,y,z,t, not t,x,y,z.

  inline void set                 ( double t );

  inline void set                 ( Tcomponent t );
  inline explicit HepLorentzVector( Tcomponent t );
  // Form 4-vector with zero space components, by supplying t component

  inline void set                 ( const Hep3Vector & v );
  inline explicit HepLorentzVector( const Hep3Vector & v );
  // Form 4-vector with zero time component, by supplying space 3-vector 

  inline HepLorentzVector & operator=( const Hep3Vector & v );
  // Form 4-vector with zero time component, equal to space 3-vector 

  inline void set ( const Hep3Vector & v, double t );
  inline void set ( double t, const Hep3Vector & v );
  // Set using specified space vector and time component

// 2 - Synonyms for accessing coordinates and properties

  inline double getX() const;
  inline double getY() const;
  inline double getZ() const;
  inline double getT() const;
  // Get position and time.

  inline Hep3Vector v() const;
  inline Hep3Vector getV() const;
  // Get spatial component.   Same as vect.

  inline void setV(const Hep3Vector &);
  // Set spatial component.   Same as setVect.

// 2a - Setting space coordinates in different ways 

  inline void setV( double x, double y, double z );

  inline void setRThetaPhi( double r, double theta, double phi);
  inline void setREtaPhi( double r, double eta, double phi);
  inline void setRhoPhiZ( double rho, double phi, double z );

// 3 - Comparisions (dictionary, near-ness, and geometric)

  int compare( const HepLorentzVector & w ) const;

  bool operator >( const HepLorentzVector & w ) const;
  bool operator <( const HepLorentzVector & w ) const;
  bool operator>=( const HepLorentzVector & w ) const;
  bool operator<=( const HepLorentzVector & w ) const;

  bool   isNear ( const HepLorentzVector & w, 
                                        double epsilon=tolerance ) const;
  double howNear( const HepLorentzVector & w ) const;
  // Is near using Euclidean measure t**2 + v**2

  bool   isNearCM ( const HepLorentzVector & w, 
                                        double epsilon=tolerance ) const;
  double howNearCM( const HepLorentzVector & w ) const;
  // Is near in CM frame:  Applicable only for two timelike HepLorentzVectors

        // If w1 and w2 are already in their CM frame, then w1.isNearCM(w2)
        // is exactly equivalent to w1.isNear(w2).
        // If w1 and w2 have T components of zero, w1.isNear(w2) is exactly
        // equivalent to w1.getV().isNear(w2.v()).  

  bool isParallel( const HepLorentzVector & w, 
                                        double epsilon=tolerance ) const;
  // Test for isParallel is within tolerance epsilon
  double howParallel (const HepLorentzVector & w) const;

  static double getTolerance();
  static double setTolerance( double tol );
  // Set the tolerance for HepLorentzVectors to be considered near
  // The same tolerance is used for determining isLightlike, and isParallel

  double deltaR(const HepLorentzVector & v) const;
  // sqrt ( (delta eta)^2 + (delta phi)^2 ) of space part

// 4 - Intrinsic properties 

         double howLightlike() const;
  // Close to zero for almost lightlike 4-vectors; up to 1.

  inline double euclideanNorm2()  const;
  // Sum of the squares of time and space components; not Lorentz invariant. 

  inline double euclideanNorm()  const; 
  // Length considering the metric as (+ + + +); not Lorentz invariant.


// 4a - Relativistic kinematic properties 

// All Relativistic kinematic properties are independent of the sense of metric

  inline double restMass2() const;
  inline double invariantMass2() const; 
  // Rest mass squared -- same as m2()

  inline double restMass() const;
  inline double invariantMass() const; 
  // Same as m().  If m2() is negative then -sqrt(-m2()) is returned.

// The following properties are rest-frame related, 
// and are applicable only to non-spacelike 4-vectors

  HepLorentzVector rest4Vector() const;
  // This 4-vector, boosted into its own rest frame:  (0, 0, 0, m()) 
          // The following relation holds by definition:
          // w.rest4Vector().boost(w.boostVector()) == w

  // Beta and gamma of the boost vector
  double beta() const;
  // Relativistic beta of the boost vector

  double gamma() const;
  // Relativistic gamma of the boost vector

  inline double eta() const;
  // Pseudorapidity (of the space part)

  inline double eta(const Hep3Vector & ref) const;
  // Pseudorapidity (of the space part) w.r.t. specified direction

  double rapidity(const Hep3Vector & ref) const;
  // Rapidity in specified direction

  double coLinearRapidity() const;
  // Rapidity, in the relativity textbook sense:  atanh (|P|/E)

  Hep3Vector findBoostToCM() const;
  // Boost needed to get to center-of-mass  frame:
          // w.findBoostToCM() == - w.boostVector()
          // w.boost(w.findBoostToCM()) == w.rest4Vector()

  Hep3Vector findBoostToCM( const HepLorentzVector & w ) const;
  // Boost needed to get to combined center-of-mass frame:
          // w1.findBoostToCM(w2) == w2.findBoostToCM(w1)
          // w.findBoostToCM(w) == w.findBoostToCM()

  inline double et2(const Hep3Vector &) const;
  // Transverse energy w.r.t. given axis squared.

  inline double et(const Hep3Vector &) const;
  // Transverse energy w.r.t. given axis.

// 4b - Methods combining two 4-vectors

  inline double diff2( const HepLorentzVector & w ) const;
  // (this - w).dot(this-w); sign depends on metric choice

  inline double delta2Euclidean ( const HepLorentzVector & w ) const;
  // Euclidean norm of differnce:  (delta_T)^2  + (delta_V)^2

// 5 - Properties releative to z axis and to arbitrary directions

  double  plus(  const Hep3Vector & ref ) const;
  // t + projection in reference direction

  double  minus( const Hep3Vector & ref ) const;
  // t - projection in reference direction

// 7 - Rotations and boosts

  HepLorentzVector & rotate ( const Hep3Vector & axis, double delta );
  // Same as rotate (delta, axis)

  HepLorentzVector & rotate ( const HepAxisAngle & ax );
  HepLorentzVector & rotate ( const HepEulerAngles & e );
  HepLorentzVector & rotate ( double phi,
                              double theta,
                              double psi );
  // Rotate using these HepEuler angles - see Goldstein page 107 for conventions

  HepLorentzVector & boost ( const Hep3Vector & axis,  double beta );
  // Normalizes the Hep3Vector to define a direction, and uses beta to
  // define the magnitude of the boost.

  friend HepLorentzVector rotationXOf
    ( const HepLorentzVector & vec, double delta );
  friend HepLorentzVector rotationYOf
    ( const HepLorentzVector & vec, double delta );
  friend HepLorentzVector rotationZOf
    ( const HepLorentzVector & vec, double delta );
  friend HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, const Hep3Vector & axis, double delta );
  friend HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, const HepAxisAngle & ax );
  friend HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, const HepEulerAngles & e );
  friend HepLorentzVector rotationOf
    ( const HepLorentzVector & vec, double phi,
                                    double theta,
                                    double psi );

  inline friend HepLorentzVector  boostXOf
    ( const HepLorentzVector & vec, double beta );
  inline friend HepLorentzVector  boostYOf
    ( const HepLorentzVector & vec, double beta );
  inline friend HepLorentzVector  boostZOf
    ( const HepLorentzVector & vec, double beta );
  inline friend HepLorentzVector  boostOf
    ( const HepLorentzVector & vec, const Hep3Vector & betaVector );
  inline friend HepLorentzVector  boostOf
    ( const HepLorentzVector & vec, const Hep3Vector & axis,  double beta );
 
private:

  Hep3Vector pp;
  double  ee;

  static double tolerance;
  static double metric;

};

// 8 - Axial Unit 4-vectors

static const HepLorentzVector X_HAT4 = HepLorentzVector( 1, 0, 0, 0 );
static const HepLorentzVector Y_HAT4 = HepLorentzVector( 0, 1, 0, 0 );
static const HepLorentzVector Z_HAT4 = HepLorentzVector( 0, 0, 1, 0 );
static const HepLorentzVector T_HAT4 = HepLorentzVector( 0, 0, 0, 1 );

// Global methods

std ::ostream & operator << (std ::ostream &, const HepLorentzVector &);
// Output to a stream.

std ::istream & operator >> (std ::istream &, HepLorentzVector &);
// Input from a stream.














typedef HepLorentzVector HepLorentzVectorD;
typedef HepLorentzVector HepLorentzVectorF;

inline HepLorentzVector operator * (const HepLorentzVector &, double a);
inline HepLorentzVector operator * (double a, const HepLorentzVector &);
// Scaling LorentzVector with a real number

       HepLorentzVector operator / (const HepLorentzVector &, double a);
// Dividing LorentzVector by a real number

// Tcomponent definition:

// Signature protection for 4-vector constructors taking 4 components
class Tcomponent {
private:
  double t_;
public:
  explicit Tcomponent(double t) : t_(t) {}
  operator double() const { return t_; }
};


# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.icc" 1
// -*- C++ -*-
// $Id$
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
// 
// This is the definitions of the inline member functions of the
// HepLorentzVector class.
//

# 1 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ZMxpv.h" 1



// ----------------------------------------------------------------------
//
//  ZMxpv.h     ZMexception's ZMthrown by classes in the PhysicsVectors
//              package.  To avoid name clashes, these start with ZMxpv.
//
//  THIS FILE CONTAINS TWO VERSIONS OF THE NECESSARY CODE:
//
//  With no special defines, this file will produce code for pure CLHEP 
//  building -- no ZOOM Exceptions are involved.
//
//  To force a build using ZOOM Exceptions where the ZMthorw macros appear,
//  compile with ENABLE_ZOOM_EXCEPTIONS defined.
//
// ----------------------------------------------------------------------

//#undef  ENABLE_ZOOM_EXCEPTIONS        // For CLHEP builds 
//#define ENABLE_ZOOM_EXCEPTIONS        // For ZOOM  builds



  // Member functions of the Vector classes are capable of ZMthrow-ing the
  // following ZMexception's:
  //
  //    ZMxPhysicsVectors       Severe  Parent exception of all ZMexceptions
  //                                    particular to classes in the package.
  //
  //    ZMxpvInfiniteVector     Error
  //                                    Mathematical operation will lead
  //                                    to infinity or NAN in a component
  //                                    of a result vector.
  //    ZMxpvZeroVector         Error
  //                                    A zero vector was used to specify
  //                                    a direction based on vector.unit().
  //    ZMxpvTachyonic          Error
  //                                    A relativistic kinematic function was
  //                                    taken, involving a vector representing
  //                                    a speed at or beyond that of light (=1).
  //    ZMxpvSpacelike          Error
  //                                    A spacelike 4-vector was used in a
  //                                    context where its restMass or gamma
  //                                    needs to be computed:  The result is
  //                                    formally imaginary (a zero result is
  //                                    supplied).
  //    ZMxpvInfinity           Error
  //                                    Mathematical operation will lead
  //                                    to infinity as a Scalar result.
  //    ZMxpvNegativeMass       Error
  //                                    Kinematic operation, e.g. invariant
  //                                    mass, rendered meaningless by an input
  //                                    with negative time component.
  //    ZMxpvVectorInputFails   Error
  //                                    Input to a SpaceVector or Lorentz
  //                                    Vector failed due to bad format or EOF.
  //    ZMxpvParallelCols       Error
  //                                    Purportedly orthogonal col's supplied
  //                                    to form a Rotation are exactly
  //                                    parallel instead.
  //    ZMxpvImproperRotation   Error
  //                                    Orthogonal col's supplied form a
  //                                    refection (determinant -1) more
  //                                    nearly than rather than a rotation.
  //    ZMxpvImproperTransformation Error
  //                                    Orthogonalized rows supplied form a
  //                                    tachyonic boost, a reflection, or
  //                                    a combination of those flaws,
  //                                    more nearly than a proper Lorentz
  //                                    transformation.
  //    ZMxpvFixedAxis          Error
  //                                    Attempt to change a RotationX,
  //                                    RotationY, or RotationZ in such a way
  //                                    that the axis might no longer be X,
  //                                    Y, or Z respectively.
  //    ZMxpvIndexRange         Error
  //                                    When using the syntax of v(i) to get
  //                                    a vector component, i is out of range.
  //    ZMxpvNotOrthogonal      Warning
  //                                    Purportedly orthogonal col's supplied
  //                                    to form a Rotation or LT are not
  //                                    orthogonal within the tolerance.
  //    ZMxpvNotSymplectic      Warning
  //                                    A row supplied to form a Lorentz
  //                                    transformation has a value of restmass
  //                                    incorrect by more than the tolerance:
  //                                    It should be -1 for rows 1-3,
  //                                    +1 for row 4.
  //    ZMxpvAmbiguousAngle     Warning
  //                                    Method involves taking an angle against
  //                                    a reference vector of zero length, or
  //                                    phi in polar coordinates of a vector
  //                                    along the Z axis.
  //    ZMxpvNegativeR          Warning
  //                                    R of a supplied vector is negative.
  //                                    The mathematical operation done is
  //                                    still formally valid.
  //    ZMxpvUnusualTheta       Warning
  //                                    Theta supplied to construct or set
  //                                    a vector is outside the range [0,PI].
  //                                    The mathematical operation done is
  //                                    still formally valid.  But note that
  //                                    when sin(theta) < 0, phi becomes an
  //                                    angle against the -X axis.
  //______________________________________________________________________



//  This is the CLHEP version.  When compiled for CLHEP, the basic CLHEP 
//  Vector classes will not (at least for now) depend on ZOOM Exceptions.  
//  Though this header lists the various sorts of Exceptions that could be 
//  thrown, ZMthrow.h in the pure CLHEP context will make ZMthrowA and 
//  ZMthrowC do what CLHEP has always done:  whine to cerr about the problem 
//  and exit (or continue in the ZMthrowC case).
//
//      If CLHEP ever embraces the ZOOM Exceptions mechanism, we will simply
//      modify this file.





























// endif for ifndef ENABLE_ZOOM_EXCEPTIONS 

// =============================================================
// =============================================================
// =============================================================

// ENABLE_ZOOM_EXCEPTIONS
# 196 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/ZMxpv.h"


// HEP_ZMXPV_H
# 11 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.icc" 2






inline double HepLorentzVector::x() const { return pp.x(); }
inline double HepLorentzVector::y() const { return pp.y(); }
inline double HepLorentzVector::z() const { return pp.z(); }
inline double HepLorentzVector::t() const { return ee; }

inline HepLorentzVector::
HepLorentzVector(double x, double y, double z, double t)
  : pp(x, y, z), ee(t) {}

inline HepLorentzVector:: HepLorentzVector(double x, double y, double z)
  : pp(x, y, z), ee(0) {}

inline HepLorentzVector:: HepLorentzVector(double t)
  : pp(0, 0, 0), ee(t) {}

inline HepLorentzVector:: HepLorentzVector()
  : pp(0, 0, 0), ee(0) {}

inline HepLorentzVector::HepLorentzVector(const Hep3Vector & p, double e)
  : pp(p), ee(e) {}

inline HepLorentzVector::HepLorentzVector(double e, const Hep3Vector & p)
  : pp(p), ee(e) {}

inline HepLorentzVector::HepLorentzVector(const HepLorentzVector & p)
  : pp(p.x(), p.y(), p.z()), ee(p.t()) {}

inline HepLorentzVector::~HepLorentzVector() {}

inline HepLorentzVector::operator const Hep3Vector & () const {return pp;}
inline HepLorentzVector::operator Hep3Vector & () { return pp; }

inline void HepLorentzVector::setX(double a) { pp.setX(a); } 
inline void HepLorentzVector::setY(double a) { pp.setY(a); }
inline void HepLorentzVector::setZ(double a) { pp.setZ(a); }
inline void HepLorentzVector::setT(double a) { ee = a;}

inline double HepLorentzVector::px() const { return pp.x(); }
inline double HepLorentzVector::py() const { return pp.y(); }
inline double HepLorentzVector::pz() const { return pp.z(); }
inline double HepLorentzVector::e()  const { return ee; }

inline void HepLorentzVector::setPx(double a) { pp.setX(a); } 
inline void HepLorentzVector::setPy(double a) { pp.setY(a); }
inline void HepLorentzVector::setPz(double a) { pp.setZ(a); }
inline void HepLorentzVector::setE(double a)  { ee = a;}

inline Hep3Vector HepLorentzVector::vect() const { return pp; } 
inline void HepLorentzVector::setVect(const Hep3Vector &p) { pp = p; } 

inline double HepLorentzVector::theta() const { return pp.theta(); }
inline double HepLorentzVector::cosTheta() const { return pp.cosTheta(); }
inline double HepLorentzVector::phi() const { return pp.phi(); }
inline double HepLorentzVector::rho() const { return pp.mag(); }

inline void HepLorentzVector::setTheta(double a) { pp.setTheta(a); }
inline void HepLorentzVector::setPhi(double a) { pp.setPhi(a); }
inline void HepLorentzVector::setRho(double a) { pp.setMag(a); }

double & HepLorentzVector::operator [] (int i)       { return (*this)(i); }
double   HepLorentzVector::operator [] (int i) const { return (*this)(i); }

inline HepLorentzVector &
HepLorentzVector::operator = (const HepLorentzVector & q) {
  pp = q.vect();
  ee = q.t();
  return *this;
}

inline HepLorentzVector
HepLorentzVector::operator + (const HepLorentzVector & q) const {
  return HepLorentzVector(x()+q.x(), y()+q.y(), z()+q.z(), t()+q.t());
}

inline HepLorentzVector &
HepLorentzVector::operator += (const HepLorentzVector & q) {
  pp += q.vect();
  ee += q.t();
  return *this;
}

inline HepLorentzVector
HepLorentzVector::operator - (const HepLorentzVector & q) const {
  return HepLorentzVector(x()-q.x(), y()-q.y(), z()-q.z(), t()-q.t());
}

inline HepLorentzVector &
HepLorentzVector::operator -= (const HepLorentzVector & q) {
  pp -= q.vect();
  ee -= q.t();
  return *this;
}

inline HepLorentzVector HepLorentzVector::operator - () const {
  return HepLorentzVector(-x(), -y(), -z(), -t());
}

inline HepLorentzVector& HepLorentzVector::operator *= (double a) {
  pp *= a;
  ee *= a;
  return *this;
}

inline bool
HepLorentzVector::operator == (const HepLorentzVector & q) const {
  return (vect()==q.vect() && t()==q.t());
}

inline bool
HepLorentzVector::operator != (const HepLorentzVector & q) const {
  return (vect()!=q.vect() || t()!=q.t());
}

inline double HepLorentzVector::perp2() const   { return pp.perp2(); }
inline double HepLorentzVector::perp()  const   { return pp.perp(); }
inline void HepLorentzVector::setPerp(double a) { pp.setPerp(a); }

inline double HepLorentzVector::perp2(const Hep3Vector &v) const {
  return pp.perp2(v);
}

inline double HepLorentzVector::perp(const Hep3Vector &v) const {
  return pp.perp(v);
}

inline double HepLorentzVector::angle(const Hep3Vector &v) const {
  return pp.angle(v);
}

inline double HepLorentzVector::mag2() const {
  return metric*(t()*t() - pp.mag2());
}

inline double HepLorentzVector::mag() const {
  double mm = m2();
  return mm < 0.0 ? -sqrt(-mm) : sqrt(mm);
}

inline double HepLorentzVector::m2() const { 
  return t()*t() - pp.mag2();
}

inline double HepLorentzVector::m() const { return mag(); }

inline double HepLorentzVector::mt2() const {
  return e()*e() - pz()*pz();
}

inline double HepLorentzVector::mt() const {
  double mm = mt2();
  return mm < 0.0 ? -sqrt(-mm) : sqrt(mm);
}

inline double HepLorentzVector::et2() const {
  double pt2 = pp.perp2();
  return pt2 == 0 ? 0 : e()*e() * pt2/(pt2+z()*z());
}

inline double HepLorentzVector::et() const {
  double etet = et2();
  return e() < 0.0 ? -sqrt(etet) : sqrt(etet);
}

inline double HepLorentzVector::et2(const Hep3Vector & v) const {
  double pt2 = pp.perp2(v);
  double pv = pp.dot(v.unit());
  return pt2 == 0 ? 0 : e()*e() * pt2/(pt2+pv*pv);
}

inline double HepLorentzVector::et(const Hep3Vector & v) const {
  double etet = et2(v);
  return e() < 0.0 ? -sqrt(etet) : sqrt(etet);
}

inline void 
HepLorentzVector::setVectMag(const Hep3Vector & spatial, double magnitude) {
  setVect(spatial);
  setT(sqrt(magnitude * magnitude + spatial * spatial));
}

inline void 
HepLorentzVector::setVectM(const Hep3Vector & spatial, double mass) {
  setVectMag(spatial, mass);
}

inline double HepLorentzVector::dot(const HepLorentzVector & q) const {
  return metric*(t()*q.t() - z()*q.z() - y()*q.y() - x()*q.x());
}

inline double
HepLorentzVector::operator * (const HepLorentzVector & q) const {
  return dot(q);
}

inline double HepLorentzVector::plus() const {
  return t() + z();
}

inline double HepLorentzVector::minus() const {
  return t() - z();
}

inline HepLorentzVector & HepLorentzVector::boost(const Hep3Vector & b) {
  return boost(b.x(), b.y(), b.z());
}

inline double HepLorentzVector::pseudoRapidity() const {
  return pp.pseudoRapidity();
}

inline double HepLorentzVector::eta() const {
  return pp.pseudoRapidity();
}

inline double HepLorentzVector::eta( const Hep3Vector & ref ) const {
  return pp.eta( ref );
}

inline HepLorentzVector &
HepLorentzVector::operator *= (const HepRotation & m) {
  pp *= m;
  return *this;
}

inline HepLorentzVector &
HepLorentzVector::transform(const HepRotation & m) {
  pp.transform(m);
  return *this;
}

inline HepLorentzVector operator * (const HepLorentzVector & p, double a) {
  return HepLorentzVector(a*p.x(), a*p.y(), a*p.z(), a*p.t());
}

inline HepLorentzVector operator * (double a, const HepLorentzVector & p) {
  return HepLorentzVector(a*p.x(), a*p.y(), a*p.z(), a*p.t());
}

// The following were added when ZOOM PhysicsVectors was merged in:

inline HepLorentzVector::HepLorentzVector( 
        double x, double y, double z, Tcomponent t ) :
        pp(x, y, z), ee(t) {}

inline void HepLorentzVector::set(
        double x, double y, double z, Tcomponent t ) {
  pp.set(x,y,z);
  ee = t;
}

inline void HepLorentzVector::set(
        double x, double y, double z, double t ) {
  set (x,y,z,Tcomponent(t));
}

inline HepLorentzVector::HepLorentzVector( 
        Tcomponent t, double x, double y, double z ) :
        pp(x, y, z), ee(t) {}   

inline void HepLorentzVector::set(
        Tcomponent t, double x, double y, double z ) {
  pp.set(x,y,z);
  ee = t;
}

inline void HepLorentzVector::set( Tcomponent t ) {
  pp.set(0, 0, 0);
  ee = t;
}

inline void HepLorentzVector::set( double t ) {
  pp.set(0, 0, 0);
  ee = t;
}

inline HepLorentzVector::HepLorentzVector( Tcomponent t ) : 
        pp(0, 0, 0), ee(t) {}

inline void HepLorentzVector::set( const Hep3Vector & v ) {
  pp = v;
  ee = 0;
}

inline HepLorentzVector::HepLorentzVector( const Hep3Vector & v ) : 
        pp(v), ee(0) {}

inline void HepLorentzVector::setV(const Hep3Vector & v) {
  pp = v;
}

inline HepLorentzVector & HepLorentzVector::operator=(const Hep3Vector & v) {
  pp = v;
  ee = 0;
  return *this;
}

inline double HepLorentzVector::getX() const { return pp.x(); }
inline double HepLorentzVector::getY() const { return pp.y(); }
inline double HepLorentzVector::getZ() const { return pp.z(); }
inline double HepLorentzVector::getT() const { return ee; }

inline Hep3Vector HepLorentzVector::getV() const { return pp; } 
inline Hep3Vector HepLorentzVector::v() const { return pp; } 

inline void HepLorentzVector::set(double t, const Hep3Vector & v) {
  pp = v;
  ee = t;
}

inline void HepLorentzVector::set(const Hep3Vector & v, double t) {
  pp = v;
  ee = t;
}

inline void HepLorentzVector::setV( double x,
             double y,
             double z ) { pp.set(x, y, z); }

inline void HepLorentzVector::setRThetaPhi 
                ( double r, double theta, double phi ) 
                         { pp.setRThetaPhi( r, theta, phi ); }

inline void HepLorentzVector::setREtaPhi 
                ( double r, double eta, double phi ) 
                         { pp.setREtaPhi( r, eta, phi ); }

inline void HepLorentzVector::setRhoPhiZ
                ( double rho, double phi, double z )
                         { pp.setRhoPhiZ ( rho, phi, z ); }

inline bool HepLorentzVector::isTimelike() const {
  return restMass2() > 0;
}  

inline bool  HepLorentzVector::isSpacelike() const {
  return restMass2() < 0;
}

inline bool  HepLorentzVector::isLightlike(double epsilon) const {
  return fabs(restMass2()) < 2.0 * epsilon * ee * ee;
}

inline double HepLorentzVector::diff2( const HepLorentzVector & w ) const {
    return metric*( (ee-w.ee)*(ee-w.ee) - (pp-w.pp).mag2() );
}

inline double HepLorentzVector::delta2Euclidean 
                                        ( const HepLorentzVector & w ) const {
    return (ee-w.ee)*(ee-w.ee) + (pp-w.pp).mag2();
}

inline double HepLorentzVector::euclideanNorm2()  const {
  return ee*ee + pp.mag2();
}

inline double HepLorentzVector::euclideanNorm()  const {
  return sqrt(euclideanNorm2());
}

inline double HepLorentzVector::restMass2()      const { return m2(); }
inline double HepLorentzVector::invariantMass2() const { return m2(); }

inline double HepLorentzVector::restMass() const {
    if( t() < 0.0 ) do { std ::cerr <<   
              "E^2-p^2 < 0 for this particle. Magnitude returned."    << "\n" << "at line " << 381 << " in file " << "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.icc" << "\n"; } while (0) ;
    return t() < 0.0 ? -m() : m();
}

inline double HepLorentzVector::invariantMass() const {
    if( t() < 0.0 ) do { std ::cerr <<   
              "E^2-p^2 < 0 for this particle. Magnitude returned."    << "\n" << "at line " << 387 << " in file " << "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.icc" << "\n"; } while (0) ;
    return t() < 0.0 ? -m() : m();
}

inline double HepLorentzVector::invariantMass2
                                        (const HepLorentzVector & w) const {
  return (*this + w).m2();
} /* invariantMass2 */

//-*********
// boostOf()
//-*********

// Each of these is a shell over a boost method.

inline HepLorentzVector boostXOf
        (const HepLorentzVector & vec, double beta) {
  HepLorentzVector vv (vec);
  return vv.boostX (beta);
}

inline HepLorentzVector boostYOf
        (const HepLorentzVector & vec, double beta) {
  HepLorentzVector vv (vec);
  return vv.boostY (beta);
}

inline HepLorentzVector boostZOf
        (const HepLorentzVector & vec, double beta) {
  HepLorentzVector vv (vec);
  return vv.boostZ (beta);
}

inline HepLorentzVector boostOf
        (const HepLorentzVector & vec, const Hep3Vector & betaVector ) {
  HepLorentzVector vv (vec);
  return vv.boost (betaVector);
}

inline HepLorentzVector boostOf
    (const HepLorentzVector & vec, const Hep3Vector & axis,  double beta) {
  HepLorentzVector vv (vec);
  return vv.boost (axis, beta);
}






# 583 "/afs/cern.ch/sw/lhcxx/specific/redhat61/gcc-2.95.2/4.0.4/include/CLHEP/Vector/LorentzVector.h" 2



/* HEP_LORENTZVECTOR_H */
# 13 "CARF/BaseSimEvent/interface/CoreSimVertex.h" 2



/**  a Vertex
 */
class CoreSimVertex {

public:
  ///
  CoreSimVertex(){}
  ///
  CoreSimVertex(float * g3vert, float tofg) {
    vert_[0] = g3vert[0];
    vert_[1] = g3vert[1]; 
    vert_[2] = g3vert[2];
    vert_[3] = tofg; 
  }


  ///
  inline HepLorentzVector position() const { 
    return HepLorentzVector(vert_[0],vert_[1],vert_[2],vert_[3]);
  }

private:

  float vert_[4];

};

# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 43 "CARF/BaseSimEvent/interface/CoreSimVertex.h" 2

ostream& operator <<(ostream& o , const CoreSimVertex& v); 


// CoreSimVertex_H
# 12 "CARF/BaseSimEvent/interface/EmbdSimVertex.h" 2



// forward declaration
class BaseSimEvent;

class EmbdSimTrack;

/**  a not so persistent Simulated Vertex
 */
class EmbdSimVertex : public CoreSimVertex {
public:

  typedef CoreSimVertex Core;
  typedef short int c_int16;
  typedef int       c_int32;

public:

  ///
  EmbdSimVertex();

  /// constructor
  EmbdSimVertex(float * g3vert, float tofg);
  /// full constructor
  EmbdSimVertex(float * g3vert, float tofg, int it);
  /// constructor from transient
  EmbdSimVertex(const CoreSimVertex& v,  int it);

  /// parent track
  // DELETE UNDEFINED const EmbdSimTrack * parent(const BaseSimEvent & imom) const;

  /// index of the parent in the BaseSimEvent SimTrack container (-1 if no parent)
  int parentIndex() const { return  itrack_;}
  /// 
  bool noParent() const { return  itrack_==-1;}

private:

  int itrack_;

};

# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 55 "CARF/BaseSimEvent/interface/EmbdSimVertex.h" 2

ostream& operator <<(ostream& o , const EmbdSimVertex& v);

// EmbdSIMVertex_H
# 26 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 2

# 1 "CARF/BaseSimEvent/interface/EmbdSimTrack.h" 1


//
//
// Persistent track
//
//  version     0.1   99/06/02
//                    persistent for G3carf'99
//  version     0.2   99/12/15
//                    name changed
//  version     0.3   00/02/14
//                    moved to int16 and simbody
//  version     0.4   01/11/18
//                    just embedded (requires contex: BaseSimEvent)


# 1 "CARF/BaseSimEvent/interface/CoreSimTrack.h" 1


//
//  a Core Simulated Track (not a great interface...)
//  version     0.1   98/10/03
//                    pure transient for G3carf'98
//  version     0.2   99/06/02
//                    abstract for G3carf'99
//  version     0.3   99/12/15
//                    still abstract
//  version     0.4   01/11/18
//                    just core infos, not abstraction..
//                      and use HepPDT...  

# 1 "/home/wmtan/root/cint/include/cmath" 1
namespace std {

#pragma include_noerr <stdcxxfunc.dll>
}

# 15 "CARF/BaseSimEvent/interface/CoreSimTrack.h" 2




class HepParticleData;

/**  a generic Simulated Track
 */
class CoreSimTrack {

public:
  ///
  CoreSimTrack(){}

  /// constructor
  CoreSimTrack(int ipart, float * g3p) :
    part_(ipart) {
    p_[0] = g3p[0];
    p_[1] = g3p[1];
    p_[2] = g3p[2];
    p_[3] = g3p[3];
  }

  CoreSimTrack(int ipart, const Hep3Vector & ip, double ie) :
    part_(ipart) {
    p_[0] = ip.x();
    p_[1] = ip.y();
    p_[2] = ip.z();
    p_[3] = ie;
   
  }
  
  /// particle info...
  const HepParticleData * particleInfo() const;

  /// four momentum
  inline HepLorentzVector momentum() const { 
    return HepLorentzVector(p_[0],p_[1],p_[2],p_[3]);
  }


  /// particle type (HEP PDT convension)
  inline int type() const { return part_;}

  /// charge
  float charge() const;

private:

  int part_;

  float p_[4];

};

# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 70 "CARF/BaseSimEvent/interface/CoreSimTrack.h" 2

ostream& operator <<(ostream& o , const CoreSimTrack& t); 


// CoreSIMTrack_H
# 17 "CARF/BaseSimEvent/interface/EmbdSimTrack.h" 2



// forward declaration
class BaseSimEvent;

class EmbdSimVertex;
class EmbdGenParticle;

/**  a not so persistent Simulated Track
 */
class EmbdSimTrack : public CoreSimTrack {

public:
  typedef CoreSimTrack Core;
  typedef short int c_int16;
  typedef int       c_int32;

  ///
  EmbdSimTrack();

  /// constructor
  EmbdSimTrack(int ipart, float * g3p);
  /// full constructor
  EmbdSimTrack(int ipart, float * g3p,  c_int16 iv, c_int16 ig);
  /// constructor from transient
  EmbdSimTrack(const CoreSimTrack& t,  c_int16 iv, c_int16 ig);

  /// vertex
  // DELETE UNDEFINED const  EmbdSimVertex * vertex(const BaseSimEvent & imom) const;
  /// the generator particle
  // DELETE UNDEFINED const  EmbdGenParticle * genparticle(const BaseSimEvent & imom) const;

  /// index of the vertex in the BaseSimEvent container (-1 if no vertex)
  c_int16 vertIndex() const { return ivert_;}
  ///
  bool  noVertex() const { return ivert_==-1;}

  ///index of the  corrsponding Generator particle in the BaseSimEvent container (-1 if no Genpart)
  c_int16 genpartIndex() const { return igenpart_;}
  ///
  bool  noGenpart() const { return igenpart_==-1;}


private:

  c_int16 ivert_;
  
  c_int16 igenpart_;
  
};

# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 69 "CARF/BaseSimEvent/interface/EmbdSimTrack.h" 2

ostream& operator <<(ostream& o , const EmbdSimTrack& t); 


// EmbdSIMTrack_H
# 27 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 2

# 1 "CARF/BaseSimEvent/interface/EmbdGenParticle.h" 1


//
//  a generator particle
//  version     0.1   99/02/09  (St.Wynhoff)
//                    pure transient for G3carf'98
//  version     0.2   00/02/14
//                    moved to int16
//  version     0.3  01/11/18
//                    just embedded (no id,  no mom2)
//  version     0.4  02/01/09
//                    mom2 back, no double, no dauthers

# 1 "/home/wmtan/root/cint/include/cmath" 1
namespace std {

#pragma include_noerr <stdcxxfunc.dll>
}

# 14 "CARF/BaseSimEvent/interface/EmbdGenParticle.h" 2







/**  a generator particle /
just a dump of a HepEvent line as is....
 */
class EmbdGenParticle {
public:
  typedef HepLorentzVector FourVector;
  typedef short int c_int16;
  typedef int       c_int32;

public:
  ///
  EmbdGenParticle(){}
  ///
  EmbdGenParticle(c_int32 pid, c_int16 istat, 
                  c_int16 imo1, c_int16 imo2, 
                 c_int16 ida1, c_int16 ida2, 
                  float px, float py, float pz, float e) :
    pid_(pid), status_(istat), 
    Mo1_(imo1), Mo2_(imo2),
    Da1_(ida1), Da2_(ida2)
  {
    p_[0] = px;
    p_[1] = py;
    p_[2] = pz;
    p_[3] = e;
  }
  ///
  EmbdGenParticle(c_int32 pid, c_int16 istat, 
                  c_int16 imo1, c_int16 imo2, 
                  c_int16 ida1, c_int16 ida2,
                  const HepLorentzVector & ip) :
    pid_(pid), status_(istat), 
    Mo1_(imo1), Mo2_(imo2) ,
    Da1_(ida1), Da2_(ida2)
  {
    p_[0] = ip.x();
    p_[1] = ip.y();
    p_[2] = ip.z();
    p_[3] = ip.t();
  }
  
  
  ///  
  c_int32 pid() const { return pid_;} 
  ///
  c_int16 status() const { return status_;}
  ///
  c_int16 mother1() const { return Mo1_;}
  ///
  c_int16 mother2() const { return Mo2_;}
  ///
  c_int16 daughter1() const { return Da1_;}
  ///
  c_int16 daughter2() const { return Da2_;}
  ///
  HepLorentzVector fourmomentum() const {
    return HepLorentzVector(p_[0],p_[1],p_[2],p_[3]);
  }
  
  ///
  void print() const;
  
private:
  
  ///
  c_int32 pid_;
  
  ///
  c_int16 status_;
  ///
  c_int16 Mo1_, Mo2_;
  
  ///
  c_int16 Da1_, Da2_;
  
  ///
  float p_[4];

};

# 1 "/home/wmtan/root/cint/include/iosfwd" 1
namespace std {

}
# 101 "CARF/BaseSimEvent/interface/EmbdGenParticle.h" 2

///
ostream& operator <<(ostream& o , const EmbdGenParticle& v); 


// EmbdGENPARTICLE_H
# 28 "CARF/BaseSimEvent/interface/BaseSimEvent.h" 2


// forward declaration;
class BaseSimEventProxy;

// forward declaration;
class TSimEvent;

// forward declaration;
class SimTrack;
// forward declaration;
class SimVertex;


/** an Abstract Class for a generic Simulated event.
interface to track, verteces...
 */
class BaseSimEvent : public CoreSimEvent {

public:
  typedef CoreSimEvent::Id  Id;

  typedef BaseSimEventProxy Proxy;
  typedef refc_ptr<BaseSimEventProxy> ProxyRef;

  typedef stdVectorConstInterface<EmbdSimTrack>  track_container;
  typedef stdVectorConstInterface<EmbdSimVertex> vertex_container;
  typedef stdVectorConstInterface<EmbdGenParticle> genpart_container;

  typedef track_container::const_iterator          track_iterator;
  typedef vertex_container::const_iterator         vertex_iterator;
  typedef genpart_container::const_iterator        genpart_iterator;

  typedef refc_ptr<track_container>  track_containerRef;
  typedef refc_ptr<vertex_container> vertex_containerRef;
  typedef refc_ptr<genpart_container> genpart_containerRef;

public:
  static track_containerRef st_tracks_;
  static vertex_containerRef st_vertices_;
  static genpart_containerRef st_genparts_;

  static void makeCaches(); 
  static void clearCaches() { st_tracks_->clear(); st_vertices_->clear(); st_genparts_->clear();}

public:
  /// default constructor
  BaseSimEvent() {}

  // constructor
  explicit BaseSimEvent(const CoreSimEvent::Id & iid, const TSimEvent& tev);

  /// virtual destructor 
  virtual ~BaseSimEvent();
 
  virtual ProxyRef proxy() const =0;

  /// return track container
  virtual track_containerRef tracks() const  = 0;
  /// return vertex container
  virtual vertex_containerRef vertices() const =0;

  /// return vertex container
  virtual genpart_containerRef genparts() const =0;

  /// return track with given id
  SimTrack track(int id) const;

  /// return track with given id
  virtual const  EmbdSimTrack & embdTrack(int id) const=0;

  /// return vertex with given id
  virtual SimVertex vertex(int id) const;

  /// return vertex with given id
  virtual const EmbdSimVertex & embdVertex(int id) const=0;

  ///
  virtual const EmbdGenParticle & embdGenpart(int i) const=0;

  /// print event;
  virtual void print(int level=1) const;


};

// BaseSIMEVENT_H
# 29 "CARF/SimEvent/interface/SimEvent.h" 2








// forward declaration;
class TSimEvent;

// forward declaration
class SmartRun;

// forward declaration
class RawEvent;

// forward declaration
class RawData;

// forward declaration
class ReadOutUnit;

// forward declaration
class GenEventBody;

class SimEventBody;



/**   a persistent Sim event 
 */
class SimEvent : public ooEvObj, public BaseSimEvent{

public:
  typedef SimEvent self;
  typedef BaseSimEvent base;
public:
   /// constructor
  SimEvent();


  /// construct from transient
  SimEvent(const Id & iid, const TSimEvent& tev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase());

  /// virtual destructor 
  virtual ~SimEvent();

  /// (deep) copy (if hint is valid, deep copy performed....
  SimEvent(const SimEvent & ev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());


  /// clone
  virtual opiRef< SimEvent >  clone(opiRefBase hint, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  ///
  void setRawEvent(const opiRef< RawEvent > & rw) { hits_ = rw;}

  /// return track with given id
  virtual const EmbdSimTrack & embdTrack(int i) const ;
  ///
  virtual const EmbdSimVertex & embdVertex(int i) const;
  ///
  virtual const EmbdGenParticle & embdGenpart(int i) const;

 
  virtual BaseSimEvent::ProxyRef proxy() const; 
   /// return track container
  virtual BaseSimEvent::track_containerRef tracks() const; 
  /// return vertex container
  /// return MC track container
  virtual BaseSimEvent::vertex_containerRef vertices() const; 

  virtual BaseSimEvent::genpart_containerRef genparts() const; 
 

  virtual const opiRef< SmartRun > & run() const;

  /// return hits (rawEvent...)
  const opiRef< RawEvent >  & rawEvent() const { return (*static_cast<const opiRef< RawEvent >*>(static_cast<const void *>(&( hits_ )))) ;}

  /// return hits for a given detector
  opiRef< RawData >  hits(const opiRef< ReadOutUnit > & roUnit);


protected:

  /// return body (GenEventBody...)
  const opiRef< GenEventBody >  & genbody() const { return (*static_cast<const opiRef< GenEventBody >*>(static_cast<const void *>(&( genbody_ )))) ;}

  /// return body (SimEventBody...)
  const opiRef< SimEventBody >  & body() const { return (*static_cast<const opiRef< SimEventBody >*>(static_cast<const void *>(&( body_ )))) ;}

  opiRefBase  genbody_;

  opiRefBase  body_;

  opiRefBase  hits_;

  private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   SimEvent  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   SimEvent  ::Streamer(b); } static const char *DeclFileName() { return "CARF/SimEvent/interface/SimEvent.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 128; } 
};


/* !defined(_SIM_EVENT_H) */
# 13 "CARF/SimEvent/interface/SimEventWithGen.h" 2



/** a SimEvent with generator infos
 */
class SimEventWithGen : public SimEvent {
public:
  typedef SimEventWithGen self;
  typedef SimEvent super;
  typedef BaseSimEvent base;

   /// constructor
  SimEventWithGen(){}


  /// construct from transient
  SimEventWithGen(const Id & iid, const TSimEvent& tev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase());

  /// virtual destructor 
  virtual ~SimEventWithGen();

  /// (deep) copy (if hint is valid, deep copy performed....
  SimEventWithGen(const SimEventWithGen & ev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  /// clone
  virtual opiRef< SimEvent >  clone(opiRefBase hint, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());
  
  private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   SimEventWithGen  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   SimEventWithGen  ::Streamer(b); } static const char *DeclFileName() { return "CARF/SimEvent/interface/SimEventWithGen.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 40; } 
};
/* !defined(_SIM_EVENT_WITH_GEN_H) */
# 13 "CARF/PythiaSimEvent/interface/PythiaSimEvent.h" 2

# 1 "CARF/PythiaSimEvent/interface/TPythiaSimEvent.h" 1


//
//
//   V 0.0 
//

# 1 "CARF/SimEvent/interface/TSimGenEvent.h" 1


//
//
//   V 0.1  VI 8/10/2000 
//          just an interface


/**
 */
class TSimGenEvent {
public:

  /// constructor
  TSimGenEvent(){}

  /// destructor
  virtual ~TSimGenEvent(){}

private:

};

// TSimGenEvent_H
# 8 "CARF/PythiaSimEvent/interface/TPythiaSimEvent.h" 2



/** speciifc Pythia stuff ...
 */
class TPythiaSimEvent : public TSimGenEvent {
public:

  /// constructor
  TPythiaSimEvent(){}

  /// constructor
  TPythiaSimEvent(float ip) : pthat_(ip){}

  /// destructor
  virtual ~TPythiaSimEvent(){}

  float pthat() const { return pthat_;}

private:

  float pthat_;

};

// TPythiaSimEvent_H
# 14 "CARF/PythiaSimEvent/interface/PythiaSimEvent.h" 2


/** a SimEvent with generator infos
 */
class PythiaSimEvent :  public SimEventWithGen, public TPythiaSimEvent {
public:
  typedef PythiaSimEvent self;
  typedef SimEventWithGen super;
  typedef BaseSimEvent base;

   /// constructor
  PythiaSimEvent(){}


  /// construct from transient
  PythiaSimEvent(const TPythiaSimEvent & pse, const Id & iid, const TSimEvent& tev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase());

  /// virtual destructor 
virtual ~PythiaSimEvent();

  /// (deep) copy (if hint is valid, deep copy performed....
  PythiaSimEvent(const PythiaSimEvent & ev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  /// clone
  virtual opiRef< SimEvent >  clone(opiRefBase hint, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   PythiaSimEvent  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   PythiaSimEvent  ::Streamer(b); } static const char *DeclFileName() { return "CARF/PythiaSimEvent/interface/PythiaSimEvent.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 40; } 
};
/* !defined(_PYTHIA_SIM_EVENT_H) */
# 1 "CARF/PythiaSimEvent/src/PythiaSimEventLinkDef.h" 2

# 1 "CARF/PythiaSimEvent/interface/BBPyth01SimEvent.h" 1
/* C++ header file: Objectivity/DB DDL version 6.1.0         */




//
//   Based on 
//   V 0.1  VI 8/10/2000 
//          a SimEvent with Phytia generator infos
//          MM 5/07/2001
//          Extended parameters for B physics




# 1 "CARF/PythiaSimEvent/interface/TBBPyth01SimEvent.h" 1


//
//
//   V 0.0 
//




/** speciifc Pythia stuff ...
 */
class TBBPyth01SimEvent : public TSimGenEvent {
public:

  /// constructor
  TBBPyth01SimEvent(){}

  /// constructor
  TBBPyth01SimEvent(float ip) : pthat_(ip){}

  /// constructor
  TBBPyth01SimEvent( int np, vector<float> parmc ) : npara_(np){
    pthat_ = parmc[0];
    xshat_ = parmc[1];
    weimc_ = parmc[2];
    isubp_ = parmc[3];
    nbhar_ = parmc[4];
    nbham_ = parmc[5];
    trflg_ = parmc[6];
  }

  /// destructor
  virtual ~TBBPyth01SimEvent(){}

  float nparam() const { return npara_;}
  float pthat()  const { return pthat_;}
  float xshat()  const { return xshat_;}
  float weimc()  const { return weimc_;}
  float isubp()  const { return isubp_;}
  float nbhar()  const { return nbhar_;}
  float nbham()  const { return nbham_;}
  float trflg()  const { return trflg_;}

private:

  int   npara_;
  float pthat_;
  float xshat_;
  float weimc_;
  float isubp_;
  float nbhar_;
  float nbham_;
  float trflg_;

};

// TBBPyth01SimEvent_H
# 16 "CARF/PythiaSimEvent/interface/BBPyth01SimEvent.h" 2


/** a SimEvent with generator infos
 */
class BBPyth01SimEvent :  public SimEventWithGen, public TBBPyth01SimEvent {
public:
  typedef BBPyth01SimEvent self;
  typedef SimEventWithGen super;
  typedef BaseSimEvent base;

   /// constructor
  BBPyth01SimEvent(){}


  /// construct from transient
  BBPyth01SimEvent(const TBBPyth01SimEvent & pse, const Id & iid, const TSimEvent& tev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase());

  /// virtual destructor 
virtual ~BBPyth01SimEvent();

  /// (deep) copy (if hint is valid, deep copy performed....
  BBPyth01SimEvent(const BBPyth01SimEvent & ev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  /// clone
  virtual opiRef< SimEvent >  clone(opiRefBase hint, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   BBPyth01SimEvent  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   BBPyth01SimEvent  ::Streamer(b); } static const char *DeclFileName() { return "CARF/PythiaSimEvent/interface/BBPyth01SimEvent.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 42; } 
};
/* !defined(_BBPYTH01_SIM_EVENT_H) */
# 2 "CARF/PythiaSimEvent/src/PythiaSimEventLinkDef.h" 2

# 1 "CARF/PythiaSimEvent/interface/MBPythSimEvent.h" 1
/* C++ header file: Objectivity/DB DDL version 6.1.0         */




//
//   Based on 
//   V 0.1  VI 8/10/2000 
//          a SimEvent with Phytia generator infos
//          MM 5/07/2001
//          Extended parameters for B physics
// Modification for Miminum Bias Studies L.S. 14/02/02
// 


# 1 "CARF/PythiaSimEvent/interface/TMBPythSimEvent.h" 1


//
//
//   V 0.0 
//   L 0.1




/** speciifc Minimum Bias Pythia stuff ...
    Information stored are pthat and MSEL and ISUB (Sub Process) values 
    used in the production
*/

class TMBPythSimEvent : public TSimGenEvent {
public:

  /// constructor
  TMBPythSimEvent(){}

  /// constructor
  TMBPythSimEvent(float ip) : pthat_(ip){}

  /// constructor
  TMBPythSimEvent( int np, vector<float> parmc ) : npara_(np){
    pthat_ = parmc[0];
    msel_ = parmc[1];
    subp_ = parmc[2];
  }

  /// destructor
  virtual ~TMBPythSimEvent(){}

  float nparam() const { return npara_;}
  float pthat()  const { return pthat_;}
  float msel()   const { return msel_;} 
  float subp()   const { return subp_;}

private:

  int   npara_;
  float pthat_;
  float msel_;
  float subp_;

};

// TBBPyth01SimEvent_H
# 16 "CARF/PythiaSimEvent/interface/MBPythSimEvent.h" 2


/** a SimEvent with generator infos
 */
class MBPythSimEvent :  public SimEventWithGen, public TMBPythSimEvent {
public:
  typedef MBPythSimEvent self;
  typedef SimEventWithGen super;
  typedef BaseSimEvent base;

   /// constructor
  MBPythSimEvent(){}


  /// construct from transient
  MBPythSimEvent(const TMBPythSimEvent & pse, const Id & iid, const TSimEvent& tev, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase());

  /// virtual destructor 
virtual ~MBPythSimEvent();

  /// (deep) copy (if hint is valid, deep copy performed....
  MBPythSimEvent(const MBPythSimEvent & ev,opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  /// clone
  virtual opiRef< SimEvent >  clone(opiRefBase hint, opiRefBase genHint=opiRefBase(), opiRefBase bodyHint=opiRefBase(), opiRefBase hitHint=opiRefBase());

  private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return   1  ; } static void Dictionary(); virtual TClass *IsA() const { return   MBPythSimEvent  ::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) {   MBPythSimEvent  ::Streamer(b); } static const char *DeclFileName() { return "CARF/PythiaSimEvent/interface/MBPythSimEvent.h"; } static int ImplFileLine(); static const char *ImplFileName();  static int DeclFileLine() { return 42; } 
};
/* !defined(_MBPYTH_SIM_EVENT_H) */
# 3 "CARF/PythiaSimEvent/src/PythiaSimEventLinkDef.h" 2




#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TPythiaSimEvent+;
#pragma link C++ class PythiaSimEvent+;
#pragma link C++ class BBPyth01SimEvent+;
#pragma link C++ class MBPythSimEvent+;

# 1 "/tmp/filesVpDZl_cint.cxx" 2



