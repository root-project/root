/* -*- C++ -*- */

/************************************************************************
 *
 * Copyright(c) 1995~2006  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/**
 **  Copyright (c) 1994-1995 Modena Software Inc.,
 **
 **  Permission to use, copy, modify, distribute and sell this software
 **  and its documentation for any purpose is hereby granted without fee,
 **  provided that the above copyright notice appear in all copies and
 **  that both that copyright notice and this permission notice appear
 **  in supporting documentation.  Modena Software, Inc. makes no 
 **  representations about the suitability of this software for any
 **  purpose.  It is provided "as is" without express or implied warranty.
 **
 **/

#ifndef  __cplusplus
#error Must use C++ for BSTRING.H
#endif

#ifndef __MBSTRING_H
#define __MBSTRING_H

extern "C" {
#include <ctype.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
}

#include <iostream.h>
#include <bool.h>

// bc4const 

#ifndef __CINT__
#ifdef __BC4_STL
#define __BC401_STL
#endif

#ifdef __BC401_STL
#define __BC401_const 
#else
#define __BC401_const const
#endif
#endif

// bndchk.h
#ifdef BOUNDS_CHECK
void check_bounds
	( 	int index, 
		int container_size, 
		int lineno,
		char *filename	);
#endif

// mexcept.h

#ifndef __CINT__
#define _THROW_NONE
#define _THROW_DOMAIN
#define _THROW_INVALIDARG
#define _THROW_LENGTH
#define _THROW_OUTRANGE
#define _THROW_RANGE
#define _THROW_OVERFLOW
#define _THROW_ALLOC
#define _THROW_CAST
#define _THROW_TYPEID
#define _THROW_ALLOC_LENGTH
#define _THROW_ALLOC_OUTRANGE
#define _THROW_LENGTH_OUTRANGE
#define _THROW_ALLOC_LENGTH_OUTRANGE
#endif

#include <vector.h>

#ifdef __MMULTITHREAD
#include "mutex.h"
#endif

const size_t NPOS  = (size_t)(-1);

enum capacity { default_size, reserve };

template<class charT>
struct  string_char_baggage {

    typedef charT char_type;

    //
    // for users to acquire the basic character type
    //
    // constraints functions
    //
    static void
    assign (char_type& c1, const char_type& c2) 
    {
        c1 = c2;
    }
    static bool
    eq (const char_type& c1, const char_type& c2) 
    {
        return (c1 == c2);
    }
    static bool
    ne (const char_type& c1, const char_type& c2) 
    {
        return !(c1 == c2);
    }
    static bool
    lt (const char_type& c1, const char_type& c2) 
    {
        return (c1 < c2);
    }
    static char_type
    eos () 
    {
        return char_type();     // the null character
    }
    static istream&
    char_in (istream& is, char_type& c) 
    {
        return is >> c;        // extractor for a char_type object
    }
    static ostream&
    char_out (ostream& os, char_type c) 
    {
        return os << c;        // inserter for a char_type object
    }
    static bool
    is_del (char_type c) 
    {
        // characteristic function for delimiters of char_type
        return isspace(c);
    }

    //
    // speed-up functions
    //
    static int
    compare (const char_type* s1, const char_type* s2, size_t n) 
    {
        for (size_t i = 0; i < n; ++i, ++s1, ++s2)
            if (ne(*s1, *s2))
            {
                return lt(*s1, *s2) ? -1 : 1;
            }
        return 0;
    }
    static size_t
    length (const char_type* s) 
    {
        size_t l = 0;
        while (ne(*s++, eos()))
            ++l;
        return l;
    }
    static char_type*
    copy (char_type* s1, const char_type* s2, size_t n) 
    {
        char_type* s = s1;
        for (size_t i = 0; i < n; ++i)
            assign(*++s1, *++s2);
        return s;
    }
};

struct string_char_baggage<char> {

    typedef char char_type;

    //
    // constraint member functions
    //
    static void
    assign (char_type& c1, const char_type& c2) 
    {
        c1 = c2;
    }
    static bool
    eq (const char_type& c1, const char_type& c2) 
    {
        return (c1 == c2);
    }
    static bool
    ne (const char_type& c1, const char_type& c2) 
    {
        return (c1 != c2);
    }
    static bool
    lt (const char_type& c1, const char_type& c2) 
    {
        return (c1 < c2);
    }
    static char_type
    eos () 
    {
        return 0;     // the null character
    }
    static istream&
    char_in (istream& is, char_type& c) 
    {
       // extractor for a char_type object
       // return is >> c;        // this does not work
       is.get(c);
       return is;
    }
    static ostream&
    char_out (ostream& os, char_type c) 
    {
        return os << c;        // inserter for a char_type object
    }
    static bool
    is_del (char_type c) 
    {
        // characteristic function for delimiters of char_type
        return isspace(c);
    }

    //
    // speed-up functions
    //
    static int
    compare (const char_type* s1, const char_type* s2, size_t n) 
    {
        return memcmp(s1, s2, n);
    }
    static size_t
    length (const char_type* s) 
    {
        return strlen(s);
    }
    static char_type*
    copy (char_type* s1, const char_type* s2, size_t n) 
    {
        // type cast required as memcpy returns void*
        return (char_type*)memcpy(s1, s2, n);
    }
};

/*
struct string_char_baggage<wchar_t> {

    typedef wchar_t char_type;

    static void
    assign (char_type& c1, const char_type& c2) 
    {
        c1 = c2;
    }
    static bool
    eq (const char_type& c1, const char_type& c2) 
    {
        return (c1 == c2);
    }
    static bool
    ne (const char_type& c1, const char_type& c2) 
    {
        return (c1 != c2);
    }
    static bool
    lt (const char_type& c1, const char_type& c2) 
    {
        return (c1 < c2);
    }
    static char_type
    eos () 
    {
        return 0;     // the null character
    }
    static istream&
    char_in (istream& is, char_type& c) 
    {
        return is >> c;        // extractor for a char_type object
    }
    static ostream&
    char_out (ostream& os, char_type c) 
    {
        return os << c;        // inserter for a char_type object
    }
    static bool
    is_del (char_type c) 
    {
        // characteristic function for delimiters of char_type
        // using templatized locale::isspace function
        return isspace(c);
    }

    //
    // speed-up functions
    //
    static int
    compare (const char_type* s1, const char_type* s2, size_t n) 
    {
        return wmemcmp(s1, s2, n);
    }
    static size_t
    length (const char_type* s) 
    {
        return wcslen(s);
        // May use Koshida's overloaded MSE function strlen(s)
    }
    static char_type*
    copy (char_type* s1, const char_type* s2, size_t n) 
    {
        return (char_type*)wmemcpy(s1, s2, n);
    }
};
*/

template <class charT>
class basic_string;

template <class charT>
class basic_string_ref {

//
// friend class declaration
//
friend class basic_string<charT>;

//
// typedef declarations
//
typedef  string_char_baggage<charT>  baggage_type;

//
// private data members
//
    charT*   ptr;
    size_t   len;
    size_t   res;

#ifdef __MMULTITHREAD
    mutex_arith<size_t, mutex>   count;   // reference count
#else
    size_t                       count;   // reference count
#endif

//
// private constructors and destructors
//
    basic_string_ref () ;

    basic_string_ref (size_t size, capacity cap)  ;

    basic_string_ref (const basic_string<charT>& str, size_t pos , size_t rlen)
                      ;

    basic_string_ref (const charT* s, size_t rlen, size_t rres) ;

    basic_string_ref (const charT* s, size_t n)  ;

    basic_string_ref (const charT* s)  ;

    basic_string_ref (charT c, size_t rep) ;

    basic_string_ref (const vector<charT>& vec) ;

    ~basic_string_ref ()  ;

    inline void
    delete_ptr ()  ;

    inline static
    charT
    eos ()  ;

    inline static
    void
    throwlength () ;

    inline static
    void
    throwrange () ;

};

template<class charT>
class basic_string {

private:

//
// typedef declaration
//
    typedef basic_string_ref<charT>   reference_class;
    typedef basic_string_ref<charT>*  reference_pointer;

//
// data member
//
    charT*             c_str_ptr;
    reference_pointer  reference;

//
// private member functions
//
    inline charT*
    point () ;

    inline size_t&
    len ()  ;

    inline size_t
    ref_count () const ;
    
    inline static
    charT
    eos () ;

    void
    assign_str (const charT* s, size_t slen) ;

    void
    append_str (const charT* s, size_t slen) ;

    void
    insert_str (size_t pos, const charT* s, size_t slen) 
                ;

    void
    replace_str (size_t xlen, size_t pos, const charT* s, size_t slen) 
                 ;

    int
    compare_str (size_t pos, const charT* str, size_t slen, size_t strlen) 
                 const ;

    size_t
    find_str (const charT* s, size_t pos, size_t len) const ;

    size_t
    rfind_str (const charT* s, size_t pos, size_t len) const ;

    size_t
    find_first_of_str (const charT* s, size_t pos, size_t len) const 
                      ;

    size_t
    find_last_of_str (const charT* s, size_t pos, size_t len) const 
                     ;

    size_t
    find_first_not_of_str (const charT* s, size_t pos, size_t len) const 
                           ;

    size_t
    find_last_not_of_str (const charT* s, size_t pos, size_t len) const 
                         ;


protected:

    basic_string (const charT* s, size_t rlen, size_t xlen) ;

    inline void
    delete_ref ()  ;

public:

    typedef  charT                       char_type;
    typedef  string_char_baggage<charT>  baggage_type;

    basic_string ()  ;

    basic_string (size_t size, capacity cap)  ;

    basic_string (const basic_string<charT>& str, size_t pos = 0, size_t n = NPOS)
                  ;

    basic_string (const charT* s, size_t n)  ;

    basic_string (const charT* s)  ;

    basic_string (charT c, size_t rep = 1)  ;

    basic_string (const vector<charT>& vec)  ;

    ~basic_string ()  ;

    basic_string<charT>&
    operator= (const basic_string<charT>& str)  ;

    basic_string<charT>&
    operator= (const charT* s)  ;

    basic_string<charT>&
    operator= (charT c)  ;

    basic_string<charT>&
    operator+= (const basic_string<charT>& rhs)  ;

    basic_string<charT>&
    operator+= (const charT* s)  ;

    basic_string<charT>&
    operator+= (charT c)  ;

    operator vector<charT> () const 
		{
			return vector<charT> (data(), data()+length());
		}


    basic_string<charT>&
    append (const basic_string<charT>& str, size_t pos = 0, size_t n = NPOS)
            ;

    basic_string<charT>&
    append (const charT* s, size_t n) ;

    basic_string<charT>&
    append (const charT* s) ;

    basic_string<charT>&
    append (charT c, size_t rep = 1) ;

    basic_string<charT>&
    assign (const basic_string<charT>& str, size_t pos = 0, size_t n = NPOS)
            ;

    basic_string<charT>&
    assign (const charT* s, size_t n) ;

    basic_string<charT>&
    assign (const charT* s)  ;

    basic_string<charT>&
    assign (charT c, size_t rep = 1)  ;

    basic_string<charT>&
    insert (size_t pos1, const basic_string<charT>& str, size_t pos2 = 0,
            size_t n = NPOS) ;

    basic_string<charT>&
    insert (size_t pos, const charT* s, size_t n) ;

    basic_string<charT>&
    insert (size_t pos, const charT* s) ;

    basic_string<charT>&
    insert (size_t pos, charT c, size_t rep = 1) ;

    basic_string<charT>&
    remove (size_t pos = 0, size_t n = NPOS) ;

    basic_string<charT>&
    replace (size_t pos1, size_t n1, const basic_string<charT>& str, size_t pos2 = 0,
             size_t n2 = NPOS) ;

    basic_string<charT>&
    replace (size_t pos, size_t n1, const charT* s, size_t n2)
             ;

    basic_string<charT>&
    replace (size_t pos, size_t n1, const charT* s)
             ;

    basic_string<charT>&
    replace (size_t pos, size_t n, charT c, size_t rep = 1)
             ;

    inline charT
    get_at (size_t pos) const ;

    void
    put_at (size_t pos, charT c) ;

    inline charT
    operator[] (size_t pos) const ;

    charT&
    operator[] (size_t pos) ;

    const charT*
    c_str () const ;

    inline const charT*
    data () const ;

    inline size_t
    length () const ;

    void
    resize (size_t n, charT c) ;

    void
    resize (size_t n) ;

    inline size_t
    reserve () const ;

    void
    reserve (size_t res_arg) ;

    size_t
    copy (charT* s, size_t n, size_t pos = 0) const ;

    size_t
    find (const basic_string<charT>& str, size_t pos = 0) const ;

    size_t
    find (const charT* s, size_t pos, size_t n) const ;

    size_t
    find (const charT* s, size_t pos = 0) const ;

    size_t
    find (charT c, size_t pos = 0) const ;

    size_t
    rfind (const basic_string<charT>& str, size_t pos = NPOS) const ;

    size_t
    rfind (const charT* s, size_t pos, size_t n) const ;

    size_t
    rfind (const charT* s, size_t pos = NPOS) const ;

    size_t
    rfind (charT c, size_t pos = NPOS) const ;

    size_t
    find_first_of (const basic_string<charT>& str, size_t pos = 0) const ;

    size_t
    find_first_of (const charT* s, size_t pos, size_t n) const ;

    size_t
    find_first_of (const charT* s, size_t pos = 0) const ;

    size_t
    find_first_of (charT c, size_t pos = 0) const ;


    size_t
    find_last_of (const basic_string<charT>& str, size_t pos = NPOS) const
                  ;

    size_t
    find_last_of (const charT* s, size_t pos, size_t n) const ;

    size_t
    find_last_of (const charT* s, size_t pos = NPOS) const ;

    size_t
    find_last_of (charT c, size_t pos = NPOS) const ;

    size_t
    find_first_not_of (const basic_string<charT>& str, size_t pos = 0) const
                       ;

    size_t
    find_first_not_of (const charT* s, size_t pos, size_t n) const ;

    size_t
    find_first_not_of (const charT* s, size_t pos = 0) const ;

    size_t
    find_first_not_of (charT c, size_t pos = 0) const ;

    size_t
    find_last_not_of (const basic_string<charT>& str, size_t pos = NPOS) const
                      ;

    size_t
    find_last_not_of (const charT* s, size_t pos, size_t n) const ;

    size_t
    find_last_not_of (const charT* s, size_t pos = NPOS) const ;

    size_t
    find_last_not_of (charT c, size_t pos = NPOS) const ;

    basic_string<charT>
    substr (size_t pos = 0,  size_t n = NPOS) const ;

    int
    compare (const basic_string<charT>& str, size_t pos = 0, size_t n = NPOS) const
             ;

    int
    compare (const charT* s, size_t pos, size_t n) const
             ;

    int
    compare (const charT* s, size_t pos = 0) const ;

    int
    compare (charT c, size_t pos = 0, size_t rep = 1) const 
             ;

    friend
    ostream&
    operator<< (ostream& o, const basic_string<charT>& s) ;

    friend
    istream&
    operator>> (istream& i, basic_string<charT>& s) ;

    friend
    basic_string<charT>
    operator+ (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
    ;

    friend
    basic_string<charT>
    operator+ (const charT* lhs, const basic_string<charT>& rhs)
    ;

    friend
    basic_string<charT>
    operator+ (charT lhs, const basic_string<charT>& rhs) ;

    friend
    basic_string<charT>
    operator+ (const basic_string<charT>& lhs, const charT* rhs)
    ;

    friend
    basic_string<charT>
    operator+ (const basic_string<charT>& lhs, charT rhs) ;

};

template <class charT>
inline void
basic_string_ref<charT>::delete_ptr () 
{
    if (res)
    {
        delete[] ptr;
        res = 0;
        ptr = 0;
    }
}

template <class charT>
inline void
basic_string_ref<charT>::throwlength ()  
{
#ifdef  __MEXCEPT
   throw   length_error("Length exception occurred");
#else
   cout << "Length exception occurred" << endl;
   exit(1);
#endif
}

template <class charT>
inline void
basic_string_ref<charT>::throwrange () 
{
#ifdef  __MEXCEPT
   throw   out_of_range("Out of range exception occurred");
#else
   cout << "Out of range exception occurred" << endl;
   exit(1);
#endif
}

template <class charT>
inline void
basic_string<charT>::delete_ref () 
{
    --(reference->count);
    if (!(reference->count))
       delete reference;
}

template <class charT>
inline size_t
basic_string<charT>::ref_count () const 
{
    return reference->count;
}

template <class charT>
inline const charT*
basic_string<charT>::data () const 
{
    if (length())
        return reference->ptr;
    else
        return 0;
}

template <class charT>
inline charT*
basic_string<charT>::point () 
{
    return reference->ptr;
}

template <class charT>
inline size_t&
basic_string<charT>::len () 
{
    return reference->len;
}

template <class charT>
inline size_t
basic_string<charT>::length () const 
{
    return reference->len;
}

template <class charT>
inline size_t
basic_string<charT>::reserve () const 
{
    return reference->res;
}

template <class charT>
inline charT
basic_string<charT>::get_at (size_t pos) const 
{
    if (pos >= length())
    {
        reference_class::throwrange();
    }
    return *(data()+pos);
}

template <class charT>
inline charT
basic_string<charT>::operator[] (size_t pos) const 
{
    if (pos < length())
        return *(data()+pos);
    else
        return 0;
}

template <class charT>
inline bool
operator== (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
            
{
    return !(lhs.compare(rhs));
}

template <class charT>
inline bool
operator== (const charT* lhs, const basic_string<charT>& rhs) 
{
    return !(rhs.compare(lhs));
}

template <class charT>
inline bool
operator== (charT lhs, const basic_string<charT>& rhs) 
{
    return !(rhs.compare(lhs));
}

template <class charT>
inline bool
operator== (const basic_string<charT>& lhs, const charT* rhs) 
{
    return !(lhs.compare(rhs));
}

template <class charT>
inline bool
operator== (const basic_string<charT>& lhs, charT rhs) 
{
    return !(lhs.compare(rhs));
}

#ifdef __MNONDEF
template <class charT>
inline bool
operator!= (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
            
{
    return lhs.compare(rhs);
}
#endif

template <class charT>
inline bool
operator!= (const charT* lhs, const basic_string<charT>& rhs) 
{
    return rhs.compare(lhs);
}

template <class charT>
inline bool
operator!= (charT lhs, const basic_string<charT>& rhs) 
{
    return rhs.compare(lhs);
}

template <class charT>
inline bool
operator!= (const basic_string<charT>& lhs, const charT* rhs) 
{
    return lhs.compare(rhs);
}

template <class charT>
inline bool
operator!= (const basic_string<charT>& lhs, charT rhs) 
{
    return lhs.compare(rhs);
}

template <class charT>
inline bool
operator< (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
           
{
    if (lhs.compare(rhs) < 0)
        return true;
    else
        return false;
}

template <class charT>
inline bool
operator< (const charT* lhs, const basic_string<charT>& rhs) 
{
    if (rhs.compare(lhs) > 0)
        return true;
    else
        return false;
}

template <class charT>
inline bool
operator< (charT lhs, const basic_string<charT>& rhs) 
{
    if (rhs.compare(lhs) > 0)
        return true;
    else
        return false;
}

template <class charT>
inline bool
operator< (const basic_string<charT>& lhs, const charT* rhs) 
{
    if (lhs.compare(rhs) < 0)
        return true;
    else
        return false;
}

template <class charT>
inline bool
operator< (const basic_string<charT>& lhs, charT rhs) 
{
    if (lhs.compare(rhs) < 0)
        return true;
    else
        return false;
}

#ifdef  __MNONDEF
template <class charT>
inline bool
operator> (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
           
{
    return (rhs < lhs);
}
#endif

template <class charT>
inline bool
operator> (const charT* lhs, const basic_string<charT>& rhs) 
{
    return (rhs < lhs);
}

template <class charT>
inline bool
operator> (charT lhs, const basic_string<charT>& rhs) 
{
    return (rhs < lhs);
}

template <class charT>
inline bool
operator> (const basic_string<charT>& lhs, const charT* rhs) 
{
    return (rhs < lhs);
}

template <class charT>
inline bool
operator> (const basic_string<charT>& lhs, charT rhs) 
{
    return (rhs < lhs);
}

#ifdef  __MNONDEF
template <class charT>
inline bool
operator>= (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
            
{
    return !(lhs < rhs);
}
#endif

template <class charT>
inline bool
operator>= (const charT* lhs, const basic_string<charT>& rhs) 
{
    return !(lhs < rhs);
}

template <class charT>
inline bool
operator>= (charT lhs, const basic_string<charT>& rhs) 
{
    return !(lhs < rhs);
}

template <class charT>
inline bool
operator>= (const basic_string<charT>& lhs, const charT* rhs) 
{
    return !(lhs < rhs);
}

template <class charT>
inline bool
operator>= (const basic_string<charT>& lhs, charT rhs) 
{
    return !(lhs < rhs);
}

#ifdef  __MNONDEF
template <class charT>
inline bool
operator<= (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
            
{
    return !(rhs < lhs);
}
#endif

template <class charT>
inline bool
operator<= (const charT* lhs, const basic_string<charT>& rhs) 
{
    return !(rhs < lhs);
}

template <class charT>
inline bool
operator<= (charT lhs, const basic_string<charT>& rhs) 
{
    return !(rhs < lhs);
}

template <class charT>
inline bool
operator<= (const basic_string<charT>& lhs, const charT* rhs) 
{
    return !(rhs < lhs);
}

template <class charT>
inline bool
operator<= (const basic_string<charT>& lhs, charT rhs) 
{
    return !(rhs < lhs);
}

// definitions : can be in a .c file
//

template <class charT>
charT
basic_string_ref<charT>::eos () 
{
    return baggage_type::eos();
}

template <class charT>
basic_string_ref<charT>::basic_string_ref () 
{
     res = len = 0;
     ptr = 0;
     count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (size_t size, capacity cap)
                                           
{
    if (cap == ::reserve)
    {
       len = 0;
       res = size;
       ptr = new charT [res];
    }
    else if ((cap == ::default_size) && (size != NPOS))
    {
       res = len = size;
       if (res)
       {
          ptr = new charT [res];
          for (size_t position = 0; position < len; ++position)
              baggage_type::assign (*(ptr+position), eos());
       }
       else
          ptr = 0;
    }
    else
    {
       throwlength();
    }
    count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (const basic_string<charT>& str,
                                           size_t pos, size_t rlen) 
{
    res = len = rlen;
    if (res)
    {
       ptr = new charT [res];
       baggage_type::copy (ptr, str.data()+pos, len);
    }
    else
       ptr = 0;
    count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (const charT* s, size_t rlen,
                                           size_t rres) 
{
    res = rres;
    len = rlen;
    if (res)
    {
       ptr = new charT [res];
       if (len)
          baggage_type::copy (ptr, s, len);
    }
    else
       ptr = 0;
    count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (const charT* s, size_t n)
                                           
{
    if (n == NPOS)
    {
       throwlength();
    }
    res = len = n;
    if (res)
    {
       ptr = new charT [res];
       baggage_type::copy (ptr, s, len);
    }
    else
       ptr = 0;
    count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (const charT* s) 
{
    res = len = baggage_type::length(s);
    if (res)
    {
       ptr = new charT [res];
       baggage_type::copy (ptr, s, len);
    }
    else
       ptr = 0;
    count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (charT c, size_t rep)
                                           
{
    if (rep == NPOS)
    {
       throwlength();
    }
    res = len = rep;
    if (res)
    {
       ptr = new charT [res];
       for (size_t  position = 0; position < len; ++position)
            baggage_type::assign (*(ptr+position), c);
    }
    else
       ptr = 0;
    count = 1;
}

template <class charT>
basic_string_ref<charT>::basic_string_ref (const vector<charT>& vec)
                                           
{
    size_t  n = vec.size();
    if (n == NPOS)
    {
        throwlength();
    }
    res = len = n;
    if (res)
    {
       ptr = new charT [res];
       baggage_type::copy (ptr, vec.begin(), len);
    }
    else
       ptr = 0;
    count = 1;
}

template <class charT>
basic_string_ref<charT>::~basic_string_ref () 
{
    delete_ptr();
}

template <class charT>
charT
basic_string<charT>::eos () 
{
    return baggage_type::eos();
}

template <class charT>
void
basic_string<charT>::assign_str (const charT* s, size_t slen)
                                 
{
    if (slen == NPOS)
    {
        reference_class::throwlength();
    }
    if ((ref_count() > 1) || (slen && (reserve() < slen)))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (s, slen);
        delete_ref();
        reference = tmp;
    }
    else if (slen)
    {
        baggage_type::copy (point(), s, slen);
    }
    reference->len = slen;
}

template <class charT>
void
basic_string<charT>::append_str (const charT* s, size_t slen)
                                 
{
#ifndef __CINT__ // ???
    if (length() >= (NPOS-slen))
    {
        reference_class::throwlength();
    }
#endif
    if ((ref_count() > 1) || (slen > (reserve()-length())))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), length(), length()+slen);
        delete_ref();
        reference = tmp;
    }
    if (slen)
        baggage_type::copy (point()+length(), s, slen);
    reference->len += slen;
}

template <class charT>
void
basic_string<charT>::insert_str (size_t pos, const charT* s, size_t slen)
                                 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
#ifndef __CINT__ // ???
    if (length() >= (NPOS-slen))
    {
        reference_class::throwlength();
    }
#endif
    if ((ref_count() > 1) || (slen > (reserve()-length())))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), pos, length()+slen);
        baggage_type::copy (tmp->ptr+pos+slen, data()+pos, length()-pos);
        tmp->len = length();
        delete_ref();
        reference = tmp;
    }
    else
    {
        for (size_t count = length()-pos; count > 0; --count)
             baggage_type::assign (*(point()+pos+slen+count-1),
                                   *(data()+pos+count-1));
    }
    if (slen)
        baggage_type::copy (point()+pos, s, slen);
    reference->len += slen;
}

template <class charT>
void
basic_string<charT>::replace_str (size_t xlen, size_t pos, const charT* s,
                     size_t slen) 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    if ((length()-xlen) >= (NPOS-slen))
    {
        reference_class::throwlength();
    }
    if ((ref_count() > 1) || (reserve() < (length()+slen-xlen)))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), pos, length()+slen-xlen);
        baggage_type::copy (tmp->ptr+pos+slen, data()+pos+xlen,
                            length()-pos-xlen);
        tmp->len = length();
        delete_ref();
        reference = tmp;
    }
    else 
    {
        if (slen < xlen)
            baggage_type::copy (point()+pos+slen, data()+pos+xlen,
                                length()-pos-xlen);
        else
        {
            for (size_t count = length()-pos-xlen; count > 0; --count)
                baggage_type::assign (*(point()+pos+slen+count-1),
                                      *(data()+pos+xlen+count-1));
        }
    }
    if (slen)
        baggage_type::copy (point()+pos, s, slen);
    reference->len += (slen-xlen);
}

template <class charT>
int
basic_string<charT>::compare_str (size_t pos, const charT* str, size_t slen,
                     size_t strlen) const 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
#ifdef __CINT__ // ???
    size_t rlen = (slen > strlen ) ?  slen : strlen;
#else
    size_t rlen = (slen > strlen ) ?  strlen  : slen;
#endif
    int result;
    if (!length())
        return str ? (eos()- *str) : eos();
    result =  baggage_type::compare (data()+pos, str, rlen);
    return result ? result : (length()-pos-strlen);
}

template <class charT>
size_t
basic_string<charT>::find_str (const charT* s, size_t pos, size_t len) 
                               const 
{
    size_t  count = pos;
    size_t  shift;
    size_t  place;
    if ((length() == 0) || (len == 0))
        return NPOS;
    while (len <= (length()-count))
    {
        for (place = 0; place < len; ++place)
        {
            if (baggage_type::ne(*(s+len-1-place), *(data()+count+(len-1-place))))
                break;
        }
        if (place == len)
            return count;
        shift = find(*(s+len-1-place), count+(len-place));
        if (shift == NPOS)
            return NPOS;
        count = shift-(len-place-1);
    }
    return NPOS;
}

template <class charT>
size_t
basic_string<charT>::rfind_str (const charT* s, size_t pos, size_t len) 
                                const 
{
    size_t  count = (pos < (length()-len)) ? (pos+1) : (length()-len);
    size_t  shift;
    size_t  place;    
    if ((length() < len) || (len == 0))
        return NPOS;
    while (count > 0)
    {
        for (place = 0; place < len; ++place)
        {
            if (baggage_type::ne(*(s+len-1-place), *(data()+count+(len-place)-2)))
                break;
        }
        if (place == len)
            return count-1;
        shift = rfind(*(s+len-1-place), count+(len-place)-3);
        if (shift == NPOS)
            return NPOS;
        count = shift+place-len+2;
    }
    return NPOS;

}

template <class charT>
size_t
basic_string<charT>::find_first_of_str (const charT* s, size_t pos, size_t len) 
                                        const 
{
    size_t temp;
    size_t count  = pos;
    size_t result = NPOS;
    while (count < length())
    {
        temp = 0;
        while ((temp < len) && baggage_type::ne(*(data()+count), *(s+temp)))
            ++temp;
        if (temp != len)
            break;
        ++count;
    }
    temp = (count >= length()) ? NPOS : count;
    return ((result > temp) ? temp : result);
}

template <class charT>
size_t
basic_string<charT>::find_last_of_str (const charT* s, size_t pos, size_t len) 
                                       const 
{
    size_t temp = 0;
    size_t count = (pos < length()) ? (pos+1) : length();
    if (length())
    {
       while (count > 0)
       {
           temp = 0;
           --count;
           while ((temp != len) && baggage_type::ne(*(data()+count), *(s+temp)))
               ++temp;
           if (temp != len)
               break;
       }
    }
    return ((temp != len) && length()) ? count : NPOS;
}

template <class charT>
size_t
basic_string<charT>::find_first_not_of_str (const charT* s, size_t pos,
                     size_t len) const 
{
    size_t count = pos;
    while (count < length())
    {
        size_t temp  = 0;
        while (temp < len)
        {
            if (baggage_type::eq(*(data()+count), *(s+temp)))
                break;
            ++temp;
        }
        if (temp == len)
            break;
        ++count;
    }
    return  ((count >= length()) ? NPOS : count);
}

template <class charT>
size_t
basic_string<charT>::find_last_not_of_str (const charT* s, size_t pos,
                     size_t len) const 
{
    size_t temp = 0;
    size_t count = (pos < length()) ? (pos+1) : length();

    if (length())
    {
        while (count > 0)
        {
           temp = 0;
           while (temp != len)
           {
               if (baggage_type::eq(*(data()+count-1), *(s+temp)))
                   break;
               ++temp;
           }
           if (temp == len)
               break;
           --count;
       }
    }
    return ((temp == len) && length()) ? count-1 : NPOS;
}

template <class charT>
basic_string<charT>::basic_string () 
{
    reference = new basic_string_ref<charT> ();
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (size_t size, capacity cap)
                                   
{
    reference = new basic_string_ref<charT> (size, cap);
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (const basic_string<charT>& str,
                                   size_t pos, size_t n) 
{
    if (pos > str.length())
    {
       reference_class::throwrange();
    }
#ifdef __CINT__ // ???
    size_t rlen =  (n > (str.length() - pos)) ? n : str.length() - pos ;
    if ((rlen == str.length()) && (str.ref_count() != NPOS)) {
       reference = str.reference;
       reference->count++;
    }
    else
        reference = new basic_string_ref<charT> (str, pos, rlen);
#else
    size_t rlen =  (n > (str.length() - pos)) ? str.length() - pos : n;
    if ((rlen == str.length()) && (str.ref_count() != NPOS))
       (reference = str.reference)->count++;
    else
        reference = new basic_string_ref<charT> (str, pos, rlen);
#endif
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (const charT* s, size_t rlen, size_t xlen)
                                   
{
#ifndef __CINT__
    if (rlen >= (NPOS - xlen))
    {
        reference_class::throwlength();
    }
#endif
    reference = new basic_string_ref<charT> (s, rlen, rlen+xlen);
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (const charT* s, size_t n) 
{
    reference = new basic_string_ref<charT> (s, n);
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (const charT* s) 
{
    reference = new basic_string_ref<charT> (s);
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (charT c, size_t rep) 
{
    reference = new basic_string_ref<charT> (c, rep);
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::basic_string (const vector<charT>& vec) 
{
    reference = new basic_string_ref<charT> (vec);
    c_str_ptr = 0;
}

template <class charT>
basic_string<charT>::~basic_string () 
{
    delete_ref();
    if (c_str_ptr)
        delete[] c_str_ptr;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::operator= (const basic_string<charT>& str) 
{
    if (this != &str)
    {
        delete_ref();
#ifdef __CINT__
        if (str.ref_count() != NPOS) {
           reference = str.reference;
           reference->count++;
	}
        else
            reference = new basic_string_ref<charT> (str, 0, str.length());
#else
        if (str.ref_count() != NPOS)
           (reference = str.reference)->count++;
        else
            reference = new basic_string_ref<charT> (str, 0, str.length());
#endif
    }
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::operator= (const charT* s) 
{
    assign_str (s, baggage_type::length(s));
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::operator= (charT c) 
{
    if ((ref_count() == 1) && (reserve() >= 1))
    {
        baggage_type::assign (*(point()), c);
        reference->len = 1;
    }
    else
    {
        delete_ref();
        reference = new basic_string_ref<charT> (c, 1);
    }
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::operator+= (const basic_string<charT>& rhs) 
{
    append_str (rhs.data(), rhs.length());
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::operator+= (const charT* s) 
{
    append_str (s, baggage_type::length(s));
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::operator+= (charT c) 
{
    if (length() >= (NPOS-1))
    {
        reference_class::throwlength();
    }
    if (!((ref_count() == 1) && (reserve() > length())))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), length(), length()+1);
        delete_ref();
        reference = tmp;
    }
    baggage_type::assign (*(point()+length()), c);
    reference->len++;
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::append (const basic_string<charT>& str, size_t pos, size_t n)
                     
{
    if (pos > str.length())
    {
        reference_class::throwrange();
    }
    append_str (str.data() + pos, (n>(str.length()-pos))?(str.length()-pos):n);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::append (const charT* s, size_t n) 
{
    append_str (s, n);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::append (const charT* s) 
{
    append_str (s, baggage_type::length(s));
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::append (charT c, size_t rep) 
{
    if (length() >= (NPOS-rep))
    {
        reference_class::throwlength();
    }
    if (rep)
    {
       if ((ref_count() > 1) || (reserve() < (length() + rep)))
       {
          reference_pointer tmp;
          tmp = new basic_string_ref<charT> (data(), length(), length()+rep);
          delete_ref();
          reference = tmp;
       }
       for (size_t count = 0; count < rep; ++count)
            baggage_type::assign (*(point()+length()+count), c);
       reference->len += rep;
    }
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::assign (const basic_string<charT>& str, size_t pos, size_t n)
                             
{
    if (pos > str.length())
    {
        reference_class::throwrange();
    }
    size_t rlen = (n > (str.length() - pos)) ? str.length() - pos : n;
    if ((rlen == str.length()) && (str.ref_count() != NPOS))
    {
       delete_ref();
       (reference = str.reference)->count++;
    }
    else
       assign_str (str.data()+pos, rlen);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::assign (const charT* s, size_t n) 
{
    assign_str (s, n);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::assign (const charT* s) 
{
    assign_str (s, baggage_type::length(s));
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::assign (charT c, size_t rep) 
{
    if (rep == NPOS)
    {
        reference_class::throwlength();
    }
    if ((ref_count() > 1) || (rep && (reserve() < rep)))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (c, rep);
        delete_ref();
        reference = tmp;
    }
    else
    {
        for (size_t count = 0; count < rep; ++count)
            baggage_type::assign (*(point()+count), c);
        reference->len = rep;
    }
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::insert (size_t pos1, const basic_string<charT>& str,
                             size_t pos2, size_t n) 
{
    if (pos2 > str.length())
    {
        reference_class::throwrange();
    }
    size_t rlen = (n > (str.length() - pos2)) ? str.length() - pos2 : n;
    insert_str (pos1, str.data()+pos2, rlen);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::insert (size_t pos, const charT* s, size_t n)
                             
{
    insert_str(pos, s, n);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::insert (size_t pos, const charT* s)
                             
{
    insert_str(pos, s, baggage_type::length(s));
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::insert (size_t pos, charT c, size_t rep)
                             
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    if ((rep == NPOS) || (length() >= (NPOS - rep)))
    {
        reference_class::throwlength();
    }
    if (rep)
    {
        size_t count;
        if ((ref_count() > 1) || (reserve() < (length()+rep)))
        {
           reference_pointer tmp;
           tmp = new basic_string_ref<charT> (data(), pos, length()+rep);
           if (length())
               for (count = length()-pos; count > 0; --count)
                    baggage_type::assign (*(tmp->ptr+pos+rep+count-1),
                                          *(data()+pos+count-1));
           tmp->len = length();
           delete_ref();
           reference = tmp;
        }
        else
        {
           for (count = length()-pos; count > 0; --count)
                baggage_type::assign (*(point()+pos+rep+count-1),
                                      *(data()+pos+count-1));
        }
        for (count = 0; count < rep; ++count)
            baggage_type::assign (*(point()+pos+count), c);
        reference->len += rep;
    }
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::remove (size_t pos, size_t n) 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    size_t xlen = (n > (length()-pos)) ? (length()-pos) : n;
    if (ref_count() > 1)
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), pos, length());
        baggage_type::copy (tmp->ptr+pos, data()+pos+xlen, length()-xlen-pos);
        tmp->len = length()-xlen;
        delete_ref();
        reference = tmp;
    }
    else if (xlen == length())
        reference->len = 0;
    else if (xlen)
    {
        baggage_type::copy (point()+pos, data()+pos+xlen, length()-xlen-pos);
        reference->len -= xlen;
    }
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::replace (size_t pos1, size_t n1, const basic_string<charT>& str,
                     size_t pos2, size_t n2) 
{
    if (pos2 > str.length())
    {
        reference_class::throwrange();
    }
    size_t xlen = (n1 > (length()-pos1)) ? (length()-pos1) : n1;
    size_t rlen = (n2 > (str.length()-pos2)) ? (str.length()-pos2) : n2;
    replace_str (xlen, pos1, str.data()+pos2, rlen);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::replace (size_t pos, size_t n1, const charT* s, size_t n2)
                              
{
    size_t xlen = (n1 > (length()-pos)) ? (length()-pos) : n1;
    replace_str (xlen, pos, s, n2);
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::replace (size_t pos, size_t n1, const charT* s)
                              
{
    size_t xlen = (n1 > (length()-pos)) ? (length()-pos) : n1;
    replace_str (xlen, pos, s, baggage_type::length(s));
    return *this;
}

template <class charT>
basic_string<charT>&
basic_string<charT>::replace (size_t pos, size_t n, charT c, size_t rep)
                              
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    size_t xlen = (n > (length()-pos)) ? (length()-pos) : n;
    if ((length()-xlen) >= (NPOS-rep))
    {
        reference_class::throwlength();
    }
    if (!rep)
        return remove (pos, n);
    else
    {
        size_t count;
        if ((ref_count() > 1) || (reserve() < (length()-xlen+rep)))
        {
            reference_pointer tmp;
            tmp = new basic_string_ref<charT> (data(), pos,
                  length()+((xlen > rep) ? (xlen-rep) : 0));
            if (rep < xlen)
                baggage_type::copy (tmp->ptr+pos+rep, data()+pos+xlen,
                                    length()-pos-xlen);
            else
            {
                for (count = length()-xlen-pos; count > 0; --count)
                    baggage_type::assign (*(tmp->ptr+pos+rep+count-1),
                                          *(data()+pos+xlen+count-1));
            }
            tmp->len = length();
            delete_ref();
            reference = tmp;
        }
        else
        {
            if (rep < xlen)
                baggage_type::copy (point()+pos+rep, data()+pos+xlen,
                                    length()-pos-xlen);
            else
            {
                for (count = length()-xlen-pos; count > 0; --count)
                    baggage_type::assign (*(point()+pos+rep+count-1),
                                          *(data()+pos+xlen+count-1));
            }
        }
        for (count = 0; count < rep; ++count)
            baggage_type::assign (*(point()+pos+count), c);
        reference->len += (rep-xlen);
    }
    return *this;
}

template <class charT>
void
basic_string<charT>::put_at (size_t pos, charT c) 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    if ((ref_count() > 1) || (pos == reserve()))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), length(),
              length()+((pos==length())?1:0));
        delete_ref();
        reference = tmp;
    }
    if (pos == length())
        ++reference->len;
    baggage_type::assign (*(point()+pos), c);
}

template <class charT>
charT&
basic_string<charT>::operator[] (size_t pos) 
{
    if (pos >= length())
    {
        reference_class::throwrange();
    }
    if (ref_count() > 1)
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(), length(), length());
        delete_ref();
        reference = tmp;
    }
    return *(point()+pos);
}

template <class charT>
const charT*
basic_string<charT>::c_str () const 
{
    if (c_str_ptr)
        delete[] ((basic_string<charT>*)this)->c_str_ptr;
    ((basic_string<charT>*)this)->c_str_ptr = new charT [length()+1];
    if (length())
        baggage_type::copy (((basic_string<charT>*)this)->c_str_ptr, data(),
                              length());
    baggage_type::assign (*(((basic_string<charT>*)this)->c_str_ptr+length()),
                          eos());
    return c_str_ptr;
}

template <class charT>
void
basic_string<charT>::resize (size_t n, charT c) 
{
    if (n == NPOS)
    {
        reference_class::throwlength();
    }
    if ((ref_count() > 1) || (n > reserve()))
    {
        reference_pointer tmp;
        tmp = new basic_string_ref<charT> (data(),
              ((n > length()) ? length() : n), n);
        delete_ref();
        reference = tmp;
    }
    while (reference->len < n)
    {
        baggage_type::assign (*(reference->ptr+length()), c);
        ++reference->len;
    }
    reference->len = n;
}

template <class charT>
void
basic_string<charT>::resize (size_t n) 
{
    resize (n, eos());
}

template <class charT>
void
basic_string<charT>::reserve (size_t res_arg) 
{
    if (res_arg == NPOS)
    {
        reference_class::throwlength();
    }
    if (res_arg > reserve())
    {
        reference_pointer tmp;
        tmp = new  basic_string_ref<charT> (data(), length(), res_arg);
        delete_ref();
        reference = tmp;
    }
}

template <class charT>
size_t
basic_string<charT>::copy (charT* s, size_t n, size_t pos) const 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    size_t  rlen = (n > (length()-pos)) ? (length()-pos) : n;
    if (length())
        baggage_type::copy (s, data()+pos, rlen);
    return rlen;
}

template <class charT>
size_t
basic_string<charT>::find (const basic_string<charT>& str, size_t pos) const
                           
{
    return find_str (str.data(), pos, str.length());
}

template <class charT>
size_t
basic_string<charT>::find (const charT* s, size_t pos, size_t n) const
                           
{
    return find_str (s, pos, n);
}

template <class charT>
size_t
basic_string<charT>::find (const charT* s, size_t pos) const 
{
    return find_str (s, pos, baggage_type::length(s));
}

template <class charT>
size_t
basic_string<charT>::find (charT c, size_t pos) const 
{
    while ((pos < length()) && (baggage_type::ne(*(data()+pos), c)))
        ++pos;
    return ((pos < length()) ? pos : NPOS);
}

template <class charT>
size_t
basic_string<charT>::rfind (const basic_string<charT>& str, size_t pos) const
                            
{
    return rfind_str (str.data(), pos, str.length());
}

template <class charT>
size_t
basic_string<charT>::rfind (const charT* s, size_t pos, size_t n) const
                            
{
    return rfind_str (s, pos, n);
}

template <class charT>
size_t
basic_string<charT>::rfind (const charT* s, size_t pos) const 
{
    return rfind_str (s, pos, baggage_type::length(s));
}

template <class charT>
size_t
basic_string<charT>::rfind (charT c, size_t pos) const 
{
    size_t count = ((pos < length()) ? pos+1 : length());
    if (length() == 0)
        return NPOS;
    while ((baggage_type::ne(*(data()+count-1), c)) && (count > 1))
        --count;
    if ((count == 1) && (baggage_type::ne(*(data()), c)))
        return NPOS;
    else
        return count-1;
}

template <class charT>
size_t
basic_string<charT>::find_first_of (const basic_string<charT>& str, size_t pos) const
                                    
{
    return find_first_of_str (str.data(), pos, str.length());
}

template <class charT>
size_t
basic_string<charT>::find_first_of (const charT* s, size_t pos, size_t n) const
                                    
{
    return find_first_of_str (s, pos, n);
}

template <class charT>
size_t
basic_string<charT>::find_first_of (const charT* s, size_t pos) const
                                    
{
    return find_first_of_str (s, pos, baggage_type::length(s));
}

template <class charT>
size_t
basic_string<charT>::find_first_of (charT c, size_t pos) const 
{
    return find (c, pos);
}

template <class charT>
size_t
basic_string<charT>::find_last_of (const basic_string<charT>& str, size_t pos) const
                                   
{
    return find_last_of_str (str.data(), pos, str.length());
}

template <class charT>
size_t
basic_string<charT>::find_last_of (const charT* s, size_t pos, size_t n) const
                                   
{
    return find_last_of_str (s, pos, n);
}

template <class charT>
size_t
basic_string<charT>::find_last_of (const charT* s, size_t pos) const 
{
    return find_last_of_str (s, pos, baggage_type::length(s));
}

template <class charT>
size_t
basic_string<charT>::find_last_of (charT c, size_t pos) const 
{
    return rfind (c, pos);
}

template <class charT>
size_t
basic_string<charT>::find_first_not_of (const basic_string<charT>& str, size_t pos)
                                        const 
{
    return find_first_not_of_str (str.data(), pos, str.length());
}

template <class charT>
size_t
basic_string<charT>::find_first_not_of (const charT* s, size_t pos, size_t n)
                                        const 
{
    return find_first_not_of_str (s, pos, n);
}

template <class charT>
size_t
basic_string<charT>::find_first_not_of (const charT* s, size_t pos) const
                                        
{
    return find_first_not_of_str (s, pos, baggage_type::length(s));
}

template <class charT>
size_t
basic_string<charT>::find_first_not_of (charT c, size_t pos) const 
{
    while ((pos < length()) && (baggage_type::eq(*(data()+pos), c)))
        ++pos;
    return ((pos < length()) ? pos : NPOS);
}

template <class charT>
size_t
basic_string<charT>::find_last_not_of (const basic_string<charT>& str, size_t pos)
                                       const 
{
    return find_last_not_of_str (str.data(), pos, str.length());
}

template <class charT>
size_t
basic_string<charT>::find_last_not_of (const charT* s, size_t pos, size_t n)
                                       const 
{
    return find_last_not_of_str (s, pos, n);
}

template <class charT>
size_t
basic_string<charT>::find_last_not_of (const charT* s, size_t pos) const
                                       
{
    return find_last_not_of_str (s, pos, baggage_type::length(s));
}

template <class charT>
size_t
basic_string<charT>::find_last_not_of (charT c, size_t pos) const 
{
    size_t count = ((pos < length()) ? pos+1 : length());
    if (length() == 0)
        return NPOS;
    while ((baggage_type::eq(*(data()+count-1), c)) && (count > 1))
        --count;
    if ((count == 1) && (baggage_type::eq(*(data()), c)))
        return NPOS;
    else
        return count-1;
}

template <class charT>
basic_string<charT>
basic_string<charT>::substr (size_t pos,  size_t n) const 
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    if (length())
        return basic_string<charT> (data()+pos,
        (n > (length()-pos)) ? (length()-pos) : n);
    else
        return basic_string<charT>();
}

template <class charT>
int
basic_string<charT>::compare (const basic_string<charT>& str, size_t pos,
                              size_t n) const 
{
    size_t slen   = (n > (length()-pos)) ? (length()-pos) : n;
    return compare_str (pos, str.data(), slen, str.length());
}

template <class charT>
int
basic_string<charT>::compare (const charT* s, size_t pos, size_t n) const
                              
{
    if (n == NPOS)
    {
        reference_class::throwlength();
    }
    return compare_str (pos, s, length()-pos, n);
}

template <class charT>
int
basic_string<charT>::compare (const charT* s, size_t pos) const 
{
    return compare_str (pos, s, length()-pos, baggage_type::length(s));
}

template <class charT>
int
basic_string<charT>::compare (charT c, size_t pos, size_t rep) const
                              
{
    if (pos > length())
    {
        reference_class::throwrange();
    }
    if (rep == NPOS)
    {
        reference_class::throwlength();
    }
    if (rep)
    {
        size_t count = 0;
        while ((count < rep) && (count < (length()-pos)) &&
                baggage_type::eq (*(data()+pos+count), c))
            ++count;
        if ((count == rep) || (count == (length()-pos)))
            return (length()-pos-count);
        else
            return (*(data()+pos+count)-c);
    }
    else
    {
        return (length()-pos);
    }
}

template <class charT>
basic_string<charT>
operator+ (const basic_string<charT>& lhs, const basic_string<charT>& rhs)
{
    typedef  basic_string<charT>::baggage_type  baggage_type;
    basic_string<charT> tmp(lhs.data(), lhs.length(), rhs.length());
    if (rhs.length())
#ifdef __CINT__
        basic_string<charT>::baggage_type::copy (tmp.point()+lhs.length(), rhs.data(), rhs.length());
#else
        baggage_type::copy (tmp.point()+lhs.length(), rhs.data(), rhs.length());
#endif
    tmp.len() += rhs.length();
    return tmp;
}

template <class charT>
basic_string<charT>
operator+ (const charT* lhs, const basic_string<charT>& rhs) 
{
    typedef  basic_string<charT>::baggage_type  baggage_type; 
#ifdef __CINT__
    size_t  slen = basic_string<charT>::baggage_type::length(lhs);
#else
    size_t  slen = baggage_type::length(lhs);
#endif
    basic_string<charT> tmp(lhs, slen, rhs.length());
    if (rhs.length())
#ifdef __CINT__
        basic_string<charT>::baggage_type::copy (tmp.point()+slen, rhs.data(), rhs.length());
#else
        baggage_type::copy (tmp.point()+slen, rhs.data(), rhs.length());
#endif
    tmp.len() += rhs.length();
    return tmp;
}

template <class charT>
basic_string<charT>
operator+ (charT lhs, const basic_string<charT>& rhs) 
{
    typedef  basic_string<charT>::baggage_type  baggage_type; 
    basic_string<charT> tmp(&lhs, 1, rhs.length());
    if (rhs.length())
#ifdef __CINT__
        basic_string<charT>::baggage_type::copy (tmp.point()+1, rhs.data(), rhs.length());
#else
        baggage_type::copy (tmp.point()+1, rhs.data(), rhs.length());
#endif
    tmp.len() += rhs.length();
    return tmp;
}

template <class charT>
basic_string<charT>
operator+ (const basic_string<charT>& lhs, const charT* rhs) 
{
    typedef  basic_string<charT>::baggage_type  baggage_type; 
#ifdef __CINT__
    size_t  slen = basic_string<charT>::baggage_type::length(rhs);
#else
    size_t  slen = baggage_type::length(rhs);
#endif
    basic_string<charT> tmp(lhs.data(), lhs.length(), slen);
    if (slen)
#ifdef __CINT__
        basic_string<charT>::baggage_type::copy (tmp.point()+lhs.length(), rhs, slen);
#else
        baggage_type::copy (tmp.point()+lhs.length(), rhs, slen);
#endif
    tmp.len() += slen;
    return tmp;
}

template <class charT>
basic_string<charT>
operator+ (const basic_string<charT>& lhs, charT rhs) 
{
    typedef  basic_string<charT>::baggage_type  baggage_type; 
    basic_string<charT> tmp(lhs.data(), lhs.length(), 1);
#ifdef __CINT__
    basic_string<charT>::baggage_type::assign (*(tmp.point()+lhs.length()), rhs);
#else
    baggage_type::assign (*(tmp.point()+lhs.length()), rhs);
#endif
    ++tmp.len();
    return tmp;
}


template <class charT>
ostream&
operator<< (ostream& o, const basic_string<charT>& s) 
{
#ifndef __CINT__
    typedef  basic_string<charT>::baggage_type  baggage_type;
#endif
    for (size_t count = 0; count < s.length(); ++count)
#ifdef __CINT__
        basic_string<charT>::baggage_type::char_out (o, *(s.data()+count));
#else
        baggage_type::char_out (o, *(s.data()+count));
#endif
    return o;
}

template <class charT>
istream&
operator>> (istream& i, basic_string<charT>& s) 
{
#ifndef __CINT__
    typedef  basic_string<charT>::baggage_type  baggage_type; 
#endif
    s.remove();
    while (true)
    {
        charT  value;
#ifdef __CINT__
        basic_string<charT>::baggage_type::char_in (i, value);
#else
        baggage_type::char_in (i, value);
#endif
        if (!i.operator void*())
            break;
#ifdef __CINT__
        if (!basic_string<charT>::baggage_type::is_del (value))
#else
        if (!baggage_type::is_del (value))
#endif
        {
            s.append(value);
            while (true)
            {
#ifdef __CINT__
                basic_string<charT>::baggage_type::char_in (i, value);
#else
                baggage_type::char_in (i, value);
#endif
                if (!i.operator void*())
                    break;
#ifdef __CINT__
                if (!basic_string<charT>::baggage_type::is_del (value)) 
#else
                if (!baggage_type::is_del (value)) 
#endif
                {
                    s.append(value);
                }
                else
                    break;
            }
            break;
        }
    }
    return i;
}

template basic_string<char>;
typedef  basic_string<char>     cstring;
typedef  basic_string<char>     string;
//typedef  basic_string<wchar_t>  wstring;

#ifdef __CINT__
ostream& operator<< (ostream& o, const basic_string<char>& s) {
  o << s.c_str() ;
  return o;
}
istream& operator>> (istream& o, basic_string<char>& s) {
  char buf[1000];
  o >> buf;
  s = buf;
  return o;
}
#endif

#endif
