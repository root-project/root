/********************************************************************
* G__cpp_kcciostream.h
********************************************************************/
#ifdef __CINT__
#error G__cpp_kcciostream.h/C is only for compilation. Abort cint.
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define G__ANSIHEADER
#define G__DICTIONARY
/*KEEP,G__ci.*/
#include "G__ci.h"
/*KEND.*/
extern "C" {
extern void G__cpp_setup_tagtable();
extern void G__cpp_setup_inheritance();
extern void G__cpp_setup_typetable();
extern void G__cpp_setup_memvar();
extern void G__cpp_setup_global();
extern void G__cpp_setup_memfunc();
extern void G__cpp_setup_func();
extern void G__set_cpp_environment();
}


#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

#ifndef G__OLDIMPLEMENTATION1635
/********************************************************************
 * static variables for iostream redirection
 ********************************************************************/
static basic_streambuf<ostream::char_type,ostream::traits_type> *G__store_cout;
static basic_streambuf<ostream::char_type,ostream::traits_type> *G__store_cerr;
static basic_streambuf<istream::char_type,istream::traits_type> *G__store_cin;
static ofstream  *G__redirected_cout;
static ofstream  *G__redirected_cerr;
static ifstream  *G__redirected_cin;
/********************************************************************
 * G__redirectcout
 ********************************************************************/
extern "C" void G__unredirectcout() {
  if(G__store_cout) {
    cout.rdbuf(G__store_cout);
    G__store_cout = 0;
  }
  if(G__redirected_cout) {
    delete G__redirected_cout;
    G__redirected_cout = 0;
  }
}
/********************************************************************
 * G__redirectcout
 ********************************************************************/
extern "C" void G__redirectcout(const char* filename) {
  G__unredirectcout();
  G__redirected_cout = new ofstream(filename,ios_base::app);
  G__store_cout = cout.rdbuf(G__redirected_cout->rdbuf()) ;
}
/********************************************************************
 * G__redirectcerr
 ********************************************************************/
extern "C" void G__unredirectcerr() {
  if(G__store_cerr) {
    cerr.rdbuf(G__store_cerr);
    G__store_cerr = 0;
  }
  if(G__redirected_cerr) {
    delete G__redirected_cerr;
    G__redirected_cerr = 0;
  }
}
/********************************************************************
 * G__redirectcerr
 ********************************************************************/
extern "C" void G__redirectcerr(const char* filename) {
  G__unredirectcerr();
  G__redirected_cerr = new ofstream(filename,ios_base::app);
  G__store_cerr = cerr.rdbuf(G__redirected_cerr->rdbuf()) ;
}
/********************************************************************
 * G__redirectcin
 ********************************************************************/
extern "C" void G__unredirectcin() {
  if(G__store_cin) {
    cin.rdbuf(G__store_cin);
    G__store_cin = 0;
  }
  if(G__redirected_cin) {
    delete G__redirected_cin;
    G__redirected_cin = 0;
  }
}
/********************************************************************
 * G__redirectcin
 ********************************************************************/
extern "C" void G__redirectcin(const char* filename) {
  G__unredirectcin();
  G__redirected_cin = new ifstream(filename,ios_base::in);
  G__store_cin = cin.rdbuf(G__redirected_cin->rdbuf()) ;
}
#endif /* 1635 */

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__LN_char_traitslEchargR;
extern G__linked_taginfo G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_allocatorlEchargR;
extern G__linked_taginfo G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
extern G__linked_taginfo G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
extern G__linked_taginfo G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
extern G__linked_taginfo G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
extern G__linked_taginfo G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
extern G__linked_taginfo G__LN_bool;
extern G__linked_taginfo G__LN_input_iterator_tag;
extern G__linked_taginfo G__LN_output_iterator_tag;
extern G__linked_taginfo G__LN_forward_iterator_tag;
extern G__linked_taginfo G__LN_bidirectional_iterator_tag;
extern G__linked_taginfo G__LN_random_access_iterator_tag;
extern G__linked_taginfo G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR;
extern G__linked_taginfo G__LN_rel_ops;
extern G__linked_taginfo G__LN_allocatorlEvoidgR;
extern G__linked_taginfo G__LN_streampos;
extern G__linked_taginfo G__LN_b_str_reflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
extern G__linked_taginfo G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLtemplate;
extern G__linked_taginfo G__LN_reverse_iteratorlEcharmUgR;
extern G__linked_taginfo G__LN_iterator_traitslEcharmUgR;
extern G__linked_taginfo G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR;
extern G__linked_taginfo G__LN_lconv;
extern G__linked_taginfo G__LN_locale;
extern G__linked_taginfo G__LN_localecLcLfacet;
extern G__linked_taginfo G__LN_localecLcLid;
extern G__linked_taginfo G__LN_localecLcLdA;
extern G__linked_taginfo G__LN_ctype_base;
extern G__linked_taginfo G__LN_ctype_basecLcLmask;
extern G__linked_taginfo G__LN_ctypelEchargR;
extern G__linked_taginfo G__LN_ctype_bynamelEchargR;
extern G__linked_taginfo G__LN_ios_base;
extern G__linked_taginfo G__LN_ios_basecLcLfmt_flags;
extern G__linked_taginfo G__LN_ios_basecLcLio_state;
extern G__linked_taginfo G__LN_ios_basecLcLopen_mode;
extern G__linked_taginfo G__LN_ios_basecLcLseekdir;
extern G__linked_taginfo G__LN_ios_basecLcLInit;
extern G__linked_taginfo G__LN_ios_basecLcLevent;
extern G__linked_taginfo G__LN_boolean_t;
extern G__linked_taginfo G__LN_codecvt_base;
extern G__linked_taginfo G__LN_codecvt_basecLcLresult;
extern G__linked_taginfo G__LN_codecvtlEcharcOcharcOintgR;
extern G__linked_taginfo G__LN_collatelEchargR;
extern G__linked_taginfo G__LN_time_base;
extern G__linked_taginfo G__LN_time_basecLcLdateorder;
extern G__linked_taginfo G__LN_time_basecLcLt_conv_spec;
extern G__linked_taginfo G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR;
extern G__linked_taginfo G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy;
extern G__linked_taginfo G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry;
extern G__linked_taginfo G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry;
extern G__linked_taginfo G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLdA;

typedef char_traits<char> G__char_traitslEchargR;
typedef basic_ios<char,char_traits<char> > G__basic_ioslEcharcOchar_traitslEchargRsPgR;
typedef basic_istream<char,char_traits<char> > G__basic_istreamlEcharcOchar_traitslEchargRsPgR;
typedef basic_ostream<char,char_traits<char> > G__basic_ostreamlEcharcOchar_traitslEchargRsPgR;
typedef basic_iostream<char,char_traits<char> > G__basic_iostreamlEcharcOchar_traitslEchargRsPgR;
typedef basic_streambuf<char,char_traits<char> > G__basic_streambuflEcharcOchar_traitslEchargRsPgR;
typedef basic_filebuf<char,char_traits<char> > G__basic_filebuflEcharcOchar_traitslEchargRsPgR;
typedef allocator<char> G__allocatorlEchargR;
typedef basic_stringbuf<char,char_traits<char>,allocator<char> > G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
typedef basic_ifstream<char,char_traits<char> > G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR;
typedef basic_ofstream<char,char_traits<char> > G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR;
typedef basic_fstream<char,char_traits<char> > G__basic_fstreamlEcharcOchar_traitslEchargRsPgR;
typedef basic_istringstream<char,char_traits<char>,allocator<char> > G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
typedef basic_ostringstream<char,char_traits<char>,allocator<char> > G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
typedef basic_stringstream<char,char_traits<char>,allocator<char> > G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
typedef basic_string<char,char_traits<char>,allocator<char> > G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR;
typedef iterator<output_iterator_tag,void,void,void,void> G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR;
typedef allocator<void> G__allocatorlEvoidgR;

typedef reverse_iterator<char*> G__reverse_iteratorlEcharmUgR;
typedef iterator_traits<char*> G__iterator_traitslEcharmUgR;
typedef iterator<long,char*,char**,char*&,random_access_iterator_tag> G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR;
typedef ctype<char> G__ctypelEchargR;
typedef ctype_byname<char> G__ctype_bynamelEchargR;
typedef codecvt<char,char,int> G__codecvtlEcharcOcharcOintgR;
typedef collate<char> G__collatelEchargR;
typedef istreambuf_iterator<char,char_traits<char> > G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR;
typedef iterator<input_iterator_tag,char,long,char*,char&> G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR;
typedef istreambuf_iterator<char,char_traits<char> >::proxy G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy;
typedef basic_istream<char,char_traits<char> >::sentry G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry;
typedef ostreambuf_iterator<char,char_traits<char> > G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR;
typedef basic_ostream<char,char_traits<char> >::sentry G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry;
