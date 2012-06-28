/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************
* cbstrm.cpp
********************************************************/
#include "cbstrm.h"

#ifdef G__MEMTEST
#undef malloc
#undef free
#endif

extern "C" void G__cpp_reset_tagtableG__stream();

extern "C" void G__set_cpp_environmentG__stream() {
  G__add_compiledheader("iostrm.h");
  G__add_compiledheader("fstrm.h");
  G__add_compiledheader("strstrm.h");
  G__add_compiledheader("linkdef.h");
  G__cpp_reset_tagtableG__stream();
}
class G__cbstrmdOcpp_tag {};

void* operator new(size_t size,G__cbstrmdOcpp_tag* p) {
  if(p && G__PVOID!=G__getgvp()) return((void*)p);
#ifndef G__ROOT
  return(malloc(size));
#else
  return(::operator new(size));
#endif
}

/* dummy, for exception */
#ifdef G__EH_DUMMY_DELETE
void operator delete(void *p,G__cbstrmdOcpp_tag* x) {
  if((long)p==G__getgvp() && G__PVOID!=G__getgvp()) return;
#ifndef G__ROOT
  free(p);
#else
  ::operator delete(p);
#endif
}
#endif

static void G__operator_delete(void *p) {
  if((long)p==G__getgvp() && G__PVOID!=G__getgvp()) return;
#ifndef G__ROOT
  free(p);
#else
  ::operator delete(p);
#endif
}

void G__DELDMY_cbstrmdOcpp() { G__operator_delete(0); }

#include "dllrev.h"
extern "C" int G__cpp_dllrevG__stream() { return(G__CREATEDLLREV); }

/*********************************************************
* Member function Interface Method
*********************************************************/

/* mbstate_t */
// automatic default constructor
static int G__G__stream_4_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   mbstate_t *p;
   if(G__getaryconstruct()) p=new mbstate_t[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) mbstate_t;
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_mbstate_t);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__G__stream_4_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   mbstate_t *p;
   void *xtmp = (void*)G__int(libp->para[0]);
   p=new mbstate_t(*(mbstate_t*)xtmp);
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_mbstate_t);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef mbstate_t G__Tmbstate_t;
static int G__G__stream_4_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (mbstate_t *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((mbstate_t *)((G__getstructoffset())+sizeof(mbstate_t)*i))->~G__Tmbstate_t();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((mbstate_t *)(G__getstructoffset()))->~G__Tmbstate_t();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* ios_base */
static int G__G__stream_5_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((ios_base*)(G__getstructoffset()))->register_callback((ios_base::event_callback)G__int(libp->para[0]),(int)G__int(libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const ios_base*)(G__getstructoffset()))->flags());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->flags((ios_base::fmtflags)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->setf((ios_base::fmtflags)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->setf((ios_base::fmtflags)G__int(libp->para[0]),(ios_base::fmtflags)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((ios_base*)(G__getstructoffset()))->unsetf((ios_base::fmtflags)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=((ios_base*)(G__getstructoffset()))->copyfmt(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((const ios_base*)(G__getstructoffset()))->precision());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((ios_base*)(G__getstructoffset()))->precision((streamsize)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((const ios_base*)(G__getstructoffset()))->width());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((ios_base*)(G__getstructoffset()))->width((streamsize)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)ios_base::xalloc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const long& obj=((ios_base*)(G__getstructoffset()))->iword((int)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        void*& obj=((ios_base*)(G__getstructoffset()))->pword((int)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((ios_base*)(G__getstructoffset()))->is_synch());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_5_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      G__letint(result7,103,(long)((ios_base*)(G__getstructoffset()))->sync_with_stdio((bool)G__int(libp->para[0])));
      break;
   case 0:
      G__letint(result7,103,(long)((ios_base*)(G__getstructoffset()))->sync_with_stdio());
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__G__stream_5_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   ios_base *p;
   void *xtmp = (void*)G__int(libp->para[0]);
   p=new ios_base(*(ios_base*)xtmp);
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_ios_base);
   return(1 || funcname || hash || result7 || libp) ;
}


/* char_traits<char> */
static int G__G__stream_12_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      char_traits<char>::assign(*(char_traits<char>::char_type*)G__Charref(&libp->para[0]),*(char_traits<char>::char_type*)G__Charref(&libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)char_traits<char>::to_char_type(*(char_traits<char>::int_type*)G__Intref(&libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)char_traits<char>::to_int_type(*(char_traits<char>::char_type*)G__Charref(&libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)char_traits<char>::eq(*(char_traits<char>::char_type*)G__Charref(&libp->para[0]),*(char_traits<char>::char_type*)G__Charref(&libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)char_traits<char>::lt(*(char_traits<char>::char_type*)G__Charref(&libp->para[0]),*(char_traits<char>::char_type*)G__Charref(&libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)char_traits<char>::compare((const char_traits<char>::char_type*)G__int(libp->para[0]),(const char_traits<char>::char_type*)G__int(libp->para[1])
,(size_t)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)char_traits<char>::find((const char_traits<char>::char_type*)G__int(libp->para[0]),(int)G__int(libp->para[1])
,*(char_traits<char>::char_type*)G__Charref(&libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)char_traits<char>::eq_int_type(*(char_traits<char>::int_type*)G__Intref(&libp->para[0]),*(char_traits<char>::int_type*)G__Intref(&libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)char_traits<char>::eof());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)char_traits<char>::not_eof(*(char_traits<char>::int_type*)G__Intref(&libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,104,(long)char_traits<char>::length((const char_traits<char>::char_type*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)char_traits<char>::copy((char_traits<char>::char_type*)G__int(libp->para[0]),(const char_traits<char>::char_type*)G__int(libp->para[1])
,(size_t)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)char_traits<char>::move((char_traits<char>::char_type*)G__int(libp->para[0]),(const char_traits<char>::char_type*)G__int(libp->para[1])
,(size_t)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_12_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)char_traits<char>::assign((char_traits<char>::char_type*)G__int(libp->para[0]),(size_t)G__int(libp->para[1])
,(const char_traits<char>::char_type)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic default constructor
static int G__G__stream_12_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   char_traits<char> *p;
   if(G__getaryconstruct()) p=new char_traits<char>[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) char_traits<char>;
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__G__stream_12_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   char_traits<char> *p;
   void *xtmp = (void*)G__int(libp->para[0]);
   p=new char_traits<char>(*(char_traits<char>*)xtmp);
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef char_traits<char> G__Tchar_traitslEchargR;
static int G__G__stream_12_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (char_traits<char> *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((char_traits<char> *)((G__getstructoffset())+sizeof(char_traits<char>)*i))->~G__Tchar_traitslEchargR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((char_traits<char> *)(G__getstructoffset()))->~G__Tchar_traitslEchargR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_istream<char,char_traits<char> > */
static int G__G__stream_13_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_istream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_istream<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->operator>>(*(basic_istream<char,char_traits<char> >::streambuf_type*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])
,(basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[2]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get(*(basic_istream<char,char_traits<char> >::char_type*)G__Charref(&libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get(*(basic_istream<char,char_traits<char> >::streambuf_type*)libp->para[0].ref,(basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get(*(basic_istream<char,char_traits<char> >::streambuf_type*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->getline((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])
,(basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[2]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->getline((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->ignore((streamsize)G__int(libp->para[0]),(basic_istream<char,char_traits<char> >::int_type)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->ignore((streamsize)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
      break;
   case 0:
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->ignore();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->read((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->readsome((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->peek());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->tellg());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->seekg((basic_istream<char,char_traits<char> >::pos_type)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->sync());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->seekg((basic_istream<char,char_traits<char> >::off_type)G__int(libp->para[0]),(ios_base::seekdir)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->putback((basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_1_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->unget();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_13_2_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((const basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->gcount());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_istream<char,char_traits<char> > G__Tbasic_istreamlEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_13_4_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_istream<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_istream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_istream<char,char_traits<char> >)*i))->~G__Tbasic_istreamlEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_istream<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_istreamlEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ios<char,char_traits<char> > */
static int G__G__stream_14_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ios<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ios<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->fill());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->fill((basic_ios<char,char_traits<char> >::char_type)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->exceptions((ios_base::iostate)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->exceptions());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      G__setnull(result7);
      ((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->clear((ios_base::iostate)G__int(libp->para[0]));
      break;
   case 0:
      G__setnull(result7);
      ((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->clear();
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->setstate((ios_base::iostate)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->rdstate());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,89,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->operator void*());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->operator!());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->good());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->eof());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->fail());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->bad());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_ios<char,char_traits<char> >::ios_type& obj=((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->copyfmt(*(basic_ios<char,char_traits<char> >::ios_type*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->tie());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->tie((basic_ios<char,char_traits<char> >::ostream_type*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf((basic_ios<char,char_traits<char> >::streambuf_type*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->narrow((char)G__int(libp->para[0]),(char)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_14_1_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->widen((char)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_ios<char,char_traits<char> > G__Tbasic_ioslEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_14_6_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ios<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ios<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ios<char,char_traits<char> >)*i))->~G__Tbasic_ioslEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_ios<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_ioslEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_streambuf<char,char_traits<char> > */
static int G__G__stream_15_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubsetbuf((basic_streambuf<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 3:
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekoff((basic_streambuf<char,char_traits<char> >::off_type)G__int(libp->para[0]),(ios_base::seekdir)G__int(libp->para[1])
,(ios_base::openmode)G__int(libp->para[2])));
      break;
   case 2:
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekoff((basic_streambuf<char,char_traits<char> >::off_type)G__int(libp->para[0]),(ios_base::seekdir)G__int(libp->para[1])));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 2:
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekpos((basic_streambuf<char,char_traits<char> >::pos_type)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekpos((basic_streambuf<char,char_traits<char> >::pos_type)G__int(libp->para[0])));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubsync());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->which_open_mode());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->in_avail());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->snextc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sbumpc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sgetc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sgetn((basic_streambuf<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sputbackc((basic_streambuf<char,char_traits<char> >::char_type)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sungetc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sputc((basic_streambuf<char,char_traits<char> >::char_type)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_15_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sputn((const basic_streambuf<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__G__stream_15_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   basic_streambuf<char,char_traits<char> > *p;
   void *xtmp = (void*)G__int(libp->para[0]);
   p=new basic_streambuf<char,char_traits<char> >(*(basic_streambuf<char,char_traits<char> >*)xtmp);
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_streambuf<char,char_traits<char> > G__Tbasic_streambuflEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_15_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_streambuf<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_streambuf<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_streambuf<char,char_traits<char> >)*i))->~G__Tbasic_streambuflEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_streambuf<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_streambuflEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ostream<char,char_traits<char> > */
static int G__G__stream_16_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ostream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ostream<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_16_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->put((basic_ostream<char,char_traits<char> >::char_type)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_16_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->write((const basic_ostream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_16_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->flush();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_16_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->seekp((basic_ostream<char,char_traits<char> >::pos_type)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_16_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->seekp((basic_ostream<char,char_traits<char> >::off_type)G__int(libp->para[0]),(ios_base::seekdir)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_16_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->tellp());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_ostream<char,char_traits<char> > G__Tbasic_ostreamlEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_16_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ostream<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ostream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ostream<char,char_traits<char> >)*i))->~G__Tbasic_ostreamlEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_ostream<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_ostreamlEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_filebuf<char,char_traits<char> > */
static int G__G__stream_19_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_filebuf<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new basic_filebuf<char,char_traits<char> >[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_filebuf<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_19_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_filebuf<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_filebuf<char,char_traits<char> >((int)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_19_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_19_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 3:
   G__letint(result7,85,(long)((basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2])));
      break;
   case 2:
   G__letint(result7,85,(long)((basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_19_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->open((int)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_19_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->close());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__G__stream_19_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   basic_filebuf<char,char_traits<char> > *p;
   void *xtmp = (void*)G__int(libp->para[0]);
   p=new basic_filebuf<char,char_traits<char> >(*(basic_filebuf<char,char_traits<char> >*)xtmp);
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_filebuf<char,char_traits<char> > G__Tbasic_filebuflEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_19_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_filebuf<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_filebuf<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_filebuf<char,char_traits<char> >)*i))->~G__Tbasic_filebuflEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_filebuf<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_filebuflEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ifstream<char,char_traits<char> > */
static int G__G__stream_20_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new basic_ifstream<char,char_traits<char> >[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ifstream<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
   switch(libp->paran) {
   case 3:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ifstream<char,char_traits<char> >(
(const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2]));
      break;
   case 2:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ifstream<char,char_traits<char> >((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ifstream<char,char_traits<char> >((const char*)G__int(libp->para[0]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ifstream<char,char_traits<char> >((int)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ifstream<char,char_traits<char> >(
(int)G__int(libp->para[0]),(basic_ifstream<char,char_traits<char> >::char_type*)G__int(libp->para[1])
,(int)G__int(libp->para[2]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 3:
      G__setnull(result7);
      ((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2]));
      break;
   case 2:
      G__setnull(result7);
      ((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      G__setnull(result7);
      ((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_20_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->close();
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_ifstream<char,char_traits<char> > G__Tbasic_ifstreamlEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_20_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ifstream<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ifstream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ifstream<char,char_traits<char> >)*i))->~G__Tbasic_ifstreamlEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_ifstream<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_ifstreamlEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ofstream<char,char_traits<char> > */
static int G__G__stream_21_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new basic_ofstream<char,char_traits<char> >[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ofstream<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
   switch(libp->paran) {
   case 3:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ofstream<char,char_traits<char> >(
(const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2]));
      break;
   case 2:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ofstream<char,char_traits<char> >((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ofstream<char,char_traits<char> >((const char*)G__int(libp->para[0]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ofstream<char,char_traits<char> >((int)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_ofstream<char,char_traits<char> >(
(int)G__int(libp->para[0]),(basic_ofstream<char,char_traits<char> >::char_type*)G__int(libp->para[1])
,(int)G__int(libp->para[2]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 3:
      G__setnull(result7);
      ((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2]));
      break;
   case 2:
      G__setnull(result7);
      ((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      G__setnull(result7);
      ((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_21_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->close();
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_ofstream<char,char_traits<char> > G__Tbasic_ofstreamlEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_21_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ofstream<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ofstream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ofstream<char,char_traits<char> >)*i))->~G__Tbasic_ofstreamlEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_ofstream<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_ofstreamlEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_fstream<char,char_traits<char> > */
static int G__G__stream_22_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_fstream<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new basic_fstream<char,char_traits<char> >[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_fstream<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_22_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_fstream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_fstream<char,char_traits<char> >((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_22_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const basic_fstream<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_22_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_fstream<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_22_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_fstream<char,char_traits<char> >*)(G__getstructoffset()))->open((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_22_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_fstream<char,char_traits<char> >*)(G__getstructoffset()))->close();
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_fstream<char,char_traits<char> > G__Tbasic_fstreamlEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_22_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_fstream<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_fstream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_fstream<char,char_traits<char> >)*i))->~G__Tbasic_fstreamlEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_fstream<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_fstreamlEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_iostream<char,char_traits<char> > */
static int G__G__stream_23_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_iostream<char,char_traits<char> > *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) basic_iostream<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef basic_iostream<char,char_traits<char> > G__Tbasic_iostreamlEcharcOchar_traitslEchargRsPgR;
static int G__G__stream_23_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_iostream<char,char_traits<char> > *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_iostream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_iostream<char,char_traits<char> >)*i))->~G__Tbasic_iostreamlEcharcOchar_traitslEchargRsPgR();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((basic_iostream<char,char_traits<char> > *)(G__getstructoffset()))->~G__Tbasic_iostreamlEcharcOchar_traitslEchargRsPgR();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* strstreambuf */
static int G__G__stream_24_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   strstreambuf *p=NULL;
   switch(libp->paran) {
   case 1:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf((streamsize)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new strstreambuf[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   strstreambuf *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf((void *(*)(size_t))G__int(libp->para[0]),(void (*)(void*))G__int(libp->para[1]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   strstreambuf *p=NULL;
   switch(libp->paran) {
   case 3:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf(
(char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])
,(char*)G__int(libp->para[2]));
      break;
   case 2:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf((char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   strstreambuf *p=NULL;
   switch(libp->paran) {
   case 3:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf(
(unsigned char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])
,(unsigned char*)G__int(libp->para[2]));
      break;
   case 2:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf((unsigned char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   strstreambuf *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf((const char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   strstreambuf *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) strstreambuf((const unsigned char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      G__setnull(result7);
      ((strstreambuf*)(G__getstructoffset()))->freeze((bool)G__int(libp->para[0]));
      break;
   case 0:
      G__setnull(result7);
      ((strstreambuf*)(G__getstructoffset()))->freeze();
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)((strstreambuf*)(G__getstructoffset()))->str());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_24_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const strstreambuf*)(G__getstructoffset()))->pcount());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__G__stream_24_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   strstreambuf *p;
   void *xtmp = (void*)G__int(libp->para[0]);
   p=new strstreambuf(*(strstreambuf*)xtmp);
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_strstreambuf);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef strstreambuf G__Tstrstreambuf;
static int G__G__stream_24_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (strstreambuf *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((strstreambuf *)((G__getstructoffset())+sizeof(strstreambuf)*i))->~G__Tstrstreambuf();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((strstreambuf *)(G__getstructoffset()))->~G__Tstrstreambuf();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* istrstream */
static int G__G__stream_25_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   istrstream *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) istrstream((const char*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_istrstream);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_25_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   istrstream *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) istrstream((const char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_istrstream);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_25_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   istrstream *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) istrstream((char*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_istrstream);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_25_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   istrstream *p=NULL;
      p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) istrstream((char*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_istrstream);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_25_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const istrstream*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_25_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)((istrstream*)(G__getstructoffset()))->str());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef istrstream G__Tistrstream;
static int G__G__stream_25_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (istrstream *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((istrstream *)((G__getstructoffset())+sizeof(istrstream)*i))->~G__Tistrstream();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((istrstream *)(G__getstructoffset()))->~G__Tistrstream();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* ostrstream */
static int G__G__stream_26_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   ostrstream *p=NULL;
   if(G__getaryconstruct()) p=new ostrstream[G__getaryconstruct()];
   else p=::new((G__cbstrmdOcpp_tag*)G__getgvp()) ostrstream;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_ostrstream);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_26_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   ostrstream *p=NULL;
   switch(libp->paran) {
   case 3:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) ostrstream(
(char*)G__int(libp->para[0]),(int)G__int(libp->para[1])
,(ios_base::openmode)G__int(libp->para[2]));
      break;
   case 2:
      p = ::new((G__cbstrmdOcpp_tag*)G__getgvp()) ostrstream((char*)G__int(libp->para[0]),(int)G__int(libp->para[1]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_ostrstream);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_26_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,85,(long)((const ostrstream*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_26_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      G__setnull(result7);
      ((ostrstream*)(G__getstructoffset()))->freeze((int)G__int(libp->para[0]));
      break;
   case 0:
      G__setnull(result7);
      ((ostrstream*)(G__getstructoffset()))->freeze();
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_26_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__letint(result7,67,(long)((ostrstream*)(G__getstructoffset()))->str());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream_26_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const ostrstream*)(G__getstructoffset()))->pcount());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef ostrstream G__Tostrstream;
static int G__G__stream_26_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (ostrstream *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((ostrstream *)((G__getstructoffset())+sizeof(ostrstream)*i))->~G__Tostrstream();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((ostrstream *)(G__getstructoffset()))->~G__Tostrstream();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* Setting up global function */
static int G__G__stream__4_12(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=dec(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__5_12(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=hex(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__6_12(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=oct(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__7_12(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=fixed(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__8_12(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=scientific(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__9_12(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=right(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__0_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=left(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__1_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=internal(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__2_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=nouppercase(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__3_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=uppercase(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__4_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=noskipws(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__5_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=skipws(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__6_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=noshowpos(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__7_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=showpos(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__8_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=noshowpoint(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__9_13(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=showpoint(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__0_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=noshowbase(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__1_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=showbase(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__2_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=noboolalpha(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__3_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ios_base& obj=boolalpha(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__4_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=ws(*(istream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__5_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=endl(*(ostream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__6_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=ends(*(ostream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__7_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=flush(*(ostream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__8_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(char)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__9_14(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(char*)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__0_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(void*)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__1_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned char)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__2_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(short)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__3_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned short)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__4_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(int)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__5_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned int)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__6_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(long)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__7_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned long)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__8_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(float)G__double(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__9_15(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(double)G__double(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__0_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(bool)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__1_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(char*)G__Charref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__2_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned char*)G__UCharref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__3_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(short*)G__Shortref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__4_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned short*)G__UShortref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__5_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(int*)G__Intref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__6_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned int*)G__UIntref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__7_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(long*)G__Longref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__8_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned long*)G__ULongref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__9_16(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(float*)G__Floatref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__0_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(double*)G__Doubleref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__1_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(bool*)G__UCharref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__2_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,(char*)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__stream__3_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,libp->para[1].ref?*(void**)libp->para[1].ref:*(void**)(&G__Mlong(libp->para[1])));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}


/*********************************************************
* Member function Stub
*********************************************************/

/* mbstate_t */

/* ios_base */

/* ios_base::Init */

/* char_traits<char> */

/* basic_istream<char,char_traits<char> > */

/* basic_ios<char,char_traits<char> > */

/* basic_streambuf<char,char_traits<char> > */

/* basic_ostream<char,char_traits<char> > */

/* basic_ostream<char,char_traits<char> >::sentry */

/* basic_istream<char,char_traits<char> >::sentry */

/* basic_filebuf<char,char_traits<char> > */

/* basic_ifstream<char,char_traits<char> > */

/* basic_ofstream<char,char_traits<char> > */

/* basic_fstream<char,char_traits<char> > */

/* basic_iostream<char,char_traits<char> > */

/* strstreambuf */

/* istrstream */

/* ostrstream */

/*********************************************************
* Global function Stub
*********************************************************/

/*********************************************************
* Get size of pointer to member function
*********************************************************/
class G__Sizep2memfuncG__stream {
 public:
  G__Sizep2memfuncG__stream() {p=&G__Sizep2memfuncG__stream::sizep2memfunc;}
    size_t sizep2memfunc() { return(sizeof(p)); }
  private:
    size_t (G__Sizep2memfuncG__stream::*p)();
};

size_t G__get_sizep2memfuncG__stream()
{
  G__Sizep2memfuncG__stream a;
  G__setsizep2memfunc((int)a.sizep2memfunc());
  return((size_t)a.sizep2memfunc());
}


/*********************************************************
* virtual base class offset calculation interface
*********************************************************/

   /* Setting up class inheritance */
static long G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0(long pobject) {
  basic_istream<char,char_traits<char> > *G__Lderived=(basic_istream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_ios_base_1(long pobject) {
  basic_istream<char,char_traits<char> > *G__Lderived=(basic_istream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0(long pobject) {
  basic_ostream<char,char_traits<char> > *G__Lderived=(basic_ostream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_ios_base_1(long pobject) {
  basic_ostream<char,char_traits<char> > *G__Lderived=(basic_ostream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  basic_ifstream<char,char_traits<char> > *G__Lderived=(basic_ifstream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2(long pobject) {
  basic_ifstream<char,char_traits<char> > *G__Lderived=(basic_ifstream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  basic_ofstream<char,char_traits<char> > *G__Lderived=(basic_ofstream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2(long pobject) {
  basic_ofstream<char,char_traits<char> > *G__Lderived=(basic_ofstream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2(long pobject) {
  basic_fstream<char,char_traits<char> > *G__Lderived=(basic_fstream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_3(long pobject) {
  basic_fstream<char,char_traits<char> > *G__Lderived=(basic_fstream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5(long pobject) {
  basic_fstream<char,char_traits<char> > *G__Lderived=(basic_fstream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_6(long pobject) {
  basic_fstream<char,char_traits<char> > *G__Lderived=(basic_fstream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  basic_iostream<char,char_traits<char> > *G__Lderived=(basic_iostream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_2(long pobject) {
  basic_iostream<char,char_traits<char> > *G__Lderived=(basic_iostream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_4(long pobject) {
  basic_iostream<char,char_traits<char> > *G__Lderived=(basic_iostream<char,char_traits<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_5(long pobject) {
  basic_iostream<char,char_traits<char> > *G__Lderived=(basic_iostream<char,char_traits<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_istrstream_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  istrstream *G__Lderived=(istrstream*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_istrstream_ios_base_2(long pobject) {
  istrstream *G__Lderived=(istrstream*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_ostrstream_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  ostrstream *G__Lderived=(ostrstream*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_ostrstream_ios_base_2(long pobject) {
  ostrstream *G__Lderived=(ostrstream*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}


/*********************************************************
* Inheritance information setup/
*********************************************************/
extern "C" void G__cpp_setup_inheritanceG__stream() {

   /* Setting up class inheritance */
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR))) {
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0,1,3);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_ios_base_1,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR))) {
     basic_ios<char,char_traits<char> > *G__Lderived;
     G__Lderived=(basic_ios<char,char_traits<char> >*)0x1000;
     {
       ios_base *G__Lpbase=(ios_base*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR))) {
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0,1,3);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_ios_base_1,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR))) {
     basic_filebuf<char,char_traits<char> > *G__Lderived;
     G__Lderived=(basic_filebuf<char,char_traits<char> >*)0x1000;
     {
       basic_streambuf<char,char_traits<char> > *G__Lpbase=(basic_streambuf<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR))) {
     basic_ifstream<char,char_traits<char> > *G__Lderived;
     G__Lderived=(basic_ifstream<char,char_traits<char> >*)0x1000;
     {
       basic_istream<char,char_traits<char> > *G__Lpbase=(basic_istream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR))) {
     basic_ofstream<char,char_traits<char> > *G__Lderived;
     G__Lderived=(basic_ofstream<char,char_traits<char> >*)0x1000;
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR))) {
     basic_fstream<char,char_traits<char> > *G__Lderived;
     G__Lderived=(basic_fstream<char,char_traits<char> >*)0x1000;
     {
       basic_iostream<char,char_traits<char> > *G__Lpbase=(basic_iostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       basic_istream<char,char_traits<char> > *G__Lpbase=(basic_istream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,0);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_3,1,2);
     }
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,0);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_6,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR))) {
     basic_iostream<char,char_traits<char> > *G__Lderived;
     G__Lderived=(basic_iostream<char,char_traits<char> >*)0x1000;
     {
       basic_istream<char,char_traits<char> > *G__Lpbase=(basic_istream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_2,1,2);
     }
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_4,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_5,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_strstreambuf))) {
     strstreambuf *G__Lderived;
     G__Lderived=(strstreambuf*)0x1000;
     {
       basic_streambuf<char,char_traits<char> > *G__Lpbase=(basic_streambuf<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_strstreambuf),G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_istrstream))) {
     istrstream *G__Lderived;
     G__Lderived=(istrstream*)0x1000;
     {
       basic_istream<char,char_traits<char> > *G__Lpbase=(basic_istream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_istrstream),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_istrstream),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_istrstream_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_istrstream),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_istrstream_ios_base_2,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_ostrstream))) {
     ostrstream *G__Lderived;
     G__Lderived=(ostrstream*)0x1000;
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_ostrstream),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_ostrstream),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),(long)G__2vbo_ostrstream_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_ostrstream),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_ostrstream_ios_base_2,1,2);
     }
   }
}

/*********************************************************
* typedef information setup/
*********************************************************/
extern "C" void G__cpp_setup_typetableG__stream() {

   /* Setting up typedef entry */
   G__search_typename2("mbstate_t",117,G__get_linked_tagnum(&G__G__streamLN_mbstate_t),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("streampos",108,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("streamoff",108,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("SZ_T",108,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("streamsize",108,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("INT_T",105,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("istream",117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ostream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("filebuf",117,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ifstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ofstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("fstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("void *(*)(size_t)",81,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("void (*)(void*)",81,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

   /* mbstate_t */
static void G__setup_memvarmbstate_t(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_mbstate_t));
   { mbstate_t *p; p=(mbstate_t*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* ios_base */
static void G__setup_memvarios_base(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_ios_base));
   { ios_base *p; p=(ios_base*)0x1000; if (p) { }
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLio_state),-1,-2,1,"goodbit=0",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLio_state),-1,-2,1,"badbit=1",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLio_state),-1,-2,1,"eofbit=2",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLio_state),-1,-2,1,"failbit=4",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),-1,-2,1,"app=1",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),-1,-2,1,"binary=2",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),-1,-2,1,"in=4",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),-1,-2,1,"out=8",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),-1,-2,1,"trunc=16",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),-1,-2,1,"ate=32",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLseek_dir),-1,-2,1,"beg=0",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLseek_dir),-1,-2,1,"cur=1",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLseek_dir),-1,-2,1,"end=2",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"boolalpha=1",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"dec=2",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"fixed=4",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"hex=8",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"internal=16",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"left=32",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"oct=64",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"right=128",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"scientific=256",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"showbase=512",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"showpoint=1024",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"showpos=2048",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"skipws=4096",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"unitbuf=8192",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"uppercase=16384",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"adjustfield=176",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"basefield=74",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),-1,-2,1,"floatfield=260",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLevent),-1,-2,1,"erase_event=1",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLevent),-1,-2,1,"imbue_event=2",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,105,0,1,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLevent),-1,-2,1,"copyfmt_event=4",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}


   /* char_traits<char> */
static void G__setup_memvarchar_traitslEchargR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR));
   { char_traits<char> *p; p=(char_traits<char>*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_istream<char,char_traits<char> > */
static void G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   { basic_istream<char,char_traits<char> > *p; p=(basic_istream<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_ios<char,char_traits<char> > */
static void G__setup_memvarbasic_ioslEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   { basic_ios<char,char_traits<char> > *p; p=(basic_ios<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_streambuf<char,char_traits<char> > */
static void G__setup_memvarbasic_streambuflEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   { basic_streambuf<char,char_traits<char> > *p; p=(basic_streambuf<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_ostream<char,char_traits<char> > */
static void G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   { basic_ostream<char,char_traits<char> > *p; p=(basic_ostream<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_filebuf<char,char_traits<char> > */
static void G__setup_memvarbasic_filebuflEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   { basic_filebuf<char,char_traits<char> > *p; p=(basic_filebuf<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_ifstream<char,char_traits<char> > */
static void G__setup_memvarbasic_ifstreamlEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   { basic_ifstream<char,char_traits<char> > *p; p=(basic_ifstream<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_ofstream<char,char_traits<char> > */
static void G__setup_memvarbasic_ofstreamlEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   { basic_ofstream<char,char_traits<char> > *p; p=(basic_ofstream<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_fstream<char,char_traits<char> > */
static void G__setup_memvarbasic_fstreamlEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   { basic_fstream<char,char_traits<char> > *p; p=(basic_fstream<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_iostream<char,char_traits<char> > */
static void G__setup_memvarbasic_iostreamlEcharcOchar_traitslEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR));
   { basic_iostream<char,char_traits<char> > *p; p=(basic_iostream<char,char_traits<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* strstreambuf */
static void G__setup_memvarstrstreambuf(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_strstreambuf));
   { strstreambuf *p; p=(strstreambuf*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* istrstream */
static void G__setup_memvaristrstream(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_istrstream));
   { istrstream *p; p=(istrstream*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* ostrstream */
static void G__setup_memvarostrstream(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_ostrstream));
   { ostrstream *p; p=(ostrstream*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}

extern "C" void G__cpp_setup_memvarG__stream() {
}
/***********************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
***********************************************************/

/*********************************************************
* Member function information setup for each class
*********************************************************/
static void G__setup_memfuncmbstate_t(void) {
   /* mbstate_t */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_mbstate_t));
   // automatic default constructor
   G__memfunc_setup("mbstate_t",963,G__G__stream_4_1_0,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_mbstate_t),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("mbstate_t",963,G__G__stream_4_2_0,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_mbstate_t),-1,0,1,1,1,0,"u 'mbstate_t' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~mbstate_t",1089,G__G__stream_4_3_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncios_base(void) {
   /* ios_base */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__memfunc_setup("register_callback",1777,G__G__stream_5_1_0,121,-1,-1,0,2,1,1,0,
"Y - 'ios_base::event_callback' 0 - fn i - - 0 - index",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("flags",525,G__G__stream_5_2_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("flags",525,G__G__stream_5_3_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,1,1,1,0,"i - 'ios_base::fmtflags' 0 - fmtfl",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("setf",434,G__G__stream_5_4_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,1,1,1,0,"i - 'ios_base::fmtflags' 0 - fmtfl",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("setf",434,G__G__stream_5_5_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,2,1,1,0,
"i - 'ios_base::fmtflags' 0 - fmtfl i - 'ios_base::fmtflags' 0 - mask",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("unsetf",661,G__G__stream_5_6_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::fmtflags' 0 - mask",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("copyfmt",770,G__G__stream_5_7_0,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 11 - rhs",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("precision",972,G__G__stream_5_8_0,108,-1,G__defined_typename("streamsize"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("precision",972,G__G__stream_5_9_0,108,-1,G__defined_typename("streamsize"),0,1,1,1,0,"l - 'streamsize' 0 - prec",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("width",544,G__G__stream_5_0_1,108,-1,G__defined_typename("streamsize"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("width",544,G__G__stream_5_1_1,108,-1,G__defined_typename("streamsize"),0,1,1,1,0,"l - 'streamsize' 0 - wide",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("xalloc",643,G__G__stream_5_2_1,105,-1,-1,0,0,3,1,0,"",(char*)NULL,(void*)(int (*)())(&ios_base::xalloc),0);
   G__memfunc_setup("iword",549,G__G__stream_5_3_1,108,-1,-1,1,1,1,1,0,"i - - 0 - index",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pword",556,G__G__stream_5_4_1,89,-1,-1,1,1,1,1,0,"i - - 0 - index",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_synch",864,G__G__stream_5_5_1,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sync_with_stdio",1626,G__G__stream_5_6_1,103,-1,-1,0,1,1,1,0,"g - - 0 true sync",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("ios_base",837,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,0,0,1,2,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,(G__InterfaceMethod)NULL,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,2,0,"u 'ios_base' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("ios_base",837,G__G__stream_5_9_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,0,1,1,1,0,"u 'ios_base' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~ios_base",963,(G__InterfaceMethod)NULL,(int)('y'),-1,-1,0,0,1,2,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncchar_traitslEchargR(void) {
   /* char_traits<char> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR));
   G__memfunc_setup("assign",645,G__G__stream_12_1_0,121,-1,-1,0,2,3,1,0,
"c - 'char_traits<char>::char_type' 1 - c1 c - 'char_traits<char>::char_type' 11 - c2",(char*)NULL,(void*)(void (*)(char_traits<char>::char_type&,const char_traits<char>::char_type&))(&char_traits<char>::assign),0);
   G__memfunc_setup("to_char_type",1281,G__G__stream_12_2_0,99,-1,G__defined_typename("char_traits<char>::char_type"),0,1,3,1,0,"i - 'char_traits<char>::int_type' 11 - c",(char*)NULL,(void*)(char_traits<char>::char_type (*)(const char_traits<char>::int_type&))(&char_traits<char>::to_char_type),0);
   G__memfunc_setup("to_int_type",1198,G__G__stream_12_3_0,105,-1,G__defined_typename("char_traits<char>::int_type"),0,1,3,1,0,"c - 'char_traits<char>::char_type' 11 - c",(char*)NULL,(void*)(char_traits<char>::int_type (*)(const char_traits<char>::char_type&))(&char_traits<char>::to_int_type),0);
   G__memfunc_setup("eq",214,G__G__stream_12_4_0,103,-1,-1,0,2,3,1,0,
"c - 'char_traits<char>::char_type' 11 - c1 c - 'char_traits<char>::char_type' 11 - c2",(char*)NULL,(void*)(bool (*)(const char_traits<char>::char_type&,const char_traits<char>::char_type&))(&char_traits<char>::eq),0);
   G__memfunc_setup("lt",224,G__G__stream_12_5_0,103,-1,-1,0,2,3,1,0,
"c - 'char_traits<char>::char_type' 11 - c1 c - 'char_traits<char>::char_type' 11 - c2",(char*)NULL,(void*)(bool (*)(const char_traits<char>::char_type&,const char_traits<char>::char_type&))(&char_traits<char>::lt),0);
   G__memfunc_setup("compare",743,G__G__stream_12_6_0,105,-1,-1,0,3,3,1,0,
"C - 'char_traits<char>::char_type' 10 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
"h - 'size_t' 0 - n",(char*)NULL,(void*)(int (*)(const char_traits<char>::char_type*,const char_traits<char>::char_type*,size_t))(&char_traits<char>::compare),0);
   G__memfunc_setup("find",417,G__G__stream_12_7_0,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,1,
"C - 'char_traits<char>::char_type' 10 - s i - - 0 - n "
"c - 'char_traits<char>::char_type' 11 - a",(char*)NULL,(void*)(const char_traits<char>::char_type* (*)(const char_traits<char>::char_type*,int,const char_traits<char>::char_type&))(&char_traits<char>::find),0);
   G__memfunc_setup("eq_int_type",1185,G__G__stream_12_8_0,103,-1,-1,0,2,3,1,0,
"i - 'char_traits<char>::int_type' 11 - c1 i - 'char_traits<char>::int_type' 11 - c2",(char*)NULL,(void*)(bool (*)(const char_traits<char>::int_type&,const char_traits<char>::int_type&))(&char_traits<char>::eq_int_type),0);
   G__memfunc_setup("eof",314,G__G__stream_12_9_0,105,-1,G__defined_typename("char_traits<char>::int_type"),0,0,3,1,0,"",(char*)NULL,(void*)(char_traits<char>::int_type (*)())(&char_traits<char>::eof),0);
   G__memfunc_setup("not_eof",746,G__G__stream_12_0_1,105,-1,G__defined_typename("char_traits<char>::int_type"),0,1,3,1,0,"i - 'char_traits<char>::int_type' 11 - c",(char*)NULL,(void*)(char_traits<char>::int_type (*)(const char_traits<char>::int_type&))(&char_traits<char>::not_eof),0);
   G__memfunc_setup("length",642,G__G__stream_12_1_1,104,-1,G__defined_typename("size_t"),0,1,3,1,0,"C - 'char_traits<char>::char_type' 10 - s",(char*)NULL,(void*)(size_t (*)(const char_traits<char>::char_type*))(&char_traits<char>::length),0);
   G__memfunc_setup("copy",443,G__G__stream_12_2_1,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,0,
"C - 'char_traits<char>::char_type' 0 - dst C - 'char_traits<char>::char_type' 10 - src "
"h - 'size_t' 0 - n",(char*)NULL,(void*)(char_traits<char>::char_type* (*)(char_traits<char>::char_type*,const char_traits<char>::char_type*,size_t))(&char_traits<char>::copy),0);
   G__memfunc_setup("move",439,G__G__stream_12_3_1,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,0,
"C - 'char_traits<char>::char_type' 0 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
"h - 'size_t' 0 - n",(char*)NULL,(void*)(char_traits<char>::char_type* (*)(char_traits<char>::char_type*,const char_traits<char>::char_type*,size_t))(&char_traits<char>::move),0);
   G__memfunc_setup("assign",645,G__G__stream_12_4_1,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,0,
"C - 'char_traits<char>::char_type' 0 - s h - 'size_t' 0 - n "
"c - 'char_traits<char>::char_type' 10 - a",(char*)NULL,(void*)NULL,0);
   // automatic default constructor
   G__memfunc_setup("char_traits<char>",1708,G__G__stream_12_5_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("char_traits<char>",1708,G__G__stream_12_6_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),-1,0,1,1,1,0,"u 'char_traits<char>' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~char_traits<char>",1834,G__G__stream_12_7_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_istream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_istream<char,char_traits<char> >",3686,G__G__stream_13_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),-1,0,1,5,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream_13_2_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"u 'basic_streambuf<char,char_traits<char> >' 'basic_istream<char,char_traits<char> >::streambuf_type' 1 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__G__stream_13_3_0,105,-1,G__defined_typename("basic_istream<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__G__stream_13_4_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,3,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n "
"c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__G__stream_13_5_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__G__stream_13_6_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"c - 'basic_istream<char,char_traits<char> >::char_type' 1 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__G__stream_13_7_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"u 'basic_streambuf<char,char_traits<char> >' 'basic_istream<char,char_traits<char> >::streambuf_type' 1 - sb c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__G__stream_13_8_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"u 'basic_streambuf<char,char_traits<char> >' 'basic_istream<char,char_traits<char> >::streambuf_type' 1 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("getline",744,G__G__stream_13_9_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,3,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n "
"c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("getline",744,G__G__stream_13_0_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("ignore",644,G__G__stream_13_1_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"l - 'streamsize' 0 - n i - 'basic_istream<char,char_traits<char> >::int_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("ignore",644,G__G__stream_13_2_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"l - 'streamsize' 0 1 n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("read",412,G__G__stream_13_3_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("readsome",848,G__G__stream_13_4_1,108,-1,G__defined_typename("streamsize"),0,2,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("peek",421,G__G__stream_13_5_1,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tellg",536,G__G__stream_13_6_1,108,-1,G__defined_typename("basic_istream<char,char_traits<char> >::pos_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekg",527,G__G__stream_13_7_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"l - 'basic_istream<char,char_traits<char> >::pos_type' 0 - pos",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sync",445,G__G__stream_13_8_1,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekg",527,G__G__stream_13_9_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"l - 'basic_istream<char,char_traits<char> >::off_type' 0 - - i - 'ios_base::seekdir' 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("putback",746,G__G__stream_13_0_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"c - 'basic_istream<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("unget",547,G__G__stream_13_1_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("gcount",656,G__G__stream_13_2_2,108,-1,G__defined_typename("streamsize"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_istream<char,char_traits<char> >",3686,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),-1,0,0,1,2,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_istream<char,char_traits<char> >",3812,G__G__stream_13_4_2,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ioslEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ios<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ios<char,char_traits<char> >",3260,G__G__stream_14_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),-1,0,1,5,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb_arg",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("fill",423,G__G__stream_14_2_0,99,-1,G__defined_typename("basic_ios<char,char_traits<char> >::char_type"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("fill",423,G__G__stream_14_3_0,99,-1,G__defined_typename("basic_ios<char,char_traits<char> >::char_type"),0,1,1,1,0,"c - 'basic_ios<char,char_traits<char> >::char_type' 0 - ch",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("exceptions",1090,G__G__stream_14_4_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::iostate' 0 - excpt",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("exceptions",1090,G__G__stream_14_5_0,105,-1,G__defined_typename("ios_base::iostate"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("clear",519,G__G__stream_14_6_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::iostate' 0 goodbit state",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("setstate",877,G__G__stream_14_7_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::iostate' 0 - state",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdstate",759,G__G__stream_14_8_0,105,-1,G__defined_typename("ios_base::iostate"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator void*",1384,G__G__stream_14_9_0,89,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator!",909,G__G__stream_14_0_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("good",425,G__G__stream_14_1_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("eof",314,G__G__stream_14_2_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("fail",412,G__G__stream_14_3_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("bad",295,G__G__stream_14_4_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("copyfmt",770,G__G__stream_14_5_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::ios_type"),1,1,1,1,0,"u 'basic_ios<char,char_traits<char> >' 'basic_ios<char,char_traits<char> >::ios_type' 11 - rhs",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tie",322,G__G__stream_14_6_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::ostream_type"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tie",322,G__G__stream_14_7_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::ostream_type"),0,1,1,1,0,"U 'basic_ostream<char,char_traits<char> >' 'basic_ios<char,char_traits<char> >::ostream_type' 0 - tie_arg",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_14_8_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::streambuf_type"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_14_9_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::streambuf_type"),0,1,1,1,0,"U 'basic_streambuf<char,char_traits<char> >' 'basic_ios<char,char_traits<char> >::streambuf_type' 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("narrow",665,G__G__stream_14_0_2,99,-1,-1,0,2,1,1,8,
"c - - 0 - - c - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("widen",535,G__G__stream_14_1_2,99,-1,-1,0,1,1,1,8,"c - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ios<char,char_traits<char> >",3260,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),-1,0,0,1,2,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ios<char,char_traits<char> >",3260,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),-1,0,1,1,4,0,"u 'basic_ios<char,char_traits<char> >' - 11 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,(G__InterfaceMethod)NULL,117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),-1,1,1,1,4,0,"u 'basic_ios<char,char_traits<char> >' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ios<char,char_traits<char> >",3386,G__G__stream_14_6_2,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_streambuflEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_streambuf<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("pubsetbuf",976,G__G__stream_15_1_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_streambuf<char,char_traits<char> >::basic_streambuf<char_type,char_traits<char> >"),0,2,1,1,0,
"C - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pubseekoff",1066,G__G__stream_15_2_0,108,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::pos_type"),0,3,1,1,0,
"l - 'basic_streambuf<char,char_traits<char> >::off_type' 0 - off i - 'ios_base::seekdir' 0 - way "
"i - 'ios_base::openmode' 0 ios_base::in|ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pubseekpos",1089,G__G__stream_15_3_0,108,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::pos_type"),0,2,1,1,0,
"l - 'basic_streambuf<char,char_traits<char> >::pos_type' 0 - sp i - 'ios_base::openmode' 0 ios_base::in|ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pubsync",772,G__G__stream_15_4_0,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("which_open_mode",1576,G__G__stream_15_5_0,105,-1,G__defined_typename("ios_base::openmode"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("in_avail",835,G__G__stream_15_6_0,108,-1,G__defined_typename("streamsize"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("snextc",661,G__G__stream_15_7_0,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sbumpc",650,G__G__stream_15_8_0,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sgetc",534,G__G__stream_15_9_0,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sgetn",545,G__G__stream_15_0_1,108,-1,G__defined_typename("streamsize"),0,2,1,1,0,
"C - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sputbackc",960,G__G__stream_15_1_1,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,1,1,1,0,"c - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sungetc",761,G__G__stream_15_2_1,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sputc",559,G__G__stream_15_3_1,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,1,1,1,0,"c - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sputn",570,G__G__stream_15_4_1,108,-1,G__defined_typename("streamsize"),0,2,1,1,0,
"C - 'basic_streambuf<char,char_traits<char> >::char_type' 10 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_streambuf<char,char_traits<char> >",3898,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,2,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,(G__InterfaceMethod)NULL,117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),-1,1,1,1,4,0,"u 'basic_streambuf<char,char_traits<char> >' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("basic_streambuf<char,char_traits<char> >",3898,G__G__stream_15_7_1,(int)('i'),
G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"u 'basic_streambuf<char,char_traits<char> >' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_streambuf<char,char_traits<char> >",4024,G__G__stream_15_8_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ostream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ostream<char,char_traits<char> >",3692,G__G__stream_16_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),-1,0,1,5,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("put",345,G__G__stream_16_2_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1,1,1,1,0,"c - 'basic_ostream<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("write",555,G__G__stream_16_3_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1,2,1,1,0,
"C - 'basic_ostream<char,char_traits<char> >::char_type' 10 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("flush",546,G__G__stream_16_4_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekp",536,G__G__stream_16_5_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1,1,1,1,0,"l - 'basic_ostream<char,char_traits<char> >::pos_type' 0 - pos",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekp",536,G__G__stream_16_6_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1,2,1,1,0,
"l - 'basic_ostream<char,char_traits<char> >::off_type' 0 - - i - 'ios_base::seekdir' 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tellp",545,G__G__stream_16_7_0,108,-1,G__defined_typename("basic_ostream<char,char_traits<char> >::pos_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ostream<char,char_traits<char> >",3692,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),-1,0,0,1,2,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ostream<char,char_traits<char> >",3818,G__G__stream_16_9_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_filebuflEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_filebuf<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >",3662,G__G__stream_19_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >",3662,G__G__stream_19_2_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__G__stream_19_3_0,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__G__stream_19_4_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 - - "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__G__stream_19_5_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__G__stream_19_6_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,(G__InterfaceMethod)NULL,117,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,1,1,1,4,0,"u 'basic_filebuf<char,char_traits<char> >' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >",3662,G__G__stream_19_7_1,(int)('i'),
G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"u 'basic_filebuf<char,char_traits<char> >' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_filebuf<char,char_traits<char> >",3788,G__G__stream_19_8_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ifstreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ifstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__G__stream_20_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__G__stream_20_2_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::in mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__G__stream_20_3_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__G__stream_20_4_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),-1,0,3,1,1,0,
"i - - 0 - fd C - 'basic_ifstream<char,char_traits<char> >::char_type' 0 - buf "
"i - - 0 - len",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_20_5_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__G__stream_20_6_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__G__stream_20_7_0,121,-1,-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::in mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__G__stream_20_8_0,121,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ifstream<char,char_traits<char> >",3914,G__G__stream_20_9_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ofstreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ofstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__G__stream_21_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__G__stream_21_2_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::out mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__G__stream_21_3_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__G__stream_21_4_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),-1,0,3,1,1,0,
"i - - 0 - fd C - 'basic_ofstream<char,char_traits<char> >::char_type' 0 - buf "
"i - - 0 - len",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_21_5_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__G__stream_21_6_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__G__stream_21_7_0,121,-1,-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::out mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__G__stream_21_8_0,121,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ofstream<char,char_traits<char> >",3920,G__G__stream_21_9_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_fstreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_fstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_fstream<char,char_traits<char> >",3683,G__G__stream_22_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_fstream<char,char_traits<char> >",3683,G__G__stream_22_2_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),-1,0,2,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 - mode",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_22_3_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__G__stream_22_4_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__G__stream_22_5_0,121,-1,-1,0,2,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 - mode",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__G__stream_22_6_0,121,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_fstream<char,char_traits<char> >",3809,G__G__stream_22_7_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_iostreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_iostream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_iostream<char,char_traits<char> >",3797,G__G__stream_23_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),-1,0,1,5,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_iostream<char,char_traits<char> >",3797,(G__InterfaceMethod)NULL,105,G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),-1,0,0,5,2,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_iostream<char,char_traits<char> >",3923,G__G__stream_23_3_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncstrstreambuf(void) {
   /* strstreambuf */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_strstreambuf));
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_1_0,105,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,1,1,1,0,"l - 'streamsize' 0 0 alsize",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_2_0,105,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,2,1,1,0,
"Q - 'void *(*)(size_t)' 0 - palloc Q - 'void (*)(void*)' 0 - pfree",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_3_0,105,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,3,1,1,0,
"C - - 0 - gnext l - 'streamsize' 0 - n "
"C - - 0 0 pbeg",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_4_0,105,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,3,1,1,0,
"B - - 0 - gnext l - 'streamsize' 0 - n "
"B - - 0 0 pbeg",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_5_0,105,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,2,1,1,0,
"C - - 10 - gnext l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_6_0,105,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,2,1,1,0,
"B - - 10 - gnext l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("freeze",641,G__G__stream_24_7_0,121,-1,-1,0,1,1,1,0,"g - - 0 1 f",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__G__stream_24_8_0,67,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pcount",665,G__G__stream_24_9_0,105,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("strstreambuf",1314,G__G__stream_24_0_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,1,1,1,0,"u 'strstreambuf' - 11 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~strstreambuf",1440,G__G__stream_24_1_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncistrstream(void) {
   /* istrstream */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_istrstream));
   G__memfunc_setup("istrstream",1102,G__G__stream_25_1_0,105,G__get_linked_tagnum(&G__G__streamLN_istrstream),-1,0,1,1,1,0,"C - - 10 - s",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("istrstream",1102,G__G__stream_25_2_0,105,G__get_linked_tagnum(&G__G__streamLN_istrstream),-1,0,2,1,1,0,
"C - - 10 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("istrstream",1102,G__G__stream_25_3_0,105,G__get_linked_tagnum(&G__G__streamLN_istrstream),-1,0,1,1,1,0,"C - - 0 - s",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("istrstream",1102,G__G__stream_25_4_0,105,G__get_linked_tagnum(&G__G__streamLN_istrstream),-1,0,2,1,1,0,
"C - - 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_25_5_0,85,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__G__stream_25_6_0,67,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~istrstream",1228,G__G__stream_25_7_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncostrstream(void) {
   /* ostrstream */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_ostrstream));
   G__memfunc_setup("ostrstream",1108,G__G__stream_26_1_0,105,G__get_linked_tagnum(&G__G__streamLN_ostrstream),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("ostrstream",1108,G__G__stream_26_2_0,105,G__get_linked_tagnum(&G__G__streamLN_ostrstream),-1,0,3,1,1,0,
"C - - 0 - s i - - 0 - n "
"i - 'ios_base::openmode' 0 ios_base::out -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__G__stream_26_3_0,85,G__get_linked_tagnum(&G__G__streamLN_strstreambuf),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("freeze",641,G__G__stream_26_4_0,121,-1,-1,0,1,1,1,0,"i - - 0 1 freezefl",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__G__stream_26_5_0,67,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pcount",665,G__G__stream_26_6_0,105,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~ostrstream",1234,G__G__stream_26_7_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}


/*********************************************************
* Member function information setup
*********************************************************/
extern "C" void G__cpp_setup_memfuncG__stream() {
}

/*********************************************************
* Global variable information setup for each class
*********************************************************/
static void G__cpp_setup_global0() {

   /* Setting up global variables */
   G__resetplocal();

   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__IOSTREAM_H=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__TMPLTIOS=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__OSTREAMBODY=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__OSTREAMGLOBALSTUB=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__MANIP_SUPPORT=0",1,(char*)NULL);
   G__memvar_setup((void*)(&cin),117,0,0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),-1,1,"cin=",0,(char*)NULL);
   G__memvar_setup((void*)(&cout),117,0,0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),-1,1,"cout=",0,(char*)NULL);
   G__memvar_setup((void*)(&cerr),117,0,0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),-1,1,"cerr=",0,(char*)NULL);
   G__memvar_setup((void*)(&clog),117,0,0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),-1,1,"clog=",0,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__FSTREAM_H=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__STRSTREAM_H=0",1,(char*)NULL);

   G__resetglobalenv();
}
extern "C" void G__cpp_setup_globalG__stream() {
  G__cpp_setup_global0();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
static void G__cpp_setup_func0() {
   G__lastifuncposition();

}

static void G__cpp_setup_func1() {
   G__memfunc_setup("dec",300,G__G__stream__4_12,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("hex",325,G__G__stream__5_12,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("oct",326,G__G__stream__6_12,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("fixed",528,G__G__stream__7_12,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("scientific",1057,G__G__stream__8_12,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("right",542,G__G__stream__9_12,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("left",427,G__G__stream__0_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("internal",861,G__G__stream__1_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("nouppercase",1189,G__G__stream__2_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("uppercase",968,G__G__stream__3_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("noskipws",894,G__G__stream__4_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("skipws",673,G__G__stream__5_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("noshowpos",1008,G__G__stream__6_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("showpos",787,G__G__stream__7_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("noshowpoint",1224,G__G__stream__8_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("showpoint",1003,G__G__stream__9_13,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("noshowbase",1081,G__G__stream__0_14,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("showbase",860,G__G__stream__1_14,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("noboolalpha",1167,G__G__stream__2_14,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("boolalpha",946,G__G__stream__3_14,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("ws",234,G__G__stream__4_14,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,1,1,1,0,"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("endl",419,G__G__stream__5_14,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - i",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("ends",426,G__G__stream__6_14,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - i",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("flush",546,G__G__stream__7_14,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__8_14,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - c - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__9_14,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - C - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__0_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - Y - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__1_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - b - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__2_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - s - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__3_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - r - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__4_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - i - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__5_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - h - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__6_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - l - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__7_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - k - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__8_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - f - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__9_15,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - d - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__G__stream__0_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - g - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__1_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - c - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__2_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - b - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__3_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - s - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__4_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - r - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__5_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - i - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__6_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - h - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__7_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - l - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__8_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - k - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__9_16,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - f - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__0_17,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - d - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__1_17,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - g - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__2_17,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - C - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__G__stream__3_17,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - Y - - 1 - -",(char*)NULL
,(void*)NULL,0);

   G__resetifuncposition();
}

extern "C" void G__cpp_setup_funcG__stream() {
  G__cpp_setup_func0();
  G__cpp_setup_func1();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__G__streamLN_mbstate_t = { "mbstate_t" , 115 , -1 };
G__linked_taginfo G__G__streamLN_ios_base = { "ios_base" , 99 , -1 };
G__linked_taginfo G__G__streamLN_ios_basecLcLio_state = { "ios_base::io_state" , 101 , -1 };
G__linked_taginfo G__G__streamLN_ios_basecLcLopen_mode = { "ios_base::open_mode" , 101 , -1 };
G__linked_taginfo G__G__streamLN_ios_basecLcLseek_dir = { "ios_base::seek_dir" , 101 , -1 };
G__linked_taginfo G__G__streamLN_ios_basecLcLfmt_flags = { "ios_base::fmt_flags" , 101 , -1 };
G__linked_taginfo G__G__streamLN_ios_basecLcLevent = { "ios_base::event" , 101 , -1 };
G__linked_taginfo G__G__streamLN_ios_basecLcLInit = { "ios_base::Init" , 99 , -1 };
G__linked_taginfo G__G__streamLN_char_traitslEchargR = { "char_traits<char>" , 115 , -1 };
G__linked_taginfo G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR = { "basic_istream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR = { "basic_ios<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR = { "basic_streambuf<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR = { "basic_ostream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry = { "basic_ostream<char,char_traits<char> >::sentry" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry = { "basic_istream<char,char_traits<char> >::sentry" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR = { "basic_filebuf<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR = { "basic_ifstream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR = { "basic_ofstream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR = { "basic_fstream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR = { "basic_iostream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_strstreambuf = { "strstreambuf" , 99 , -1 };
G__linked_taginfo G__G__streamLN_istrstream = { "istrstream" , 99 , -1 };
G__linked_taginfo G__G__streamLN_ostrstream = { "ostrstream" , 99 , -1 };

/* Reset class/struct taginfo */
extern "C" void G__cpp_reset_tagtableG__stream() {
  G__G__streamLN_mbstate_t.tagnum = -1 ;
  G__G__streamLN_ios_base.tagnum = -1 ;
  G__G__streamLN_ios_basecLcLio_state.tagnum = -1 ;
  G__G__streamLN_ios_basecLcLopen_mode.tagnum = -1 ;
  G__G__streamLN_ios_basecLcLseek_dir.tagnum = -1 ;
  G__G__streamLN_ios_basecLcLfmt_flags.tagnum = -1 ;
  G__G__streamLN_ios_basecLcLevent.tagnum = -1 ;
  G__G__streamLN_ios_basecLcLInit.tagnum = -1 ;
  G__G__streamLN_char_traitslEchargR.tagnum = -1 ;
  G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry.tagnum = -1 ;
  G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry.tagnum = -1 ;
  G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_strstreambuf.tagnum = -1 ;
  G__G__streamLN_istrstream.tagnum = -1 ;
  G__G__streamLN_ostrstream.tagnum = -1 ;
}


extern "C" void G__cpp_setup_tagtableG__stream() {

   /* Setting up class,struct,union tag entry */
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_mbstate_t),sizeof(mbstate_t),-1,0,(char*)NULL,G__setup_memvarmbstate_t,G__setup_memfuncmbstate_t);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_base),sizeof(ios_base),-1,3328,(char*)NULL,G__setup_memvarios_base,G__setup_memfuncios_base);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLio_state),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLseek_dir),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLevent),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit),0,-1,1280,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),sizeof(char_traits<char>),-1,0,(char*)NULL,G__setup_memvarchar_traitslEchargR,G__setup_memfuncchar_traitslEchargR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_istream<char,char_traits<char> >),-1,34048,(char*)NULL,G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),sizeof(basic_ios<char,char_traits<char> >),-1,36608,(char*)NULL,G__setup_memvarbasic_ioslEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_ioslEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),sizeof(basic_streambuf<char,char_traits<char> >),-1,3328,(char*)NULL,G__setup_memvarbasic_streambuflEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_streambuflEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_ostream<char,char_traits<char> >),-1,34048,(char*)NULL,G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),0,-1,36352,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),0,-1,33792,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),sizeof(basic_filebuf<char,char_traits<char> >),-1,36096,(char*)NULL,G__setup_memvarbasic_filebuflEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_filebuflEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_ifstream<char,char_traits<char> >),-1,34048,(char*)NULL,G__setup_memvarbasic_ifstreamlEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_ofstream<char,char_traits<char> >),-1,34048,(char*)NULL,G__setup_memvarbasic_ofstreamlEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_fstream<char,char_traits<char> >),-1,33024,(char*)NULL,G__setup_memvarbasic_fstreamlEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_fstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_iostream<char,char_traits<char> >),-1,34048,(char*)NULL,G__setup_memvarbasic_iostreamlEcharcOchar_traitslEchargRsPgR,G__setup_memfuncbasic_iostreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_strstreambuf),sizeof(strstreambuf),-1,34048,(char*)NULL,G__setup_memvarstrstreambuf,G__setup_memfuncstrstreambuf);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_istrstream),sizeof(istrstream),-1,33792,(char*)NULL,G__setup_memvaristrstream,G__setup_memfuncistrstream);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ostrstream),sizeof(ostrstream),-1,34048,(char*)NULL,G__setup_memvarostrstream,G__setup_memfuncostrstream);
}
extern "C" void G__cpp_setupG__stream(void) {
  G__check_setup_version(G__CREATEDLLREV,"G__cpp_setupG__stream()");
  G__set_cpp_environmentG__stream();
  G__cpp_setup_tagtableG__stream();

  G__cpp_setup_inheritanceG__stream();

  G__cpp_setup_typetableG__stream();

  G__cpp_setup_memvarG__stream();

  G__cpp_setup_memfuncG__stream();
  G__cpp_setup_globalG__stream();
  G__cpp_setup_funcG__stream();

   if(0==G__getsizep2memfunc()) G__get_sizep2memfuncG__stream();
  return;
}
