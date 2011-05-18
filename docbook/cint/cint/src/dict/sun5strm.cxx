/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************
* sunstrm.cxx
********************************************************/
#include "sunstrm.h"

#ifdef G__MEMTEST
#undef malloc
#undef free
#endif

extern "C" void G__cpp_reset_tagtableG__stream();

extern "C" void G__set_cpp_environmentG__stream() {
  G__add_compiledheader("iostrm.h");
  G__add_compiledheader("fstrm.h");
  G__add_compiledheader("sstrm.h");
  G__cpp_reset_tagtableG__stream();
}

class cintnew_t {};

const cintnew_t cintnew = cintnew_t(); 
// empty classtype to help us define a non-standard 
//new operator. See the class nothrow_t in the standard.

static void * operator new(size_t size, const cintnew_t &) {
  if(G__PVOID!=G__getgvp()) return((void*)G__getgvp());
#ifndef G__ROOT
  return(malloc(size));
#else
    return operator new(size);
#endif
}       
static void * operator new[](size_t size, const cintnew_t &) {
  if(G__PVOID!=G__getgvp()) return((void*)G__getgvp());
#ifndef G__ROOT
  return(malloc(size));
#else
    return operator new[](size);
#endif
}

static void operator delete(void *p, const cintnew_t &) {
   if ((long) p == G__getgvp() && G__PVOID != G__getgvp())
      return;
#ifndef G__ROOT
   free(p);
#else
   delete p;
#endif   
}

static void operator delete[](void *p, const cintnew_t &) {
   if ((long) p == G__getgvp() && G__PVOID != G__getgvp())
      return;
#ifndef G__ROOT
   free(p);
#else
   delete[] p;
#endif   
}



#include "dllrev.h"
extern "C" int G__cpp_dllrevG__stream() { return(G__CREATEDLLREV); }

/*********************************************************
* Member function Interface Method
*********************************************************/

/* ios_base */
static int G__ios_base_register_callback_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((ios_base*)(G__getstructoffset()))->register_callback((ios_base::event_callback)G__int(libp->para[0]),(int)G__int(libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_flags_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const ios_base*)(G__getstructoffset()))->flags());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_flags_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->flags((ios_base::fmtflags)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_setf_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->setf((ios_base::fmtflags)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_setf_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->setf((ios_base::fmtflags)G__int(libp->para[0]),(ios_base::fmtflags)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_unsetf_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((ios_base*)(G__getstructoffset()))->unsetf((ios_base::fmtflags)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_copyfmt_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ios_base& obj=((ios_base*)(G__getstructoffset()))->copyfmt(*(ios_base*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_precision_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((const ios_base*)(G__getstructoffset()))->precision());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_precision_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((ios_base*)(G__getstructoffset()))->precision((streamsize)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_width_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((const ios_base*)(G__getstructoffset()))->width());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_width_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((ios_base*)(G__getstructoffset()))->width((streamsize)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_xalloc_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base*)(G__getstructoffset()))->xalloc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_iword_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        long& obj=((ios_base*)(G__getstructoffset()))->iword((int)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_pword_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        void*& obj=((ios_base*)(G__getstructoffset()))->pword((int)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_is_synch_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((ios_base*)(G__getstructoffset()))->is_synch());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_base_sync_with_stdio_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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
static int G__ios_base_ios_base_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   ios_base *p;
   p=new(cintnew) ios_base(*(ios_base*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_ios_base);
   return(1 || funcname || hash || result7 || libp) ;
}


/* ios_base::Init */
static int G__ios_basecLcLInit_getinit_cnt__0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((ios_base::Init*)(G__getstructoffset()))->getinit_cnt_());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ios_basecLcLInit_Init_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   ios_base::Init *p=NULL;
   if(G__getaryconstruct()) p=new(cintnew) ios_base::Init[G__getaryconstruct()];
   else                    p=new(cintnew) ios_base::Init;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__ios_basecLcLInit_Init_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   ios_base::Init *p;
   p=new(cintnew) ios_base::Init(*(ios_base::Init*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit);
   return(1 || funcname || hash || result7 || libp) ;
}


/* char_traits<char> */
static int G__char_traitslEchargR_assign_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((char_traits<char>*)(G__getstructoffset()))->assign(*(char_traits<char>::char_type*)G__Charref(&libp->para[0]),*(char_traits<char>::char_type*)G__Charref(&libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_to_char_type_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((char_traits<char>*)(G__getstructoffset()))->to_char_type(*(char_traits<char>::int_type*)G__Intref(&libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_to_int_type_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((char_traits<char>*)(G__getstructoffset()))->to_int_type(*(char_traits<char>::char_type*)G__Charref(&libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_eq_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((char_traits<char>*)(G__getstructoffset()))->eq(*(char_traits<char>::char_type*)G__Charref(&libp->para[0]),*(char_traits<char>::char_type*)G__Charref(&libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_lt_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((char_traits<char>*)(G__getstructoffset()))->lt(*(char_traits<char>::char_type*)G__Charref(&libp->para[0]),*(char_traits<char>::char_type*)G__Charref(&libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_compare_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((char_traits<char>*)(G__getstructoffset()))->compare((const char_traits<char>::char_type*)G__int(libp->para[0]),(const char_traits<char>::char_type*)G__int(libp->para[1])
,(size_t)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_find_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((char_traits<char>*)(G__getstructoffset()))->find((const char_traits<char>::char_type*)G__int(libp->para[0]),(int)G__int(libp->para[1])
,*(char_traits<char>::char_type*)G__Charref(&libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_eq_int_type_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((char_traits<char>*)(G__getstructoffset()))->eq_int_type(*(char_traits<char>::int_type*)G__Intref(&libp->para[0]),*(char_traits<char>::int_type*)G__Intref(&libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_eof_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((char_traits<char>*)(G__getstructoffset()))->eof());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_not_eof_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((char_traits<char>*)(G__getstructoffset()))->not_eof(*(char_traits<char>::int_type*)G__Intref(&libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_length_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,104,(long)((char_traits<char>*)(G__getstructoffset()))->length((const char_traits<char>::char_type*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_copy_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((char_traits<char>*)(G__getstructoffset()))->copy((char_traits<char>::char_type*)G__int(libp->para[0]),(const char_traits<char>::char_type*)G__int(libp->para[1])
,(size_t)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_move_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((char_traits<char>*)(G__getstructoffset()))->move((char_traits<char>::char_type*)G__int(libp->para[0]),(const char_traits<char>::char_type*)G__int(libp->para[1])
,(size_t)G__int(libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__char_traitslEchargR_assign_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((char_traits<char>*)(G__getstructoffset()))->assign((char_traits<char>::char_type*)G__int(libp->para[0]),(size_t)G__int(libp->para[1])
,*(char_traits<char>::char_type*)G__Charref(&libp->para[2])));
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic default constructor
static int G__char_traitslEchargR_char_traitslEchargR_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   char_traits<char> *p;
   if(0!=libp->paran) ;
   if(G__getaryconstruct()) p=new(cintnew) char_traits<char>[G__getaryconstruct()];
   else                    p=new(cintnew) char_traits<char>;
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__char_traitslEchargR_char_traitslEchargR_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   char_traits<char> *p;
   p=new(cintnew) char_traits<char>(*(char_traits<char>*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__char_traitslEchargR_wAchar_traitslEchargR_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (char_traits<char> *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((char_traits<char> *)((G__getstructoffset())+sizeof(char_traits<char>)*i))->~char_traits<char>();
   else if(G__PVOID==G__getgvp()) delete (char_traits<char> *)(G__getstructoffset()),cintnew;
   else ((char_traits<char> *)(G__getstructoffset()))->~char_traits<char>();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_istream<char,char_traits<char> > */
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_istreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_istream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_istream<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->operator>>(*(basic_istream<char,char_traits<char> >::streambuf_type*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])
,(basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[2]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get(*(basic_istream<char,char_traits<char> >::char_type*)G__Charref(&libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get(*(basic_istream<char,char_traits<char> >::streambuf_type*)libp->para[0].ref,(basic_istream<char
,char_traits<char> >::char_type)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->get(*(basic_istream<char,char_traits<char> >::streambuf_type*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->getline((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])
,(basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[2]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->getline((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_ignore_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->ignore((streamsize)G__int(libp->para[0]),(basic_istream<char,char_traits<char> >::int_type)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_ignore_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->ignore((streamsize)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
      break;
   case 0:
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->ignore();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_read_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->read((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_readsome_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->readsome((basic_istream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_peek_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->peek());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_tellg_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,117,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->tellg());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_seekg_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->seekg(*((basic_istream<char,char_traits<char> >::pos_type*)G__int(libp->para[0])));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_sync_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->sync());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_putback_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->putback((basic_istream<char,char_traits<char> >::char_type)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_unget_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_istream<char,char_traits<char> >::istream_type& obj=((basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->unget();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_gcount_1_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((const basic_istream<char,char_traits<char> >*)(G__getstructoffset()))->gcount());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_wAbasic_istreamlEcharcOchar_traitslEchargRsPgR_3_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_istream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_istream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_istream<char,char_traits<char> >)*i))->~basic_istream<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) 
     delete (basic_istream<char,char_traits<char> > *)(G__getstructoffset()), cintnew;
   else ((basic_istream<char,char_traits<char> > *)(G__getstructoffset()))->~basic_istream<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ios<char,char_traits<char> > */
static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ios<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_ios<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->fill());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->fill((basic_ios<char,char_traits<char> >::char_type)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->exceptions((ios_base::iostate)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->exceptions());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_clear_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_setstate_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->setstate((ios_base::iostate)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdstate_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->rdstate());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatorsPvoidmU_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,89,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->operator void*());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatornO_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->operator!());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_good_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->good());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_eof_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->eof());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_fail_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->fail());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_bad_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->bad());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_copyfmt_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_ios<char,char_traits<char> >::ios_type& obj=((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->copyfmt(*(basic_ios<char,char_traits<char> >::ios_type*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->tie());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->tie((basic_ios<char,char_traits<char> >::ostream_type*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf((basic_ios<char,char_traits<char> >::streambuf_type*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_narrow_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->narrow((char)G__int(libp->para[0]),(char)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_widen_1_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,99,(long)((const basic_ios<char,char_traits<char> >*)(G__getstructoffset()))->widen((char)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_wAbasic_ioslEcharcOchar_traitslEchargRsPgR_6_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ios<char,char_traits<char> > *)(G__getstructoffset()), cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ios<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ios<char,char_traits<char> >)*i))->~basic_ios<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete(basic_ios<char,char_traits<char> > *)(G__getstructoffset()), cintnew;
   else ((basic_ios<char,char_traits<char> > *)(G__getstructoffset()))->~basic_ios<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_streambuf<char,char_traits<char> > */
static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsetbuf_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubsetbuf((basic_streambuf<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekoff_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 3:
      G__letint(result7,117,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekoff(*((basic_streambuf<char,char_traits<char> >::off_type*)G__int(libp->para[0])),(ios_base::seekdir)G__int(libp->para[1])
,(ios_base::openmode)G__int(libp->para[2])));
      break;
   case 2:
      G__letint(result7,117,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekoff(*((basic_streambuf<char,char_traits<char> >::off_type*)G__int(libp->para[0])),(ios_base::seekdir)G__int(libp->para[1])));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekpos_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 2:
      G__letint(result7,117,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekpos(*((basic_streambuf<char,char_traits<char> >::pos_type*)G__int(libp->para[0])),(ios_base::openmode)G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7,117,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubseekpos(*((basic_streambuf<char,char_traits<char> >::pos_type*)G__int(libp->para[0]))));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsync_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->pubsync());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_which_open_mode_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->which_open_mode());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_in_avail_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->in_avail());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_snextc_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->snextc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sbumpc_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sbumpc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetc_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sgetc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetn_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sgetn((basic_streambuf<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputbackc_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sputbackc((basic_streambuf<char,char_traits<char> >::char_type)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sungetc_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sungetc());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputc_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sputc((basic_streambuf<char,char_traits<char> >::char_type)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputn_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((basic_streambuf<char,char_traits<char> >*)(G__getstructoffset()))->sputn((const basic_streambuf<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_basic_streambuflEcharcOchar_traitslEchargRsPgR_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   basic_streambuf<char,char_traits<char> > *p;
   p=new(cintnew) basic_streambuf<char,char_traits<char> >(*(basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_wAbasic_streambuflEcharcOchar_traitslEchargRsPgR_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_streambuf<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_streambuf<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_streambuf<char,char_traits<char> >)*i))->~basic_streambuf<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete(basic_streambuf<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_streambuf<char,char_traits<char> > *)(G__getstructoffset()))->~basic_streambuf<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ostream<char,char_traits<char> > */
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ostreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ostream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_ostream<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_put_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->put((basic_ostream<char,char_traits<char> >::char_type)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_write_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->write((const basic_ostream<char,char_traits<char> >::char_type*)G__int(libp->para[0]),(streamsize)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_flush_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->flush();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->seekp(*((basic_ostream<char,char_traits<char> >::pos_type*)G__int(libp->para[0])));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        basic_ostream<char,char_traits<char> >::ostream_type& obj=((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->seekp(*((basic_ostream<char,char_traits<char> >::off_type*)G__int(libp->para[0]))
,(ios_base::seekdir)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_tellp_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,117,(long)((basic_ostream<char,char_traits<char> >*)(G__getstructoffset()))->tellp());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ostreamlEcharcOchar_traitslEchargRsPgR_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ostream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ostream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ostream<char,char_traits<char> >)*i))->~basic_ostream<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_ostream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_ostream<char,char_traits<char> > *)(G__getstructoffset()))->~basic_ostream<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ostream<char,char_traits<char> >::sentry */
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ostream<char,char_traits<char> >::sentry *p=NULL;
      p = new(cintnew) basic_ostream<char,char_traits<char> >::sentry(*(basic_ostream<char,char_traits<char> >*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_ostream<char,char_traits<char> >::sentry*)(G__getstructoffset()))->operator bool());
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_istream<char,char_traits<char> >::sentry */
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_istream<char,char_traits<char> >::sentry *p=NULL;
   switch(libp->paran) {
   case 2:
      p = new(cintnew) basic_istream<char,char_traits<char> >::sentry(*(basic_istream<char,char_traits<char> >*)libp->para[0].ref,(bool)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_istream<char,char_traits<char> >::sentry(*(basic_istream<char,char_traits<char> >*)libp->para[0].ref);
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_istream<char,char_traits<char> >::sentry*)(G__getstructoffset()))->operator bool());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   basic_istream<char,char_traits<char> >::sentry *p;
   p=new(cintnew) basic_istream<char,char_traits<char> >::sentry(*(basic_istream<char,char_traits<char> >::sentry*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_filebuf<char,char_traits<char> > */
static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_filebuf<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new(cintnew) basic_filebuf<char,char_traits<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_filebuf<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_filebuf<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_filebuf<char,char_traits<char> >((int)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_is_open_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((const basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_open_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_open_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->open((int)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_close_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((basic_filebuf<char,char_traits<char> >*)(G__getstructoffset()))->close());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   basic_filebuf<char,char_traits<char> > *p;
   p=new(cintnew) basic_filebuf<char,char_traits<char> >(*(basic_filebuf<char,char_traits<char> >*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_wAbasic_filebuflEcharcOchar_traitslEchargRsPgR_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_filebuf<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_filebuf<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_filebuf<char,char_traits<char> >)*i))->~basic_filebuf<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete(basic_filebuf<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_filebuf<char,char_traits<char> > *)(G__getstructoffset()))->~basic_filebuf<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ifstream<char,char_traits<char> > */
static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new(cintnew) basic_ifstream<char,char_traits<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_ifstream<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
   switch(libp->paran) {
   case 3:
      p = new(cintnew) basic_ifstream<char,char_traits<char> >(
(const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2]));
      break;
   case 2:
      p = new(cintnew) basic_ifstream<char,char_traits<char> >((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_ifstream<char,char_traits<char> >((const char*)G__int(libp->para[0]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_ifstream<char,char_traits<char> >((int)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ifstream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_ifstream<char,char_traits<char> >(
(int)G__int(libp->para[0]),(basic_ifstream<char,char_traits<char> >::char_type*)G__int(libp->para[1])
,(int)G__int(libp->para[2]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_is_open_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_open_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_close_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ifstream<char,char_traits<char> >*)(G__getstructoffset()))->close();
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ifstreamlEcharcOchar_traitslEchargRsPgR_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ifstream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ifstream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ifstream<char,char_traits<char> >)*i))->~basic_ifstream<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_ifstream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_ifstream<char,char_traits<char> > *)(G__getstructoffset()))->~basic_ifstream<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ofstream<char,char_traits<char> > */
static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
   if(G__getaryconstruct()) p=new(cintnew) basic_ofstream<char,char_traits<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_ofstream<char,char_traits<char> >;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
   switch(libp->paran) {
   case 3:
      p = new(cintnew) basic_ofstream<char,char_traits<char> >(
(const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1])
,(long)G__int(libp->para[2]));
      break;
   case 2:
      p = new(cintnew) basic_ofstream<char,char_traits<char> >((const char*)G__int(libp->para[0]),(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_ofstream<char,char_traits<char> >((const char*)G__int(libp->para[0]));
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_ofstream<char,char_traits<char> >((int)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ofstream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_ofstream<char,char_traits<char> >(
(int)G__int(libp->para[0]),(basic_ofstream<char,char_traits<char> >::char_type*)G__int(libp->para[1])
,(int)G__int(libp->para[2]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_is_open_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,103,(long)((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->is_open());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_open_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_close_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ofstream<char,char_traits<char> >*)(G__getstructoffset()))->close();
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ofstreamlEcharcOchar_traitslEchargRsPgR_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ofstream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ofstream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_ofstream<char,char_traits<char> >)*i))->~basic_ofstream<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_ofstream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_ofstream<char,char_traits<char> > *)(G__getstructoffset()))->~basic_ofstream<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_stringbuf<char,char_traits<char>,allocator<char> > */
static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_stringbuf<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 1:
      p = new(cintnew) basic_stringbuf<char,char_traits<char>,allocator<char> >((ios_base::openmode)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new(cintnew) basic_stringbuf<char,char_traits<char>,allocator<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_stringbuf<char,char_traits<char>,allocator<char> >;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_stringbuf<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 2:
      p = new(cintnew) basic_stringbuf<char,char_traits<char>,allocator<char> >(*(basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref,(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_stringbuf<char,char_traits<char>,allocator<char> >(*(basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
const         string *pobj,xobj=((const basic_stringbuf<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str();
        pobj=new(cintnew) string(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_stringbuf<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str(*(basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   basic_stringbuf<char,char_traits<char>,allocator<char> > *p;
   p=new(cintnew) basic_stringbuf<char,char_traits<char>,allocator<char> >(*(basic_stringbuf<char,char_traits<char>,allocator<char> >*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_stringbuf<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_stringbuf<char,char_traits<char>,allocator<char> > *)((G__getstructoffset())+sizeof(basic_stringbuf<char,char_traits<char>,allocator<char> >)*i))->~basic_stringbuf<char,char_traits<char>,allocator<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_stringbuf<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_stringbuf<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()))->~basic_stringbuf<char,char_traits<char>,allocator<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_istringstream<char,char_traits<char>,allocator<char> > */
static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_istringstream<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 1:
      p = new(cintnew) basic_istringstream<char,char_traits<char>,allocator<char> >((ios_base::openmode)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new(cintnew) basic_istringstream<char,char_traits<char>,allocator<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_istringstream<char,char_traits<char>,allocator<char> >;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_istringstream<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 2:
      p = new(cintnew) basic_istringstream<char,char_traits<char>,allocator<char> >(*(basic_istringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref,(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_istringstream<char,char_traits<char>,allocator<char> >(*(basic_istringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_istringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
const         string *pobj,xobj=((const basic_istringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str();
        pobj=new(cintnew) string(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_istringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str(*(basic_istringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_istringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_istringstream<char,char_traits<char>,allocator<char> > *)((G__getstructoffset())+sizeof(basic_istringstream<char,char_traits<char>,allocator<char> >)*i))->~basic_istringstream<char,char_traits<char>,allocator<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_istringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_istringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()))->~basic_istringstream<char,char_traits<char>,allocator<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_ostringstream<char,char_traits<char>,allocator<char> > */
static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ostringstream<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 1:
      p = new(cintnew) basic_ostringstream<char,char_traits<char>,allocator<char> >((ios_base::openmode)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new(cintnew) basic_ostringstream<char,char_traits<char>,allocator<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_ostringstream<char,char_traits<char>,allocator<char> >;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_ostringstream<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 2:
      p = new(cintnew) basic_ostringstream<char,char_traits<char>,allocator<char> >(*(basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref,(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_ostringstream<char,char_traits<char>,allocator<char> >(*(basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_ostringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
const         string *pobj,xobj=((const basic_ostringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str();
        pobj=new(cintnew) string(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_ostringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str(*(basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_ostringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_ostringstream<char,char_traits<char>,allocator<char> > *)((G__getstructoffset())+sizeof(basic_ostringstream<char,char_traits<char>,allocator<char> >)*i))->~basic_ostringstream<char,char_traits<char>,allocator<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_ostringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_ostringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()))->~basic_ostringstream<char,char_traits<char>,allocator<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_stringstream<char,char_traits<char>,allocator<char> > */
static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_stringstream<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 1:
      p = new(cintnew) basic_stringstream<char,char_traits<char>,allocator<char> >((ios_base::openmode)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new(cintnew) basic_stringstream<char,char_traits<char>,allocator<char> >[G__getaryconstruct()];
   else                    p=new(cintnew) basic_stringstream<char,char_traits<char>,allocator<char> >;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_stringstream<char,char_traits<char>,allocator<char> > *p=NULL;
   switch(libp->paran) {
   case 2:
      p = new(cintnew) basic_stringstream<char,char_traits<char>,allocator<char> >(*(basic_stringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref,(ios_base::openmode)G__int(libp->para[1]));
      break;
   case 1:
      p = new(cintnew) basic_stringstream<char,char_traits<char>,allocator<char> >(*(basic_stringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((const basic_stringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->rdbuf());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
const         string *pobj,xobj=((const basic_stringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str();
        pobj=new(cintnew) string(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((basic_stringstream<char,char_traits<char>,allocator<char> >*)(G__getstructoffset()))->str(*(basic_stringstream<char,char_traits<char>,allocator<char> >::string_type*)libp->para[0].ref);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_stringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_stringstream<char,char_traits<char>,allocator<char> > *)((G__getstructoffset())+sizeof(basic_stringstream<char,char_traits<char>,allocator<char> >)*i))->~basic_stringstream<char,char_traits<char>,allocator<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_stringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_stringstream<char,char_traits<char>,allocator<char> > *)(G__getstructoffset()))->~basic_stringstream<char,char_traits<char>,allocator<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* basic_iostream<char,char_traits<char> > */
static int G__basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_iostreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   basic_iostream<char,char_traits<char> > *p=NULL;
      p = new(cintnew) basic_iostream<char,char_traits<char> >((basic_streambuf<char,char_traits<char> >*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
static int G__basic_iostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_iostreamlEcharcOchar_traitslEchargRsPgR_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (basic_iostream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((basic_iostream<char,char_traits<char> > *)((G__getstructoffset())+sizeof(basic_iostream<char,char_traits<char> >)*i))->~basic_iostream<char,char_traits<char> >();
   else if(G__PVOID==G__getgvp()) delete (basic_iostream<char,char_traits<char> > *)(G__getstructoffset()),cintnew;
   else ((basic_iostream<char,char_traits<char> > *)(G__getstructoffset()))->~basic_iostream<char,char_traits<char> >();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* Setting up global function */
static int G___ws_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=ws(*(istream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___endl_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=endl(*(ostream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___ends_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=ends(*(ostream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___flush_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=flush(*(ostream*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(char)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(char*)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(void*)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned char)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(short)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned short)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(int)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned int)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(long)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(unsigned long)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(float)G__double(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(double)G__double(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        ostream& obj=operator<<(*(ostream*)libp->para[0].ref,(bool)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(char*)G__Charref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned char*)G__UCharref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(short*)G__Shortref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned short*)G__UShortref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_1_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(int*)G__Intref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_2_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned int*)G__UIntref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_3_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(long*)G__Longref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_4_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(unsigned long*)G__ULongref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_5_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(float*)G__Floatref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_6_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(double*)G__Doubleref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_7_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,*(bool*)G__Boolref(&libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_8_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,(char*)G__int(libp->para[1]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_9_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        istream& obj=operator>>(*(istream*)libp->para[0].ref,libp->para[1].ref?*(void**)libp->para[1].ref:*(void**)(&G__Mlong(libp->para[1])));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}


/*********************************************************
* Member function Stub
*********************************************************/

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

/* basic_stringbuf<char,char_traits<char>,allocator<char> > */

/* basic_istringstream<char,char_traits<char>,allocator<char> > */

/* basic_ostringstream<char,char_traits<char>,allocator<char> > */

/* basic_stringstream<char,char_traits<char>,allocator<char> > */

/* basic_iostream<char,char_traits<char> > */

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

static long G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  basic_istringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_istringstream<char,char_traits<char>,allocator<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2(long pobject) {
  basic_istringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_istringstream<char,char_traits<char>,allocator<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject) {
  basic_ostringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_ostringstream<char,char_traits<char>,allocator<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2(long pobject) {
  basic_ostringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_ostringstream<char,char_traits<char>,allocator<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2(long pobject) {
  basic_stringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_stringstream<char,char_traits<char>,allocator<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_3(long pobject) {
  basic_stringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_stringstream<char,char_traits<char>,allocator<char> >*)pobject;
  ios_base *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5(long pobject) {
  basic_stringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_stringstream<char,char_traits<char>,allocator<char> >*)pobject;
  basic_ios<char,char_traits<char> > *G__Lbase=G__Lderived;
  return((long)G__Lbase-(long)G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_6(long pobject) {
  basic_stringstream<char,char_traits<char>,allocator<char> > *G__Lderived=(basic_stringstream<char,char_traits<char>,allocator<char> >*)pobject;
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


/*********************************************************
* Inheritance information setup/
*********************************************************/
extern "C" void G__cpp_setup_inheritanceG__stream() {

   /* Setting up class inheritance */
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR))) {
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0,1,3);
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
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0,1,3);
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
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
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
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
     basic_stringbuf<char,char_traits<char>,allocator<char> > *G__Lderived;
     G__Lderived=(basic_stringbuf<char,char_traits<char>,allocator<char> >*)0x1000;
     {
       basic_streambuf<char,char_traits<char> > *G__Lpbase=(basic_streambuf<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1
,1);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
     basic_istringstream<char,char_traits<char>,allocator<char> > *G__Lderived;
     G__Lderived=(basic_istringstream<char,char_traits<char>,allocator<char> >*)0x1000;
     {
       basic_istream<char,char_traits<char> > *G__Lpbase=(basic_istream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1
,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base)
,(long)G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
     basic_ostringstream<char,char_traits<char>,allocator<char> > *G__Lderived;
     G__Lderived=(basic_ostringstream<char,char_traits<char>,allocator<char> >*)0x1000;
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1
,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base)
,(long)G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2,1,2);
     }
   }
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
     basic_stringstream<char,char_traits<char>,allocator<char> > *G__Lderived;
     G__Lderived=(basic_stringstream<char,char_traits<char>,allocator<char> >*)0x1000;
     {
       basic_iostream<char,char_traits<char> > *G__Lpbase=(basic_iostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1
,1);
     }
     {
       basic_istream<char,char_traits<char> > *G__Lpbase=(basic_istream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1
,0);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base)
,(long)G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_3,1,2);
     }
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1
,0);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base)
,(long)G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_6,1,2);
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
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_2,1,2);
     }
     {
       basic_ostream<char,char_traits<char> > *G__Lpbase=(basic_ostream<char,char_traits<char> >*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR)
,(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_4,1,2);
     }
     {
       G__inheritance_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),G__get_linked_tagnum(&G__G__streamLN_ios_base),(long)G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_5,1,2);
     }
   }
}

/*********************************************************
* typedef information setup/
*********************************************************/
extern "C" void G__cpp_setup_typetableG__stream() {

   /* Setting up typedef entry */
   G__search_typename2("size_t",104,-1,0,
-1);
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
   G__search_typename2("iostate",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("openmode",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("seekdir",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("fmtflags",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("event_callback",89,-1,0,
G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("INT_T",105,-1,0,
-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("basic_streambuf<char_type,char_traits<char> >",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,"/* /% C++ %/ */",0);
   G__search_typename2("streambuf_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ostream_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ostream_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("istream_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("streambuf_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("istream",117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ostream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("filebuf",117,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ifstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ofstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("cstring",117,G__get_linked_tagnum(&G__G__streamLN_string),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("string_type",117,G__get_linked_tagnum(&G__G__streamLN_string),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("stringbuf",117,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("sb_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("string_type",117,G__get_linked_tagnum(&G__G__streamLN_string),0,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("istringstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("sb_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("string_type",117,G__get_linked_tagnum(&G__G__streamLN_string),0,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ostringstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("char_type",99,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("int_type",105,-1,0,
G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("pos_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("off_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("traits_type",117,G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("sb_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("ios_type",117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("string_type",117,G__get_linked_tagnum(&G__G__streamLN_string),0,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("stringstream",117,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

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


   /* ios_base::io_state */

   /* ios_base::open_mode */

   /* ios_base::seek_dir */

   /* ios_base::fmt_flags */

   /* ios_base::event */

   /* ios_base::Init */
static void G__setup_memvarios_basecLcLInit(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit));
   { ios_base::Init *p; p=(ios_base::Init*)0x1000; if (p) { }
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


   /* basic_ostream<char,char_traits<char> >::sentry */
static void G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   { basic_ostream<char,char_traits<char> >::sentry *p; p=(basic_ostream<char,char_traits<char> >::sentry*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_istream<char,char_traits<char> >::sentry */
static void G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   { basic_istream<char,char_traits<char> >::sentry *p; p=(basic_istream<char,char_traits<char> >::sentry*)0x1000; if (p) { }
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


   /* basic_stringbuf<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   { basic_stringbuf<char,char_traits<char>,allocator<char> > *p; p=(basic_stringbuf<char,char_traits<char>,allocator<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_istringstream<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   { basic_istringstream<char,char_traits<char>,allocator<char> > *p; p=(basic_istringstream<char,char_traits<char>,allocator<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_ostringstream<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   { basic_ostringstream<char,char_traits<char>,allocator<char> > *p; p=(basic_ostringstream<char,char_traits<char>,allocator<char> >*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* basic_stringstream<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   { basic_stringstream<char,char_traits<char>,allocator<char> > *p; p=(basic_stringstream<char,char_traits<char>,allocator<char> >*)0x1000; if (p) { }
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
static void G__setup_memfuncios_base(void) {
   /* ios_base */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_ios_base));
   G__memfunc_setup("register_callback",1777,G__ios_base_register_callback_0_0,121,-1,-1,0,2,1,1,0,
"Y - 'ios_base::event_callback' 0 - fn i - - 0 - index",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("flags",525,G__ios_base_flags_1_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("flags",525,G__ios_base_flags_2_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,1,1,1,0,"i - 'ios_base::fmtflags' 0 - fmtfl",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("setf",434,G__ios_base_setf_3_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,1,1,1,0,"i - 'ios_base::fmtflags' 0 - fmtfl",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("setf",434,G__ios_base_setf_4_0,105,-1,G__defined_typename("ios_base::fmtflags"),0,2,1,1,0,
"i - 'ios_base::fmtflags' 0 - fmtfl i - 'ios_base::fmtflags' 0 - mask",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("unsetf",661,G__ios_base_unsetf_5_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::fmtflags' 0 - mask",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("copyfmt",770,G__ios_base_copyfmt_6_0,117,G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,1,1,1,1,0,"u 'ios_base' - 11 - rhs",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("precision",972,G__ios_base_precision_7_0,108,-1,G__defined_typename("streamsize"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("precision",972,G__ios_base_precision_8_0,108,-1,G__defined_typename("streamsize"),0,1,1,1,0,"l - 'streamsize' 0 - prec",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("width",544,G__ios_base_width_9_0,108,-1,G__defined_typename("streamsize"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("width",544,G__ios_base_width_0_1,108,-1,G__defined_typename("streamsize"),0,1,1,1,0,"l - 'streamsize' 0 - wide",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("xalloc",643,G__ios_base_xalloc_1_1,105,-1,-1,0,0,3,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("iword",549,G__ios_base_iword_2_1,108,-1,-1,1,1,1,1,0,"i - - 0 - index",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pword",556,G__ios_base_pword_3_1,89,-1,-1,1,1,1,1,0,"i - - 0 - index",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_synch",864,G__ios_base_is_synch_4_1,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sync_with_stdio",1626,G__ios_base_sync_with_stdio_5_1,103,-1,-1,0,1,1,1,0,"g - - 0 true sync",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("ios_base",837,G__ios_base_ios_base_8_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_ios_base),-1,0,1,1,1,0,"u 'ios_base' - 1 - -",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncios_basecLcLInit(void) {
   /* ios_base::Init */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit));
   G__memfunc_setup("getinit_cnt_",1271,G__ios_basecLcLInit_getinit_cnt__0_0,105,-1,-1,0,0,3,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("Init",404,G__ios_basecLcLInit_Init_1_0,105,G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("Init",404,G__ios_basecLcLInit_Init_3_0,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit),-1,0,1,1,1,0,"u 'ios_base::Init' - 1 - -",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncchar_traitslEchargR(void) {
   /* char_traits<char> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR));
   G__memfunc_setup("assign",645,G__char_traitslEchargR_assign_0_0,121,-1,-1,0,2,3,1,0,
"c - 'char_traits<char>::char_type' 1 - c1 c - 'char_traits<char>::char_type' 11 - c2",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("to_char_type",1281,G__char_traitslEchargR_to_char_type_1_0,99,-1,G__defined_typename("char_traits<char>::char_type"),0,1,3,1,0,"i - 'char_traits<char>::int_type' 11 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("to_int_type",1198,G__char_traitslEchargR_to_int_type_2_0,105,-1,G__defined_typename("char_traits<char>::int_type"),0,1,3,1,0,"c - 'char_traits<char>::char_type' 11 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("eq",214,G__char_traitslEchargR_eq_3_0,103,-1,-1,0,2,3,1,0,
"c - 'char_traits<char>::char_type' 11 - c1 c - 'char_traits<char>::char_type' 11 - c2",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("lt",224,G__char_traitslEchargR_lt_4_0,103,-1,-1,0,2,3,1,0,
"c - 'char_traits<char>::char_type' 11 - c1 c - 'char_traits<char>::char_type' 11 - c2",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("compare",743,G__char_traitslEchargR_compare_5_0,105,-1,-1,0,3,3,1,0,
"C - 'char_traits<char>::char_type' 10 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
"h - 'size_t' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("find",417,G__char_traitslEchargR_find_6_0,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,1,
"C - 'char_traits<char>::char_type' 10 - s i - - 0 - n "
"c - 'char_traits<char>::char_type' 11 - a",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("eq_int_type",1185,G__char_traitslEchargR_eq_int_type_7_0,103,-1,-1,0,2,3,1,0,
"i - 'char_traits<char>::int_type' 11 - c1 i - 'char_traits<char>::int_type' 11 - c2",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("eof",314,G__char_traitslEchargR_eof_8_0,105,-1,G__defined_typename("char_traits<char>::int_type"),0,0,3,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("not_eof",746,G__char_traitslEchargR_not_eof_9_0,105,-1,G__defined_typename("char_traits<char>::int_type"),0,1,3,1,0,"i - 'char_traits<char>::int_type' 11 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("length",642,G__char_traitslEchargR_length_0_1,104,-1,G__defined_typename("size_t"),0,1,3,1,0,"C - 'char_traits<char>::char_type' 10 - s",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("copy",443,G__char_traitslEchargR_copy_1_1,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,0,
"C - 'char_traits<char>::char_type' 0 - dst C - 'char_traits<char>::char_type' 10 - src "
"h - 'size_t' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("move",439,G__char_traitslEchargR_move_2_1,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,0,
"C - 'char_traits<char>::char_type' 0 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
"h - 'size_t' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("assign",645,G__char_traitslEchargR_assign_3_1,67,-1,G__defined_typename("char_traits<char>::char_type"),0,3,3,1,0,
"C - 'char_traits<char>::char_type' 0 - s h - 'size_t' 0 - n "
"c - 'char_traits<char>::char_type' 11 - a",(char*)NULL,(void*)NULL,0);
   // automatic default constructor
   G__memfunc_setup("char_traits<char>",1708,G__char_traitslEchargR_char_traitslEchargR_4_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("char_traits<char>",1708,G__char_traitslEchargR_char_traitslEchargR_5_1,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),-1,0,1,1,1,0,"u 'char_traits<char>' - 1 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~char_traits<char>",1834,G__char_traitslEchargR_wAchar_traitslEchargR_6_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_istream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_istream<char,char_traits<char> >",3686,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_istreamlEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),-1,0
,1,1,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_2_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR)
,G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"u 'basic_streambuf<char,char_traits<char> >' 'basic_istream<char,char_traits<char> >::streambuf_type' 1 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_3_0,105,-1,G__defined_typename("basic_istream<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_4_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,3,1
,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n "
"c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_5_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1
,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_6_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1
,1,0,"c - 'basic_istream<char,char_traits<char> >::char_type' 1 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_7_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1
,1,0,
"u 'basic_streambuf<char,char_traits<char> >' 'basic_istream<char,char_traits<char> >::streambuf_type' 1 - sb c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("get",320,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_8_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1
,1,0,"u 'basic_streambuf<char,char_traits<char> >' 'basic_istream<char,char_traits<char> >::streambuf_type' 1 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("getline",744,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_9_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR)
,G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,3,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n "
"c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("getline",744,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_0_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR)
,G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("ignore",644,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_ignore_1_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type")
,1,2,1,1,0,
"l - 'streamsize' 0 - n i - 'basic_istream<char,char_traits<char> >::int_type' 0 - delim",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("ignore",644,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_ignore_2_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type")
,1,1,1,1,0,"l - 'streamsize' 0 1 n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("read",412,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_read_3_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,2
,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("readsome",848,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_readsome_4_1,108,-1,G__defined_typename("streamsize"),0,2,1,1,0,
"C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("peek",421,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_peek_5_1,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tellg",536,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_tellg_6_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type)
,G__defined_typename("basic_istream<char,char_traits<char> >::pos_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekg",527,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_seekg_7_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1
,1,1,1,0,"u 'basic_ios<char,char_traits<char> >::pos_type' 'basic_istream<char,char_traits<char> >::pos_type' 0 - pos",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sync",445,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_sync_8_1,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("putback",746,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_putback_9_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR)
,G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1,1,1,1,0,"c - 'basic_istream<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("unget",547,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_unget_0_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_istream<char,char_traits<char> >::istream_type"),1
,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("gcount",656,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_gcount_1_2,108,-1,G__defined_typename("streamsize"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_istream<char,char_traits<char> >",3812,G__basic_istreamlEcharcOchar_traitslEchargRsPgR_wAbasic_istreamlEcharcOchar_traitslEchargRsPgR_3_2,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ioslEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ios<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ios<char,char_traits<char> >",3260,G__basic_ioslEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0
,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb_arg",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("fill",423,G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_2_0,99,-1,G__defined_typename("basic_ios<char,char_traits<char> >::char_type"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("fill",423,G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_3_0,99,-1,G__defined_typename("basic_ios<char,char_traits<char> >::char_type"),0,1,1,1,0,"c - 'basic_ios<char,char_traits<char> >::char_type' 0 - ch",(char*)NULL,(void*)NULL
,0);
   G__memfunc_setup("exceptions",1090,G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_4_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::iostate' 0 - excpt",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("exceptions",1090,G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_5_0,105,-1,G__defined_typename("ios_base::iostate"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("clear",519,G__basic_ioslEcharcOchar_traitslEchargRsPgR_clear_6_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::iostate' 0 goodbit state",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("setstate",877,G__basic_ioslEcharcOchar_traitslEchargRsPgR_setstate_7_0,121,-1,-1,0,1,1,1,0,"i - 'ios_base::iostate' 0 - state",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdstate",759,G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdstate_8_0,105,-1,G__defined_typename("ios_base::iostate"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator void*",1384,G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatorsPvoidmU_9_0,89,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator!",909,G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatornO_0_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("good",425,G__basic_ioslEcharcOchar_traitslEchargRsPgR_good_1_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("eof",314,G__basic_ioslEcharcOchar_traitslEchargRsPgR_eof_2_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("fail",412,G__basic_ioslEcharcOchar_traitslEchargRsPgR_fail_3_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("bad",295,G__basic_ioslEcharcOchar_traitslEchargRsPgR_bad_4_1,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("copyfmt",770,G__basic_ioslEcharcOchar_traitslEchargRsPgR_copyfmt_5_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::ios_type"),1,1,1,1,0
,"u 'basic_ios<char,char_traits<char> >' 'basic_ios<char,char_traits<char> >::ios_type' 11 - rhs",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tie",322,G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_6_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::ostream_type"),0,0,1,1,8,""
,(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tie",322,G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_7_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::ostream_type"),0,1,1,1,0
,"U 'basic_ostream<char,char_traits<char> >' 'basic_ios<char,char_traits<char> >::ostream_type' 0 - tie_arg",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_8_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::streambuf_type"),0,0,1,1
,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_9_1,85,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ios<char,char_traits<char> >::streambuf_type"),0,1,1,1
,0,"U 'basic_streambuf<char,char_traits<char> >' 'basic_ios<char,char_traits<char> >::streambuf_type' 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("narrow",665,G__basic_ioslEcharcOchar_traitslEchargRsPgR_narrow_0_2,99,-1,-1,0,2,1,1,8,
"c - - 0 - - c - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("widen",535,G__basic_ioslEcharcOchar_traitslEchargRsPgR_widen_1_2,99,-1,-1,0,1,1,1,8,"c - - 0 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ios<char,char_traits<char> >",3386,G__basic_ioslEcharcOchar_traitslEchargRsPgR_wAbasic_ioslEcharcOchar_traitslEchargRsPgR_6_2,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_streambuflEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_streambuf<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("pubsetbuf",976,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsetbuf_1_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR)
,G__defined_typename("basic_streambuf<char_type,char_traits<char> >"),0,2,1,1,0,
"C - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pubseekoff",1066,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekoff_2_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type)
,G__defined_typename("basic_streambuf<char,char_traits<char> >::pos_type"),0,3,1,1,0,
"u 'basic_streambuf<char,char_traits<char> >::off_type' 'basic_streambuf<char,char_traits<char> >::off_type' 0 - off i - 'ios_base::seekdir' 0 - way "
"i - 'ios_base::openmode' 0 ios_base::in|ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pubseekpos",1089,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekpos_3_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type)
,G__defined_typename("basic_streambuf<char,char_traits<char> >::pos_type"),0,2,1,1,0,
"u 'basic_streambuf<char,char_traits<char> >::pos_type' 'basic_streambuf<char,char_traits<char> >::pos_type' 0 - sp i - 'ios_base::openmode' 0 ios_base::in|ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("pubsync",772,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsync_4_0,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("which_open_mode",1576,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_which_open_mode_5_0,105,-1,G__defined_typename("ios_base::openmode"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("in_avail",835,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_in_avail_6_0,108,-1,G__defined_typename("streamsize"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("snextc",661,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_snextc_7_0,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sbumpc",650,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sbumpc_8_0,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sgetc",534,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetc_9_0,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sgetn",545,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetn_0_1,108,-1,G__defined_typename("streamsize"),0,2,1,1,0,
"C - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sputbackc",960,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputbackc_1_1,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,1,1,1,0
,"c - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sungetc",761,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sungetc_2_1,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sputc",559,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputc_3_1,105,-1,G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"),0,1,1,1,0,"c - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - c"
,(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("sputn",570,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputn_4_1,108,-1,G__defined_typename("streamsize"),0,2,1,1,0,
"C - 'basic_streambuf<char,char_traits<char> >::char_type' 10 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("basic_streambuf<char,char_traits<char> >",3898,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_basic_streambuflEcharcOchar_traitslEchargRsPgR_6_1,(int)('i'),
G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"u 'basic_streambuf<char,char_traits<char> >' - 1 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_streambuf<char,char_traits<char> >",4024,G__basic_streambuflEcharcOchar_traitslEchargRsPgR_wAbasic_streambuflEcharcOchar_traitslEchargRsPgR_7_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ostream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ostream<char,char_traits<char> >",3692,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ostreamlEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),-1,0
,1,1,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("put",345,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_put_2_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1,1,1
,1,0,"c - 'basic_ostream<char,char_traits<char> >::char_type' 0 - c",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("write",555,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_write_3_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1
,2,1,1,0,
"C - 'basic_ostream<char,char_traits<char> >::char_type' 10 - s l - 'streamsize' 0 - n",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("flush",546,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_flush_4_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1
,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekp",536,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_5_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1
,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >::pos_type' 'basic_ostream<char,char_traits<char> >::pos_type' 0 - pos",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("seekp",536,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_6_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"),1
,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >::off_type' 'basic_ostream<char,char_traits<char> >::off_type' 0 - - i - 'ios_base::seekdir' 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("tellp",545,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_tellp_7_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLpos_type)
,G__defined_typename("basic_ostream<char,char_traits<char> >::pos_type"),0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ostream<char,char_traits<char> >",3818,G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ostreamlEcharcOchar_traitslEchargRsPgR_9_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void) {
   /* basic_ostream<char,char_traits<char> >::sentry */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   G__memfunc_setup("sentry",677,G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),-1,0,1,1,1,0
,"u 'basic_ostream<char,char_traits<char> >' - 1 - stream",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator bool",1336,G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void) {
   /* basic_istream<char,char_traits<char> >::sentry */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   G__memfunc_setup("sentry",677,G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),-1,0,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' - 1 - stream g - - 0 0 noskipws",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator bool",1336,G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("sentry",677,G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_3_0,(int)('i'),G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),-1,0,1,1,1,0
,"u 'basic_istream<char,char_traits<char> >::sentry' - 1 - -",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_filebuflEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_filebuf<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >",3662,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0
,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >",3662,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0
,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_is_open_3_0,103,-1,-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_open_4_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 - - "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_open_5_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_close_6_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >",3662,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_6_1,(int)('i'),
G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,1,1,1,0,"u 'basic_filebuf<char,char_traits<char> >' - 1 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_filebuf<char,char_traits<char> >",3788,G__basic_filebuflEcharcOchar_traitslEchargRsPgR_wAbasic_filebuflEcharcOchar_traitslEchargRsPgR_7_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ifstreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ifstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::in mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_2_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >",3788,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_3_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,3,1,1,0,
"i - - 0 - fd C - 'basic_ifstream<char,char_traits<char> >::char_type' 0 - buf "
"i - - 0 - len",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_5_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_is_open_6_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_open_7_0,121,-1,-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::in mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_close_8_0,121,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ifstream<char,char_traits<char> >",3914,G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ifstreamlEcharcOchar_traitslEchargRsPgR_9_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ofstreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_ofstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_1_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::out mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_2_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,1,1,1,0,"i - - 0 - fd",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >",3794,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_3_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,3,1,1,0,
"i - - 0 - fd C - 'basic_ofstream<char,char_traits<char> >::char_type' 0 - buf "
"i - - 0 - len",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_5_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("is_open",749,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_is_open_6_0,103,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("open",434,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_open_7_0,121,-1,-1,0,3,1,1,0,
"C - - 10 - s i - 'ios_base::openmode' 0 ios_base::out mode "
"l - - 0 0666 protection",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("close",534,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_close_8_0,121,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_ofstream<char,char_traits<char> >",3920,G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ofstreamlEcharcOchar_traitslEchargRsPgR_9_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   /* basic_stringbuf<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("basic_stringbuf<char,char_traits<char>,allocator<char> >",5450,G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,1,1,1,0,"i - 'ios_base::openmode' 0 ios_base::in|ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_stringbuf<char,char_traits<char>,allocator<char> >",5450,G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,2,1,1,0,
"u 'string' 'basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type' 11 - str i - 'ios_base::openmode' 0 ios_base::in|ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_3_0,117,G__get_linked_tagnum(&G__G__streamLN_string),G__defined_typename("basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type"),0,0,1
,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0,121,-1,-1,0,1,1,1,0,"u 'string' 'basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type' 11 - str_arg",(char*)NULL,(void*)NULL,0);
   // automatic copy constructor
   G__memfunc_setup("basic_stringbuf<char,char_traits<char>,allocator<char> >",5450,G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_2_1,(int)('i'),
G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,1,1,1,0,"u 'basic_stringbuf<char,char_traits<char>,allocator<char> >' - 1 - -",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_stringbuf<char,char_traits<char>,allocator<char> >",5576,G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_3_1,(int)('y'),-1,-1,0,0,1,1,0,""
,(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   /* basic_istringstream<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("basic_istringstream<char,char_traits<char>,allocator<char> >",5890,G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,1,1,1,0,"i - 'ios_base::openmode' 0 ios_base::in which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_istringstream<char,char_traits<char>,allocator<char> >",5890,G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,2,1,1,0,
"u 'string' 'basic_istringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str i - 'ios_base::openmode' 0 ios_base::in which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0,117,G__get_linked_tagnum(&G__G__streamLN_string)
,G__defined_typename("basic_istringstream<char,char_traits<char>,allocator<char> >::string_type"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0,121,-1,-1,0,1,1,1,0,"u 'string' 'basic_istringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str",(char*)NULL,(void*)NULL
,0);
   // automatic destructor
   G__memfunc_setup("~basic_istringstream<char,char_traits<char>,allocator<char> >",6016,G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0,(int)('y'),-1,-1
,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   /* basic_ostringstream<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("basic_ostringstream<char,char_traits<char>,allocator<char> >",5896,G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,1,1,1,0,"i - 'ios_base::openmode' 0 ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_ostringstream<char,char_traits<char>,allocator<char> >",5896,G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,2,1,1,0,
"u 'string' 'basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str i - 'ios_base::openmode' 0 ios_base::out which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0,117,G__get_linked_tagnum(&G__G__streamLN_string)
,G__defined_typename("basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type"),0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0,121,-1,-1,0,1,1,1,0,"u 'string' 'basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str",(char*)NULL,(void*)NULL
,0);
   // automatic destructor
   G__memfunc_setup("~basic_ostringstream<char,char_traits<char>,allocator<char> >",6022,G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0,(int)('y'),-1,-1
,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void) {
   /* basic_stringstream<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("basic_stringstream<char,char_traits<char>,allocator<char> >",5785,G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_0_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,1,1,1,0,"i - 'ios_base::openmode' 0 ios_base::out|ios_base::in which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("basic_stringstream<char,char_traits<char>,allocator<char> >",5785,G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0,105
,G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,2,1,1,0,
"u 'string' 'basic_stringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str i - 'ios_base::openmode' 0 ios_base::out|ios_base::in which",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("rdbuf",531,G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0,85,G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),-1,0,0,1,1,8,"",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0,117,G__get_linked_tagnum(&G__G__streamLN_string),G__defined_typename("basic_stringstream<char,char_traits<char>,allocator<char> >::string_type")
,0,0,1,1,8,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("str",345,G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0,121,-1,-1,0,1,1,1,0,"u 'string' 'basic_stringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_stringstream<char,char_traits<char>,allocator<char> >",5911,G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0,(int)('y'),-1,-1,0,0
,1,1,0,"",(char*)NULL,(void*)NULL,1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_iostreamlEcharcOchar_traitslEchargRsPgR(void) {
   /* basic_iostream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_iostream<char,char_traits<char> >",3797,G__basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_iostreamlEcharcOchar_traitslEchargRsPgR_0_0,105,G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR)
,-1,0,1,1,1,0,"U 'basic_streambuf<char,char_traits<char> >' - 0 - sb",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~basic_iostream<char,char_traits<char> >",3923,G__basic_iostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_iostreamlEcharcOchar_traitslEchargRsPgR_3_0,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,1);
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
extern "C" void G__cpp_setup_globalG__stream() {

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
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__SSTREAM_H=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__STRING_DLL=0",1,(char*)NULL);

   G__resetglobalenv();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
extern "C" void G__cpp_setup_funcG__stream() {
   G__lastifuncposition();

   G__memfunc_setup("ws",234,G___ws_0_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,1,1,1,0,"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("endl",419,G___endl_1_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - i",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("ends",426,G___ends_2_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - i",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("flush",546,G___flush_3_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,1,1,1,0,"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_4_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - c - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_5_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - C - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_6_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - Y - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_7_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - b - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_8_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - s - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_9_0,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - r - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_0_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - i - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_1_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - h - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_2_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - l - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_3_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - k - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_4_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - f - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_5_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - d - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_6_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),1,2,1,1,0,
"u 'basic_ostream<char,char_traits<char> >' 'ostream' 1 - - g - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_7_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - c - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_8_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - b - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_9_1,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - s - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_0_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - r - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_1_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - i - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_2_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - h - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_3_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - l - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_4_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - k - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_5_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - f - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_6_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - d - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_7_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - g - - 1 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_8_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - C - - 0 - -",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_9_2,117,G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("istream"),1,2,1,1,0,
"u 'basic_istream<char,char_traits<char> >' 'istream' 1 - - Y - - 1 - -",(char*)NULL
,(void*)NULL,0);

   G__resetifuncposition();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
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
G__linked_taginfo G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type = { "basic_streambuf<char,char_traits<char> >::pos_type" , 0 , -1 };
G__linked_taginfo G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLoff_type = { "basic_streambuf<char,char_traits<char> >::off_type" , 0 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR = { "basic_ostream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLpos_type = { "basic_ostream<char,char_traits<char> >::pos_type" , 0 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLoff_type = { "basic_ostream<char,char_traits<char> >::off_type" , 0 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry = { "basic_ostream<char,char_traits<char> >::sentry" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type = { "basic_ios<char,char_traits<char> >::off_type" , 0 , -1 };
G__linked_taginfo G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type = { "basic_ios<char,char_traits<char> >::pos_type" , 0 , -1 };
G__linked_taginfo G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry = { "basic_istream<char,char_traits<char> >::sentry" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR = { "basic_filebuf<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR = { "basic_ifstream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR = { "basic_ofstream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_string = { "string" , 99 , -1 };
G__linked_taginfo G__G__streamLN_allocatorlEchargR = { "allocator<char>" , 99 , -1 };
G__linked_taginfo G__G__streamLN_allocatorlEwchar_tgR = { "allocator<wchar_t>" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR = { "basic_stringbuf<char,char_traits<char>,allocator<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR = { "basic_istringstream<char,char_traits<char>,allocator<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR = { "basic_ostringstream<char,char_traits<char>,allocator<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR = { "basic_stringstream<char,char_traits<char>,allocator<char> >" , 99 , -1 };
G__linked_taginfo G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR = { "basic_iostream<char,char_traits<char> >" , 99 , -1 };

/* Reset class/struct taginfo */
extern "C" void G__cpp_reset_tagtableG__stream() {
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
  G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type.tagnum = -1 ;
  G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLoff_type.tagnum = -1 ;
  G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLpos_type.tagnum = -1 ;
  G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLoff_type.tagnum = -1 ;
  G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry.tagnum = -1 ;
  G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type.tagnum = -1 ;
  G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type.tagnum = -1 ;
  G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry.tagnum = -1 ;
  G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_string.tagnum = -1 ;
  G__G__streamLN_allocatorlEchargR.tagnum = -1 ;
  G__G__streamLN_allocatorlEwchar_tgR.tagnum = -1 ;
  G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1 ;
  G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
}

extern "C" void G__cpp_setup_tagtableG__stream() {

   /* Setting up class,struct,union tag entry */
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_base),sizeof(ios_base),-1,0,(char*)NULL,G__setup_memvarios_base,G__setup_memfuncios_base);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLio_state),sizeof(int),-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLopen_mode),sizeof(int),-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLseek_dir),sizeof(int),-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLfmt_flags),sizeof(int),-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLevent),sizeof(int),-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_ios_basecLcLInit),sizeof(ios_base::Init),-1,0,(char*)NULL,G__setup_memvarios_basecLcLInit,G__setup_memfuncios_basecLcLInit);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_char_traitslEchargR),sizeof(char_traits<char>),-1,0,(char*)NULL,G__setup_memvarchar_traitslEchargR,G__setup_memfuncchar_traitslEchargR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_istream<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgR),sizeof(basic_ios<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_ioslEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_ioslEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgR),sizeof(basic_streambuf<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_streambuflEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_streambuflEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_streambuflEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_ostream<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),sizeof(basic_ostream<char,char_traits<char> >::sentry),-1,0,(char*)NULL,G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry
,G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLoff_type),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ioslEcharcOchar_traitslEchargRsPgRcLcLpos_type),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),sizeof(basic_istream<char,char_traits<char> >::sentry),-1,0,(char*)NULL,G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry
,G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_filebuflEcharcOchar_traitslEchargRsPgR),sizeof(basic_filebuf<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_filebuflEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_filebuflEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_ifstream<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_ifstreamlEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_ofstream<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_ofstreamlEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_string),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_allocatorlEchargR),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_allocatorlEwchar_tgR),0,-1,0,(char*)NULL,NULL,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),sizeof(basic_stringbuf<char,char_traits<char>,allocator<char> >),-1,0,(char*)NULL
,G__setup_memvarbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,G__setup_memfuncbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),sizeof(basic_istringstream<char,char_traits<char>,allocator<char> >),-1,0,(char*)NULL
,G__setup_memvarbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,G__setup_memfuncbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),sizeof(basic_ostringstream<char,char_traits<char>,allocator<char> >),-1,0,(char*)NULL
,G__setup_memvarbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,G__setup_memfuncbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),sizeof(basic_stringstream<char,char_traits<char>,allocator<char> >),-1,0,(char*)NULL
,G__setup_memvarbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,G__setup_memfuncbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__streamLN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),sizeof(basic_iostream<char,char_traits<char> >),-1,0,(char*)NULL,G__setup_memvarbasic_iostreamlEcharcOchar_traitslEchargRsPgR
,G__setup_memfuncbasic_iostreamlEcharcOchar_traitslEchargRsPgR);
}
extern "C" void G__cpp_setupG__stream() {
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

