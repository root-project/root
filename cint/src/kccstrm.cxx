//*CMZ :  2.23/10 10/12/99  18.00.11  by  Masaharu Goto
//*-- Author :    Masaharu Goto   10/12/99
/********************************************************
* G__cpp_kcciostream.C
********************************************************/
//*KEEP,kccstrm.
#include "kccstrm.h"
//*KEND.

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(__osf__) || defined(__sun__)
// actually only needed for KCC 3.4
#define MBSTATE_IS_STRUCT
#endif

#ifdef G__MEMTEST
#undef malloc
#undef free
#endif

extern "C" void G__cpp_reset_tagtable();

extern "C" void G__set_cpp_environment()
{
   G__cpp_reset_tagtable();
}

class G__kccstrmdOcxx_tag {};

/* dummy, for exception */
#ifdef G__EH_DUMMY_DELETE
void operator delete(void *p,G__kccstrmdOcxx_tag* x) {
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

#if 0
static void operator delete(void *p)
{
   if ((long) p == G__getgvp() && G__PVOID != G__getgvp())
      return;
   free(p);
}
#endif

#include "dllrev.h"
extern "C" int G__cpp_dllrev()
{
   return (G__CREATEDLLREV);
}

/*********************************************************
* Member function Interface Method
*********************************************************/

/* char_traits<char> */
static int G__char_traitslEchargR_assign_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((char_traits < char >*) (G__getstructoffset()))->assign(*(char_traits < char >:: char_type *) G__Charref(&libp->para[0]), *(char_traits < char >::char_type *) G__Charref(&libp->para[1]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_eq_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((char_traits < char >*) (G__getstructoffset()))->eq(*(char_traits < char >:: char_type *) G__Charref(&libp->para[0]), *(char_traits < char >::char_type *) G__Charref(&libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_lt_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((char_traits < char >*) (G__getstructoffset()))->lt(*(char_traits < char >:: char_type *) G__Charref(&libp->para[0]), *(char_traits < char >::char_type *) G__Charref(&libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_compare_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((char_traits < char >*) (G__getstructoffset()))->compare((const char_traits < char >:: char_type *) G__int(libp->para[0]), (const char_traits < char >::char_type *) G__int(libp->para[1])
                                       ,(size_t) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_length_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((char_traits < char >*) (G__getstructoffset()))->length((const char_traits < char >::char_type *) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_copy_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((char_traits < char >*) (G__getstructoffset()))->copy((char_traits < char >:: char_type *) G__int(libp->para[0]), (const char_traits < char >::char_type *) G__int(libp->para[1])
                                       ,(size_t) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_find_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((char_traits < char >*) (G__getstructoffset()))->find((const char_traits < char >::char_type *) G__int(libp->para[0]), (size_t) G__int(libp->para[1])
 ,*(char_traits < char >::   char_type *) G__Charref(&libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_move_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((char_traits < char >*) (G__getstructoffset()))->move((char_traits < char >:: char_type *) G__int(libp->para[0]), (const char_traits < char >::char_type *) G__int(libp->para[1])
                                       ,(size_t) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_assign_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((char_traits < char >*) (G__getstructoffset()))->assign((char_traits < char >::char_type *) G__int(libp->para[0]), (size_t) G__int(libp->para[1])
 ,(char_traits < char >::           char_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_not_eof_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((char_traits < char >*) (G__getstructoffset()))->not_eof(*(char_traits < char >::int_type *) G__Intref(&libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_to_char_type_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((char_traits < char >*) (G__getstructoffset()))->to_char_type(*(char_traits < char >::int_type *) G__Intref(&libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_to_int_type_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((char_traits < char >*) (G__getstructoffset()))->to_int_type(*(char_traits < char >::char_type *) G__Charref(&libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__char_traitslEchargR_eq_int_type_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((char_traits < char >*) (G__getstructoffset()))->eq_int_type(*(char_traits < char >:: int_type *) G__Intref(&libp->para[0]), *(char_traits < char >::int_type *) G__Intref(&libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

// automatic default constructor
static int G__char_traitslEchargR_char_traitslEchargR_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   char_traits < char >*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new char_traits < char >[G__getaryconstruct()];
   else
      p = new char_traits < char >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_char_traitslEchargR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__char_traitslEchargR_char_traitslEchargR_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   char_traits < char >*p;
   if (1 != libp->paran);
   p = new char_traits < char >(*(char_traits < char >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_char_traitslEchargR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__char_traitslEchargR_wAchar_traitslEchargR_0_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](char_traits < char >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(char_traits < char >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_ios<char,char_traits<char> > */
static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatorsPvoidmU_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 89, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->operator void *());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatornO_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->operator ! ());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdstate_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 115, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->rdstate());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_clear_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 1:
      G__setnull(result7);
    ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->clear((ios_base::iostate) G__int(libp->para[0]));
      break;
   case 0:
      G__setnull(result7);
      ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->clear();
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_setstate_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->setstate((ios_base::iostate) G__int(libp->para[0]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_good_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->good());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_eof_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->eof());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_fail_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->fail());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_bad_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->bad());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 115, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->exceptions());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->exceptions((ios_base::iostate) G__int(libp->para[0]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->tie());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->tie((basic_ostream < char, char_traits < char > >*) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->rdbuf((basic_streambuf < char, char_traits < char > >*) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_copyfmt_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ios < char,
       char_traits < char > >&obj = ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->copyfmt(*(basic_ios < char, char_traits < char > >*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 99, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->fill());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->fill((basic_ios < char, char_traits < char > >::char_type) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_imbue_0_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      locale *pobj,
       xobj = ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->imbue(*(locale *) libp->para[0].ref);
      pobj = new locale(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_narrow_1_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->narrow((basic_ios < char, char_traits < char > >::char_type) G__int(libp->para[0]), (char) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_widen_2_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 99, (long) ((basic_ios < char, char_traits < char > >*) (G__getstructoffset()))->widen((char) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_7_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ios < char,
    char_traits < char > >*p = NULL;
   p = new basic_ios < char,
    char_traits < char > >((basic_streambuf < char, char_traits < char > >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_ioslEcharcOchar_traitslEchargRsPgR_wAbasic_ioslEcharcOchar_traitslEchargRsPgR_8_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_ios < char, char_traits < char > >*) (G__getstructoffset());
      else 
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_ios < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_istream<char,char_traits<char> > */
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> ((ios_base & (*)(ios_base &)) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

#ifdef BOOL_NOT_YET_FUNCTIONNING
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(bool *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}
#endif

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(short *) G__Shortref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(unsigned short *) G__UShortref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(int *) G__Intref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(unsigned int *) G__UIntref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(long *) G__Longref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(unsigned long *) G__ULongref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(float *) G__Floatref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (*(double *) G__Doubleref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> (libp->para[0].ref ? *(void **) libp->para[0].ref : *(void **) (&G__Mlong(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->operator >> ((basic_streambuf < char, char_traits < char > >*) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >*what = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_istream < char,
       char_traits < char > >&obj = *what >> ((char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >*what = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_istream < char,
       char_traits < char > >&obj = *what >> ((unsigned char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >*what = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_istream < char,
       char_traits < char > >&obj = *what >> (*(char *) G__Charref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >*what = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_istream < char,
       char_traits < char > >&obj = *what >> (*(unsigned char *) G__UCharref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_gcount_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->gcount());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->get());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->get(*(basic_istream < char, char_traits < char > >::char_type *) G__Charref(&libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->get((basic_istream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->get((basic_istream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1])
                                                                                                                ,(basic_istream < char, char_traits < char > >::char_type) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->get(*(basic_streambuf < char, char_traits < char > >*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_0_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->get(*(basic_streambuf < char, char_traits < char > >*) libp->para[0].ref, (basic_istream < char, char_traits < char > >::char_type) 
       G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_1_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->getline((basic_istream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_2_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->getline((basic_istream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1])
                                                                                                                    ,(basic_istream < char, char_traits < char > >::char_type) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_ignore_3_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      {
         basic_istream < char,
          char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->ignore((streamsize) G__int(libp->para[0]), (basic_istream < char, char_traits < char > >::int_type) G__int(libp->para[1]));
         result7->ref = (long) (&obj);
         result7->obj.i = (long) (&obj);
      }
      break;
   case 1:
      {
         basic_istream < char,
          char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->ignore((streamsize) G__int(libp->para[0]));
         result7->ref = (long) (&obj);
         result7->obj.i = (long) (&obj);
      }
      break;
   case 0:
      {
         basic_istream < char,
          char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->ignore();
         result7->ref = (long) (&obj);
         result7->obj.i = (long) (&obj);
      }
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_peek_4_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->peek());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_read_5_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->read((basic_istream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_readsome_6_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 108, (long) ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->readsome((basic_istream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_putback_7_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->putback((basic_istream < char, char_traits < char > >::char_type) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_unget_8_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->unget();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_sync_9_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->sync());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_tellg_0_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos *pobj,
       xobj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->tellg();
      pobj = new streampos(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_seekg_1_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->seekg(*((basic_istream < char, char_traits < char > >::pos_type *) G__int(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_seekg_2_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ((basic_istream < char, char_traits < char > >*) (G__getstructoffset()))->seekg((basic_istream < char, char_traits < char > >::off_type) G__int(libp->para[0]), (ios_base::seekdir) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_istreamlEcharcOchar_traitslEchargRsPgR_4_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_istream < char,
    char_traits < char > >*p = NULL;
   p = new basic_istream < char,
    char_traits < char > >((basic_streambuf < char, char_traits < char > >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgR_wAbasic_istreamlEcharcOchar_traitslEchargRsPgR_5_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_istream < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_istream < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_ostream<char,char_traits<char> > */
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((basic_ostream < char, char_traits < char > >&(*)(basic_ostream < char, char_traits < char > >&)) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

#ifdef BOOL_NOT_YET_FUNCTIONNING
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((bool) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}
#endif

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((short) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((unsigned short) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((int) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((unsigned int) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((long) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((unsigned long) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((float) G__double(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((double) G__double(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((const void *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->operator << ((basic_streambuf < char, char_traits < char > >*) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_2_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >*what = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()));

      basic_ostream < char,
       char_traits < char > >&obj = operator << (*what, (char) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_3_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >*what = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_ostream < char,
       char_traits < char > >&obj = (*what << ((unsigned char) G__int(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_4_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >*what = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_ostream < char,
       char_traits < char > >&obj = *what << ((const char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_5_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >*what = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()));
      basic_ostream < char,
       char_traits < char > >&obj = *what << ((const unsigned char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_put_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->put((basic_ostream < char, char_traits < char > >::char_type) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_write_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->write((const basic_ostream < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_flush_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->flush();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_tellp_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos *pobj,
       xobj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->tellp();
      pobj = new streampos(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_0_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->seekp(*((basic_ostream < char, char_traits < char > >::pos_type *) G__int(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_1_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ((basic_ostream < char, char_traits < char > >*) (G__getstructoffset()))->seekp((basic_ostream < char, char_traits < char > >::off_type) G__int(libp->para[0]), (ios_base::seekdir) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ostreamlEcharcOchar_traitslEchargRsPgR_3_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_ostream < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_ostream < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_iostream<char,char_traits<char> > */

// automatic destructor
static int G__basic_iostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_iostreamlEcharcOchar_traitslEchargRsPgR_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_iostream < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_iostream < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_streambuf<char,char_traits<char> > */
static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubimbue_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      locale *pobj,
       xobj = ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubimbue(*(locale *) libp->para[0].ref);
      pobj = new locale(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_getloc_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      locale *pobj,
       xobj = ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->getloc();
      pobj = new locale(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsetbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 85, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubsetbuf((basic_streambuf < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekoff_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 3:
      {
         streampos *pobj,
          xobj = ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubseekoff((basic_streambuf < char, char_traits < char > >::off_type) G__int(libp->para[0]), (ios_base::seekdir) G__int(libp->para[1])
                            ,(ios_base::openmode) G__int(libp->para[2]));
         pobj = new streampos(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   case 2:
      {
         streampos *pobj,
          xobj = ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubseekoff((basic_streambuf < char, char_traits < char > >::off_type) G__int(libp->para[0]), (ios_base::seekdir) G__int(libp->para[1]));
         pobj = new streampos(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekpos_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      {
         streampos *pobj,
          xobj = ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubseekpos(*((basic_streambuf < char, char_traits < char > >::pos_type *) G__int(libp->para[0])), (ios_base::openmode) G__int(libp->para[1]));
         pobj = new streampos(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   case 1:
      {
         streampos *pobj,
          xobj = ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubseekpos(*((basic_streambuf < char, char_traits < char > >::pos_type *) G__int(libp->para[0])));
         pobj = new streampos(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsync_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->pubsync());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_in_avail_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->in_avail());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_snextc_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->snextc());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sbumpc_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sbumpc());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetc_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sgetc());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetn_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 108, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sgetn((basic_streambuf < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputbackc_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sputbackc((basic_streambuf < char, char_traits < char > >::char_type) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sungetc_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sungetc());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputc_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sputc((basic_streambuf < char, char_traits < char > >::char_type) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputn_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 108, (long) ((basic_streambuf < char, char_traits < char > >*) (G__getstructoffset()))->sputn((const basic_streambuf < char, char_traits < char > >::char_type *) G__int(libp->para[0]), (streamsize) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_streambuflEcharcOchar_traitslEchargRsPgR_wAbasic_streambuflEcharcOchar_traitslEchargRsPgR_1_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_streambuf < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_streambuf < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_filebuf<char,char_traits<char> > */
static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_filebuf < char,
    char_traits < char > >*p = NULL;
   if (G__getaryconstruct())
      p = new basic_filebuf < char,
       char_traits < char > >[G__getaryconstruct()];
   else
      p = new basic_filebuf < char,
       char_traits < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_is_open_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_filebuf < char, char_traits < char > >*) (G__getstructoffset()))->is_open());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_open_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 85, (long) ((basic_filebuf < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_close_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_filebuf < char, char_traits < char > >*) (G__getstructoffset()))->close());
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_filebuflEcharcOchar_traitslEchargRsPgR_wAbasic_filebuflEcharcOchar_traitslEchargRsPgR_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_filebuf < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_filebuf < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* allocator<char> */
static int G__allocatorlEchargR_allocate_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 67, (long) ((allocator < char >*) (G__getstructoffset()))->allocate((allocator < char >::size_type) G__int(libp->para[0]), (const void *) G__int(libp->para[1])));
      break;
   case 1:
    G__letint(result7, 67, (long) ((allocator < char >*) (G__getstructoffset()))->allocate((allocator < char >::size_type) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__allocatorlEchargR_deallocate_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((allocator < char >*) (G__getstructoffset()))->deallocate((allocator < char >:: pointer) G__int(libp->para[0]), (allocator < char >::size_type) G__int(libp->para[1]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__allocatorlEchargR_address_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((allocator < char >*) (G__getstructoffset()))->address(*(char *) G__Charref(&libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__allocatorlEchargR_max_size_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 104, (long) ((allocator < char >*) (G__getstructoffset()))->max_size());
   return (1 || funcname || hash || result7 || libp);
}

static int G__allocatorlEchargR_construct_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((allocator < char >*) (G__getstructoffset()))->construct((allocator < char >::pointer) G__int(libp->para[0]), *(char *) G__Charref(&libp->para[1]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__allocatorlEchargR_destroy_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((allocator < char >*) (G__getstructoffset()))->destroy((allocator < char >::pointer) G__int(libp->para[0]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__allocatorlEchargR_allocatorlEchargR_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   allocator < char >*p = NULL;
   if (G__getaryconstruct())
      p = new allocator < char >[G__getaryconstruct()];
   else
      p = new allocator < char >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_allocatorlEchargR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__allocatorlEchargR_allocatorlEchargR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   allocator < char >*p;
   if (1 != libp->paran);
   p = new allocator < char >(*(allocator < char >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_allocatorlEchargR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__allocatorlEchargR_wAallocatorlEchargR_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](allocator < char >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(allocator < char >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((basic_stringbuf < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str(*(basic_string< char, char_traits < char >, allocator < char > > *) libp->para[0].ref);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_stringbuf < char, char_traits < char >, allocator < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_stringbuf < char, char_traits < char >, allocator < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_ifstream<char,char_traits<char> > */
static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ifstream < char,
    char_traits < char > >*p = NULL;
   if (G__getaryconstruct())
      p = new basic_ifstream < char,
       char_traits < char > >[G__getaryconstruct()];
   else
      p = new basic_ifstream < char,
       char_traits < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ifstream < char,
    char_traits < char > >*p = NULL;
   switch (libp->paran) {
   case 2:
      p = new basic_ifstream < char,
       char_traits < char > >((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
      break;
   case 1:
      p = new basic_ifstream < char,
       char_traits < char > >((const char *) G__int(libp->para[0]));
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ifstream < char, char_traits < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_is_open_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ifstream < char, char_traits < char > >*) (G__getstructoffset()))->is_open());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_open_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      G__setnull(result7);
    ((basic_ifstream < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
      break;
   case 1:
      G__setnull(result7);
      ((basic_ifstream < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_close_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
   ((basic_ifstream < char, char_traits < char > >*) (G__getstructoffset()))->close();
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ifstream < char,
    char_traits < char > >*p = NULL;
   p = new basic_ifstream < char,
    char_traits < char > >((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ifstreamlEcharcOchar_traitslEchargRsPgR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_ifstream < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_ifstream < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_ofstream<char,char_traits<char> > */

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ofstream < char,
    char_traits < char > >*p = NULL;
   if (G__getaryconstruct())
      p = new basic_ofstream < char,
       char_traits < char > >[G__getaryconstruct()];
   else
      p = new basic_ofstream < char,
       char_traits < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ofstream < char,
    char_traits < char > >*p = NULL;
   switch (libp->paran) {
   case 2:
      p = new basic_ofstream < char,
       char_traits < char > >((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
      break;
   case 1:
      p = new basic_ofstream < char,
       char_traits < char > >((const char *) G__int(libp->para[0]));
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ofstream < char, char_traits < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_is_open_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_ofstream < char, char_traits < char > >*) (G__getstructoffset()))->is_open());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_open_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      G__setnull(result7);
    ((basic_ofstream < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
      break;
   case 1:
      G__setnull(result7);
      ((basic_ofstream < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_close_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
   ((basic_ofstream < char, char_traits < char > >*) (G__getstructoffset()))->close();
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ofstream < char,
    char_traits < char > >*p = NULL;
   p = new basic_ofstream < char,
    char_traits < char > >((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ofstreamlEcharcOchar_traitslEchargRsPgR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_ofstream < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_ofstream < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_fstream<char,char_traits<char> > */
static int G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_fstream < char, char_traits < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_is_open_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((basic_fstream < char, char_traits < char > >*) (G__getstructoffset()))->is_open());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_open_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      G__setnull(result7);
    ((basic_fstream < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
      break;
   case 1:
      G__setnull(result7);
      ((basic_fstream < char, char_traits < char > >*) (G__getstructoffset()))->open((const char *) G__int(libp->para[0]));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_close_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
   ((basic_fstream < char, char_traits < char > >*) (G__getstructoffset()))->close();
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_fstreamlEcharcOchar_traitslEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_fstream < char,
    char_traits < char > >*p = NULL;
   p = new basic_fstream < char,
    char_traits < char > >((const char *) G__int(libp->para[0]), (ios_base::openmode) G__int(libp->para[1]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_fstreamlEcharcOchar_traitslEchargRsPgR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_fstream < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_fstream < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_istringstream<char,char_traits<char>,allocator<char> > */
static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_istringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >*pobj,
       xobj = ((basic_istringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str();
      pobj = new basic_string < char,
       char_traits < char >,
       allocator < char > >(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((basic_istringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str(*(basic_string< char, char_traits < char >, allocator < char > > *) libp->para[0].ref);
   return (1 || funcname || hash || result7 || libp);
}

// automatic default constructor
static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_istringstream < char,
    char_traits < char >,
    allocator < char > >*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new basic_istringstream < char,
       char_traits < char >,
       allocator < char > >[G__getaryconstruct()];
   else
      p = new basic_istringstream < char,
       char_traits < char >,
       allocator < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_istringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_istringstream < char, char_traits < char >, allocator < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_ostringstream<char,char_traits<char>,allocator<char> > */
static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_ostringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >*pobj,
       xobj = ((basic_ostringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str();
      pobj = new basic_string < char,
       char_traits < char >,
       allocator < char > >(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((basic_ostringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str(*(basic_string < char, char_traits < char >, allocator < char > > *) libp->para[0].ref);
   return (1 || funcname || hash || result7 || libp);
}

// automatic default constructor
static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_ostringstream < char,
    char_traits < char >,
    allocator < char > >*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new basic_ostringstream < char,
       char_traits < char >,
       allocator < char > >[G__getaryconstruct()];
   else
      p = new basic_ostringstream < char,
       char_traits < char >,
       allocator < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_ostringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_ostringstream < char, char_traits < char >, allocator < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_stringstream<char,char_traits<char>,allocator<char> > */
static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) ((basic_stringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->rdbuf());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >*pobj,
       xobj = ((basic_stringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str();
      pobj = new basic_string < char,
       char_traits < char >,
       allocator < char > >(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((basic_stringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset()))->str(*(basic_string < char, char_traits < char >, allocator < char > > *) libp->para[0].ref);
   return (1 || funcname || hash || result7 || libp);
}

// automatic default constructor
static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_stringstream < char,
    char_traits < char >,
    allocator < char > >*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new basic_stringstream < char,
       char_traits < char >,
       allocator < char > >[G__getaryconstruct()];
   else
      p = new basic_stringstream < char,
       char_traits < char >,
       allocator < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](basic_stringstream < char, char_traits < char >, allocator < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_stringstream < char, char_traits < char >, allocator < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_string<char,char_traits<char>,allocator<char> > */

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_string < char,
    char_traits < char >,
    allocator < char > >*p = NULL;
   switch (libp->paran) {
   case 4:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(
                             *(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])
                             ,(string::size_type) G__int(libp->para[2]), *(allocator < char >*) libp->para[3].ref);
      break;
   case 3:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(
                             *(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])
                             ,(string::size_type) G__int(libp->para[2]));
      break;
   case 2:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1]));
      break;
   case 1:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(*(string*) libp->para[0].ref);
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_string < char,
    char_traits < char >,
    allocator < char > >*p = NULL;
   switch (libp->paran) {
   case 4:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(
                             *(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])
                             ,(string::size_type) G__int(libp->para[2]), *(allocator < char >*) libp->para[3].ref);
      break;
   case 3:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(
                             *(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])
                             ,(string::size_type) G__int(libp->para[2]));
      break;
   case 2:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1]));
      break;
   case 1:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(*(string*) libp->para[0].ref);
      break;
   case 0:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >();
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_string < char,
    char_traits < char >,
    allocator < char > >*p = NULL;
   switch (libp->paran) {
   case 3:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(
                              (const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
                             ,*(allocator < char >*) libp->para[2].ref);
      break;
   case 2:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1]));
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_string < char,
    char_traits < char >,
    allocator < char > >*p = NULL;
   switch (libp->paran) {
   case 2:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >((const char *) G__int(libp->para[0]), *(allocator < char >*) libp->para[1].ref);
      break;
   case 1:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >((const char *) G__int(libp->para[0]));
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_string < char,
    char_traits < char >,
    allocator < char > >*p = NULL;
   switch (libp->paran) {
   case 3:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >(
                              (string::size_type) G__int(libp->para[0]), (char) G__int(libp->para[1])
                             ,*(allocator < char >*) libp->para[2].ref);
      break;
   case 2:
      p = new basic_string < char,
       char_traits < char >,
       allocator < char > >((string::size_type) G__int(libp->para[0]), (char) G__int(libp->para[1]));
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoreQ_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->operator = (*(string*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoreQ_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->operator = ((const char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoreQ_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->operator = ((char) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_begin_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))->begin());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_end_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))->end());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rbegin_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = ((string*) (G__getstructoffset()))->rbegin();
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rend_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = ((string*) (G__getstructoffset()))->rend();
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_size_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->size());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_length_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->length());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_max_size_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->max_size());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_resize_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((string*) (G__getstructoffset()))->resize((string::size_type) G__int(libp->para[0]), (char) G__int(libp->para[1]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_resize_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((string*) (G__getstructoffset()))->resize((string::size_type) G__int(libp->para[0]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_capacity_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->capacity());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_reserve_0_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 1:
      G__setnull(result7);
    ((string*) (G__getstructoffset()))->reserve((string::size_type) G__int(libp->para[0]));
      break;
   case 0:
      G__setnull(result7);
      ((string*) (G__getstructoffset()))->reserve();
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_empty_1_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->empty());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_clear_2_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
   ((string*) (G__getstructoffset()))->clear();
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoroBcB_3_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      char &obj = ((string*) (G__getstructoffset()))->operator[]((string::size_type) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_at_4_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      char &obj = ((string*) (G__getstructoffset()))->at((string::size_type) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatorpLeQ_5_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->operator += (*(string*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatorpLeQ_6_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->operator += ((const char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatorpLeQ_7_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->operator += ((char) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_8_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->append(*(string*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_9_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = 
        ((string*) (G__getstructoffset()))
          ->append( *(basic_string < char, 
                                     char_traits < char >, 
                                     allocator < char > >*) libp->para[0].ref, 
                    (string::size_type) G__int(libp->para[1]),
                    (string::size_type) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_0_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = 
	((string*) (G__getstructoffset()))
	  -> append((const char *) G__int(libp->para[0]), 
		    (string::size_type) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_1_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->append((const char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_2_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = 
	((string*) (G__getstructoffset()))
	->append( (string::size_type) G__int(libp->para[0]), 
		  (char) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_3_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->assign(*(string*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_4_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = 
	((string*) (G__getstructoffset()))
	 ->assign( *(string*) libp->para[0].ref, 
		   (basic_string < char, char_traits < char >, 
		    allocator < char > >::size_type) G__int(libp->para[1]),
		   (string::size_type) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_5_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = 
	((string*) (G__getstructoffset()))
	->assign( (const char *) G__int(libp->para[0]), 
		  (string::size_type) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_6_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))->assign((const char *) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_7_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->assign((string::size_type) G__int(libp->para[0]), (char) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_8_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
       ->insert((string::size_type) G__int(libp->para[0]), 
		*(string*) libp->para[1].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_9_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->insert((string::size_type) G__int(libp->para[0]), 
		 *(string*) libp->para[1].ref,
		 (string::size_type) G__int(libp->para[2]), 
		 (string::size_type) G__int(libp->para[3]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_0_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->insert((string::size_type) G__int(libp->para[0]),
		 (const char *) G__int(libp->para[1]),
		 (string::size_type) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_1_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->insert((string::size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
		 ,(char) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_2_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->insert((string::size_type) G__int(libp->para[0]), 
		 (const char *) G__int(libp->para[1]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_3_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))
	      ->insert((string::iterator) G__int(libp->para[0]), 
		       (char) G__int(libp->para[1])));
      break;
   case 1:
    G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))->insert((string::iterator) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_4_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((string*) (G__getstructoffset()))
   ->insert((string:: iterator) G__int(libp->para[0]), 
	    (string::size_type) G__int(libp->para[1])
                                          ,(char) G__int(libp->para[2]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_erase_5_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      {
         basic_string < char,
          char_traits < char >,
          allocator < char > >&obj = ((string*) (G__getstructoffset()))
	   ->erase((string::size_type) G__int(libp->para[0]), 
		   (string::size_type) G__int(libp->para[1]));
         result7->ref = (long) (&obj);
         result7->obj.i = (long) (&obj);
      }
      break;
   case 1:
      {
         basic_string < char,
          char_traits < char >,
          allocator < char > >&obj = ((string*) (G__getstructoffset()))
	   ->erase((string::size_type) G__int(libp->para[0]));
         result7->ref = (long) (&obj);
         result7->obj.i = (long) (&obj);
      }
      break;
   case 0:
      {
         basic_string < char,
          char_traits < char >,
          allocator < char > >&obj = ((string*) (G__getstructoffset()))->erase();
         result7->ref = (long) (&obj);
         result7->obj.i = (long) (&obj);
      }
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_erase_6_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))->erase((string::iterator) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_erase_7_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))
	   ->erase((string:: iterator) G__int(libp->para[0]), 
		   (string::iterator) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_8_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::size_type) G__int(libp->para[0]), 
		  (string::size_type) G__int(libp->para[1])
                                                                                                                                     ,*(string*) libp->para[2].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_9_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::size_type) G__int(libp->para[0]), 
		  (string::size_type) G__int(libp->para[1])
		  ,*(string*) libp->para[2].ref
		  ,(string::size_type) G__int(libp->para[3])
		  ,(string::size_type) G__int(libp->para[4]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_0_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::size_type) G__int(libp->para[0]), 
		  (string::size_type) G__int(libp->para[1])
		  ,(const char *) G__int(libp->para[2])
		  , (string::size_type) G__int(libp->para[3]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_1_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
		  ,(const char *) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_2_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::size_type) G__int(libp->para[0]), 
		  (string::size_type) G__int(libp->para[1])
		  ,(string::size_type) G__int(libp->para[2]), 
		  (char) G__int(libp->para[3]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_3_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::iterator) G__int(libp->para[0]), 
		  (string::iterator) G__int(libp->para[1])
		  ,*(string*) libp->para[2].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_4_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::iterator) G__int(libp->para[0]), 
		  (string::iterator) G__int(libp->para[1]),
		  (const char *) G__int(libp->para[2]), 
		  (string::size_type) G__int(libp->para[3]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_5_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::iterator) G__int(libp->para[0]), 
		  (string::iterator) G__int(libp->para[1])
		  ,(const char *) G__int(libp->para[2]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_6_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >&obj = ((string*) (G__getstructoffset()))
	->replace((string::iterator) G__int(libp->para[0]), 
		  (string::iterator) G__int(libp->para[1])
		  ,(string::size_type) G__int(libp->para[2]), 
		  (char) G__int(libp->para[3]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_copy_7_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 3:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	      ->copy((char *) G__int(libp->para[0]), 
		     (string::size_type) G__int(libp->para[1])
		     ,(string::size_type) G__int(libp->para[2])));
      break;
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	      ->copy((char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_swap_8_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
   ((string*) (G__getstructoffset()))->swap(*(string*) libp->para[0].ref);
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_c_str_9_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))->c_str());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_data_0_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((string*) (G__getstructoffset()))->data());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_get_allocator_1_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      allocator < char >*pobj,
       xobj = ((string*) (G__getstructoffset()))->get_allocator();
      pobj = new allocator < char >(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_2_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	      ->find(*(string*) libp->para[0].ref, 
		     (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find(*(string*) libp->para[0].ref));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_3_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	   ->find((const char *) G__int(libp->para[0]), 
		  (string::size_type) G__int(libp->para[1])
 ,(string::size_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_4_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	      ->find((const char *) G__int(libp->para[0]), 
		     (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find((const char *) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_5_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	      ->find((char) G__int(libp->para[0]), 
		     (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find((char) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_6_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))
	      ->rfind(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->rfind(*(string*) libp->para[0].ref));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_7_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->rfind((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
 ,(string::size_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_8_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->rfind((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->rfind((const char *) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_9_6(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->rfind((char) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->rfind((char) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_0_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of(*(string*) libp->para[0].ref));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_1_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
 ,(string::size_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_2_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of((const char *) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_3_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of((char) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_of((char) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_4_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of(*(string*) libp->para[0].ref));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_5_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
 ,(string::size_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_6_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of((const char *) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_7_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of((char) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_of((char) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_8_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of(*(string*) libp->para[0].ref));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_9_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
 ,(string::size_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_0_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of((const char *) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_1_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of((char) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_first_not_of((char) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_2_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of(*(string*) libp->para[0].ref, (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of(*(string*) libp->para[0].ref));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_3_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
 ,(string::size_type) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_4_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of((const char *) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of((const char *) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_5_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
    G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of((char) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7, 104, (long) ((string*) (G__getstructoffset()))->find_last_not_of((char) G__int(libp->para[0])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_substr_6_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 2:
      {
         basic_string < char,
          char_traits < char >,
          allocator < char > >*pobj,
          xobj = ((string*) (G__getstructoffset()))->substr((string::size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1]));
         pobj = new basic_string < char,
          char_traits < char >,
          allocator < char > >(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   case 1:
      {
         basic_string < char,
          char_traits < char >,
          allocator < char > >*pobj,
          xobj = ((string*) (G__getstructoffset()))->substr((string::size_type) G__int(libp->para[0]));
         pobj = new basic_string < char,
          char_traits < char >,
          allocator < char > >(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   case 0:
      {
         basic_string < char,
          char_traits < char >,
          allocator < char > >*pobj,
          xobj = ((string*) (G__getstructoffset()))->substr();
         pobj = new basic_string < char,
          char_traits < char >,
          allocator < char > >(xobj);
         result7->obj.i = (long) ((void *) pobj);
         result7->ref = result7->obj.i;
         G__store_tempobject(*result7);
      }
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_7_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->compare(*(string*) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_8_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->compare((string:: size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
                                                                                                                                      ,*(string*) libp->para[2].ref));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_9_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->compare((string:: size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
 ,*(string*) libp->para[2].ref, (string::size_type) G__int(libp->para[3])
 ,(string::size_type) G__int(libp->para[4])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_0_9(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->compare((const char *) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_1_9(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 4:
    G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->compare((string:: size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
    ,(const char *) G__int(libp->para[2]), (string::size_type) G__int(libp->para[3])));
      break;
   case 3:
    G__letint(result7, 105, (long) ((string*) (G__getstructoffset()))->compare((string:: size_type) G__int(libp->para[0]), (string::size_type) G__int(libp->para[1])
                                 ,(const char *) G__int(libp->para[2])));
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_2_9(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](string*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(string) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}

/* input_iterator_tag */
// automatic default constructor
static int G__input_iterator_tag_input_iterator_tag_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   input_iterator_tag *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new input_iterator_tag[G__getaryconstruct()];
   else
      p = new input_iterator_tag;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_input_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__input_iterator_tag_input_iterator_tag_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   input_iterator_tag *p;
   if (1 != libp->paran);
   p = new input_iterator_tag(*(input_iterator_tag *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_input_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__input_iterator_tag_wAinput_iterator_tag_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](input_iterator_tag *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(input_iterator_tag) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* output_iterator_tag */
// automatic default constructor
static int G__output_iterator_tag_output_iterator_tag_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   output_iterator_tag *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new output_iterator_tag[G__getaryconstruct()];
   else
      p = new output_iterator_tag;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_output_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__output_iterator_tag_output_iterator_tag_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   output_iterator_tag *p;
   if (1 != libp->paran);
   p = new output_iterator_tag(*(output_iterator_tag *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_output_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__output_iterator_tag_wAoutput_iterator_tag_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](output_iterator_tag *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(output_iterator_tag) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* forward_iterator_tag */
// automatic default constructor
static int G__forward_iterator_tag_forward_iterator_tag_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   forward_iterator_tag *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new forward_iterator_tag[G__getaryconstruct()];
   else
      p = new forward_iterator_tag;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_forward_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__forward_iterator_tag_forward_iterator_tag_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   forward_iterator_tag *p;
   if (1 != libp->paran);
   p = new forward_iterator_tag(*(forward_iterator_tag *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_forward_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__forward_iterator_tag_wAforward_iterator_tag_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](forward_iterator_tag *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(forward_iterator_tag) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* bidirectional_iterator_tag */
// automatic default constructor
static int G__bidirectional_iterator_tag_bidirectional_iterator_tag_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   bidirectional_iterator_tag *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new bidirectional_iterator_tag[G__getaryconstruct()];
   else
      p = new bidirectional_iterator_tag;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__bidirectional_iterator_tag_bidirectional_iterator_tag_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   bidirectional_iterator_tag *p;
   if (1 != libp->paran);
   p = new bidirectional_iterator_tag(*(bidirectional_iterator_tag *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__bidirectional_iterator_tag_wAbidirectional_iterator_tag_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](bidirectional_iterator_tag *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(bidirectional_iterator_tag) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* random_access_iterator_tag */
// automatic default constructor
static int G__random_access_iterator_tag_random_access_iterator_tag_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   random_access_iterator_tag *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new random_access_iterator_tag[G__getaryconstruct()];
   else
      p = new random_access_iterator_tag;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_random_access_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__random_access_iterator_tag_random_access_iterator_tag_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   random_access_iterator_tag *p;
   if (1 != libp->paran);
   p = new random_access_iterator_tag(*(random_access_iterator_tag *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_random_access_iterator_tag);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__random_access_iterator_tag_wArandom_access_iterator_tag_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](random_access_iterator_tag *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(random_access_iterator_tag) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* iterator<output_iterator_tag,void,void,void,void> */
// automatic default constructor
static int G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator < output_iterator_tag, void,
   void,
   void,
   void >*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new iterator < output_iterator_tag, void,
      void,
      void,
      void >[G__getaryconstruct()];
   else
      p = new iterator < output_iterator_tag, void,
      void,
      void,
      void >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator < output_iterator_tag, void,
   void,
   void,
   void >*p;
   if (1 != libp->paran);
   p = new iterator < output_iterator_tag, void,
   void,
   void,
   void >(*(iterator < output_iterator_tag, void, void, void, void >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_wAiteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](iterator < output_iterator_tag, void, void, void, void >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(iterator < output_iterator_tag, void, void, void, void >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* rel_ops */

/* allocator<void> */
// automatic default constructor
static int G__allocatorlEvoidgR_allocatorlEvoidgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   allocator < void >*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new allocator < void >[G__getaryconstruct()];
   else
      p = new allocator < void >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_allocatorlEvoidgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__allocatorlEvoidgR_allocatorlEvoidgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   allocator < void >*p;
   if (1 != libp->paran);
   p = new allocator < void >(*(allocator < void >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_allocatorlEvoidgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__allocatorlEvoidgR_wAallocatorlEvoidgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](allocator < void >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(allocator < void >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}

/* streampos */
static int G__streampos_streampos_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   streampos *p = NULL;
   switch (libp->paran) {
   case 1:
      p = new streampos((streamoff) G__int(libp->para[0]));
      break;
   case 0:
      if (G__getaryconstruct())
         p = new streampos[G__getaryconstruct()];
      else
         p = new streampos;
      break;
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_streampos);
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_streampos_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   streampos *p = NULL;
   p = new streampos(*(streampos *) libp->para[0].ref);
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_streampos);
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatorsPstreamoff_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((streampos *) (G__getstructoffset()))->operator streamoff());
   return (1 || funcname || hash || result7 || libp);
}

#ifndef MBSTATE_IS_STRUCT
// Disable because of KCC 3.4 on linux :(
static int G__streampos_state_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((streampos *) (G__getstructoffset()))->state());
   return (1 || funcname || hash || result7 || libp);
}
#endif

static int G__streampos_operatoreQ_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos & obj = ((streampos *) (G__getstructoffset()))->operator = (*(streampos *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatormI_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((streampos *) (G__getstructoffset()))->operator - (*(streampos *) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatorpLeQ_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos & obj = ((streampos *) (G__getstructoffset()))->operator += ((streamoff) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatormIeQ_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos & obj = ((streampos *) (G__getstructoffset()))->operator -= ((streamoff) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatorpL_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos *pobj,
       xobj = ((streampos *) (G__getstructoffset()))->operator + ((streamoff) G__int(libp->para[0]));
      pobj = new streampos(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatormI_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      streampos *pobj,
       xobj = ((streampos *) (G__getstructoffset()))->operator - ((streamoff) G__int(libp->para[0]));
      pobj = new streampos(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatoreQeQ_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((streampos *) (G__getstructoffset()))->operator == (*(streampos *) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

static int G__streampos_operatornOeQ_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((streampos *) (G__getstructoffset()))->operator != (*(streampos *) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__streampos_wAstreampos_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](streampos *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(streampos) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* reverse_iterator<char*> */
static int G__reverse_iteratorlEcharmUgR_reverse_iteratorlEcharmUgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   reverse_iterator < char *>*p = NULL;
   if (G__getaryconstruct())
      p = new reverse_iterator < char *>[G__getaryconstruct()];
   else
      p = new reverse_iterator < char *>;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR);
   return (1 || funcname || hash || result7 || libp);
}

#if 0
static int G__reverse_iteratorlEcharmUgR_base_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((reverse_iterator < char *>*) (G__getstructoffset()))->base());
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatormU_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      char *&obj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator * ();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (obj);
   }
   return (1 || funcname || hash || result7 || libp);
}
#endif

static int G__reverse_iteratorlEcharmUgR_operatormIgR_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ((reverse_iterator < char *>*) (G__getstructoffset()))->operator->());
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatorpLpL_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>&obj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator++ ();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatorpLpL_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator++ ((int) G__int(libp->para[0]));
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatormImI_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>&obj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator-- ();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatormImI_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator-- ((int) G__int(libp->para[0]));
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatorpL_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator + (*((reverse_iterator < char *>::difference_type *) G__int(libp->para[0])));
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatorpLeQ_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>&obj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator += (*((reverse_iterator < char *>::difference_type *) G__int(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatormI_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator - (*((reverse_iterator < char *>::difference_type *) G__int(libp->para[0])));
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatormIeQ_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>&obj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator -= (*((reverse_iterator < char *>::difference_type *) G__int(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__reverse_iteratorlEcharmUgR_operatoroBcB_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      // NOT Thread safe...
      static char obj;
    obj = ((reverse_iterator < char *>*) (G__getstructoffset()))->operator[](*((reverse_iterator < char *>::difference_type *) G__int(libp->para[0])));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__reverse_iteratorlEcharmUgR_reverse_iteratorlEcharmUgR_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   reverse_iterator < char *>*p;
   if (1 != libp->paran);
   p = new reverse_iterator < char *>(*(reverse_iterator < char *>*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__reverse_iteratorlEcharmUgR_wAreverse_iteratorlEcharmUgR_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](reverse_iterator < char *>*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(reverse_iterator < char *>) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* iterator_traits<char*> */
// automatic default constructor
static int G__iterator_traitslEcharmUgR_iterator_traitslEcharmUgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator_traits < char *>*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new iterator_traits < char *>[G__getaryconstruct()];
   else
      p = new iterator_traits < char *>;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__iterator_traitslEcharmUgR_iterator_traitslEcharmUgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator_traits < char *>*p;
   if (1 != libp->paran);
   p = new iterator_traits < char *>(*(iterator_traits < char *>*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__iterator_traitslEcharmUgR_wAiterator_traitslEcharmUgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](iterator_traits < char *>*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(iterator_traits < char *>) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* iterator<long,char*,char**,char*&,random_access_iterator_tag> */
// automatic default constructor
static int G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator < long,
   char *,
   char **,
   char *&,
    random_access_iterator_tag > *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new iterator < long,
      char *,
      char **,
      char *&,
       random_access_iterator_tag >[G__getaryconstruct()];
   else
      p = new iterator < long,
      char *,
      char **,
      char *&,
       random_access_iterator_tag >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator < long,
   char *,
   char **,
   char *&,
    random_access_iterator_tag > *p;
   if (1 != libp->paran);
   p = new iterator < long,
   char *,
   char **,
   char *&,
    random_access_iterator_tag > (*(iterator < long, char *, char **, char *&, random_access_iterator_tag > *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_wAiteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](iterator < long, char *, char **, char *&, random_access_iterator_tag > *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(iterator < long, char *, char **, char *&, random_access_iterator_tag >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* lconv */
// automatic default constructor
static int G__lconv_lconv_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   lconv *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new lconv[G__getaryconstruct()];
   else
      p = new lconv;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_lconv);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__lconv_lconv_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   lconv *p;
   if (1 != libp->paran);
   p = new lconv(*(lconv *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_lconv);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__lconv_wAlconv_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](lconv *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(lconv) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* locale */
static int G__locale_locale_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   locale *p = NULL;
   if (G__getaryconstruct())
      p = new locale[G__getaryconstruct()];
   else
      p = new locale;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_locale);
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_locale_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   locale *p = NULL;
   p = new locale(*(locale *) libp->para[0].ref);
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_locale);
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_locale_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   locale *p = NULL;
   p = new locale((const char *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_locale);
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_locale_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   locale *p = NULL;
   p = new locale(
      *(locale *) libp->para[0].ref, (const char *) G__int(libp->para[1])
 ,(locale::       category) G__int(libp->para[2]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_locale);
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_locale_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   locale *p = NULL;
   p = new locale(
             *(locale *) libp->para[0].ref, *(locale *) libp->para[1].ref
 ,(locale::       category) G__int(libp->para[2]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_locale);
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_operatoreQ_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      const locale & obj = ((locale *) (G__getstructoffset()))->operator = (*(locale *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_name_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >*pobj,
       xobj = ((locale *) (G__getstructoffset()))->name();
      pobj = new basic_string < char,
       char_traits < char >,
       allocator < char > >(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_operatoreQeQ_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((locale *) (G__getstructoffset()))->operator == (*(locale *) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_operatornOeQ_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((locale *) (G__getstructoffset()))->operator != (*(locale *) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_global_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      locale *pobj,
       xobj = ((locale *) (G__getstructoffset()))->global (*(locale *) libp->para[0].ref);
      pobj = new locale(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__locale_classic_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      const locale & obj = ((locale *) (G__getstructoffset()))->classic();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__locale_wAlocale_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](locale *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(locale) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* locale::facet */

/* locale::id */
static int G__localecLcLid_id_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 locale::id * p = NULL;
   if (G__getaryconstruct())
    p = new locale::id[G__getaryconstruct()];
   else
    p = new locale::id;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_localecLcLid);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__localecLcLid_wAid_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
       delete[](locale::id *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
          G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(locale::id) * i));
   else
    G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* locale::imp */



/* ctype_base */
// automatic default constructor
static int G__ctype_base_ctype_base_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   ctype_base *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new ctype_base[G__getaryconstruct()];
   else
      p = new ctype_base;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ctype_base);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__ctype_base_ctype_base_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   ctype_base *p;
   if (1 != libp->paran);
   p = new ctype_base(*(ctype_base *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ctype_base);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__ctype_base_wActype_base_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](ctype_base *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(ctype_base) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* ctype<char> */
static int G__ctypelEchargR_is_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 105, (long) ((ctype < char >*) (G__getstructoffset()))->is((ctype_base:: mask) G__int(libp->para[0]), (ctype < char >::char_type) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_is_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->is((const ctype < char >:: char_type *) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])
 ,(ctype_base::                        mask *) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_scan_is_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->scan_is((ctype_base:: mask) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])
 ,(const ctype < char >::         char_type *) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_scan_not_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->scan_not((ctype_base:: mask) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])
 ,(const ctype < char >::         char_type *) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_toupper_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((ctype < char >*) (G__getstructoffset()))->toupper((ctype < char >::char_type) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_toupper_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->toupper((ctype < char >:: char_type *) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_tolower_4_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((ctype < char >*) (G__getstructoffset()))->tolower((ctype < char >::char_type) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_tolower_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->tolower((ctype < char >:: char_type *) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_widen_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((ctype < char >*) (G__getstructoffset()))->widen((ctype < char >::char_type) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_widen_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->widen((const ctype < char >:: char_type *) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])
 ,(ctype < char >::               char_type *) G__int(libp->para[2])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_narrow_8_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((ctype < char >*) (G__getstructoffset()))->narrow((ctype < char >:: char_type) G__int(libp->para[0]), (ctype < char >::char_type) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ctypelEchargR_narrow_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 67, (long) ((ctype < char >*) (G__getstructoffset()))->narrow((const ctype < char >:: char_type *) G__int(libp->para[0]), (const ctype < char >::char_type *) G__int(libp->para[1])
 ,(ctype < char >:: char_type) G__int(libp->para[2]), (ctype < char >::char_type *) G__int(libp->para[3])));
   return (1 || funcname || hash || result7 || libp);
}


/* ios_base */
static int G__ios_base_flags_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->flags());
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_flags_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->flags((ios_base::fmtflags) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_setf_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->setf((ios_base::fmtflags) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_setf_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->setf((ios_base:: fmtflags) G__int(libp->para[0]), (ios_base::fmtflags) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_unsetf_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((ios_base *) (G__getstructoffset()))->unsetf((ios_base::fmtflags) G__int(libp->para[0]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_precision_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->precision());
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_precision_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->precision((streamsize) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_width_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->width());
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_width_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((ios_base *) (G__getstructoffset()))->width((streamsize) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_imbue_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      locale *pobj,
       xobj = ((ios_base *) (G__getstructoffset()))->imbue(*(locale *) libp->para[0].ref);
      pobj = new locale(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_getloc_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      locale *pobj,
       xobj = ((ios_base *) (G__getstructoffset()))->getloc();
      pobj = new locale(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_xalloc_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((ios_base *) (G__getstructoffset()))->xalloc());
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_iword_2_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      long &obj = ((ios_base *) (G__getstructoffset()))->iword((int) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_pword_3_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      void *&obj = ((ios_base *) (G__getstructoffset()))->pword((int) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_register_callback_5_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__setnull(result7);
 ((ios_base *) (G__getstructoffset()))->register_callback((ios_base::event_callback) G__int(libp->para[0]), (int) G__int(libp->para[1]));
   return (1 || funcname || hash || result7 || libp);
}

static int G__ios_base_sync_with_stdio_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   switch (libp->paran) {
   case 1:
      G__letint(result7, 105, (long) ((ios_base *) (G__getstructoffset()))->sync_with_stdio((bool) G__int(libp->para[0])));
      break;
   case 0:
      G__letint(result7, 105, (long) ((ios_base *) (G__getstructoffset()))->sync_with_stdio());
      break;
   }
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__ios_base_wAios_base_1_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](ios_base *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(ios_base) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* ios_base::Init */
static int G__ios_basecLcLInit_Init_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 ios_base::Init * p = NULL;
   if (G__getaryconstruct())
    p = new ios_base::Init[G__getaryconstruct()];
   else
    p = new ios_base::Init;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ios_basecLcLInit);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__ios_basecLcLInit_Init_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 ios_base::Init * p;
   if (1 != libp->paran);
 p = new ios_base:: Init(*(ios_base::Init *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ios_basecLcLInit);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__ios_basecLcLInit_wAInit_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
       delete[](ios_base::Init *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
          G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(ios_base::Init) * i));
   else
    G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* codecvt_base */
// automatic default constructor
static int G__codecvt_base_codecvt_base_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   codecvt_base *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new codecvt_base[G__getaryconstruct()];
   else
      p = new codecvt_base;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_codecvt_base);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__codecvt_base_codecvt_base_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   codecvt_base *p;
   if (1 != libp->paran);
   p = new codecvt_base(*(codecvt_base *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_codecvt_base);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__codecvt_base_wAcodecvt_base_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](codecvt_base *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(codecvt_base) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


#ifdef HAS_PROPER_DO_LENGTH
static int G__codecvtlEcharcOcharcOintgR_length_0_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
  G__letint(result7, 105, (long) ((codecvt < char, char, int >*) (G__getstructoffset()))->length(*(codecvt < char, char, int >:: state_type *) G__Intref(&libp->para[0]), (const codecvt < char, char, int >::extern_type *) G__int(libp->para[1])
 ,(const codecvt < char, char, int >::extern_type *) G__int(libp->para[2]), (size_t) G__int(libp->para[3])));
   return (1 || funcname || hash || result7 || libp);
}
#endif

static int G__codecvtlEcharcOcharcOintgR_max_length_1_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((codecvt < char, char, int >*) (G__getstructoffset()))->max_length());
   return (1 || funcname || hash || result7 || libp);
}

/* collate<char> */
static int G__collatelEchargR_compare_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((collate < char >*) (G__getstructoffset()))->compare((const char *) G__int(libp->para[0]), (const char *) G__int(libp->para[1])
                                                                                        ,(const char *) G__int(libp->para[2]), (const char *) G__int(libp->para[3])));
   return (1 || funcname || hash || result7 || libp);
}

static int G__collatelEchargR_transform_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >*pobj,
       xobj = ((collate < char >*) (G__getstructoffset()))->transform((const char *) G__int(libp->para[0]), (const char *) G__int(libp->para[1]));
      pobj = new basic_string < char,
       char_traits < char >,
       allocator < char > >(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__collatelEchargR_hash_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 108, (long) ((collate < char >*) (G__getstructoffset()))->hash((const char *) G__int(libp->para[0]), (const char *) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

/* time_base */
// automatic default constructor
static int G__time_base_time_base_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   time_base *p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new time_base[G__getaryconstruct()];
   else
      p = new time_base;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_time_base);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__time_base_time_base_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   time_base *p;
   if (1 != libp->paran);
   p = new time_base(*(time_base *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_time_base);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__time_base_wAtime_base_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](time_base *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(time_base) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* istreambuf_iterator<char,char_traits<char> > */
static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   istreambuf_iterator < char,
    char_traits < char > >*p = NULL;
   if (G__getaryconstruct())
      p = new istreambuf_iterator < char,
       char_traits < char > >[G__getaryconstruct()];
   else
      p = new istreambuf_iterator < char,
       char_traits < char > >;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   istreambuf_iterator < char,
    char_traits < char > >*p = NULL;
   p = new istreambuf_iterator < char,
    char_traits < char > >(*(istreambuf_iterator < char, char_traits < char > >::istream_type *) libp->para[0].ref);
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   istreambuf_iterator < char,
    char_traits < char > >*p = NULL;
   p = new istreambuf_iterator < char,
    char_traits < char > >((istreambuf_iterator < char, char_traits < char > >::streambuf_type *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatormU_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 99, (long) ((istreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator * ());
   return (1 || funcname || hash || result7 || libp);
}

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      istreambuf_iterator < char,
       char_traits < char > >&obj = ((istreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator++ ();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      istreambuf_iterator < char,
       char_traits < char > >::proxy * pobj,
       xobj = ((istreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator++ ((int) G__int(libp->para[0]));
      pobj = new istreambuf_iterator < char,
       char_traits < char > >::proxy(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_equal_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((istreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->equal(*(istreambuf_iterator < char, char_traits < char > >*) libp->para[0].ref));
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   istreambuf_iterator < char,
    char_traits < char > >*p;
   if (1 != libp->paran);
   p = new istreambuf_iterator < char,
    char_traits < char > >(*(istreambuf_iterator < char, char_traits < char > >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_wAistreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_9_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](istreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(istreambuf_iterator < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* iterator<input_iterator_tag,char,long,char*,char&> */
// automatic default constructor
static int G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator < input_iterator_tag, char,
   long,
   char *,
   char &>*p;
   if (0 != libp->paran);
   if (G__getaryconstruct())
      p = new iterator < input_iterator_tag, char,
      long,
      char *,
      char &>[G__getaryconstruct()];
   else
      p = new iterator < input_iterator_tag, char,
      long,
      char *,
      char &>;
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   iterator < input_iterator_tag, char,
   long,
   char *,
   char &>*p;
   if (1 != libp->paran);
   p = new iterator < input_iterator_tag, char,
   long,
   char *,
   char &>(*(iterator < input_iterator_tag, char, long, char *, char &>*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_wAiteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](iterator < input_iterator_tag, char, long, char *, char &>*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(iterator < input_iterator_tag, char, long, char *, char &>) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* istreambuf_iterator<char,char_traits<char> >::proxy */

static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy_operatormU_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 99, (long) ((istreambuf_iterator < char, char_traits < char > >::proxy *) (G__getstructoffset()))->operator * ());
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy_proxy_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   istreambuf_iterator < char,
    char_traits < char > >::proxy * p;
   if (1 != libp->paran);
   p = new istreambuf_iterator < char,
    char_traits < char > >::proxy(*(istreambuf_iterator < char, char_traits < char > >::proxy *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy_wAproxy_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
       delete[](istreambuf_iterator < char, char_traits < char > >::proxy *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(istreambuf_iterator < char, char_traits < char > >::proxy) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_istream<char,char_traits<char> >::sentry */
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 117, (long) ((basic_istream < char, char_traits < char > >::sentry *) (G__getstructoffset()))->operator bool());
   return (1 || funcname || hash || result7 || libp);
}

static int G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   basic_istream < char,
    char_traits < char > >::sentry * p = NULL;
   p = new basic_istream < char,
    char_traits < char > >::sentry(*(basic_istream < char, char_traits < char > >*) libp->para[0].ref, (bool) G__int(libp->para[1]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_wAsentry_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
       delete[](basic_istream < char, char_traits < char > >::sentry *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_istream < char, char_traits < char > >::sentry) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* ostreambuf_iterator<char,char_traits<char> > */
static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_0_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   ostreambuf_iterator < char,
    char_traits < char > >*p = NULL;
   p = new ostreambuf_iterator < char,
    char_traits < char > >(*(ostreambuf_iterator < char, char_traits < char > >::ostream_type *) libp->para[0].ref);
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_1_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   ostreambuf_iterator < char,
    char_traits < char > >*p = NULL;
   p = new ostreambuf_iterator < char,
    char_traits < char > >((ostreambuf_iterator < char, char_traits < char > >::streambuf_type *) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatormU_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ostreambuf_iterator < char,
       char_traits < char > >&obj = ((ostreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator * ();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_3_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ostreambuf_iterator < char,
       char_traits < char > >&obj = ((ostreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator++ ();
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_4_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ostreambuf_iterator < char,
       char_traits < char > >&obj = ((ostreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator++ ((int) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatoreQ_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ostreambuf_iterator < char,
       char_traits < char > >&obj = ((ostreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->operator = ((ostreambuf_iterator < char, char_traits < char > >::char_type) G__int(libp->para[0]));
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_failed_6_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) ((ostreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset()))->failed());
   return (1 || funcname || hash || result7 || libp);
}

// automatic copy constructor
static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_7_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   ostreambuf_iterator < char,
    char_traits < char > >*p;
   if (1 != libp->paran);
   p = new ostreambuf_iterator < char,
    char_traits < char > >(*(ostreambuf_iterator < char, char_traits < char > >*) G__int(libp->para[0]));
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_wAostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_8_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
         delete[](ostreambuf_iterator < char, char_traits < char > >*) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(ostreambuf_iterator < char, char_traits < char > >) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* basic_ostream<char,char_traits<char> >::sentry */

static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
 G__letint(result7, 117, (long) ((basic_ostream < char, char_traits < char > >::sentry *) (G__getstructoffset()))->operator bool());
   return (1 || funcname || hash || result7 || libp);
}

// automatic destructor
static int G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_wAsentry_5_0(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   if (G__getaryconstruct())
      if (G__PVOID == G__getgvp())
       delete[](basic_ostream < char, char_traits < char > >::sentry *) (G__getstructoffset());
      else
         for (int i = G__getaryconstruct() - 1; i >= 0; i--)
            G__operator_delete( (void*) ((G__getstructoffset()) + sizeof(basic_ostream < char, char_traits < char > >::sentry) * i));
   else
      G__operator_delete( (void*) (G__getstructoffset()));
   G__setnull(result7);
   return (1 || funcname || hash || result7 || libp);
}


/* Setting up global function */

static int G___operatorpL_6_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      reverse_iterator < char *>*pobj,
       xobj = operator + (*((reverse_iterator < char *>::difference_type *) G__int(libp->para[0])), *(reverse_iterator < char *>*) libp->para[1].ref);
      pobj = new reverse_iterator < char *>(xobj);
      result7->obj.i = (long) ((void *) pobj);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
   }
   return (1 || funcname || hash || result7 || libp);
}

#if 0
static int G___reverse_iteratorlEcharmUgRcLcLdifference_typeoperatormI_7_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) reverse_iterator < char *>::difference_typeoperator - (*(reverse_iterator < char *>*) libp->para[0].ref, *(reverse_iterator < char *>*) libp->para[1].ref));
   return (1 || funcname || hash || result7 || libp);
}
#endif

static int G___localeconv_9_1(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 85, (long) localeconv());
   return (1 || funcname || hash || result7 || libp);
}

static int G___boolalpha_4_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = boolalpha(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___noboolalpha_5_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = noboolalpha(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___showbase_6_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = showbase(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___noshowbase_7_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = noshowbase(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___showpoint_8_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = showpoint(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___noshowpoint_9_2(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = noshowpoint(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___showpos_0_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = showpos(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___noshowpos_1_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = noshowpos(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___skipws_2_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = skipws(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___noskipws_3_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = noskipws(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___uppercase_4_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = uppercase(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___nouppercase_5_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = nouppercase(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___internal_6_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = internal(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___left_7_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = left(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___right_8_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = right(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___dec_9_3(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = dec(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___hex_0_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = hex(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___oct_1_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = oct(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___fixed_2_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = fixed(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___scientific_3_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = scientific(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___unitbuf_4_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = unitbuf(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___nounitbuf_5_4(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      ios_base & obj = nounitbuf(*(ios_base *) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___difftime_1_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letdouble(result7, 100, (double) difftime((time_t) G__int(libp->para[0]), (time_t) G__int(libp->para[1])));
   return (1 || funcname || hash || result7 || libp);
}

static int G___mktime_2_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) mktime((tm *) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G___time_3_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 105, (long) time((time_t *) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G___asctime_4_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) asctime((const tm *) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G___ctime_5_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 67, (long) ctime((const time_t *) G__int(libp->para[0])));
   return (1 || funcname || hash || result7 || libp);
}

static int G___strftime_8_5(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   G__letint(result7, 104, (long) strftime((char *) G__int(libp->para[0]), (size_t) G__int(libp->para[1])
                                           ,(const char *) G__int(libp->para[2]), (const tm *) G__int(libp->para[3])));
   return (1 || funcname || hash || result7 || libp);
}

static int G___endl_8_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = endl(*(basic_ostream < char, char_traits < char > >*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___ends_9_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = ends(*(basic_ostream < char, char_traits < char > >*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___flush_0_8(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_ostream < char,
       char_traits < char > >&obj = flush(*(basic_ostream < char, char_traits < char > >*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

static int G___ws_9_7(G__value * result7, const char *funcname, struct G__param *libp, int hash)
{
   {
      basic_istream < char,
       char_traits < char > >&obj = ws(*(basic_istream < char, char_traits < char > >*) libp->para[0].ref);
      result7->ref = (long) (&obj);
      result7->obj.i = (long) (&obj);
   }
   return (1 || funcname || hash || result7 || libp);
}

/*********************************************************
* Member function Stub
*********************************************************/

/* char_traits<char> */

/* basic_ios<char,char_traits<char> > */

/* basic_istream<char,char_traits<char> > */

/* basic_ostream<char,char_traits<char> > */

/* basic_iostream<char,char_traits<char> > */

/* basic_streambuf<char,char_traits<char> > */

/* basic_filebuf<char,char_traits<char> > */

/* allocator<char> */

/* basic_stringbuf<char,char_traits<char>,allocator<char> > */

/* basic_ifstream<char,char_traits<char> > */

/* basic_ofstream<char,char_traits<char> > */

/* basic_fstream<char,char_traits<char> > */

/* basic_istringstream<char,char_traits<char>,allocator<char> > */

/* basic_ostringstream<char,char_traits<char>,allocator<char> > */

/* basic_stringstream<char,char_traits<char>,allocator<char> > */

/* basic_string<char,char_traits<char>,allocator<char> > */

/* input_iterator_tag */

/* output_iterator_tag */

/* forward_iterator_tag */

/* bidirectional_iterator_tag */

/* random_access_iterator_tag */

/* output_iterator */

/* iterator<output_iterator_tag,void,void,void,void> */

/* rel_ops */

/* allocator<void> */

/* streampos */

/* b_str_ref<char,char_traits<char>,allocator<char> > */

/* reverse_iterator<char*> */

/* iterator_traits<char*> */

/* iterator<long,char*,char**,char*&,random_access_iterator_tag> */

/* lconv */

/* locale */

/* locale::facet */

/* locale::id */

/* locale::imp */

/* ctype_base */

/* ctype<char> */

/* ctype_byname<char> */

/* ios_base */

/* ios_base::Init */

/* codecvt_base */

/* codecvt<char,char,int> */

/* collate<char> */

/* time_base */

/* istreambuf_iterator<char,char_traits<char> > */

/* iterator<input_iterator_tag,char,long,char*,char&> */

/* istreambuf_iterator<char,char_traits<char> >::proxy */

/* basic_istream<char,char_traits<char> >::sentry */

/* ostreambuf_iterator<char,char_traits<char> > */

/* basic_ostream<char,char_traits<char> >::sentry */

/*********************************************************
* Global function Stub
*********************************************************/

/*********************************************************
* Get size of pointer to member function
*********************************************************/
class G__Sizep2memfunc {
   public:
   G__Sizep2memfunc() {
    p = &G__Sizep2memfunc::sizep2memfunc;
   }
   size_t sizep2memfunc() {
      return (sizeof(p));
   }
   private:
    size_t(G__Sizep2memfunc::*p) ();
};

size_t G__get_sizep2memfunc()
{
   G__Sizep2memfunc a;
   G__setsizep2memfunc((int) a.sizep2memfunc());
   return ((size_t) a.sizep2memfunc());
}


/*********************************************************
* virtual base class offset calculation interface
*********************************************************/

   /* Setting up class inheritance */
static long G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0(long pobject)
{
   basic_istream < char,
    char_traits < char > >*G__Lderived = (basic_istream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_ios_base_1(long pobject)
{
   basic_istream < char,
    char_traits < char > >*G__Lderived = (basic_istream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0(long pobject)
{
   basic_ostream < char,
    char_traits < char > >*G__Lderived = (basic_ostream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_ios_base_1(long pobject)
{
   basic_ostream < char,
    char_traits < char > >*G__Lderived = (basic_ostream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject)
{
   basic_iostream < char,
    char_traits < char > >*G__Lderived = (basic_iostream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_2(long pobject)
{
   basic_iostream < char,
    char_traits < char > >*G__Lderived = (basic_iostream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_4(long pobject)
{
   basic_iostream < char,
    char_traits < char > >*G__Lderived = (basic_iostream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_5(long pobject)
{
   basic_iostream < char,
    char_traits < char > >*G__Lderived = (basic_iostream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject)
{
   basic_ifstream < char,
    char_traits < char > >*G__Lderived = (basic_ifstream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2(long pobject)
{
   basic_ifstream < char,
    char_traits < char > >*G__Lderived = (basic_ifstream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject)
{
   basic_ofstream < char,
    char_traits < char > >*G__Lderived = (basic_ofstream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2(long pobject)
{
   basic_ofstream < char,
    char_traits < char > >*G__Lderived = (basic_ofstream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2(long pobject)
{
   basic_fstream < char,
    char_traits < char > >*G__Lderived = (basic_fstream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_3(long pobject)
{
   basic_fstream < char,
    char_traits < char > >*G__Lderived = (basic_fstream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5(long pobject)
{
   basic_fstream < char,
    char_traits < char > >*G__Lderived = (basic_fstream < char, char_traits < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_6(long pobject)
{
   basic_fstream < char,
    char_traits < char > >*G__Lderived = (basic_fstream < char, char_traits < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject)
{
   basic_istringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_istringstream < char, char_traits < char >, allocator < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2(long pobject)
{
   basic_istringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_istringstream < char, char_traits < char >, allocator < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1(long pobject)
{
   basic_ostringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_ostringstream < char, char_traits < char >, allocator < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2(long pobject)
{
   basic_ostringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_ostringstream < char, char_traits < char >, allocator < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2(long pobject)
{
   basic_stringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_stringstream < char, char_traits < char >, allocator < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_3(long pobject)
{
   basic_stringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_stringstream < char, char_traits < char >, allocator < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5(long pobject)
{
   basic_stringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_stringstream < char, char_traits < char >, allocator < char > >*) pobject;
   basic_ios < char,
    char_traits < char > >*G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}

static long G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_6(long pobject)
{
   basic_stringstream < char,
    char_traits < char >,
    allocator < char > >*G__Lderived = (basic_stringstream < char, char_traits < char >, allocator < char > >*) pobject;
   ios_base *G__Lbase = G__Lderived;
   return ((long) G__Lbase - (long) G__Lderived);
}


/*********************************************************
* Inheritance information setup/
*********************************************************/
extern "C" void G__cpp_setup_inheritance()
{

   /* Setting up class inheritance */
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR))) {
      basic_ios < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (basic_ios < char, char_traits < char > >*) 0x1000;
      {
         ios_base *G__Lpbase = (ios_base *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), G__get_linked_tagnum(&G__LN_ios_base), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR))) {
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0, 1, 3);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), G__get_linked_tagnum(&G__LN_ios_base), (long) G__2vbo_basic_istreamlEcharcOchar_traitslEchargRsPgR_ios_base_1, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR))) {
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_0, 1, 3);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), G__get_linked_tagnum(&G__LN_ios_base), (long) G__2vbo_basic_ostreamlEcharcOchar_traitslEchargRsPgR_ios_base_1, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR))) {
      basic_iostream < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (basic_iostream < char, char_traits < char > >*) 0x1000;
      {
         basic_istream < char,
          char_traits < char > >*G__Lpbase = (basic_istream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_2, 1, 2);
      }
      {
         basic_ostream < char,
          char_traits < char > >*G__Lpbase = (basic_ostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_4, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_iostreamlEcharcOchar_traitslEchargRsPgR_ios_base_5, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR))) {
      basic_filebuf < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (basic_filebuf < char, char_traits < char > >*) 0x1000;
      {
         basic_streambuf < char,
          char_traits < char > >*G__Lpbase = (basic_streambuf < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
      basic_stringbuf < char,
       char_traits < char >,
       allocator < char > >*G__Lderived;
      G__Lderived = (basic_stringbuf < char, char_traits < char >, allocator < char > >*) 0x1000;
      {
         basic_streambuf < char,
          char_traits < char > >*G__Lpbase = (basic_streambuf < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR))) {
      basic_ifstream < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (basic_ifstream < char, char_traits < char > >*) 0x1000;
      {
         basic_istream < char,
          char_traits < char > >*G__Lpbase = (basic_istream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR))) {
      basic_ofstream < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (basic_ofstream < char, char_traits < char > >*) 0x1000;
      {
         basic_ostream < char,
          char_traits < char > >*G__Lpbase = (basic_ostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }

      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_ios_base_2, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR))) {
      basic_fstream < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (basic_fstream < char, char_traits < char > >*) 0x1000;
      {
         basic_iostream < char,
          char_traits < char > >*G__Lpbase = (basic_iostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         basic_istream < char,
          char_traits < char > >*G__Lpbase = (basic_istream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_3, 1, 2);
      }
      {
         basic_ostream < char,
          char_traits < char > >*G__Lpbase = (basic_ostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_fstreamlEcharcOchar_traitslEchargRsPgR_ios_base_6, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
      basic_istringstream < char,
       char_traits < char >,
       allocator < char > >*G__Lderived;
      G__Lderived = (basic_istringstream < char, char_traits < char >, allocator < char > >*) 0x1000;
      {
         basic_istream < char,
          char_traits < char > >*G__Lpbase = (basic_istream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
      basic_ostringstream < char,
       char_traits < char >,
       allocator < char > >*G__Lderived;
      G__Lderived = (basic_ostringstream < char, char_traits < char >, allocator < char > >*) 0x1000;
      {
         basic_ostream < char,
          char_traits < char > >*G__Lpbase = (basic_ostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
			      (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_1, 
			      1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_2, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR))) {
      basic_stringstream < char,
       char_traits < char >,
       allocator < char > >*G__Lderived;
      G__Lderived = (basic_stringstream < char, char_traits < char >, allocator < char > >*) 0x1000;
      {
         basic_iostream < char,
          char_traits < char > >*G__Lpbase = (basic_iostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         basic_istream < char,
          char_traits < char > >*G__Lpbase = (basic_istream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_2, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_3, 1, 2);
      }
      {
         basic_ostream < char,
          char_traits < char > >*G__Lpbase = (basic_ostream < char, char_traits < char > >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
			      G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 
			      (long) G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_5, 1, 2);
      }
      {
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__get_linked_tagnum(&G__LN_ios_base), 
			      (long) G__2vbo_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_ios_base_6, 1, 2);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_forward_iterator_tag))) {
      forward_iterator_tag *G__Lderived;
      G__Lderived = (forward_iterator_tag *) 0x1000;
      {
         input_iterator_tag *G__Lpbase = (input_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_forward_iterator_tag), G__get_linked_tagnum(&G__LN_input_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         output_iterator_tag *G__Lpbase = (output_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_forward_iterator_tag), G__get_linked_tagnum(&G__LN_output_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag))) {
      bidirectional_iterator_tag *G__Lderived;
      G__Lderived = (bidirectional_iterator_tag *) 0x1000;
      {
         forward_iterator_tag *G__Lpbase = (forward_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), G__get_linked_tagnum(&G__LN_forward_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         input_iterator_tag *G__Lpbase = (input_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), G__get_linked_tagnum(&G__LN_input_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         output_iterator_tag *G__Lpbase = (output_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), G__get_linked_tagnum(&G__LN_output_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_random_access_iterator_tag))) {
      random_access_iterator_tag *G__Lderived;
      G__Lderived = (random_access_iterator_tag *) 0x1000;
      {
         bidirectional_iterator_tag *G__Lpbase = (bidirectional_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag), G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         forward_iterator_tag *G__Lpbase = (forward_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag), G__get_linked_tagnum(&G__LN_forward_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         input_iterator_tag *G__Lpbase = (input_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag), G__get_linked_tagnum(&G__LN_input_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
         output_iterator_tag *G__Lpbase = (output_iterator_tag *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag), G__get_linked_tagnum(&G__LN_output_iterator_tag), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR))) {
      reverse_iterator < char *>*G__Lderived;
      G__Lderived = (reverse_iterator < char *>*) 0x1000;
      {
         iterator < long,
         char *,
         char **,
         char *&,
          random_access_iterator_tag > *G__Lpbase = (iterator < long, char *, char **, char *&, random_access_iterator_tag > *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_ctypelEchargR))) {
      ctype < char >*G__Lderived;
      G__Lderived = (ctype < char >*) 0x1000;
      {
         ctype_base *G__Lpbase = (ctype_base *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_ctypelEchargR), G__get_linked_tagnum(&G__LN_ctype_base), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
       locale:: facet * G__Lpbase = (locale::facet *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_ctypelEchargR), G__get_linked_tagnum(&G__LN_localecLcLfacet), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_ctype_bynamelEchargR))) {
      ctype_byname < char >*G__Lderived;
      G__Lderived = (ctype_byname < char >*) 0x1000;
      {
         ctype < char >*G__Lpbase = (ctype < char >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_ctype_bynamelEchargR), G__get_linked_tagnum(&G__LN_ctypelEchargR), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         ctype_base *G__Lpbase = (ctype_base *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_ctype_bynamelEchargR), G__get_linked_tagnum(&G__LN_ctype_base), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
      {
       locale:: facet * G__Lpbase = (locale::facet *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_ctype_bynamelEchargR), G__get_linked_tagnum(&G__LN_localecLcLfacet), (long) G__Lpbase - (long) G__Lderived, 1, 0);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR))) {
      codecvt < char,
      char,
      int >*G__Lderived;
      G__Lderived = (codecvt < char, char, int >*) 0x1000;
      {
       locale:: facet * G__Lpbase = (locale::facet *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR), G__get_linked_tagnum(&G__LN_localecLcLfacet), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
      {
         codecvt_base *G__Lpbase = (codecvt_base *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR), G__get_linked_tagnum(&G__LN_codecvt_base), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_collatelEchargR))) {
      collate < char >*G__Lderived;
      G__Lderived = (collate < char >*) 0x1000;
      {
       locale:: facet * G__Lpbase = (locale::facet *) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_collatelEchargR), G__get_linked_tagnum(&G__LN_localecLcLfacet), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR))) {
      istreambuf_iterator < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (istreambuf_iterator < char, char_traits < char > >*) 0x1000;
      {
         iterator < input_iterator_tag, char,
         long,
         char *,
         char &>*G__Lpbase = (iterator < input_iterator_tag, char, long, char *, char &>*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
   if (0 == G__getnumbaseclass(G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR))) {
      ostreambuf_iterator < char,
       char_traits < char > >*G__Lderived;
      G__Lderived = (ostreambuf_iterator < char, char_traits < char > >*) 0x1000;
      {
         iterator < output_iterator_tag, void,
         void,
         void,
         void >*G__Lpbase = (iterator < output_iterator_tag, void, void, void, void >*) G__Lderived;
         G__inheritance_setup(G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR), (long) G__Lpbase - (long) G__Lderived, 1, 1);
      }
   }
}

/*********************************************************
* typedef information setup/
*********************************************************/
extern "C" void G__cpp_setup_typetable()
{

   /* Setting up typedef entry */
   G__search_typename2("ios", 117, G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("istream", 117, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ostream", 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iostream", 117, G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streambuf", 117, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("filebuf", 117, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("stringbuf", 117, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ifstream", 117, G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ofstream", 117, G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fstream", 117, G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("istringstream", 117, G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ostringstream", 117, G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("stringstream", 117, G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streamoff", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("size_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("unexpected_handler", 89, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("terminate_handler", 89, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("new_handler", 89, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__new_handler", 89, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("difference_type", 121, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 121, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 121, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reference", 121, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator_category", 117, G__get_linked_tagnum(&G__LN_output_iterator_tag), 0, G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("size_type", 104, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("difference_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 89, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_pointer", 89, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 121, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("size_type", 104, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("difference_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_pointer", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reference", 99, -1, 1,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_reference", 99, -1, 1,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fstate_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("state_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_b_str_reflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("size_type", 104, -1, 0,
                       G__get_linked_tagnum(&G__LN_b_str_reflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("buff_t", 117, G__get_linked_tagnum(&G__LN_b_str_reflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("template", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLtemplate), 0, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("allocator_type", 117, G__get_linked_tagnum(&G__LN_allocatorlEchargR), 0, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("size_type", 104, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("difference_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reference", 99, -1, 1,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_reference", 99, -1, 1,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_pointer", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_iterator", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reverse_iterator<const_iterator>", 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("difference_type", 108, -1, 0,
                 G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 67, -1, 0,
                 G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 67, -1, 2,
                 G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reference", 67, -1, 1,
                 G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator_category", 117, G__get_linked_tagnum(&G__LN_random_access_iterator_tag), 0, G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("distance_type", 108, -1, 0,
                 G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2(
	     "iterator<iterator_traits<char*>::difference_type,iterator_traits<char*>::value_type,iterator_traits<char*>::pointer,iterator_traits<char*>::reference,iterator_traits<char*>::iterator_category>", 
	     117, G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("difference_type", 67, -1, 2,
                       G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 67, -1, 1,
                       G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reference", 117, G__get_linked_tagnum(&G__LN_random_access_iterator_tag), 0, G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator_category", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("distance_type", 67, -1, 2,
                       G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("difference_type", 108, -1, 0,
                G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 67, -1, 0,
                G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("Pointer", 67, -1, 2,
                G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("Reference", 67, -1, 1,
                G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator_category", 117, G__get_linked_tagnum(&G__LN_random_access_iterator_tag), 0, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator_type", 67, -1, 0,
                G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("const_reverse_iterator", 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), 0, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reverse_iterator<iterator>", 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("reverse_iterator", 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), 0, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("category", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_locale));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_ctypelEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streamsize", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fmtflags", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_ios_base));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("openmode", 115, -1, 0,
                       G__get_linked_tagnum(&G__LN_ios_base));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iostate", 115, -1, 0,
                       G__get_linked_tagnum(&G__LN_ios_base));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("event_callback", 89, -1, 0,
                       G__get_linked_tagnum(&G__LN_ios_base));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
   G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
   G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
   G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_streambuf<char_type,char_traits<char> >", 117, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("basic_ostream<char_type,char_traits<char> >", 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("basic_ios<char_type,char_traits<char> >", 117, G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("__int32_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__uint32_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__int64_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__uint64_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__psint_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__psunsigned_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__scint_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__scunsigned_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uchar_t", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ushort_t", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uint_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ulong_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("addr_t", 67, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("caddr_t", 67, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("daddr_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pgno_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pfn_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("cnt_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pgcnt_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("boolean_t", 117, G__get_linked_tagnum(&G__LN_boolean_t), 0, -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("id_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("major_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("minor_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_mode_t", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_dev_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_uid_t", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_gid_t", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_nlink_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_pid_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("o_ino_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("mode_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("dev_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uid_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("gid_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("nlink_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pid_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ino_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ino64_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off64_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("__scoff_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("scoff_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("blkcnt_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fsblkcnt_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fsfilcnt_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("swblk_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("paddr_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("key_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("use_t", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("sysid_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("index_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("lock_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("cpuid_t", 99, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pri_t", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("accum_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("prid_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ash_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ssize_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("time_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("clockid_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("timer_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("useconds_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("bitnum_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("bitlen_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("processorid_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("toid_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("qaddr_t", 76, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("inst_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("machreg_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fpreg_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int8_t", 99, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uint8_t", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int16_t", 115, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uint16_t", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int32_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uint32_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int64_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uint64_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("intmax_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uintmax_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("intptr_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uintptr_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_int8_t", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_int16_t", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_int32_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("hostid_t", 108, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("unchar", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_char", 98, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ushort", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_short", 114, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("uint", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_int", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ulong", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("u_long", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fd_mask_t", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ufd_mask_t", 104, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("fd_mask", 105, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("k_sigset_t", 107, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("SIG_PF", 89, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("state_type", 105, -1, 0,
                G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("intern_type", 99, -1, 0,
                G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("extern_type", 99, -1, 0,
                G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_collatelEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_type", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_collatelEchargR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streambuf_type", 117, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator<input_iterator_tag,char,char_traits<char>::off_type,char*,char&>", 117, G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("difference_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("value_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pointer", 67, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("reference", 99, -1, 1,
                       G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iterator_category", 117, G__get_linked_tagnum(&G__LN_input_iterator_tag), 0, G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("distance_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streambuf_type", 117, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("istream_type", 117, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("phil_interm", 117, G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ctype_facet", 117, G__get_linked_tagnum(&G__LN_ctypelEchargR), 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ctype<char_type>", 117, G__get_linked_tagnum(&G__LN_ctypelEchargR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("ctype_facet", 117, G__get_linked_tagnum(&G__LN_ctypelEchargR), 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_istream<char,char_traits<char> >& (*) (basic_istream<char,char_traits<char> >&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ios_base& (*) (ios_base&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_ios<char,char_traits<char> >& (*) (basic_ios<char, char_traits<char>>&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_ios<char,char_traits<char> >& (*)(basic_ios<char, char_traits<char>>&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ostream_type", 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streambuf_type", 117, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ostream_type", 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("streambuf_type", 117, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("iter_type", 117, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ostream_type& (*)(ostream_type&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_ios<char,char_traits<char> >& (*)(ios_base&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ios_base& (*)(ios_base&)", 81, -1, 0,
                       -1);
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_traits<char_type>", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("string_traits", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_string<char_type,string_traits,allocator<char> >", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("string_type", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_traits", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_type", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("basic_stringbuf<char_type,char_traits<char>,allocator<char> >", 117, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("sb_type", 117, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_traits", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_type", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("sb_type", 117, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_traits", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("string_type", 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("sb_type", 117, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("result", 117, G__get_linked_tagnum(&G__LN_codecvt_basecLcLresult), 0, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("state_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("filebuf_type", 117, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("codecvt<char,char,state_type>", 117, G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR), 0, -1);
   G__setnewtype(-1, "#line 1 \"list.C\"", 0);
   G__search_typename2("ofacet_type", 117, G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR), 0, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("ifacet_type", 117, G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR), 0, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("filebuf_type", 117, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("filebuf_type", 117, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("filebuf_type", 117, G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 0, G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("char_type", 99, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("pos_type", 117, G__get_linked_tagnum(&G__LN_streampos), 0, G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("off_type", 108, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("int_type", 105, -1, 0,
                       G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
   G__search_typename2("traits_type", 117, G__get_linked_tagnum(&G__LN_char_traitslEchargR), 0, G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__setnewtype(-1, NULL, 0);
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

   /* char_traits<char> */
static void G__setup_memvarchar_traitslEchargR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   {
      char_traits < char >*p;
      p = (char_traits < char >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_ios<char,char_traits<char> > */
static void G__setup_memvarbasic_ioslEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   {
      basic_ios < char,
       char_traits < char > >*p;
      p = (basic_ios < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_istream<char,char_traits<char> > */
static void G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
   {
      basic_istream < char,
       char_traits < char > >*p;
      p = (basic_istream < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_ostream<char,char_traits<char> > */
static void G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   {
      basic_ostream < char,
       char_traits < char > >*p;
      p = (basic_ostream < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_iostream<char,char_traits<char> > */
static void G__setup_memvarbasic_iostreamlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR));
   {
      basic_iostream < char,
       char_traits < char > >*p;
      p = (basic_iostream < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_streambuf<char,char_traits<char> > */
static void G__setup_memvarbasic_streambuflEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   {
      basic_streambuf < char,
       char_traits < char > >*p;
      p = (basic_streambuf < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_filebuf<char,char_traits<char> > */
static void G__setup_memvarbasic_filebuflEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   {
      basic_filebuf < char,
       char_traits < char > >*p;
      p = (basic_filebuf < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* allocator<char> */
static void G__setup_memvarallocatorlEchargR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   {
      allocator < char >*p;
      p = (allocator < char >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_stringbuf<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   {
      basic_stringbuf < char,
       char_traits < char >,
       allocator < char > >*p;
      p = (basic_stringbuf < char, char_traits < char >, allocator < char > >*) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLdA), -1, -2, 1, "inc_size=64", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* basic_ifstream<char,char_traits<char> > */
static void G__setup_memvarbasic_ifstreamlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   {
      basic_ifstream < char,
       char_traits < char > >*p;
      p = (basic_ifstream < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_ofstream<char,char_traits<char> > */
static void G__setup_memvarbasic_ofstreamlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   {
      basic_ofstream < char,
       char_traits < char > >*p;
      p = (basic_ofstream < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_fstream<char,char_traits<char> > */
static void G__setup_memvarbasic_fstreamlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   {
      basic_fstream < char,
       char_traits < char > >*p;
      p = (basic_fstream < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_istringstream<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   {
      basic_istringstream < char,
       char_traits < char >,
       allocator < char > >*p;
      p = (basic_istringstream < char, char_traits < char >, allocator < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_ostringstream<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   {
      basic_ostringstream < char,
       char_traits < char >,
       allocator < char > >*p;
      p = (basic_ostringstream < char, char_traits < char >, allocator < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_stringstream<char,char_traits<char>,allocator<char> > */
static void G__setup_memvarbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   {
      basic_stringstream < char,
       char_traits < char >,
       allocator < char > >*p;
      p = (basic_stringstream < char, char_traits < char >, allocator < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_string<char,char_traits<char>,allocator<char> > */
static const long npos = (long) -1;

static void G__setup_memvarbasic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   {
      basic_string < char,
       char_traits < char >,
       allocator < char > >*p;
      p = (string*) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) (&npos), 108, 0, 1, -1, -1, -2, 1, "npos=", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}



   /* input_iterator_tag */
static void G__setup_memvarinput_iterator_tag(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_input_iterator_tag));
   {
      input_iterator_tag *p;
      p = (input_iterator_tag *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* output_iterator_tag */
static void G__setup_memvaroutput_iterator_tag(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_output_iterator_tag));
   {
      output_iterator_tag *p;
      p = (output_iterator_tag *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* forward_iterator_tag */
static void G__setup_memvarforward_iterator_tag(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_forward_iterator_tag));
   {
      forward_iterator_tag *p;
      p = (forward_iterator_tag *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* bidirectional_iterator_tag */
static void G__setup_memvarbidirectional_iterator_tag(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag));
   {
      bidirectional_iterator_tag *p;
      p = (bidirectional_iterator_tag *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* random_access_iterator_tag */
static void G__setup_memvarrandom_access_iterator_tag(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag));
   {
      random_access_iterator_tag *p;
      p = (random_access_iterator_tag *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* iterator<output_iterator_tag,void,void,void,void> */
static void G__setup_memvariteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   {
      iterator < output_iterator_tag, void,
      void,
      void,
      void >*p;
      p = (iterator < output_iterator_tag, void, void, void, void >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* rel_ops */
static void G__setup_memvarrel_ops(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_rel_ops));
   {
   }
   G__tag_memvar_reset();
}


   /* allocator<void> */
static void G__setup_memvarallocatorlEvoidgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   {
      allocator < void >*p;
      p = (allocator < void >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* streampos */
static void G__setup_memvarstreampos(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_streampos));
   {
      streampos *p;
      p = (streampos *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* reverse_iterator<char*> */
static void G__setup_memvarreverse_iteratorlEcharmUgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   {
      reverse_iterator < char *>*p;
      p = (reverse_iterator < char *>*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* iterator_traits<char*> */
static void G__setup_memvariterator_traitslEcharmUgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   {
      iterator_traits < char *>*p;
      p = (iterator_traits < char *>*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* iterator<long,char*,char**,char*&,random_access_iterator_tag> */
static void G__setup_memvariteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   {
      iterator < long,
      char *,
      char **,
      char *&,
       random_access_iterator_tag > *p;
      p = (iterator < long, char *, char **, char *&, random_access_iterator_tag > *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* lconv */
static void G__setup_memvarlconv(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_lconv));
   {
      lconv *p;
      p = (lconv *) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) ((long) (&p->decimal_point) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "decimal_point=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->thousands_sep) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "thousands_sep=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->grouping) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "grouping=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->int_curr_symbol) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "int_curr_symbol=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->currency_symbol) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "currency_symbol=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->mon_decimal_point) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "mon_decimal_point=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->mon_thousands_sep) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "mon_thousands_sep=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->mon_grouping) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "mon_grouping=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->positive_sign) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "positive_sign=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->negative_sign) - (long) (p)), 67, 0, 0, -1, -1, -1, 1, "negative_sign=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->int_frac_digits) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "int_frac_digits=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->frac_digits) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "frac_digits=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->p_cs_precedes) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "p_cs_precedes=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->p_sep_by_space) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "p_sep_by_space=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->n_cs_precedes) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "n_cs_precedes=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->n_sep_by_space) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "n_sep_by_space=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->p_sign_posn) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "p_sign_posn=", 0, (char *) NULL);
      G__memvar_setup((void *) ((long) (&p->n_sign_posn) - (long) (p)), 99, 0, 0, -1, -1, -1, 1, "n_sign_posn=", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* locale */
static void G__setup_memvarlocale(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_locale));
   {
      locale *p;
      p = (locale *) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "none=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "collate=16", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "ctype=32", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "monetary=64", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "numeric=128", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "time=256", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "messages=512", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_localecLcLdA), -1, -2, 1, "all=1008", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* locale::facet */
static void G__setup_memvarlocalecLcLfacet(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_localecLcLfacet));
   {
    locale::facet * p;
    p = (locale::facet *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* locale::id */
static void G__setup_memvarlocalecLcLid(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_localecLcLid));
   {
    locale::id * p;
    p = (locale::id *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* enum locale:: */



   /* ctype_base */
static void G__setup_memvarctype_base(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_ctype_base));
   {
      ctype_base *p;
      p = (ctype_base *) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "space=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "print=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "cntrl=4", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "upper=8", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "lower=16", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "alpha=32", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "digit=64", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "punct=128", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "xdigit=256", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "alnum=96", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "graph=224", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "spcnt=5", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "spprn=3", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "prpug=130", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "prxag=322", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "puxag=298", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "pualg=42", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "plxag=306", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), -1, -2, 1, "plalg=50", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* ctype_base::mask */

   /* ctype<char> */
static void G__setup_memvarctypelEchargR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_ctypelEchargR));
   {
      ctype < char >*p;
      p = (ctype < char >*) 0x1000;
      if (p) {
      }
    G__memvar_setup((void *) (&ctype < char >::id), 117, 0, 0, G__get_linked_tagnum(&G__LN_localecLcLid), -1, -2, 1, "id=", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}




   /* ios_base */
static void G__setup_memvarios_base(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_ios_base));
   {
      ios_base *p;
      p = (ios_base *) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLio_state), -1, -2, 1, "goodbit=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLio_state), -1, -2, 1, "eofbit=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLio_state), -1, -2, 1, "failbit=4", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLio_state), -1, -2, 1, "badbit=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), -1, -2, 1, "in=8", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), -1, -2, 1, "out=16", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), -1, -2, 1, "ate=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), -1, -2, 1, "app=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), -1, -2, 1, "trunc=32", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), -1, -2, 1, "binary=4", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLseekdir), -1, -2, 1, "beg=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLseekdir), -1, -2, 1, "cur=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLseekdir), -1, -2, 1, "end=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "boolalpha=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "adjustfield=112", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "basefield=14", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "floatfield=384", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "skipws=4096", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "left=32", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "right=64", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "internal=16", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "dec=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "oct=8", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "hex=4", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "showbase=512", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "showpoint=1024", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "uppercase=16384", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "showpos=2048", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "scientific=256", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "fixed=128", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "unitbuf=8192", 0, (char *) NULL);

      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), -1, -2, 1, "null=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLevent), -1, -2, 1, "erase_event=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLevent), -1, -2, 1, "imbue_event=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_ios_basecLcLevent), -1, -2, 1, "copyfmt_event=2", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* ios_base::io_state */

   /* ios_base::open_mode */

   /* ios_base::seekdir */

   /* ios_base::Init */
static void G__setup_memvarios_basecLcLInit(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLInit));
   {
    ios_base::Init * p;
    p = (ios_base::Init *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* ios_base::event */




   /* boolean_t */


   /* codecvt_base */
static void G__setup_memvarcodecvt_base(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_codecvt_base));
   {
      codecvt_base *p;
      p = (codecvt_base *) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_codecvt_basecLcLresult), -1, -2, 1, "ok=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_codecvt_basecLcLresult), -1, -2, 1, "partial=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_codecvt_basecLcLresult), -1, -2, 1, "error=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_codecvt_basecLcLresult), -1, -2, 1, "noconv=3", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* codecvt_base::result */

   /* codecvt<char,char,int> */
static void G__setup_memvarcodecvtlEcharcOcharcOintgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR));
   {
      codecvt < char,
      char,
      int >*p;
      p = (codecvt < char, char, int >*) 0x1000;
      if (p) {
      }
    G__memvar_setup((void *) (&codecvt < char, char, int >::id), 117, 0, 0, G__get_linked_tagnum(&G__LN_localecLcLid), -1, -2, 1, "id=", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* collate<char> */
static void G__setup_memvarcollatelEchargR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_collatelEchargR));
   {
      collate < char >*p;
      p = (collate < char >*) 0x1000;
      if (p) {
      }
    G__memvar_setup((void *) (&collate < char >::id), 117, 0, 0, G__get_linked_tagnum(&G__LN_localecLcLid), -1, -2, 1, "id=", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}




   /* time_base */
static void G__setup_memvartime_base(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_time_base));
   {
      time_base *p;
      p = (time_base *) 0x1000;
      if (p) {
      }
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLdateorder), -1, -2, 1, "no_order=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLdateorder), -1, -2, 1, "dmy=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLdateorder), -1, -2, 1, "mdy=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLdateorder), -1, -2, 1, "ymd=3", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLdateorder), -1, -2, 1, "ydm=4", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "a=0", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "A=1", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "b=2", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "B=3", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "c=4", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "d=5", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "H=6", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "I=7", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "j=8", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "m=9", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "M=10", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "p=11", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "S=12", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "U=13", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "w=14", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "W=15", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "x=16", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "X=17", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "y=18", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "Y=19", 0, (char *) NULL);
      G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), -1, -2, 1, "END=20", 0, (char *) NULL);
   }
   G__tag_memvar_reset();
}


   /* time_base::dateorder */

   /* time_base::t_conv_spec */

   /* istreambuf_iterator<char,char_traits<char> > */
static void G__setup_memvaristreambuf_iteratorlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   {
      istreambuf_iterator < char, char_traits < char > >*p;
      p = (istreambuf_iterator < char, char_traits < char > >*) 0x1000;
      p = p;
   }
   G__tag_memvar_reset();
}


   /* iterator<input_iterator_tag,char,long,char*,char&> */
static void G__setup_memvariteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   {
      iterator < input_iterator_tag, char,
      long,
      char *,
      char &>*p;
      p = (iterator < input_iterator_tag, char, long, char *, char &>*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* istreambuf_iterator<char,char_traits<char> >::proxy */
static void G__setup_memvaristreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy));
   {
      istreambuf_iterator < char,
       char_traits < char > >::proxy * p;
    p = (istreambuf_iterator < char, char_traits < char > >::proxy *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_istream<char,char_traits<char> >::sentry */
static void G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   {
      basic_istream < char,
       char_traits < char > >::sentry * p;
    p = (basic_istream < char, char_traits < char > >::sentry *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* ostreambuf_iterator<char,char_traits<char> > */
static void G__setup_memvarostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   {
      ostreambuf_iterator < char,
       char_traits < char > >*p;
      p = (ostreambuf_iterator < char, char_traits < char > >*) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* basic_ostream<char,char_traits<char> >::sentry */
static void G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void)
{
   G__tag_memvar_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   {
      basic_ostream < char,
       char_traits < char > >::sentry * p;
    p = (basic_ostream < char, char_traits < char > >::sentry *) 0x1000;
      if (p) {
      }
   }
   G__tag_memvar_reset();
}


   /* enum basic_stringbuf<char,char_traits<char>,allocator<char> >:: */
extern "C" void G__cpp_setup_memvar()
{
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
static void G__setup_memfuncchar_traitslEchargR(void)
{
   /* char_traits<char> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_char_traitslEchargR));
   G__memfunc_setup("assign", 645, G__char_traitslEchargR_assign_0_0, 121, -1, -1, 0, 2, 1, 1, 0,
                    "c - 'char_traits<char>::char_type' 1 - c1 c - 'char_traits<char>::char_type' 11 - c2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("eq", 214, G__char_traitslEchargR_eq_1_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 2, 1, 1, 0,
                    "c - 'char_traits<char>::char_type' 11 - c1 c - 'char_traits<char>::char_type' 11 - c2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("lt", 224, G__char_traitslEchargR_lt_2_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 2, 1, 1, 0,
                    "c - 'char_traits<char>::char_type' 11 - c1 c - 'char_traits<char>::char_type' 11 - c2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("compare", 743, G__char_traitslEchargR_compare_3_0, 105, -1, -1, 0, 3, 1, 1, 0,
                    "C - 'char_traits<char>::char_type' 10 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
                  "h - 'size_t' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("length", 642, G__char_traitslEchargR_length_4_0, 104, -1, G__defined_typename("size_t"), 0, 1, 1, 1, 0, "C - 'char_traits<char>::char_type' 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("copy", 443, G__char_traitslEchargR_copy_5_0, 67, -1, G__defined_typename("char_traits<char>::char_type"), 0, 3, 1, 1, 0,
                    "C - 'char_traits<char>::char_type' 0 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
                  "h - 'size_t' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find", 417, G__char_traitslEchargR_find_6_0, 67, -1, G__defined_typename("char_traits<char>::char_type"), 0, 3, 1, 1, 1,
          "C - 'char_traits<char>::char_type' 10 - s h - 'size_t' 0 - n "
                    "c - 'char_traits<char>::char_type' 11 - a", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("move", 439, G__char_traitslEchargR_move_7_0, 67, -1, G__defined_typename("char_traits<char>::char_type"), 0, 3, 1, 1, 0,
                    "C - 'char_traits<char>::char_type' 0 - s1 C - 'char_traits<char>::char_type' 10 - s2 "
                  "h - 'size_t' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("assign", 645, G__char_traitslEchargR_assign_8_0, 67, -1, G__defined_typename("char_traits<char>::char_type"), 0, 3, 1, 1, 0,
           "C - 'char_traits<char>::char_type' 0 - s h - 'size_t' 0 - n "
                    "c - 'char_traits<char>::char_type' 0 - a", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("not_eof", 746, G__char_traitslEchargR_not_eof_9_0, 105, -1, G__defined_typename("char_traits<char>::int_type"), 0, 1, 1, 1, 0, "i - 'char_traits<char>::int_type' 11 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("to_char_type", 1281, G__char_traitslEchargR_to_char_type_0_1, 99, -1, G__defined_typename("char_traits<char>::char_type"), 0, 1, 1, 1, 0, "i - 'char_traits<char>::int_type' 11 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("to_int_type", 1198, G__char_traitslEchargR_to_int_type_1_1, 105, -1, G__defined_typename("char_traits<char>::int_type"), 0, 1, 1, 1, 0, "c - 'char_traits<char>::char_type' 11 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("eq_int_type", 1185, G__char_traitslEchargR_eq_int_type_2_1, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 2, 1, 1, 0,
                    "i - 'char_traits<char>::int_type' 11 - c1 i - 'char_traits<char>::int_type' 11 - c2", (char *) NULL, (void *) NULL, 0);
   // automatic default constructor
   G__memfunc_setup("char_traits<char>", 1708, G__char_traitslEchargR_char_traitslEchargR_8_1, (int) ('i'), G__get_linked_tagnum(&G__LN_char_traitslEchargR), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("char_traits<char>", 1708, G__char_traitslEchargR_char_traitslEchargR_9_1, (int) ('i'), G__get_linked_tagnum(&G__LN_char_traitslEchargR), -1, 0, 1, 1, 1, 0, "u 'char_traits<char>' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~char_traits<char>", 1834, G__char_traitslEchargR_wAchar_traitslEchargR_0_2, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ioslEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_ios<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("operator void*", 1384, G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatorsPvoidmU_0_0, 89, -1, -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator!", 909, G__basic_ioslEcharcOchar_traitslEchargRsPgR_operatornO_1_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rdstate", 759, G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdstate_2_0, 115, -1, G__defined_typename("ios_base::iostate"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("clear", 519, G__basic_ioslEcharcOchar_traitslEchargRsPgR_clear_3_0, 121, -1, -1, 0, 1, 1, 1, 0, "s - 'ios_base::iostate' 0 goodbit state_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("setstate", 877, G__basic_ioslEcharcOchar_traitslEchargRsPgR_setstate_4_0, 121, -1, -1, 0, 1, 1, 1, 0, "s - 'ios_base::iostate' 0 - state_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("good", 425, G__basic_ioslEcharcOchar_traitslEchargRsPgR_good_5_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("eof", 314, G__basic_ioslEcharcOchar_traitslEchargRsPgR_eof_6_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("fail", 412, G__basic_ioslEcharcOchar_traitslEchargRsPgR_fail_7_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("bad", 295, G__basic_ioslEcharcOchar_traitslEchargRsPgR_bad_8_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("exceptions", 1090, G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_9_0, 115, -1, G__defined_typename("ios_base::iostate"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("exceptions", 1090, G__basic_ioslEcharcOchar_traitslEchargRsPgR_exceptions_0_1, 121, -1, -1, 0, 1, 1, 1, 0, "s - 'ios_base::iostate' 0 - except_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("tie", 322, G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_3_1, 85, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("tie", 322, 
		    G__basic_ioslEcharcOchar_traitslEchargRsPgR_tie_4_1, 85, 
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "U 'basic_ostream<char,char_traits<char> >' - 0 - tiestr_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rdbuf", 531, G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_5_1, 85, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rdbuf", 531, G__basic_ioslEcharcOchar_traitslEchargRsPgR_rdbuf_6_1, 85, 
		    G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "U 'basic_streambuf<char,char_traits<char> >' - 0 - sb_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("copyfmt", 770, G__basic_ioslEcharcOchar_traitslEchargRsPgR_copyfmt_7_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "u 'basic_ios<char,char_traits<char> >' - 11 - rhs", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("fill", 423, G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_8_1, 99, -1, G__defined_typename("basic_ios<char,char_traits<char> >::char_type"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("fill", 423, G__basic_ioslEcharcOchar_traitslEchargRsPgR_fill_9_1, 99, -1, 
		    G__defined_typename("basic_ios<char,char_traits<char> >::char_type"), 0, 1, 1, 1, 0, 
		    "c - 'basic_ios<char,char_traits<char> >::char_type' 0 - ch", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("imbue", 530, G__basic_ioslEcharcOchar_traitslEchargRsPgR_imbue_0_2, 117, G__get_linked_tagnum(&G__LN_locale), -1, 0, 1, 1, 1, 0, "u 'locale' - 11 - loc_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("narrow", 665, G__basic_ioslEcharcOchar_traitslEchargRsPgR_narrow_1_2, 99, -1, -1, 0, 2, 1, 1, 8,
                    "c - 'basic_ios<char,char_traits<char> >::char_type' 0 - c c - - 0 - dfault", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("widen", 535, G__basic_ioslEcharcOchar_traitslEchargRsPgR_widen_2_2, 99, -1, 
		    G__defined_typename("basic_ios<char,char_traits<char> >::char_type"), 0, 1, 1, 1, 8, 
		    "c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_ios<char,char_traits<char> >", 3260, 
		    G__basic_ioslEcharcOchar_traitslEchargRsPgR_basic_ioslEcharcOchar_traitslEchargRsPgR_7_2, 105, 
		    G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "U 'basic_streambuf<char,char_traits<char> >' - 0 - sb_arg", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_ios<char,char_traits<char> >", 3386, G__basic_ioslEcharcOchar_traitslEchargRsPgR_wAbasic_ioslEcharcOchar_traitslEchargRsPgR_8_2, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_istream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR));
#ifdef BOOL_NOT_YET_FUNCTIONNING
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_3_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "i 'bool' - 1 - n", (char *) NULL, (void *) NULL, 0);
#endif
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "s - - 1 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_5_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "r - - 1 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_6_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "i - - 1 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_7_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "h - - 1 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_8_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "l - - 1 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_9_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "k - - 1 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_0_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "f - - 1 - f", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_1_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "d - - 1 - f", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_2_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "Y - - 1 - p", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_3_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "U 'basic_streambuf<char,char_traits<char> >' 'basic_streambuf<char,char_traits<char> >' 0 - sb", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_4_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "C - - 0 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_5_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "B - - 0 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_6_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "c - - 1 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_7_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "b - - 1 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator>>", 1000, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_operatorgRgR_2_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "Q - 'ios_base& (*) (ios_base&)' 0 - pf", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("gcount", 656, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_gcount_4_1, 108, -1, 
		    G__defined_typename("streamsize"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get", 320, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_5_1, 105, -1, 
		    G__defined_typename("basic_istream<char,char_traits<char> >::int_type"), 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get", 320, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_6_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "c - 'basic_istream<char,char_traits<char> >::char_type' 1 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get", 320, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_7_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get", 320, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_8_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n "
                    "c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get", 320, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_9_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "u 'basic_streambuf<char,char_traits<char> >' 'basic_streambuf<char,char_traits<char> >' 1 - sb", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get", 320, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_get_0_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "u 'basic_streambuf<char,char_traits<char> >' 'basic_streambuf<char,char_traits<char> >' 1 - buf c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("getline", 744, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_1_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("getline", 744, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_getline_2_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n "
                    "c - 'basic_istream<char,char_traits<char> >::char_type' 0 - delim", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("ignore", 644, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_ignore_3_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "l - 'streamsize' 0 1 n i - 'basic_istream<char,char_traits<char> >::int_type' 0 char_traits<char>::eof() delim", (char *) NULL, 
		    (void *) NULL, 0);
   G__memfunc_setup("peek", 421, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_peek_4_2, 105, -1, 
		    G__defined_typename("basic_istream<char,char_traits<char> >::int_type"), 0, 0, 1, 1, 0, "", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("read", 412, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_read_5_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("readsome", 848, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_readsome_6_2, 108, -1, 
		    G__defined_typename("streamsize"), 0, 2, 1, 1, 0,
                    "C - 'basic_istream<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("putback", 746, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_putback_7_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "c - 'basic_istream<char,char_traits<char> >::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("unget", 547, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_unget_8_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sync", 445, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_sync_9_2, 105, -1, -1, 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("tellg", 536, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_tellg_0_3, 117, 
		    G__get_linked_tagnum(&G__LN_streampos), 
		    G__defined_typename("basic_istream<char,char_traits<char> >::pos_type"), 0, 0, 1, 1, 0, "", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("seekg", 527, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_seekg_1_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "u 'streampos' 'basic_istream<char,char_traits<char> >::pos_type' 0 - pos", (char *) NULL, 
		    (void *) NULL, 0);
   G__memfunc_setup("seekg", 527, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_seekg_2_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "l - 'basic_istream<char,char_traits<char> >::off_type' 0 - off i 'ios_base::seekdir' - 0 - dir", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_istream<char,char_traits<char> >", 3686, 
		    G__basic_istreamlEcharcOchar_traitslEchargRsPgR_basic_istreamlEcharcOchar_traitslEchargRsPgR_4_3, 105, 
		    G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "U 'basic_streambuf<char,char_traits<char> >' 'basic_streambuf<char,char_traits<char> >' 0 - sb", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_istream<char,char_traits<char> >", 3812, G__basic_istreamlEcharcOchar_traitslEchargRsPgR_wAbasic_istreamlEcharcOchar_traitslEchargRsPgR_5_3, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_ostream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_2_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0, 
		    "Q - 'ostream_type& (*)(ostream_type&)' 0 - pf", (char *) NULL, (void *) NULL, 0);
#ifdef BOOL_NOT_YET_FUNCTIONNING
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_5_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0, 
		    "i 'bool' - 0 - n", (char *) NULL, (void *) NULL, 0);
#endif
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_6_0, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "s - - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_7_0, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "r - - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_8_0, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "i - - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_9_0, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "h - - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_0_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "l - - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_1_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "k - - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_2_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "f - - 0 - f", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_3_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "d - - 0 - f", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_4_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "Y - - 10 - p", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_5_1, 117, 
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "U 'basic_streambuf<char,char_traits<char> >' 'basic_streambuf<char,char_traits<char> >' 0 - sb",  
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_2_2, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"), 1, 1, 1, 1, 0,  
		    "c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_3_2, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"), 1, 1, 1, 1, 0,  
		    "b - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_4_2, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"), 1, 1, 1, 1, 0,  
		    "C - - 10 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator<<", 996, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_operatorlElE_5_2, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >::ostream_type"), 1, 1, 1, 1, 0,  
		    "B - - 10 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("put", 345, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_put_6_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "c - 'basic_ostream<char,char_traits<char> >::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("write", 555, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_write_7_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 2, 1, 1, 0,
                    "C - 'basic_ostream<char,char_traits<char> >::char_type' 10 - s l - 'streamsize' 0 - n",  
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("flush", 546, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_flush_8_1, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 0, 1, 1, 0, "",  
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("tellp", 545, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_tellp_9_1, 117,  
		    G__get_linked_tagnum(&G__LN_streampos),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >::pos_type"), 0, 0, 1, 1, 0, "",  
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("seekp", 536, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_0_2, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 1, 1, 1, 0,  
		    "u 'streampos' 'basic_ostream<char,char_traits<char> >::pos_type' 0 - -", (char *) NULL,  
		    (void *) NULL, 0);
   G__memfunc_setup("seekp", 536, G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_seekp_1_2, 117,  
		    G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),  
		    G__defined_typename("basic_ostream<char,char_traits<char> >"), 1, 2, 1, 1, 0,
                    "l - 'basic_ostream<char,char_traits<char> >::off_type' 0 - - i 'ios_base::seekdir' - 0 - -",  
		    (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_ostream<char,char_traits<char> >", 3818,  
		    G__basic_ostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ostreamlEcharcOchar_traitslEchargRsPgR_3_2,  
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_iostreamlEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_iostream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR));
   // automatic destructor
   G__memfunc_setup("~basic_iostream<char,char_traits<char> >", 3923, G__basic_iostreamlEcharcOchar_traitslEchargRsPgR_wAbasic_iostreamlEcharcOchar_traitslEchargRsPgR_3_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_streambuflEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_streambuf<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("pubimbue", 857, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubimbue_1_0, 117, G__get_linked_tagnum(&G__LN_locale), -1, 0, 1, 1, 1, 0, "u 'locale' - 11 - loc_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("getloc", 638, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_getloc_2_0, 117, G__get_linked_tagnum(&G__LN_locale), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("pubsetbuf", 976, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsetbuf_3_0, 85, G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR), -1, 0, 2, 1, 1, 0,
                    "C - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("pubseekoff", 1066, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekoff_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_streampos), 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::pos_type"), 0, 3, 1, 1, 0,
                    "l - 'basic_streambuf<char,char_traits<char> >::off_type' 0 - off i 'ios_base::seekdir' - 0 - way "
                    "s - 'ios_base::openmode' 0 ios_base::in|ios_base::out which", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("pubseekpos", 1089, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubseekpos_5_0, 117, 
		    G__get_linked_tagnum(&G__LN_streampos), 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::pos_type"), 0, 2, 1, 1, 0,
                    "u 'streampos' 'basic_streambuf<char,char_traits<char> >::pos_type' 0 - sp s - 'ios_base::openmode' 0 ios_base::in|ios_base::out which", (
		    char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("pubsync", 772, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_pubsync_6_0, 105, -
		    1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("in_avail", 835, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_in_avail_7_0, 108, -1, 
		    G__defined_typename("streamsize"), 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("snextc", 661, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_snextc_8_0, 105, -1, 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"), 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sbumpc", 650, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sbumpc_9_0, 105, -1, 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"), 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sgetc", 534, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetc_0_1, 105, -1, 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"), 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sgetn", 545, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sgetn_1_1, 108, -1, 
		    G__defined_typename("streamsize"), 0, 2, 1, 1, 0,
                    "C - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - s l - 'streamsize' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sputbackc", 960, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputbackc_2_1, 105, -1, 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"), 0, 1, 1, 1, 0, 
		    "c - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sungetc", 761, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sungetc_3_1, 105, -1, 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"), 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sputc", 559, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputc_4_1, 105, -1, 
		    G__defined_typename("basic_streambuf<char,char_traits<char> >::int_type"), 0, 1, 1, 1, 0, 
		    "c - 'basic_streambuf<char,char_traits<char> >::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sputn", 570, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_sputn_5_1, 108, -1, G__defined_typename("streamsize"), 0, 2, 1, 1, 0,
                    "C - 'basic_streambuf<char,char_traits<char> >::char_type' 10 - s l - 'streamsize' 0 - n", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_streambuf<char,char_traits<char> >", 4024, G__basic_streambuflEcharcOchar_traitslEchargRsPgR_wAbasic_streambuflEcharcOchar_traitslEchargRsPgR_1_4, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_filebuflEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_filebuf<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_filebuf<char,char_traits<char> >", 3662, 
		    G__basic_filebuflEcharcOchar_traitslEchargRsPgR_basic_filebuflEcharcOchar_traitslEchargRsPgR_0_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), -1, 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("is_open", 749, G__basic_filebuflEcharcOchar_traitslEchargRsPgR_is_open_3_0, 105, 
		    G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("open", 434, G__basic_filebuflEcharcOchar_traitslEchargRsPgR_open_4_0, 85, 
		    G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_filebuf<char,char_traits<char> >::filebuf_type"), 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 - mode", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("close", 534, G__basic_filebuflEcharcOchar_traitslEchargRsPgR_close_5_0, 85, 
		    G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_filebuf<char,char_traits<char> >::filebuf_type"), 
		    0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_filebuf<char,char_traits<char> >", 3788, G__basic_filebuflEcharcOchar_traitslEchargRsPgR_wAbasic_filebuflEcharcOchar_traitslEchargRsPgR_8_1, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncallocatorlEchargR(void)
{
   /* allocator<char> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_allocatorlEchargR));
   G__memfunc_setup("allocate", 837, G__allocatorlEchargR_allocate_0_0, 67, -1, G__defined_typename("allocator<char>::pointer"), 0, 2, 1, 1, 0,
                    "h - 'allocator<char>::size_type' 0 - n Y - - 10 0 hint", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("deallocate", 1038, G__allocatorlEchargR_deallocate_1_0, 121, -1, -1, 0, 2, 1, 1, 0,
                    "C - 'allocator<char>::pointer' 0 - p h - 'allocator<char>::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("address", 742, G__allocatorlEchargR_address_2_0, 67, -1, G__defined_typename("allocator<char>::const_pointer"), 0, 1, 1, 1, 8, "c - 'allocator<char>::const_reference' 0 - x", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("max_size", 864, G__allocatorlEchargR_max_size_3_0, 104, -1, G__defined_typename("allocator<char>::size_type"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("construct", 997, G__allocatorlEchargR_construct_4_0, 121, -1, -1, 0, 2, 1, 1, 0,
                    "C - 'allocator<char>::pointer' 0 - p c - - 11 - val", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("destroy", 778, G__allocatorlEchargR_destroy_5_0, 121, -1, -1, 0, 1, 1, 1, 0, "C - 'allocator<char>::pointer' 0 - p", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("allocator<char>", 1497, G__allocatorlEchargR_allocatorlEchargR_6_0, 105, G__get_linked_tagnum(&G__LN_allocatorlEchargR), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("allocator<char>", 1497, G__allocatorlEchargR_allocatorlEchargR_8_0, (int) ('i'), G__get_linked_tagnum(&G__LN_allocatorlEchargR), -1, 0, 1, 1, 1, 0, "u 'allocator<char>' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~allocator<char>", 1623, G__allocatorlEchargR_wAallocatorlEchargR_9_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   /* basic_stringbuf<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("str", 345, G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0, 
		    121, -1, -1, 0, 1, 1, 1, 0, 
		    "u 'basic_string<char,char_traits<char>,allocator<char> >' 'basic_stringbuf<char,char_traits<char>,allocator<char> >::string_type' 11 - str_arg", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_stringbuf<char,char_traits<char>,allocator<char> >", 5576, 
		    G__basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_5_1, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ifstreamlEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_ifstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >", 3788, 
		    G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_0_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >", 3788, 
		    G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_1_0, 105,
		    G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 ios_base::in mode", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rdbuf", 531, G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_3_0, 85, 
		    G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_ifstream<char,char_traits<char> >::filebuf_type"), 0, 0, 1, 1, 8, "", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("is_open", 749, G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_is_open_4_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("open", 434, G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_open_5_0, 121, -1, -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 ios_base::in mode", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("close", 534, G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_close_6_0, 121, -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_ifstream<char,char_traits<char> >", 3788, 
		    G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_basic_ifstreamlEcharcOchar_traitslEchargRsPgR_7_0, 
		    105, G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 - mode", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_ifstream<char,char_traits<char> >", 3914, 
		    G__basic_ifstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ifstreamlEcharcOchar_traitslEchargRsPgR_8_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ofstreamlEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_ofstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >", 3794, 
		    G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_0_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 0, 1, 1, 0, 
"", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >", 3794, 
		    G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_1_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 ios_base::out|ios_base::trunc mode", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rdbuf", 531, G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_3_0, 85, 
		    G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_ofstream<char,char_traits<char> >::filebuf_type"), 0, 0, 1, 1, 8, "", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("is_open", 749, G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_is_open_4_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("open", 434, G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_open_5_0, 121, -1, -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 ios_base::out|ios_base::trunc mode", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("close", 534, G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_close_6_0, 121, -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_ofstream<char,char_traits<char> >", 3794, 
		    G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_basic_ofstreamlEcharcOchar_traitslEchargRsPgR_7_0, 
		    105, G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 - mode", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_ofstream<char,char_traits<char> >", 3920, G__basic_ofstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_ofstreamlEcharcOchar_traitslEchargRsPgR_8_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_fstreamlEcharcOchar_traitslEchargRsPgR(void)
{
   /* basic_fstream<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("rdbuf", 531, G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_rdbuf_3_0, 85, 
		    G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR), 
		    G__defined_typename("basic_fstream<char,char_traits<char> >::filebuf_type"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("is_open", 749, G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_is_open_4_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("open", 434, G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_open_5_0, 121, -1, -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 ios_base::in|ios_base::out mode", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("close", 534, G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_close_6_0, 121, -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_fstream<char,char_traits<char> >", 3683, 
		    G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_basic_fstreamlEcharcOchar_traitslEchargRsPgR_7_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR), -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s s - 'ios_base::openmode' 0 - mode", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_fstream<char,char_traits<char> >", 3809, G__basic_fstreamlEcharcOchar_traitslEchargRsPgR_wAbasic_fstreamlEcharcOchar_traitslEchargRsPgR_8_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   /* basic_istringstream<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("rdbuf", 531, G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0, 85,
		     G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    G__defined_typename("basic_istringstream<char,char_traits<char>,allocator<char> >::sb_type"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("str", 345, G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    G__defined_typename("basic_istringstream<char,char_traits<char>,allocator<char> >::string_type"),
		     0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("str", 345, G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0, 121, -1, -1, 0, 1, 1, 1, 0, 
		    "u 'basic_string<char,char_traits<char>,allocator<char> >' 'basic_istringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic default constructor
   G__memfunc_setup("basic_istringstream<char,char_traits<char>,allocator<char> >", 5890, 
		    G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_istringstream<char,char_traits<char>,allocator<char> >", 6016, 
		    G__basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_7_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   /* basic_ostringstream<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("rdbuf", 531, G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0, 85,
		     G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),
		     G__defined_typename("basic_ostringstream<char,char_traits<char>,allocator<char> >::sb_type"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("str", 345, G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    G__defined_typename("basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type"),
		     0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("str", 345, G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0, 
		    121, -1, -1, 0, 1, 1, 1, 0,
		     "u 'basic_string<char,char_traits<char>,allocator<char> >' 'basic_ostringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic default constructor
   G__memfunc_setup("basic_ostringstream<char,char_traits<char>,allocator<char> >", 5896, 
		    G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0, 
		    (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_ostringstream<char,char_traits<char>,allocator<char> >", 6022, 
		    G__basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_7_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   /* basic_stringstream<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("rdbuf", 531, G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rdbuf_3_0, 
		    85, G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    G__defined_typename("basic_stringstream<char,char_traits<char>,allocator<char> >::sb_type"), 0, 0, 1, 1,
		     8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("str", 345, G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    G__defined_typename("basic_stringstream<char,char_traits<char>,allocator<char> >::string_type"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("str", 345, G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_str_5_0, 121, -1, -1,
		     0, 1, 1, 1, 0, 
		    "u 'basic_string<char,char_traits<char>,allocator<char> >' 'basic_stringstream<char,char_traits<char>,allocator<char> >::string_type' 11 - str", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic default constructor
   G__memfunc_setup("basic_stringstream<char,char_traits<char>,allocator<char> >", 5785, 
		    G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_6_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_stringstream<char,char_traits<char>,allocator<char> >", 5911, 
		    G__basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_7_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", 
		    (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR(void)
{
   /* basic_string<char,char_traits<char>,allocator<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR));
   G__memfunc_setup("basic_string<char,char_traits<char>,allocator<char> >", 5133, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_1_0, 
		    105, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 1, 1, 1, 0,
                    "u 'allocator<char>' - 11 allocator<char>() alloc", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_string<char,char_traits<char>,allocator<char> >", 5133, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_2_0, 
		    105, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 4, 1, 1, 0,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos n u 'allocator<char>' - 11 allocator<char>() alloc", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_string<char,char_traits<char>,allocator<char> >", 5133, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_3_0, 
		    105, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 3, 1, 1, 0,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n "
                    "u 'allocator<char>' - 11 allocator<char>() alloc", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_string<char,char_traits<char>,allocator<char> >", 5133, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_4_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 2, 1, 1, 0,
                    "C - - 10 - s u 'allocator<char>' - 11 allocator<char>() alloc", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("basic_string<char,char_traits<char>,allocator<char> >", 5133, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_5_0, 105, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 0, 3, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n c - - 0 - c "
                    "u 'allocator<char>' - 11 allocator<char>() alloc", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator=", 937, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoreQ_7_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 1, 1,
		     1, 0, "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str", (char *) NULL, 
		    (void *) NULL, 0);
   G__memfunc_setup("operator=", 937, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoreQ_8_0, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 1, 1,
		     1, 0, "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator=", 937, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoreQ_9_0, 117,
		     G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 1, 1, 1, 0, "c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("begin", 517, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_begin_0_1, 67, -1, 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::const_iterator"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("end", 311, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_end_1_1, 67, -1,
		     G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::const_iterator"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rbegin", 631, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rbegin_2_1, 117, 
		    G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::const_reverse_iterator"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rend", 425, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rend_3_1, 117, 
		    G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::const_reverse_iterator"), 
		    0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("size", 443, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_size_4_1, 104, -1, 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 0, 1, 1, 8, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("length", 642, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_length_5_1, 104, -1, 
		     G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 0, 1, 1, 8,
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("max_size", 864, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_max_size_6_1, 104, -1,
		     G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 0, 1, 1, 8, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("resize", 658, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_resize_7_1, 121, -1, -1,
		     0, 2, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n c - - 0 - c", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("resize", 658, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_resize_8_1, 121, -1, -1,
		     0, 1, 1, 1, 0, "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("capacity", 846, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_capacity_9_1, 104, -1,
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 0, 1, 1, 8, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("reserve", 764, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_reserve_0_2, 
		    121, -1, -1, 0, 1, 1, 1, 0, 
		    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 res_arg", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("empty", 559, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_empty_1_2, 105, 
		    G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("clear", 519, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_clear_2_2, 
		    121, -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator[]", 1060, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatoroBcB_3_2, 
		    99, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::reference"), 
		    1, 1, 1, 1, 8, "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("at", 213, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_at_4_2, 99, -1, 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::reference"),
		     1, 1, 1, 1, 8, "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+=", 980, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatorpLeQ_5_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 1, 1, 1, 0, "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+=", 980, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatorpLeQ_6_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 1, 1, 1, 0, "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+=", 980, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_operatorpLeQ_7_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 1, 1, 1, 0, "c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("append", 632, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_8_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 1, 1, 1, 0, "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("append", 632, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_9_2, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 3, 1, 1, 0,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, 
		    (void *) NULL, 0);
   G__memfunc_setup("append", 632, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_0_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 2, 1,
		     1, 0,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("append", 632, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_1_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 1, 1,
		     1, 0, "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("append", 632, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_append_2_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 2, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - rep c - - 0 - c", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("assign", 645, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_3_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), 
		    -1, 1, 1, 1, 1, 0, "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str",
		     (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("assign", 645, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_4_3, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("assign", 645, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_5_3, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("assign", 645, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_6_3, 117, 
		    G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 1, 1,
		     1, 0, "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("assign", 645, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_assign_7_3, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_8_3, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_9_3, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 4, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos1 u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos2 h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_0_4, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos C - - 10 - s "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_1_4, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n "
                    "c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_2_4, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_3_4, 67, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::iterator"), 0, 2, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - p c - - 0 char() c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("insert", 661, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_insert_4_4, 121, -1, -1, 0, 3, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - p h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n "
                    "c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("erase", 528, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_erase_5_4, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 2, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("erase", 528, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_erase_6_4, 67, -1, 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::iterator"), 0, 1, 1, 1, 0, 
		    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - p", (char *) NULL, 
		    (void *) NULL, 0);
   G__memfunc_setup("erase", 528, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_erase_7_4, 67, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::iterator"), 0, 2, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - first C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - last", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_8_4, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - p h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n "
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_9_4, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 5, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - p1 h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n1 "
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - p2 "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_0_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 4, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n1 "
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_1_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - p h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n "
                    "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_2_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 4, 1, 1, 0,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n1 "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n2 c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_3_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i1 C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i2 "
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_4_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 4, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i1 C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i2 "
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_5_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 3, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i1 C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i2 "
                    "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("replace", 732, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_replace_6_5, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 1, 4, 1, 1, 0,
                    "C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i1 C - 'basic_string<char,char_traits<char>,allocator<char> >::iterator' 0 - i2 "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n c - - 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("copy", 443, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_copy_7_5, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 0 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("swap", 443, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_swap_8_5, 121, -1, -1, 0, 1, 1, 1, 0, "u 'basic_string<char,char_traits<char>,allocator<char> >' - 1 - rhs", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("c_str", 539, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_c_str_9_5, 67, -1, -1, 0, 0, 1, 1, 9, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("data", 410, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_data_0_6, 67, -1, -1, 0, 0, 1, 1, 9, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("get_allocator", 1376, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_get_allocator_1_6, 117, G__get_linked_tagnum(&G__LN_allocatorlEchargR), 
		    G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::allocator_type"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find", 417, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_2_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find", 417, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_3_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find", 417, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_4_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find", 417, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_5_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "c - - 0 - c h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rfind", 531, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_6_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rfind", 531, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_7_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rfind", 531, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_8_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("rfind", 531, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_rfind_9_6, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "c - - 0 - c h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_of", 1372, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_0_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_of", 1372, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_1_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_of", 1372, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_2_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_of", 1372, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_of_3_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "c - - 0 - c h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_of", 1256, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_4_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos p", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_of", 1256, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_5_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_of", 1256, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_6_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_of", 1256, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_of_7_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "c - - 0 - c h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_not_of", 1804, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_8_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 p", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_not_of", 1804, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_9_7, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_not_of", 1804, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_0_8, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_first_not_of", 1804, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_first_not_of_1_8, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "c - - 0 - c h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_not_of", 1688, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_2_8, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos p", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_not_of", 1688, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_3_8, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 3, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_not_of", 1688, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_4_8, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "C - - 10 - s h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("find_last_not_of", 1688, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_find_last_not_of_5_8, 104, -1, G__defined_typename("basic_string<char,char_traits<char>,allocator<char> >::size_type"), 0, 2, 1, 1, 8,
                    "c - - 0 - c h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos pos", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("substr", 675, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_substr_6_8, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), -1, 0, 2, 1, 1, 8,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 0 pos h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("compare", 743, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_7_8, 105, -1, -1, 0, 1, 1, 1, 8, "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("compare", 743, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_8_8, 105, -1, -1, 0, 3, 1, 1, 8,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - p h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n1 "
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("compare", 743, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_9_8, 105, -1, -1, 0, 5, 1, 1, 8,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos1 h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n1 "
                    "u 'basic_string<char,char_traits<char>,allocator<char> >' - 11 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos2 "
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("compare", 743, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_0_9, 105, -1, -1, 0, 1, 1, 1, 8, "C - - 10 - s", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("compare", 743, G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_compare_1_9, 105, -1, -1, 0, 4, 1, 1, 8,
                    "h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - pos h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 - n1 "
                    "C - - 10 - str h - 'basic_string<char,char_traits<char>,allocator<char> >::size_type' 0 npos n2", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~basic_string<char,char_traits<char>,allocator<char> >", 5259, 
		    G__basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_wAbasic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR_2_9, (
		    int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}


static void G__setup_memfuncinput_iterator_tag(void)
{
   /* input_iterator_tag */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_input_iterator_tag));
   // automatic default constructor
   G__memfunc_setup("input_iterator_tag", 1940, G__input_iterator_tag_input_iterator_tag_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_input_iterator_tag), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("input_iterator_tag", 1940, G__input_iterator_tag_input_iterator_tag_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_input_iterator_tag), -1, 0, 1, 1, 1, 0, "u 'input_iterator_tag' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~input_iterator_tag", 2066, G__input_iterator_tag_wAinput_iterator_tag_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncoutput_iterator_tag(void)
{
   /* output_iterator_tag */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_output_iterator_tag));
   // automatic default constructor
   G__memfunc_setup("output_iterator_tag", 2069, G__output_iterator_tag_output_iterator_tag_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_output_iterator_tag), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("output_iterator_tag", 2069, G__output_iterator_tag_output_iterator_tag_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_output_iterator_tag), -1, 0, 1, 1, 1, 0, "u 'output_iterator_tag' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~output_iterator_tag", 2195, G__output_iterator_tag_wAoutput_iterator_tag_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncforward_iterator_tag(void)
{
   /* forward_iterator_tag */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_forward_iterator_tag));
   // automatic default constructor
   G__memfunc_setup("forward_iterator_tag", 2137, G__forward_iterator_tag_forward_iterator_tag_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_forward_iterator_tag), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("forward_iterator_tag", 2137, G__forward_iterator_tag_forward_iterator_tag_1_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_forward_iterator_tag), -1, 0, 1, 1, 1, 0, 
		    "u 'forward_iterator_tag' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~forward_iterator_tag", 2263, G__forward_iterator_tag_wAforward_iterator_tag_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbidirectional_iterator_tag(void)
{
   /* bidirectional_iterator_tag */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag));
   // automatic default constructor
   G__memfunc_setup("bidirectional_iterator_tag", 2749, 
		    G__bidirectional_iterator_tag_bidirectional_iterator_tag_0_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), -1, 0, 0, 1, 1, 0, "", (char *) NULL, 
		    (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("bidirectional_iterator_tag", 2749, 
		    G__bidirectional_iterator_tag_bidirectional_iterator_tag_1_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), -1, 0, 1, 1, 1, 0, 
		    "u 'bidirectional_iterator_tag' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~bidirectional_iterator_tag", 2875, G__bidirectional_iterator_tag_wAbidirectional_iterator_tag_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncrandom_access_iterator_tag(void)
{
   /* random_access_iterator_tag */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag));
   // automatic default constructor
   G__memfunc_setup("random_access_iterator_tag", 2742, 
		    G__random_access_iterator_tag_random_access_iterator_tag_0_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_random_access_iterator_tag), -1, 0, 0, 1, 1, 0, "", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("random_access_iterator_tag", 2742, 
		    G__random_access_iterator_tag_random_access_iterator_tag_1_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_random_access_iterator_tag), -1, 0, 1, 1, 1, 0, 
		    "u 'random_access_iterator_tag' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~random_access_iterator_tag", 2868, G__random_access_iterator_tag_wArandom_access_iterator_tag_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunciteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR(void)
{
   /* iterator<output_iterator_tag,void,void,void,void> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR));
   // automatic default constructor
   G__memfunc_setup("iterator<output_iterator_tag,void,void,void,void>", 4977, 
		    G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_0_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR), 
		    -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("iterator<output_iterator_tag,void,void,void,void>", 4977,
		     G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_1_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR), 
		    -1, 0, 1, 1, 1, 0, "u 'iterator<output_iterator_tag,void,void,void,void>' - 1 - -", (char *) NULL, 
		    (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~iterator<output_iterator_tag,void,void,void,void>", 5103, 
		    G__iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_wAiteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR_2_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncrel_ops(void)
{
   /* rel_ops */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_rel_ops));
   G__tag_memfunc_reset();
}

static void G__setup_memfuncallocatorlEvoidgR(void)
{
   /* allocator<void> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_allocatorlEvoidgR));
   // automatic default constructor
   G__memfunc_setup("allocator<void>", 1517, G__allocatorlEvoidgR_allocatorlEvoidgR_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_allocatorlEvoidgR), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("allocator<void>", 1517, G__allocatorlEvoidgR_allocatorlEvoidgR_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_allocatorlEvoidgR), -1, 0, 1, 1, 1, 0, "u 'allocator<void>' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~allocator<void>", 1643, G__allocatorlEvoidgR_wAallocatorlEvoidgR_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncstreampos(void)
{
   /* streampos */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_streampos));
   // NOTE: I should change the next line.  I removed by hand the ability to use the 2 argument
   // constructor of streampos (it does not exist in kai v3.4)
   G__memfunc_setup("streampos", 990, G__streampos_streampos_0_0, 105, G__get_linked_tagnum(&G__LN_streampos), -1, 0, 2, 1, 1, 0,
                    "l - 'streamoff' 0 0 off_arg i - 'fstate_t' 0 0 fst_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("streampos", 990, G__streampos_streampos_1_0, 105, G__get_linked_tagnum(&G__LN_streampos), -1, 0, 1, 1, 1, 0, "u 'streampos' - 11 - rhs", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator streamoff", 1875, G__streampos_operatorsPstreamoff_2_0, 108, -1, G__defined_typename("streamoff"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
#ifndef MBSTATE_IS_STRUCT
    G__memfunc_setup("state", 545, G__streampos_state_4_0, 105, -1, G__defined_typename("fstate_t"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
#endif
   G__memfunc_setup("operator=", 937, G__streampos_operatoreQ_5_0, 117, G__get_linked_tagnum(&G__LN_streampos), -1, 1, 1, 1, 1, 0, "u 'streampos' - 11 - rhs", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator-", 921, G__streampos_operatormI_6_0, 108, -1, G__defined_typename("streamoff"), 0, 1, 1, 1, 8, "u 'streampos' - 11 - rhs", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+=", 980, G__streampos_operatorpLeQ_7_0, 117, G__get_linked_tagnum(&G__LN_streampos), -1, 1, 1, 1, 1, 0, "l - 'streamoff' 0 - off_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator-=", 982, G__streampos_operatormIeQ_8_0, 117, G__get_linked_tagnum(&G__LN_streampos), -1, 1, 1, 1, 1, 0, "l - 'streamoff' 0 - off_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+", 919, G__streampos_operatorpL_9_0, 117, G__get_linked_tagnum(&G__LN_streampos), -1, 0, 1, 1, 1, 8, "l - 'streamoff' 0 - off_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator-", 921, G__streampos_operatormI_0_1, 117, G__get_linked_tagnum(&G__LN_streampos), -1, 0, 1, 1, 1, 8, "l - 'streamoff' 0 - off_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator==", 998, G__streampos_operatoreQeQ_1_1, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 1, 1, 1, 8, "u 'streampos' - 11 - rhs", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator!=", 970, G__streampos_operatornOeQ_2_1, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 1, 1, 1, 8, "u 'streampos' - 11 - rhs", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~streampos", 1116, G__streampos_wAstreampos_3_1, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncreverse_iteratorlEcharmUgR(void)
{
   /* reverse_iterator<char*> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR));
   G__memfunc_setup("reverse_iterator<char*>", 2311, G__reverse_iteratorlEcharmUgR_reverse_iteratorlEcharmUgR_0_0, 105, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
#if 0
   G__memfunc_setup("reverse_iterator<char*>", 3177, G__reverse_iteratorlEcharmUgR_reverse_iteratorlEcharmUgR_1_0, 105, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 1, 1, 1, 0, "C - - 0 - x", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("base", 411, G__reverse_iteratorlEcharmUgR_base_2_0, 67, -1, -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator*", 918, G__reverse_iteratorlEcharmUgR_operatormU_3_0, 67, -1, G__defined_typename("reverse_iterator<char*>::Reference"), 1, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
#endif
   G__memfunc_setup("operator->", 983, G__reverse_iteratorlEcharmUgR_operatormIgR_4_0, 67, -1, G__defined_typename("reverse_iterator<char*>::Pointer"), 2, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator++", 962, G__reverse_iteratorlEcharmUgR_operatorpLpL_5_0, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 1, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator++", 962, G__reverse_iteratorlEcharmUgR_operatorpLpL_6_0, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 1, 1, 1, 0, "i - - 0 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator--", 966, G__reverse_iteratorlEcharmUgR_operatormImI_7_0, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 1, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator--", 966, G__reverse_iteratorlEcharmUgR_operatormImI_8_0, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 1, 1, 1, 0, "i - - 0 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+", 919, G__reverse_iteratorlEcharmUgR_operatorpL_9_0, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 1, 1, 1, 8, "u 'reverse_iterator<char*>' - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator+=", 980, G__reverse_iteratorlEcharmUgR_operatorpLeQ_0_1, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 1, 1, 1, 1, 0, "u 'reverse_iterator<char*>' - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator-", 921, G__reverse_iteratorlEcharmUgR_operatormI_1_1, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 1, 1, 1, 8, "u 'reverse_iterator<char*>' - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator-=", 982, G__reverse_iteratorlEcharmUgR_operatormIeQ_2_1, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 1, 1, 1, 1, 0, "u 'reverse_iterator<char*>' - 0 - n", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator[]", 1060, G__reverse_iteratorlEcharmUgR_operatoroBcB_3_1, 67, -1, G__defined_typename("reverse_iterator<char*>::Reference"), 1, 1, 1, 1, 8, "u 'reverse_iterator<char*>' - 0 - n", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("reverse_iterator<char*>", 2311, 
		    G__reverse_iteratorlEcharmUgR_reverse_iteratorlEcharmUgR_4_1, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 1, 1, 1, 0, 
		    "u 'reverse_iterator<char*>' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~reverse_iterator<char*>", 2437, G__reverse_iteratorlEcharmUgR_wAreverse_iteratorlEcharmUgR_5_1, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunciterator_traitslEcharmUgR(void)
{
   /* iterator_traits<char*> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR));
   // automatic default constructor
   G__memfunc_setup("iterator_traits<char*>", 2210, 
		    G__iterator_traitslEcharmUgR_iterator_traitslEcharmUgR_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("iterator_traits<char*>", 2210, 
		    G__iterator_traitslEcharmUgR_iterator_traitslEcharmUgR_1_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR), -1, 0, 1, 1, 1, 0, "u 'iterator_traits<char*>' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~iterator_traits<char*>", 2336, G__iterator_traitslEcharmUgR_wAiterator_traitslEcharmUgR_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunciteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR(void)
{
   /* iterator<long,char*,char**,char*&,random_access_iterator_tag> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR));
   // automatic default constructor
   G__memfunc_setup("iterator<long,char*,char**,char*&,random_access_iterator_tag>", 5794, 
		    G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_0_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR), 
		    -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("iterator<long,char*,char**,char*&,random_access_iterator_tag>", 5794, 
		    G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_1_0, 
		    (int) ('i'), G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR), 
		    -1, 0, 1, 1, 1, 0, "u 'iterator<long,char*,char**,char*&,random_access_iterator_tag>' - 1 - -", 
		    (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~iterator<long,char*,char**,char*&,random_access_iterator_tag>", 5920, 
		    G__iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_wAiteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR_2_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunclconv(void)
{
   /* lconv */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_lconv));
   // automatic default constructor
   G__memfunc_setup("lconv", 546, G__lconv_lconv_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_lconv), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("lconv", 546, G__lconv_lconv_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_lconv), -1, 0, 1, 1, 1, 0, "u 'lconv' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~lconv", 672, G__lconv_wAlconv_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunclocale(void)
{
   /* locale */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_locale));
   G__memfunc_setup("locale", 624, G__locale_locale_2_0, 105, G__get_linked_tagnum(&G__LN_locale), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("locale", 624, G__locale_locale_3_0, 105, G__get_linked_tagnum(&G__LN_locale), -1, 0, 1, 1, 1, 0, "u 'locale' - 11 - other", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("locale", 624, G__locale_locale_4_0, 105, G__get_linked_tagnum(&G__LN_locale), -1, 0, 1, 1, 1, 0, "C - - 10 - std_name", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("locale", 624, G__locale_locale_5_0, 105, G__get_linked_tagnum(&G__LN_locale), -1, 0, 3, 1, 1, 0,
                    "u 'locale' - 11 - other C - - 10 - std_name "
      "i - 'locale::category' 0 - cat", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("locale", 624, G__locale_locale_6_0, 105, G__get_linked_tagnum(&G__LN_locale), -1, 0, 3, 1, 1, 0,
                    "u 'locale' - 11 - one u 'locale' - 11 - other "
      "i - 'locale::category' 0 - cat", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator=", 937, G__locale_operatoreQ_8_0, 117, G__get_linked_tagnum(&G__LN_locale), -1, 1, 1, 1, 1, 1, "u 'locale' - 11 - other", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("name", 417, G__locale_name_9_0, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__defined_typename("string"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator==", 998, G__locale_operatoreQeQ_0_1, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 1, 1, 1, 8, "u 'locale' - 11 - other", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator!=", 970, G__locale_operatornOeQ_1_1, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 1, 1, 1, 8, "u 'locale' - 11 - other", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("global", 625, G__locale_global_2_1, 117, G__get_linked_tagnum(&G__LN_locale), -1, 0, 1, 1, 1, 0, "u 'locale' - 11 - loc", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("classic", 738, G__locale_classic_3_1, 117, G__get_linked_tagnum(&G__LN_locale), -1, 1, 0, 1, 1, 1, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~locale", 750, G__locale_wAlocale_5_1, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunclocalecLcLfacet(void)
{
   /* locale::facet */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_localecLcLfacet));
   G__tag_memfunc_reset();
}

static void G__setup_memfunclocalecLcLid(void)
{
   /* locale::id */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_localecLcLid));
   G__memfunc_setup("id", 205, G__localecLcLid_id_2_0, 105, G__get_linked_tagnum(&G__LN_localecLcLid), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~id", 331, G__localecLcLid_wAid_7_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncctype_base(void)
{
   /* ctype_base */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_ctype_base));
   // automatic default constructor
   G__memfunc_setup("ctype_base", 1055, G__ctype_base_ctype_base_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_ctype_base), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("ctype_base", 1055, G__ctype_base_ctype_base_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_ctype_base), -1, 0, 1, 1, 1, 0, "u 'ctype_base' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~ctype_base", 1181, G__ctype_base_wActype_base_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncctypelEchargR(void)
{
   /* ctype<char> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_ctypelEchargR));
   G__memfunc_setup("is", 220, G__ctypelEchargR_is_8_0, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 2, 1, 1, 8,
                    "i 'ctype_base::mask' - 0 - mask_ c - 'ctype<char>::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("is", 220, G__ctypelEchargR_is_9_0, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 3, 1, 1, 9,
                    "C - 'ctype<char>::char_type' 10 - lo C - 'ctype<char>::char_type' 10 - hi "
      "I 'ctype_base::mask' - 0 - vec", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("scan_is", 736, G__ctypelEchargR_scan_is_0_1, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 3, 1, 1, 9,
                    "i 'ctype_base::mask' - 0 - mask_ C - 'ctype<char>::char_type' 10 - low "
                    "C - 'ctype<char>::char_type' 10 - high", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("scan_not", 853, G__ctypelEchargR_scan_not_1_1, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 3, 1, 1, 9,
                    "i 'ctype_base::mask' - 0 - mask_ C - 'ctype<char>::char_type' 10 - low "
                    "C - 'ctype<char>::char_type' 10 - high", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("toupper", 783, G__ctypelEchargR_toupper_2_1, 99, -1, G__defined_typename("ctype<char>::char_type"), 0, 1, 1, 1, 8, "c - 'ctype<char>::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("toupper", 783, G__ctypelEchargR_toupper_3_1, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 2, 1, 1, 9,
                    "C - 'ctype<char>::char_type' 0 - low C - 'ctype<char>::char_type' 10 - high", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("tolower", 780, G__ctypelEchargR_tolower_4_1, 99, -1, G__defined_typename("ctype<char>::char_type"), 0, 1, 1, 1, 8, "c - 'ctype<char>::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("tolower", 780, G__ctypelEchargR_tolower_5_1, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 2, 1, 1, 9,
                    "C - 'ctype<char>::char_type' 0 - low C - 'ctype<char>::char_type' 10 - high", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("widen", 535, G__ctypelEchargR_widen_6_1, 99, -1, G__defined_typename("ctype<char>::char_type"), 0, 1, 1, 1, 8, "c - 'ctype<char>::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("widen", 535, G__ctypelEchargR_widen_7_1, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 3, 1, 1, 9,
                    "C - 'ctype<char>::char_type' 10 - lo C - 'ctype<char>::char_type' 10 - hi "
                    "C - 'ctype<char>::char_type' 0 - to", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("narrow", 665, G__ctypelEchargR_narrow_8_1, 99, -1, G__defined_typename("ctype<char>::char_type"), 0, 2, 1, 1, 8,
                    "c - 'ctype<char>::char_type' 0 - c c - 'ctype<char>::char_type' 0 - dfault", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("narrow", 665, G__ctypelEchargR_narrow_9_1, 67, -1, G__defined_typename("ctype<char>::char_type"), 0, 4, 1, 1, 9,
                    "C - 'ctype<char>::char_type' 10 - lo C - 'ctype<char>::char_type' 10 - hi "
                    "c - 'ctype<char>::char_type' 0 - dfault C - 'ctype<char>::char_type' 0 - to", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncios_base(void)
{
   /* ios_base */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_ios_base));
   G__memfunc_setup("flags", 525, G__ios_base_flags_0_0, 108, -1, G__defined_typename("ios_base::fmtflags"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("flags", 525, G__ios_base_flags_1_0, 108, -1, G__defined_typename("ios_base::fmtflags"), 0, 1, 1, 1, 0, "l - 'ios_base::fmtflags' 0 - fmtfl_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("setf", 434, G__ios_base_setf_2_0, 108, -1, G__defined_typename("ios_base::fmtflags"), 0, 1, 1, 1, 0, "l - 'ios_base::fmtflags' 0 - fmtfl_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("setf", 434, G__ios_base_setf_3_0, 108, -1, G__defined_typename("ios_base::fmtflags"), 0, 2, 1, 1, 0,
                    "l - 'ios_base::fmtflags' 0 - fmtfl_arg l - 'ios_base::fmtflags' 0 - mask", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("unsetf", 661, G__ios_base_unsetf_4_0, 121, -1, -1, 0, 1, 1, 1, 0, "l - 'ios_base::fmtflags' 0 - mask", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("precision", 972, G__ios_base_precision_5_0, 108, -1, G__defined_typename("streamsize"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("precision", 972, G__ios_base_precision_6_0, 108, -1, G__defined_typename("streamsize"), 0, 1, 1, 1, 0, "l - 'streamsize' 0 - prec_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("width", 544, G__ios_base_width_7_0, 108, -1, G__defined_typename("streamsize"), 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("width", 544, G__ios_base_width_8_0, 108, -1, G__defined_typename("streamsize"), 0, 1, 1, 1, 0, "l - 'streamsize' 0 - wide_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("imbue", 530, G__ios_base_imbue_9_0, 117, G__get_linked_tagnum(&G__LN_locale), -1, 0, 1, 1, 1, 0, "u 'locale' - 11 - loc_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("getloc", 638, G__ios_base_getloc_0_1, 117, G__get_linked_tagnum(&G__LN_locale), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("xalloc", 643, G__ios_base_xalloc_1_1, 105, -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("iword", 549, G__ios_base_iword_2_1, 108, -1, -1, 1, 1, 1, 1, 0, "i - - 0 - index_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("pword", 556, G__ios_base_pword_3_1, 89, -1, -1, 1, 1, 1, 1, 0, "i - - 0 - index_arg", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("register_callback", 1777, G__ios_base_register_callback_5_1, 121, -1, -1, 0, 2, 1, 1, 0,
                    "Y - 'ios_base::event_callback' 0 - fn i - - 0 - index", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sync_with_stdio", 1626, G__ios_base_sync_with_stdio_6_1, 105, G__get_linked_tagnum(&G__LN_bool), -1, 0, 1, 1, 1, 0, "i 'bool' - 0 true sync", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~ios_base", 963, G__ios_base_wAios_base_1_2, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 1);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncios_basecLcLInit(void)
{
   /* ios_base::Init */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLInit));
   G__memfunc_setup("Init", 404, G__ios_basecLcLInit_Init_0_0, 105, G__get_linked_tagnum(&G__LN_ios_basecLcLInit), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("Init", 404, G__ios_basecLcLInit_Init_2_0, (int) ('i'), G__get_linked_tagnum(&G__LN_ios_basecLcLInit), -1, 0, 1, 1, 1, 0, "u 'ios_base::Init' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~Init", 530, G__ios_basecLcLInit_wAInit_3_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}


static void G__setup_memfunccodecvt_base(void)
{
   /* codecvt_base */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_codecvt_base));
   // automatic default constructor
   G__memfunc_setup("codecvt_base", 1250, G__codecvt_base_codecvt_base_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_codecvt_base), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("codecvt_base", 1250, G__codecvt_base_codecvt_base_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_codecvt_base), -1, 0, 1, 1, 1, 0, "u 'codecvt_base' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~codecvt_base", 1376, G__codecvt_base_wAcodecvt_base_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunccodecvtlEcharcOcharcOintgR(void)
{
   /* codecvt<char,char,int> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR));
#ifdef HAS_PROPER_DO_LENGTH
   G__memfunc_setup("length", 642, G__codecvtlEcharcOcharcOintgR_length_0_1, 105, -1, -1, 0, 4, 1, 1, 8,
                    "i - 'codecvt<char,char,int>::state_type' 11 - state C - 'codecvt<char,char,int>::extern_type' 10 - from "
                    "C - 'codecvt<char,char,int>::extern_type' 10 - from_end h - 'size_t' 0 - max", (char *) NULL, (void *) NULL, 0);
#endif
   G__memfunc_setup("max_length", 1063, G__codecvtlEcharcOcharcOintgR_max_length_1_1, 105, -1, -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   // automatic default constructor
   G__tag_memfunc_reset();
}

static void G__setup_memfunccollatelEchargR(void)
{
   /* collate<char> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_collatelEchargR));
   G__memfunc_setup("compare", 743, G__collatelEchargR_compare_3_0, 105, -1, -1, 0, 4, 1, 1, 8,
                    "C - - 10 - low1 C - - 10 - high1 "
    "C - - 10 - low2 C - - 10 - high2", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("transform", 988, G__collatelEchargR_transform_4_0, 117, G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR), G__defined_typename("collate<char>::string_type"), 0, 2, 1, 1, 8,
      "C - - 10 - low C - - 10 - high", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("hash", 420, G__collatelEchargR_hash_5_0, 108, -1, -1, 0, 2, 1, 1, 8,
      "C - - 10 - low C - - 10 - high", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunctime_base(void)
{
   /* time_base */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_time_base));
   // automatic default constructor
   G__memfunc_setup("time_base", 937, G__time_base_time_base_0_0, (int) ('i'), G__get_linked_tagnum(&G__LN_time_base), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("time_base", 937, G__time_base_time_base_1_0, (int) ('i'), G__get_linked_tagnum(&G__LN_time_base), -1, 0, 1, 1, 1, 0, "u 'time_base' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~time_base", 1063, G__time_base_wAtime_base_2_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncistreambuf_iteratorlEcharcOchar_traitslEchargRsPgR(void)
{
   /* istreambuf_iterator<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("istreambuf_iterator<char,char_traits<char> >", 4363, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_0_0, 
		    105, G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 0, 1, 1, 0, 
		    "", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("istreambuf_iterator<char,char_traits<char> >", 4363, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_1_0, 105, 
		    G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "u 'basic_istream<char,char_traits<char> >' 'istreambuf_iterator<char,char_traits<char> >::istream_type' 1 - is", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("istreambuf_iterator<char,char_traits<char> >", 4363, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_2_0, 105, 
		    G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "U 'basic_streambuf<char,char_traits<char> >' 'istreambuf_iterator<char,char_traits<char> >::streambuf_type' 0 - sb", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator*", 918, G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatormU_3_0, 99, -1, 
		    G__defined_typename("istreambuf_iterator<char,char_traits<char> >::char_type"), 0, 0, 1, 1, 8, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator++", 962, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 1, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator++", 962, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_5_0, 117,
		     G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy), 
		    -1, 0, 1, 1, 1, 0, "i - - 0 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("equal", 536, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_equal_6_0, 105, 
		    G__get_linked_tagnum(&G__LN_bool), -1, 0, 1, 1, 1, 8, 
		    "u 'istreambuf_iterator<char,char_traits<char> >' - 11 - b", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("istreambuf_iterator<char,char_traits<char> >", 4363, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_8_0, (int) ('i'), G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "u 'istreambuf_iterator<char,char_traits<char> >' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~istreambuf_iterator<char,char_traits<char> >", 4489, 
		    G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_wAistreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_9_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfunciteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR(void)
{
   /* iterator<input_iterator_tag,char,long,char*,char&> */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR));
   // automatic default constructor
   G__memfunc_setup("iterator<input_iterator_tag,char,long,char*,char&>", 4866, 
		    G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_0_0, 
		    (int) ('i'), G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR), 
		    -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("iterator<input_iterator_tag,char,long,char*,char&>", 4866,
		     G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_1_0, 
		    (int) ('i'), G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR), 
		    -1, 0, 1, 1, 1, 0, "u 'iterator<input_iterator_tag,char,long,char*,char&>' - 1 - -", (char *) NULL, 
		    (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~iterator<input_iterator_tag,char,long,char*,char&>", 4992, 
		    G__iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_wAiteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR_2_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncistreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy(void)
{
   /* istreambuf_iterator<char,char_traits<char> >::proxy */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy));
   G__memfunc_setup("operator*", 918, G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy_operatormU_1_0, 99, -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("proxy", 578, G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy_proxy_2_0, 
		    (int) ('i'), G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy), 
		    -1, 0, 1, 1, 1, 0, "u 'istreambuf_iterator<char,char_traits<char> >::proxy' - 1 - -", (char *) NULL, 
		    (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~proxy", 704, G__istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy_wAproxy_3_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void)
{
   /* basic_istream<char,char_traits<char> >::sentry */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   G__memfunc_setup("operator bool", 1336, G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0, 117, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("sentry", 677, G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_sentry_5_0, 105, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry), -1, 0, 2, 1, 1, 0,
                    "u 'basic_istream<char,char_traits<char> >' - 1 - is i 'bool' - 0 - noskipws", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~sentry", 803, G__basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_wAsentry_6_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR(void)
{
   /* ostreambuf_iterator<char,char_traits<char> > */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR));
   G__memfunc_setup("ostreambuf_iterator<char,char_traits<char> >", 4369, 
		    G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_0_0, 
		    105, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "u 'basic_ostream<char,char_traits<char> >' 'ostreambuf_iterator<char,char_traits<char> >::ostream_type' 1 - s", 
		    (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("ostreambuf_iterator<char,char_traits<char> >", 4369, 
		    G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_1_0, 
		    105, G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "U 'basic_streambuf<char,char_traits<char> >' 'ostreambuf_iterator<char,char_traits<char> >::streambuf_type' 0 - s", 
		    (char *) NULL, 
		    (void *) NULL, 0);
   G__memfunc_setup("operator*", 918, G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatormU_2_0, 117, 
		    G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 1, 0, 1, 1, 0, 
		    "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator++", 962, G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_3_0, 117, 
		    G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 1, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator++", 962, G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatorpLpL_4_0, 117, 
		    G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "i - - 0 - -", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("operator=", 937, G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_operatoreQ_5_0, 117, 
		    G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, 
		    "c - 'ostreambuf_iterator<char,char_traits<char> >::char_type' 0 - c", (char *) NULL, (void *) NULL, 0);
   G__memfunc_setup("failed", 613, G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_failed_6_0, 105, 
		    G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 8, "", (char *) NULL, (void *) NULL, 0);
   // automatic copy constructor
   G__memfunc_setup("ostreambuf_iterator<char,char_traits<char> >", 4369, 
		    G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_7_0, (int) ('i'), 
		    G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR), -1, 0, 1, 1, 1, 0, 
		    "u 'ostreambuf_iterator<char,char_traits<char> >' - 1 - -", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~ostreambuf_iterator<char,char_traits<char> >", 4495, 
		    G__ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_wAostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR_8_0, 
		    (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry(void)
{
   /* basic_ostream<char,char_traits<char> >::sentry */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry));
   G__memfunc_setup("operator bool", 1336, G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_operatorsPbool_2_0, 117, G__get_linked_tagnum(&G__LN_bool), -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   // automatic destructor
   G__memfunc_setup("~sentry", 803, G__basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry_wAsentry_5_0, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char *) NULL, (void *) NULL, 0);
   G__tag_memfunc_reset();
}


/*********************************************************
* Member function information setup
*********************************************************/
extern "C" void G__cpp_setup_memfunc()
{
}

/*********************************************************
* Global variable information setup for each class
*********************************************************/
extern "C" void G__cpp_setup_global()
{

   /* Setting up global variables */
   G__resetplocal();

#ifdef REMOVED
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__KCC=1", 1, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__UNIX=1", 1, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__REGEXP1=1", 1, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__OSFDLL=1", 1, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__SIGNEDCHAR=1", 1, (char *) NULL);
   G__memvar_setup((void *) (&nothrow), 117, 0, 1, G__get_linked_tagnum(&G__LN_nothrow_t), -1, -1, 1, "nothrow=", 0, (char *) NULL);
   G__memvar_setup((void *) (&__stl_temp_buffer), 100, 0, 0, -1, -1, -1, 1, "__stl_temp_buffer[2048]=", 0, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_boolean_t), -1, -1, 1, "B_FALSE=0", 0, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 105, 0, 1, G__get_linked_tagnum(&G__LN_boolean_t), -1, -1, 1, "B_TRUE=1", 0, (char *) NULL);
   G__memvar_setup((void *) (&tzname), 67, 0, 0, -1, -1, -1, 1, "tzname[2]=", 0, (char *) NULL);
   G__memvar_setup((void *) (&timezone), 105, 0, 0, -1, G__defined_typename("time_t"), -1, 1, "timezone=", 0, (char *) NULL);
   G__memvar_setup((void *) (&daylight), 105, 0, 0, -1, -1, -1, 1, "daylight=", 0, (char *) NULL);
   G__memvar_setup((void *) (&getdate_err), 105, 0, 0, -1, -1, -1, 1, "getdate_err=", 0, (char *) NULL);
   G__memvar_setup((void *) (&altzone), 105, 0, 0, -1, G__defined_typename("time_t"), -1, 1, "altzone=", 0, (char *) NULL);
#endif
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__IOSTREAM_H=0", 1, (char *) NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__TMPLTIOS=0",1,(char*)NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__MANIP_SUPPORT=0", 1, (char *) NULL);

   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__KCC=1", 1, (char *) NULL);
   G__memvar_setup((void *) (&cin), 117, 0, 0, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), G__defined_typename("istream"), -1, 1, "cin=", 0, (char *) NULL);
   G__memvar_setup((void *) (&cout), 117, 0, 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), G__defined_typename("ostream"), -1, 1, "cout=", 0, (char *) NULL);
   G__memvar_setup((void *) (&clog), 117, 0, 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), G__defined_typename("ostream"), -1, 1, "clog=", 0, (char *) NULL);
   G__memvar_setup((void *) (&cerr), 117, 0, 0, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), G__defined_typename("ostream"), -1, 1, "cerr=", 0, (char *) NULL);

   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__FSTREAM_H=0", 1, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__STDSTREAM_H=0", 1, (char *) NULL);
   G__memvar_setup((void *) G__PVOID, 112, 0, 0, -1, -1, -1, 1, "G__STRSTREAM_H=0", 1, (char *) NULL);

   G__resetglobalenv();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
extern "C" void G__cpp_setup_func()
{
   G__lastifuncposition();

   G__memfunc_setup("operator+", 919, G___operatorpL_6_1, 117, G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), -1, 0, 2, 1, 1, 0,
                    "u 'reverse_iterator<char*>' - 0 - n u 'reverse_iterator<char*>' - 1 - x", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("localeconv", 1062, G___localeconv_9_1, 85, G__get_linked_tagnum(&G__LN_lconv), -1, 0, 0, 1, 1, 0, "", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("boolalpha", 946, G___boolalpha_4_2, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("noboolalpha", 1167, G___noboolalpha_5_2, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("showbase", 860, G___showbase_6_2, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("noshowbase", 1081, G___noshowbase_7_2, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("showpoint", 1003, G___showpoint_8_2, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("noshowpoint", 1224, G___noshowpoint_9_2, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("showpos", 787, G___showpos_0_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("noshowpos", 1008, G___noshowpos_1_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("skipws", 673, G___skipws_2_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("noskipws", 894, G___noskipws_3_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("uppercase", 968, G___uppercase_4_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("nouppercase", 1189, G___nouppercase_5_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("internal", 861, G___internal_6_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("left", 427, G___left_7_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("right", 542, G___right_8_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("dec", 300, G___dec_9_3, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("hex", 325, G___hex_0_4, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("oct", 326, G___oct_1_4, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("fixed", 528, G___fixed_2_4, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("scientific", 1057, G___scientific_3_4, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("unitbuf", 765, G___unitbuf_4_4, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("nounitbuf", 986, G___nounitbuf_5_4, 117, G__get_linked_tagnum(&G__LN_ios_base), -1, 1, 1, 1, 1, 0, "u 'ios_base' - 1 - str", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("difftime", 840, G___difftime_1_5, 100, -1, -1, 0, 2, 1, 1, 0,
                    "i - 'time_t' 0 - - i - 'time_t' 0 - -", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("mktime", 647, G___mktime_2_5, 105, -1, G__defined_typename("time_t"), 0, 1, 1, 1, 0, "U 'tm' - 0 - -", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("time", 431, G___time_3_5, 105, -1, G__defined_typename("time_t"), 0, 1, 1, 1, 0, "I - 'time_t' 0 - -", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("asctime", 742, G___asctime_4_5, 67, -1, -1, 0, 1, 1, 1, 0, "U 'tm' - 0 - -", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("ctime", 530, G___ctime_5_5, 67, -1, -1, 0, 1, 1, 1, 0, "I - 'time_t' 0 - -", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("strftime", 878, G___strftime_8_5, 104, -1, G__defined_typename("size_t"), 0, 4, 1, 1, 0,
                    "C - - 0 - - h - 'size_t' 0 - - "
                    "C - - 0 - - U 'tm' - 0 - -", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("endl", 419, G___endl_8_7, 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, "u 'basic_ostream<char,char_traits<char> >' - 1 - os", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("ends", 426, G___ends_9_7, 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, "u 'basic_ostream<char,char_traits<char> >' - 1 - os", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("flush", 546, G___flush_0_8, 117, G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, "u 'basic_ostream<char,char_traits<char> >' - 1 - os", (char *) NULL
                    ,(void *) NULL, 0);
   G__memfunc_setup("ws", 234, G___ws_9_7, 117, G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR), -1, 1, 1, 1, 1, 0, "u 'basic_istream<char,char_traits<char> >' - 1 - is", (char *) NULL
                    ,(void *) NULL, 0);


   G__resetifuncposition();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__LN_char_traitslEchargR =
{"char_traits<char>", 99, -1};
G__linked_taginfo G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR =
{"basic_ios<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR =
{"basic_istream<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR =
{"basic_ostream<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR =
{"basic_iostream<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR =
{"basic_streambuf<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR =
{"basic_filebuf<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_allocatorlEchargR =
{"allocator<char>", 99, -1};
G__linked_taginfo G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR =
{"basic_stringbuf<char,char_traits<char>,allocator<char> >", 99, -1};
G__linked_taginfo G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR =
{"basic_ifstream<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR =
{"basic_ofstream<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR =
{"basic_fstream<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR =
{"basic_istringstream<char,char_traits<char>,allocator<char> >", 99, -1};
G__linked_taginfo G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR =
{"basic_ostringstream<char,char_traits<char>,allocator<char> >", 99, -1};
G__linked_taginfo G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR =
{"basic_stringstream<char,char_traits<char>,allocator<char> >", 99, -1};
G__linked_taginfo G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR =
{"basic_string<char,char_traits<char>,allocator<char> >", 99, -1};
G__linked_taginfo G__LN_bool =
{"bool", 101, -1};
G__linked_taginfo G__LN_input_iterator_tag =
{"input_iterator_tag", 115, -1};
G__linked_taginfo G__LN_output_iterator_tag =
{"output_iterator_tag", 115, -1};
G__linked_taginfo G__LN_forward_iterator_tag =
{"forward_iterator_tag", 115, -1};
G__linked_taginfo G__LN_bidirectional_iterator_tag =
{"bidirectional_iterator_tag", 115, -1};
G__linked_taginfo G__LN_random_access_iterator_tag =
{"random_access_iterator_tag", 115, -1};
G__linked_taginfo G__LN_output_iterator =
{"output_iterator", 115, -1};
G__linked_taginfo G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR =
{"iterator<output_iterator_tag,void,void,void,void>", 115, -1};
G__linked_taginfo G__LN_rel_ops =
{"rel_ops", 110, -1};
G__linked_taginfo G__LN_allocatorlEvoidgR =
{"allocator<void>", 99, -1};
G__linked_taginfo G__LN_streampos =
{"streampos", 99, -1};
G__linked_taginfo G__LN_b_str_reflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR =
{"b_str_ref<char,char_traits<char>,allocator<char> >", 99, -1};
G__linked_taginfo G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLtemplate =
{"basic_string<char,char_traits<char>,allocator<char> >::template", 0, -1};
G__linked_taginfo G__LN_reverse_iteratorlEcharmUgR =
{"reverse_iterator<char*>", 99, -1};
G__linked_taginfo G__LN_iterator_traitslEcharmUgR =
{"iterator_traits<char*>", 115, -1};
G__linked_taginfo G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR =
{"iterator<long,char*,char**,char*&,random_access_iterator_tag>", 115, -1};
G__linked_taginfo G__LN_lconv =
{"lconv", 115, -1};
G__linked_taginfo G__LN_locale =
{"locale", 99, -1};
G__linked_taginfo G__LN_localecLcLfacet =
{"locale::facet", 99, -1};
G__linked_taginfo G__LN_localecLcLid =
{"locale::id", 99, -1};
G__linked_taginfo G__LN_localecLcLimp =
{"locale::imp", 99, -1};
G__linked_taginfo G__LN_localecLcLdA =
{"locale::$", 101, -1};
G__linked_taginfo G__LN_ctype_base =
{"ctype_base", 99, -1};
G__linked_taginfo G__LN_ctype_basecLcLmask =
{"ctype_base::mask", 101, -1};
G__linked_taginfo G__LN_ctypelEchargR =
{"ctype<char>", 99, -1};
G__linked_taginfo G__LN_ctype_bynamelEchargR =
{"ctype_byname<char>", 99, -1};
G__linked_taginfo G__LN_ios_base =
{"ios_base", 99, -1};
G__linked_taginfo G__LN_ios_basecLcLfmt_flags =
{"ios_base::fmt_flags", 101, -1};
G__linked_taginfo G__LN_ios_basecLcLio_state =
{"ios_base::io_state", 101, -1};
G__linked_taginfo G__LN_ios_basecLcLopen_mode =
{"ios_base::open_mode", 101, -1};
G__linked_taginfo G__LN_ios_basecLcLseekdir =
{"ios_base::seekdir", 101, -1};
G__linked_taginfo G__LN_ios_basecLcLInit =
{"ios_base::Init", 99, -1};
G__linked_taginfo G__LN_ios_basecLcLevent =
{"ios_base::event", 101, -1};
G__linked_taginfo G__LN_boolean_t =
{"boolean_t", 101, -1};
G__linked_taginfo G__LN_codecvt_base =
{"codecvt_base", 99, -1};
G__linked_taginfo G__LN_codecvt_basecLcLresult =
{"codecvt_base::result", 101, -1};
G__linked_taginfo G__LN_codecvtlEcharcOcharcOintgR =
{"codecvt<char,char,int>", 99, -1};
G__linked_taginfo G__LN_collatelEchargR =
{"collate<char>", 99, -1};
G__linked_taginfo G__LN_time_base =
{"time_base", 99, -1};
G__linked_taginfo G__LN_time_basecLcLdateorder =
{"time_base::dateorder", 101, -1};
G__linked_taginfo G__LN_time_basecLcLt_conv_spec =
{"time_base::t_conv_spec", 101, -1};
G__linked_taginfo G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR =
{"istreambuf_iterator<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR =
{"iterator<input_iterator_tag,char,long,char*,char&>", 115, -1};
G__linked_taginfo G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy =
{"istreambuf_iterator<char,char_traits<char> >::proxy", 99, -1};
G__linked_taginfo G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry =
{"basic_istream<char,char_traits<char> >::sentry", 99, -1};
G__linked_taginfo G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR =
{"ostreambuf_iterator<char,char_traits<char> >", 99, -1};
G__linked_taginfo G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry =
{"basic_ostream<char,char_traits<char> >::sentry", 99, -1};
G__linked_taginfo G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLdA =
{"basic_stringbuf<char,char_traits<char>,allocator<char> >::$", 101, -1};

/* Reset class/struct taginfo */
extern "C" void G__cpp_reset_tagtable()
{
   G__LN_char_traitslEchargR.tagnum = -1;
   G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_allocatorlEchargR.tagnum = -1;
   G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1;
   G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1;
   G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1;
   G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1;
   G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1;
   G__LN_bool.tagnum = -1;
   G__LN_input_iterator_tag.tagnum = -1;
   G__LN_output_iterator_tag.tagnum = -1;
   G__LN_forward_iterator_tag.tagnum = -1;
   G__LN_bidirectional_iterator_tag.tagnum = -1;
   G__LN_random_access_iterator_tag.tagnum = -1;
   G__LN_output_iterator.tagnum = -1;
   G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR.tagnum = -1;
   G__LN_rel_ops.tagnum = -1;
   G__LN_allocatorlEvoidgR.tagnum = -1;
   G__LN_streampos.tagnum = -1;
   G__LN_b_str_reflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR.tagnum = -1;
   G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLtemplate.tagnum = -1;
   G__LN_reverse_iteratorlEcharmUgR.tagnum = -1;
   G__LN_iterator_traitslEcharmUgR.tagnum = -1;
   G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR.tagnum = -1;
   G__LN_lconv.tagnum = -1;
   G__LN_locale.tagnum = -1;
   G__LN_localecLcLfacet.tagnum = -1;
   G__LN_localecLcLid.tagnum = -1;
   G__LN_localecLcLimp.tagnum = -1;
   G__LN_localecLcLdA.tagnum = -1;
   G__LN_ctype_base.tagnum = -1;
   G__LN_ctype_basecLcLmask.tagnum = -1;
   G__LN_ctypelEchargR.tagnum = -1;
   G__LN_ctype_bynamelEchargR.tagnum = -1;
   G__LN_ios_base.tagnum = -1;
   G__LN_ios_basecLcLfmt_flags.tagnum = -1;
   G__LN_ios_basecLcLio_state.tagnum = -1;
   G__LN_ios_basecLcLopen_mode.tagnum = -1;
   G__LN_ios_basecLcLseekdir.tagnum = -1;
   G__LN_ios_basecLcLInit.tagnum = -1;
   G__LN_ios_basecLcLevent.tagnum = -1;
   G__LN_boolean_t.tagnum = -1;
   G__LN_codecvt_base.tagnum = -1;
   G__LN_codecvt_basecLcLresult.tagnum = -1;
   G__LN_codecvtlEcharcOcharcOintgR.tagnum = -1;
   G__LN_collatelEchargR.tagnum = -1;
   G__LN_time_base.tagnum = -1;
   G__LN_time_basecLcLdateorder.tagnum = -1;
   G__LN_time_basecLcLt_conv_spec.tagnum = -1;
   G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR.tagnum = -1;
   G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy.tagnum = -1;
   G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry.tagnum = -1;
   G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR.tagnum = -1;
   G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry.tagnum = -1;
   G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLdA.tagnum = -1;
}

extern "C" void G__cpp_setup_tagtable()
{

   /* Setting up class,struct,union tag entry */
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_char_traitslEchargR),   
		     sizeof(char_traits < char >), -1, 0, (char *) NULL, G__setup_memvarchar_traitslEchargR,   
		     G__setup_memfuncchar_traitslEchargR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_ioslEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_ios < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_ioslEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_ioslEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_istream < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_ostream < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_iostreamlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_iostream < char, char_traits < char > >), -1, 0, (char *) NULL,  
		      G__setup_memvarbasic_iostreamlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_iostreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_streambuflEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_streambuf < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_streambuflEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_streambuflEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_filebuflEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_filebuf < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_filebuflEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_filebuflEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_allocatorlEchargR), sizeof(allocator < char >), -1, 0, (char *) NULL,   
		     G__setup_memvarallocatorlEchargR, G__setup_memfuncallocatorlEchargR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),   
		     sizeof(basic_stringbuf < char, char_traits < char >, allocator < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,   
		     G__setup_memfuncbasic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_ifstreamlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_ifstream < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_ifstreamlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_ifstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_ofstreamlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_ofstream < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_ofstreamlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_ofstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_fstreamlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(basic_fstream < char, char_traits < char > >), -1, 0, (char *) NULL,  
		      G__setup_memvarbasic_fstreamlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncbasic_fstreamlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),   
		     sizeof(basic_istringstream < char, char_traits < char >, allocator < char > >), -1, 0,   
		     (char *) NULL, G__setup_memvarbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,   
		     G__setup_memfuncbasic_istringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),  
		      sizeof(basic_ostringstream < char, char_traits < char >, allocator < char > >), -1, 0,   
		     (char *) NULL, G__setup_memvarbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,   
		     G__setup_memfuncbasic_ostringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),   
		     sizeof(basic_stringstream < char, char_traits < char >, allocator < char > >), -1, 0,  
		      (char *) NULL, G__setup_memvarbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,   
		     G__setup_memfuncbasic_stringstreamlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR),   
		     sizeof(string), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR,   
		     G__setup_memfuncbasic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_input_iterator_tag), sizeof(input_iterator_tag), -1, 0, (char *) NULL,   
		     G__setup_memvarinput_iterator_tag, G__setup_memfuncinput_iterator_tag);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_output_iterator_tag), sizeof(output_iterator_tag), -1, 0, (char *) NULL,   
		     G__setup_memvaroutput_iterator_tag, G__setup_memfuncoutput_iterator_tag);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_forward_iterator_tag), sizeof(forward_iterator_tag), -1, 0, (char *) NULL,   
		     G__setup_memvarforward_iterator_tag, G__setup_memfuncforward_iterator_tag);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_bidirectional_iterator_tag), sizeof(bidirectional_iterator_tag), -1, 0,   
		     (char *) NULL, G__setup_memvarbidirectional_iterator_tag, G__setup_memfuncbidirectional_iterator_tag);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_random_access_iterator_tag), sizeof(random_access_iterator_tag), -1, 0,   
		     (char *) NULL, G__setup_memvarrandom_access_iterator_tag, G__setup_memfuncrandom_access_iterator_tag);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_iteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR),   
		     sizeof(iterator < output_iterator_tag, void, void, void, void >), -1, 0, (char *) NULL,   
		     G__setup_memvariteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR,   
		     G__setup_memfunciteratorlEoutput_iterator_tagcOvoidcOvoidcOvoidcOvoidgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_rel_ops), 0, -1, 0, (char *) NULL, G__setup_memvarrel_ops,   
		     G__setup_memfuncrel_ops);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_allocatorlEvoidgR), sizeof(allocator < void >), -1, 0, (char *) NULL,   
		     G__setup_memvarallocatorlEvoidgR, G__setup_memfuncallocatorlEvoidgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_streampos), sizeof(streampos), -1, 0, (char *) NULL,   
		     G__setup_memvarstreampos, G__setup_memfuncstreampos);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_stringlEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLtemplate), 
		     0, -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_reverse_iteratorlEcharmUgR), sizeof(reverse_iterator < char *>), -1, 0,   
		     (char *) NULL, G__setup_memvarreverse_iteratorlEcharmUgR, G__setup_memfuncreverse_iteratorlEcharmUgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_iterator_traitslEcharmUgR), sizeof(iterator_traits < char *>), -1, 0,   
		     (char *) NULL, G__setup_memvariterator_traitslEcharmUgR, G__setup_memfunciterator_traitslEcharmUgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_iteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR),   
		     sizeof(iterator < long, char *, char **, char *&, random_access_iterator_tag >), -1, 0, (char *) NULL,  
		     G__setup_memvariteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR,   
		     G__setup_memfunciteratorlElongcOcharmUcOcharmUmUcOcharmUaNcOrandom_access_iterator_taggR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_lconv), sizeof(lconv), -1, 0, (char *) NULL,   
		     G__setup_memvarlconv, G__setup_memfunclconv);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_locale), sizeof(locale), -1, 0, (char *) NULL,   
		     G__setup_memvarlocale, G__setup_memfunclocale);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_localecLcLfacet), sizeof(locale::facet), -1, 0, (char *) NULL,   
		     G__setup_memvarlocalecLcLfacet, G__setup_memfunclocalecLcLfacet);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_localecLcLid), sizeof(locale::id), -1, 0, (char *) NULL,   
		     G__setup_memvarlocalecLcLid, G__setup_memfunclocalecLcLid);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_localecLcLdA), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ctype_base), sizeof(ctype_base), -1, 0, (char *) NULL,   
		     G__setup_memvarctype_base, G__setup_memfuncctype_base);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ctype_basecLcLmask), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ctypelEchargR), sizeof(ctype < char >), -1, 0, (char *) NULL,   
		     G__setup_memvarctypelEchargR, G__setup_memfuncctypelEchargR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_base), sizeof(ios_base), -1, 0, (char *) NULL,  
		      G__setup_memvarios_base, G__setup_memfuncios_base);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLfmt_flags), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLio_state), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLopen_mode), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLseekdir), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLInit), sizeof(ios_base::Init), -1, 0, (char *) NULL,   
		     G__setup_memvarios_basecLcLInit, G__setup_memfuncios_basecLcLInit);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ios_basecLcLevent), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_boolean_t), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_codecvt_base), sizeof(codecvt_base), -1, 0, (char *) NULL,
		     G__setup_memvarcodecvt_base, G__setup_memfunccodecvt_base);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_codecvt_basecLcLresult), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_codecvtlEcharcOcharcOintgR), sizeof(codecvt < char, char, int >), -1, 0,   
		     (char *) NULL, G__setup_memvarcodecvtlEcharcOcharcOintgR, G__setup_memfunccodecvtlEcharcOcharcOintgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_collatelEchargR), sizeof(collate < char >), -1, 0, (char *) NULL,   
		     G__setup_memvarcollatelEchargR, G__setup_memfunccollatelEchargR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_time_base), sizeof(time_base), -1, 0, (char *) NULL,
		     G__setup_memvartime_base, G__setup_memfunctime_base);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_time_basecLcLdateorder), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_time_basecLcLt_conv_spec), sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(istreambuf_iterator < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvaristreambuf_iteratorlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncistreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_iteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR),   
		     sizeof(iterator < input_iterator_tag, char, long, char *, char &>), -1, 0, (char *) NULL,   
		     G__setup_memvariteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR,   
		     G__setup_memfunciteratorlEinput_iterator_tagcOcharcOlongcOcharmUcOcharaNgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_istreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy),   
		     sizeof(istreambuf_iterator < char, char_traits < char > >::proxy), -1, 0, (char *) NULL,   
		     G__setup_memvaristreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy,   
		     G__setup_memfuncistreambuf_iteratorlEcharcOchar_traitslEchargRsPgRcLcLproxy);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),   
		     sizeof(basic_istream < char, char_traits < char > >::sentry), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry,   
		     G__setup_memfuncbasic_istreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_ostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR),   
		     sizeof(ostreambuf_iterator < char, char_traits < char > >), -1, 0, (char *) NULL,   
		     G__setup_memvarostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR,   
		     G__setup_memfuncostreambuf_iteratorlEcharcOchar_traitslEchargRsPgR);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry),   
		     sizeof(basic_ostream < char, char_traits < char > >::sentry), -1, 0, (char *) NULL,   
		     G__setup_memvarbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry,   
		     G__setup_memfuncbasic_ostreamlEcharcOchar_traitslEchargRsPgRcLcLsentry);
   G__tagtable_setup(G__get_linked_tagnum(&G__LN_basic_stringbuflEcharcOchar_traitslEchargRcOallocatorlEchargRsPgRcLcLdA),  
		     sizeof(int), -1, 0, (char *) NULL, NULL, NULL);
}
extern "C" void G__cpp_setupG__stream()
{
   G__check_setup_version(G__CREATEDLLREV, "G__cpp_setupG__stream()");
   G__set_cpp_environment();
   G__cpp_setup_tagtable();

   G__cpp_setup_inheritance();

   G__cpp_setup_typetable();

   G__cpp_setup_memvar();

   G__cpp_setup_memfunc();
   G__cpp_setup_global();
   G__cpp_setup_func();

   if (0 == G__getsizep2memfunc())
      G__get_sizep2memfunc();
   return;
}
