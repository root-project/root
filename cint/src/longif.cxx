/********************************************************
* longif.cxx
********************************************************/
#include "longif.h"

#ifdef G__MEMTEST
#undef malloc
#undef free
#endif

extern "C" void G__cpp_reset_tagtablelongif();

extern "C" void G__set_cpp_environmentlongif() {
  G__add_compiledheader("longlong.h");
  G__add_compiledheader("<iostream.h");
  G__add_compiledheader("<iosenum.h");
  G__add_compiledheader("<bool.h");
  G__add_compiledheader("<_iostream");
  G__add_compiledheader("<stdio.h");
  G__add_compiledheader("<stdfunc.dll");
  G__add_compiledheader("longdbl.h");
  G__cpp_reset_tagtablelongif();
}
class G__longifdOcxx_tag {};

void* operator new(size_t size,G__longifdOcxx_tag* p) {
  if(p && G__PVOID!=G__getgvp()) return((void*)p);
#ifndef G__ROOT
  return(malloc(size));
#else
  return(::operator new(size));
#endif
}

/* dummy, for exception */
#ifdef G__EH_DUMMY_DELETE
void operator delete(void *p,G__longifdOcxx_tag* x) {
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

void G__DELDMY_longifdOcxx() { G__operator_delete(0); }

extern "C" int G__cpp_dllrevlongif() { return(30051515); }

/*********************************************************
* Member function Interface Method
*********************************************************/

/* G__ulonglong */
static int G__longif_28_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__ulonglong *p=NULL;
   switch(libp->paran) {
   case 1:
      p = ::new((G__longifdOcxx_tag*)G__getgvp()) G__ulonglong((unsigned long)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new G__ulonglong[G__getaryconstruct()];
   else p=::new((G__longifdOcxx_tag*)G__getgvp()) G__ulonglong;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__ulonglong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__ulonglong *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__ulonglong(*(G__ulonglong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__ulonglong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__ulonglong *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__ulonglong(*(G__longlong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__ulonglong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__ulonglong *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__ulonglong((const char*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__ulonglong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((G__ulonglong*)(G__getstructoffset()))->operator long());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((G__ulonglong*)(G__getstructoffset()))->operator int());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator++();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=((G__ulonglong*)(G__getstructoffset()))->operator++((int)G__int(libp->para[0]));
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator--();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=((G__ulonglong*)(G__getstructoffset()))->operator--((int)G__int(libp->para[0]));
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator=((long)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator+=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator-=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator*=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator&=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator|=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator<<=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_28_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator>>=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef G__ulonglong G__TG__ulonglong;
static int G__longif_28_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (G__ulonglong *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((G__ulonglong *)((G__getstructoffset())+sizeof(G__ulonglong)*i))->~G__TG__ulonglong();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((G__ulonglong *)(G__getstructoffset()))->~G__TG__ulonglong();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* G__longlong */
static int G__longif_29_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longlong *p=NULL;
   switch(libp->paran) {
   case 1:
      p = ::new((G__longifdOcxx_tag*)G__getgvp()) G__longlong((long)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new G__longlong[G__getaryconstruct()];
   else p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longlong;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longlong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longlong *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longlong(*(G__longlong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longlong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longlong *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longlong(*(G__ulonglong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longlong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longlong *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longlong((const char*)G__int(libp->para[0]));
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longlong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((G__longlong*)(G__getstructoffset()))->operator long());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((G__longlong*)(G__getstructoffset()))->operator int());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letdouble(result7,100,(double)((G__longlong*)(G__getstructoffset()))->operator double());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator++();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=((G__longlong*)(G__getstructoffset()))->operator++((int)G__int(libp->para[0]));
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator--();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=((G__longlong*)(G__getstructoffset()))->operator--((int)G__int(libp->para[0]));
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator=((long)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator+=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator-=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator*=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator&=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator|=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator<<=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_29_0_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator>>=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef G__longlong G__TG__longlong;
static int G__longif_29_1_2(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (G__longlong *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((G__longlong *)((G__getstructoffset())+sizeof(G__longlong)*i))->~G__TG__longlong();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((G__longlong *)(G__getstructoffset()))->~G__TG__longlong();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* G__longdouble */
static int G__longif_35_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longdouble *p=NULL;
   switch(libp->paran) {
   case 1:
      p = ::new((G__longifdOcxx_tag*)G__getgvp()) G__longdouble((double)G__double(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new G__longdouble[G__getaryconstruct()];
   else p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longdouble;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longdouble);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longdouble *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longdouble(*(G__longdouble*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longdouble);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_2_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longdouble *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longdouble(*(G__longlong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longdouble);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longdouble *p=NULL;
      p=::new((G__longifdOcxx_tag*)G__getgvp()) G__longdouble(*(G__ulonglong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longdouble);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letdouble(result7,100,(double)((G__longdouble*)(G__getstructoffset()))->operator double());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator++();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longdouble *pobj,xobj=((G__longdouble*)(G__getstructoffset()))->operator++((int)G__int(libp->para[0]));
        pobj=new G__longdouble(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator--();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longdouble *pobj,xobj=((G__longdouble*)(G__getstructoffset()))->operator--((int)G__int(libp->para[0]));
        pobj=new G__longdouble(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator=((double)G__double(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator=(*(G__longdouble*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator+=(*(G__longdouble*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator-=(*(G__longdouble*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator*=(*(G__longdouble*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif_35_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longdouble& obj=((G__longdouble*)(G__getstructoffset()))->operator/=(*(G__longdouble*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef G__longdouble G__TG__longdouble;
static int G__longif_35_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   if(0==G__getstructoffset()) return(1);
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (G__longdouble *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         ((G__longdouble *)((G__getstructoffset())+sizeof(G__longdouble)*i))->~G__TG__longdouble();
   else {
     long G__Xtmp=G__getgvp();
     G__setgvp(G__PVOID);
     ((G__longdouble *)(G__getstructoffset()))->~G__TG__longdouble();
     G__setgvp(G__Xtmp);
     G__operator_delete((void*)G__getstructoffset());
   }
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* Setting up global function */
static int G__longif__3_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator+(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__4_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator-(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__5_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator*(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__6_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator/(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__7_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator%(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__8_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator&(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__9_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator|(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__0_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator<<(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__1_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator>>(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__2_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator&&(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__3_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator||(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__4_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__5_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__6_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<=(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__7_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>=(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__8_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator!=(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__9_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator==(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__0_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__1_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__2_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator+(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__3_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator-(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__4_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator*(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__5_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator/(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__6_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator%(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__7_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator&(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__8_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator|(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__9_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator<<(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__0_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator>>(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__1_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator&&(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__2_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator||(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__3_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__4_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__5_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<=(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__6_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>=(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__7_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator!=(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__8_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator==(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__9_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__0_7(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__5_19(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      G__printformatll((char*)G__int(libp->para[0]),(const char*)G__int(libp->para[1])
,(void*)G__int(libp->para[2]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__6_19(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      G__printformatull((char*)G__int(libp->para[0]),(const char*)G__int(libp->para[1])
,(void*)G__int(libp->para[2]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__7_19(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)G__ateval(*(G__longlong*)libp->para[0].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__8_19(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)G__ateval(*(G__ulonglong*)libp->para[0].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__9_19(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longdouble *pobj,xobj=operator+(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref);
        pobj=new G__longdouble(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__0_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longdouble *pobj,xobj=operator-(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref);
        pobj=new G__longdouble(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__1_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longdouble *pobj,xobj=operator*(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref);
        pobj=new G__longdouble(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__2_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longdouble *pobj,xobj=operator/(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref);
        pobj=new G__longdouble(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__3_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__4_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__5_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<=(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__6_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>=(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__7_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator!=(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__8_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator==(*(G__longdouble*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__9_20(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__0_21(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(G__longdouble*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__1_21(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      G__printformatld((char*)G__int(libp->para[0]),(const char*)G__int(libp->para[1])
,(void*)G__int(libp->para[2]));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__longif__2_21(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)G__ateval(*(G__longdouble*)libp->para[0].ref));
   return(1 || funcname || hash || result7 || libp) ;
}


/*********************************************************
* Member function Stub
*********************************************************/

/* G__ulonglong */

/* G__longlong */

/* G__longdouble */

/*********************************************************
* Global function Stub
*********************************************************/

/*********************************************************
* Get size of pointer to member function
*********************************************************/
class G__Sizep2memfunclongif {
 public:
  G__Sizep2memfunclongif() {p=&G__Sizep2memfunclongif::sizep2memfunc;}
    size_t sizep2memfunc() { return(sizeof(p)); }
  private:
    size_t (G__Sizep2memfunclongif::*p)();
};

size_t G__get_sizep2memfunclongif()
{
  G__Sizep2memfunclongif a;
  G__setsizep2memfunc((int)a.sizep2memfunc());
  return((size_t)a.sizep2memfunc());
}


/*********************************************************
* virtual base class offset calculation interface
*********************************************************/

   /* Setting up class inheritance */

/*********************************************************
* Inheritance information setup/
*********************************************************/
extern "C" void G__cpp_setup_inheritancelongif() {

   /* Setting up class inheritance */
}

/*********************************************************
* typedef information setup/
*********************************************************/
extern "C" void G__cpp_setup_typetablelongif() {

   /* Setting up typedef entry */
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

   /* G__ulonglong */
static void G__setup_memvarG__ulonglong(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__longifLN_G__ulonglong));
   { G__ulonglong *p; p=(G__ulonglong*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* G__longlong */
static void G__setup_memvarG__longlong(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__longifLN_G__longlong));
   { G__longlong *p; p=(G__longlong*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* G__longdouble */
static void G__setup_memvarG__longdouble(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__longifLN_G__longdouble));
   { G__longdouble *p; p=(G__longdouble*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}

extern "C" void G__cpp_setup_memvarlongif() {
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
static void G__setup_memfuncG__ulonglong(void) {
   /* G__ulonglong */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__longifLN_G__ulonglong));
   G__memfunc_setup("G__ulonglong",1242,G__longif_28_0_0,105,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"k - - 0 0 l",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__ulonglong",1242,G__longif_28_1_0,105,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__ulonglong",1242,G__longif_28_2_0,105,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__ulonglong",1242,G__longif_28_3_0,105,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"C - - 10 - s",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator long",1340,G__longif_28_5_0,108,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator int",1239,G__longif_28_6_0,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__longif_28_7_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__longif_28_8_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"i - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__longif_28_9_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__longif_28_0_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"i - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__longif_28_1_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"l - - 0 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__longif_28_2_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator+=",980,G__longif_28_3_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator-=",982,G__longif_28_4_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator*=",979,G__longif_28_5_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator&=",975,G__longif_28_6_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator|=",1061,G__longif_28_7_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator<<=",1057,G__longif_28_8_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator>>=",1061,G__longif_28_9_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~G__ulonglong",1368,G__longif_28_0_2,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncG__longlong(void) {
   /* G__longlong */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__longifLN_G__longlong));
   G__memfunc_setup("G__longlong",1125,G__longif_29_0_0,105,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"l - - 0 0 l",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longlong",1125,G__longif_29_1_0,105,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longlong",1125,G__longif_29_2_0,105,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longlong",1125,G__longif_29_3_0,105,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"C - - 10 - s",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator long",1340,G__longif_29_5_0,108,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator int",1239,G__longif_29_6_0,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator double",1543,G__longif_29_7_0,100,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__longif_29_8_0,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__longif_29_9_0,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"i - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__longif_29_0_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__longif_29_1_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"i - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__longif_29_2_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"l - - 0 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__longif_29_3_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator+=",980,G__longif_29_4_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator-=",982,G__longif_29_5_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator*=",979,G__longif_29_6_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator&=",975,G__longif_29_7_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator|=",1061,G__longif_29_8_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator<<=",1057,G__longif_29_9_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator>>=",1061,G__longif_29_0_2,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~G__longlong",1251,G__longif_29_1_2,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncG__longdouble(void) {
   /* G__longdouble */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__longifLN_G__longdouble));
   G__memfunc_setup("G__longdouble",1328,G__longif_35_0_0,105,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,1,1,1,0,"d - - 0 0 l",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longdouble",1328,G__longif_35_1_0,105,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,1,1,1,0,"u 'G__longdouble' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longdouble",1328,G__longif_35_2_0,105,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longdouble",1328,G__longif_35_3_0,105,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator double",1543,G__longif_35_5_0,100,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__longif_35_6_0,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__longif_35_7_0,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,1,1,1,0,"i - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__longif_35_8_0,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__longif_35_9_0,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,1,1,1,0,"i - - 0 - -",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__longif_35_0_1,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,1,1,1,0,"d - - 0 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__longif_35_1_1,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,1,1,1,0,"u 'G__longdouble' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator+=",980,G__longif_35_2_1,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,1,1,1,0,"u 'G__longdouble' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator-=",982,G__longif_35_3_1,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,1,1,1,0,"u 'G__longdouble' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator*=",979,G__longif_35_4_1,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,1,1,1,0,"u 'G__longdouble' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator/=",984,G__longif_35_5_1,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,1,1,1,1,0,"u 'G__longdouble' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~G__longdouble",1454,G__longif_35_6_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}


/*********************************************************
* Member function information setup
*********************************************************/
extern "C" void G__cpp_setup_memfunclongif() {
}

/*********************************************************
* Global variable information setup for each class
*********************************************************/
static void G__cpp_setup_global0() {

   /* Setting up global variables */
   G__resetplocal();

   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__LONGLONG_H=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"IOS=0",1,(char*)NULL);
   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__LONGDOUBLE_H=0",1,(char*)NULL);

   G__resetglobalenv();
}
extern "C" void G__cpp_setup_globallongif() {
  G__cpp_setup_global0();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
static void G__cpp_setup_func0() {
   G__lastifuncposition();

   G__memfunc_setup("operator+",919,G__longif__3_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator-",921,G__longif__4_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator*",918,G__longif__5_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator/",923,G__longif__6_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator%",913,G__longif__7_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&",914,G__longif__8_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator|",1000,G__longif__9_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__longif__0_4,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__longif__1_4,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&&",952,G__longif__2_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator||",1124,G__longif__3_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<",936,G__longif__4_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>",938,G__longif__5_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<=",997,G__longif__6_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>=",999,G__longif__7_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator!=",970,G__longif__8_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator==",998,G__longif__9_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__longif__0_5,117,G__get_linked_tagnum(&G__longifLN_ostream),-1,1,2,1,1,0,
"u 'ostream' - 1 - ost u 'G__longlong' - 11 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__longif__1_5,117,G__get_linked_tagnum(&G__longifLN_istream),-1,1,2,1,1,0,
"u 'istream' - 1 - ist u 'G__longlong' - 1 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator+",919,G__longif__2_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator-",921,G__longif__3_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator*",918,G__longif__4_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator/",923,G__longif__5_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator%",913,G__longif__6_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&",914,G__longif__7_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator|",1000,G__longif__8_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__longif__9_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__longif__0_6,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&&",952,G__longif__1_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator||",1124,G__longif__2_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<",936,G__longif__3_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>",938,G__longif__4_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<=",997,G__longif__5_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>=",999,G__longif__6_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator!=",970,G__longif__7_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator==",998,G__longif__8_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__longif__9_6,117,G__get_linked_tagnum(&G__longifLN_ostream),-1,1,2,1,1,0,
"u 'ostream' - 1 - ost u 'G__ulonglong' - 11 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__longif__0_7,117,G__get_linked_tagnum(&G__longifLN_istream),-1,1,2,1,1,0,
"u 'istream' - 1 - ist u 'G__ulonglong' - 1 - a",(char*)NULL
,(void*)NULL,0);
}

static void G__cpp_setup_func1() {
   G__memfunc_setup("G__printformatll",1683,G__longif__5_19,121,-1,-1,0,3,1,1,0,
"C - - 0 - out C - - 10 - fmt "
"Y - - 0 - p",(char*)NULL
#ifndef G__printformatll
,(void*)(void (*)(char*,const char*,void*))G__printformatll,0);
#else
,(void*)NULL,0);
#endif
   G__memfunc_setup("G__printformatull",1800,G__longif__6_19,121,-1,-1,0,3,1,1,0,
"C - - 0 - out C - - 10 - fmt "
"Y - - 0 - p",(char*)NULL
#ifndef G__printformatull
,(void*)(void (*)(char*,const char*,void*))G__printformatull,0);
#else
,(void*)NULL,0);
#endif
   G__memfunc_setup("G__ateval",898,G__longif__7_19,105,-1,-1,0,1,1,1,0,"u 'G__longlong' - 11 - a",(char*)NULL
#ifndef G__ateval
,(void*)(int (*)(const G__longlong&))G__ateval,0);
#else
,(void*)NULL,0);
#endif
   G__memfunc_setup("G__ateval",898,G__longif__8_19,105,-1,-1,0,1,1,1,0,"u 'G__ulonglong' - 11 - a",(char*)NULL
#ifndef G__ateval
,(void*)(int (*)(const G__ulonglong&))G__ateval,0);
#else
,(void*)NULL,0);
#endif
   G__memfunc_setup("operator+",919,G__longif__9_19,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator-",921,G__longif__0_20,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator*",918,G__longif__1_20,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator/",923,G__longif__2_20,117,G__get_linked_tagnum(&G__longifLN_G__longdouble),-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
}

static void G__cpp_setup_func2() {
   G__memfunc_setup("operator<",936,G__longif__3_20,105,-1,-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>",938,G__longif__4_20,105,-1,-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<=",997,G__longif__5_20,105,-1,-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>=",999,G__longif__6_20,105,-1,-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator!=",970,G__longif__7_20,105,-1,-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator==",998,G__longif__8_20,105,-1,-1,0,2,1,1,0,
"u 'G__longdouble' - 11 - a u 'G__longdouble' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G__longif__9_20,117,G__get_linked_tagnum(&G__longifLN_ostream),-1,1,2,1,1,0,
"u 'ostream' - 1 - ost u 'G__longdouble' - 11 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G__longif__0_21,117,G__get_linked_tagnum(&G__longifLN_istream),-1,1,2,1,1,0,
"u 'istream' - 1 - ist u 'G__longdouble' - 1 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("G__printformatld",1675,G__longif__1_21,121,-1,-1,0,3,1,1,0,
"C - - 0 - out C - - 10 - fmt "
"Y - - 0 - p",(char*)NULL
#ifndef G__printformatld
,(void*)(void (*)(char*,const char*,void*))G__printformatld,0);
#else
,(void*)NULL,0);
#endif
   G__memfunc_setup("G__ateval",898,G__longif__2_21,105,-1,-1,0,1,1,1,0,"u 'G__longdouble' - 11 - a",(char*)NULL
#ifndef G__ateval
,(void*)(int (*)(const G__longdouble&))G__ateval,0);
#else
,(void*)NULL,0);
#endif

   G__resetifuncposition();
}

extern "C" void G__cpp_setup_funclongif() {
  G__cpp_setup_func0();
  G__cpp_setup_func1();
  G__cpp_setup_func2();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__longifLN_ostream = { "ostream" , 99 , -1 };
G__linked_taginfo G__longifLN_istream = { "istream" , 99 , -1 };
G__linked_taginfo G__longifLN_G__ulonglong = { "G__ulonglong" , 99 , -1 };
G__linked_taginfo G__longifLN_G__longlong = { "G__longlong" , 99 , -1 };
G__linked_taginfo G__longifLN_G__longdouble = { "G__longdouble" , 99 , -1 };

/* Reset class/struct taginfo */
extern "C" void G__cpp_reset_tagtablelongif() {
  G__longifLN_ostream.tagnum = -1 ;
  G__longifLN_istream.tagnum = -1 ;
  G__longifLN_G__ulonglong.tagnum = -1 ;
  G__longifLN_G__longlong.tagnum = -1 ;
  G__longifLN_G__longdouble.tagnum = -1 ;
}


extern "C" void G__cpp_setup_tagtablelongif() {

   /* Setting up class,struct,union tag entry */
   G__get_linked_tagnum(&G__longifLN_ostream);
   G__get_linked_tagnum(&G__longifLN_istream);
   G__tagtable_setup(G__get_linked_tagnum(&G__longifLN_G__ulonglong),sizeof(G__ulonglong),-1,36608,(char*)NULL,G__setup_memvarG__ulonglong,G__setup_memfuncG__ulonglong);
   G__tagtable_setup(G__get_linked_tagnum(&G__longifLN_G__longlong),sizeof(G__longlong),-1,36608,(char*)NULL,G__setup_memvarG__longlong,G__setup_memfuncG__longlong);
   G__tagtable_setup(G__get_linked_tagnum(&G__longifLN_G__longdouble),sizeof(G__longdouble),-1,36608,(char*)NULL,G__setup_memvarG__longdouble,G__setup_memfuncG__longdouble);
}
extern "C" void G__cpp_setuplongif(void) {
  G__check_setup_version(30051515,"G__cpp_setuplongif()");
  G__set_cpp_environmentlongif();
  G__cpp_setup_tagtablelongif();

  G__cpp_setup_inheritancelongif();

  G__cpp_setup_typetablelongif();

  G__cpp_setup_memvarlongif();

  G__cpp_setup_memfunclongif();
  G__cpp_setup_globallongif();
  G__cpp_setup_funclongif();

   if(0==G__getsizep2memfunc()) G__get_sizep2memfunclongif();
  return;
}
