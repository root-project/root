/********************************************************
* longif.C
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
  G__cpp_reset_tagtablelongif();
}
class G__longif_tag {};

void* operator new(size_t size,G__longif_tag* p) {
  if(p && G__PVOID!=G__getgvp()) return((void*)p);
#ifndef G__ROOT
  return(malloc(size));
#else
  return(::operator new(size));
#endif
}

/* dummy, for exception */
#ifdef G__EH_DUMMY_DELETE
void operator delete(void *p,G__longif_tag* x) {
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

void G__DELDMY_longif() { G__operator_delete(0); }

extern "C" int G__cpp_dllrevlongif() { return(30051515); }

/*********************************************************
* Member function Interface Method
*********************************************************/

/* G__longlong */
static int G__G__longlong_G__longlong_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longlong *p=NULL;
   switch(libp->paran) {
   case 1:
      p = ::new((G__longif_tag*)G__getgvp()) G__longlong((long)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new G__longlong[G__getaryconstruct()];
   else p=::new((G__longif_tag*)G__getgvp()) G__longlong;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longlong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_G__longlong_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__longlong *p=NULL;
      p=::new((G__longif_tag*)G__getgvp()) G__longlong(*(G__longlong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__longlong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorsPlong_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((G__longlong*)(G__getstructoffset()))->operator long());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorsPint_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((G__longlong*)(G__getstructoffset()))->operator int());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorsPdouble_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letdouble(result7,100,(double)((G__longlong*)(G__getstructoffset()))->operator double());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorpLpL_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator++();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorpLpL_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=((G__longlong*)(G__getstructoffset()))->operator++((int)G__int(libp->para[0]));
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatormImI_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator--();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatormImI_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=((G__longlong*)(G__getstructoffset()))->operator--((int)G__int(libp->para[0]));
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatoreQ_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator=((const long)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatoreQ_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorpLeQ_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator+=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatormIeQ_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator-=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatormUeQ_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator*=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatoraNeQ_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator&=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatoroReQ_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator|=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorlElEeQ_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator<<=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__longlong_operatorgRgReQ_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__longlong& obj=((G__longlong*)(G__getstructoffset()))->operator>>=(*(G__longlong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef G__longlong G__TG__longlong;
static int G__G__longlong_wAG__longlong_9_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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


/* G__ulonglong */
static int G__G__ulonglong_G__ulonglong_0_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__ulonglong *p=NULL;
   switch(libp->paran) {
   case 1:
      p = ::new((G__longif_tag*)G__getgvp()) G__ulonglong((long)G__int(libp->para[0]));
      break;
   case 0:
   if(G__getaryconstruct()) p=new G__ulonglong[G__getaryconstruct()];
   else p=::new((G__longif_tag*)G__getgvp()) G__ulonglong;
      break;
   }
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__ulonglong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_G__ulonglong_1_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   G__ulonglong *p=NULL;
      p=::new((G__longif_tag*)G__getgvp()) G__ulonglong(*(G__ulonglong*)libp->para[0].ref);
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__longifLN_G__ulonglong);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorsPlong_3_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,108,(long)((G__ulonglong*)(G__getstructoffset()))->operator long());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorsPint_4_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((G__ulonglong*)(G__getstructoffset()))->operator int());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorpLpL_5_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator++();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorpLpL_6_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=((G__ulonglong*)(G__getstructoffset()))->operator++((int)G__int(libp->para[0]));
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatormImI_7_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator--();
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatormImI_8_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=((G__ulonglong*)(G__getstructoffset()))->operator--((int)G__int(libp->para[0]));
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatoreQ_9_0(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator=((const long)G__int(libp->para[0]));
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatoreQ_0_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorpLeQ_1_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator+=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatormIeQ_2_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator-=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatormUeQ_3_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator*=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatoraNeQ_4_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator&=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatoroReQ_5_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator|=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorlElEeQ_6_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator<<=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__ulonglong_operatorgRgReQ_7_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const G__ulonglong& obj=((G__ulonglong*)(G__getstructoffset()))->operator>>=(*(G__ulonglong*)libp->para[0].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef G__ulonglong G__TG__ulonglong;
static int G__G__ulonglong_wAG__ulonglong_8_1(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
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


/* Setting up global function */
static int G___operatorpL_2_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator+(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatormI_3_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator-(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatormU_4_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator*(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatordI_5_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator/(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorpE_6_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator%(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoraN_7_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator&(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoroR_8_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator|(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_9_3(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator<<(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_0_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__longlong *pobj,xobj=operator>>(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
        pobj=new G__longlong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoraNaN_1_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator&&(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoroRoR_2_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator||(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlE_3_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgR_4_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlEeQ_5_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<=(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgReQ_6_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>=(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatornOeQ_7_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator!=(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoreQeQ_8_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator==(*(G__longlong*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_9_4(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_0_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(G__longlong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___G__ateval_1_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)G__ateval(*(G__longlong*)libp->para[0].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorpL_2_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator+(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatormI_3_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator-(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatormU_4_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator*(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatordI_5_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator/(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorpE_6_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator%(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoraN_7_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator&(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoroR_8_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator|(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_9_5(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator<<(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_0_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        G__ulonglong *pobj,xobj=operator>>(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
        pobj=new G__ulonglong(xobj);
        result7->obj.i=(long)((void*)pobj); result7->ref=result7->obj.i;
        G__store_tempobject(*result7);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoraNaN_1_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator&&(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoroRoR_2_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator||(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlE_3_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgR_4_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlEeQ_5_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator<=(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgReQ_6_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator>=(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatornOeQ_7_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator!=(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatoreQeQ_8_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)operator==(*(G__ulonglong*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorlElE_9_6(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const ostream& obj=operator<<(*(ostream*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___operatorgRgR_0_7(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      {
        const istream& obj=operator>>(*(istream*)libp->para[0].ref,*(G__ulonglong*)libp->para[1].ref);
         result7->ref=(long)(&obj); result7->obj.i=(long)(&obj);
      }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G___G__ateval_1_7(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)G__ateval(*(G__ulonglong*)libp->para[0].ref));
   return(1 || funcname || hash || result7 || libp) ;
}


/*********************************************************
* Member function Stub
*********************************************************/

/* G__longlong */

/* G__ulonglong */

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

   /* G__longlong */
static void G__setup_memvarG__longlong(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__longifLN_G__longlong));
   { G__longlong *p; p=(G__longlong*)0x1000; if (p) { }
   }
   G__tag_memvar_reset();
}


   /* G__ulonglong */
static void G__setup_memvarG__ulonglong(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__longifLN_G__ulonglong));
   { G__ulonglong *p; p=(G__ulonglong*)0x1000; if (p) { }
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
static void G__setup_memfuncG__longlong(void) {
   /* G__longlong */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__longifLN_G__longlong));
   G__memfunc_setup("G__longlong",1125,G__G__longlong_G__longlong_0_0,105,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"l - - 0 0 l",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__longlong",1125,G__G__longlong_G__longlong_1_0,105,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator long",1340,G__G__longlong_operatorsPlong_3_0,108,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator int",1239,G__G__longlong_operatorsPint_4_0,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator double",1543,G__G__longlong_operatorsPdouble_5_0,100,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__G__longlong_operatorpLpL_6_0,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__G__longlong_operatorpLpL_7_0,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"i - - 0 - dmy",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__G__longlong_operatormImI_8_0,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__G__longlong_operatormImI_9_0,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,1,1,1,0,"i - - 0 - dmy",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__G__longlong_operatoreQ_0_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"l - - 10 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__G__longlong_operatoreQ_1_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator+=",980,G__G__longlong_operatorpLeQ_2_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator-=",982,G__G__longlong_operatormIeQ_3_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator*=",979,G__G__longlong_operatormUeQ_4_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator&=",975,G__G__longlong_operatoraNeQ_5_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator|=",1061,G__G__longlong_operatoroReQ_6_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator<<=",1057,G__G__longlong_operatorlElEeQ_7_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator>>=",1061,G__G__longlong_operatorgRgReQ_8_1,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,1,1,1,1,0,"u 'G__longlong' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~G__longlong",1251,G__G__longlong_wAG__longlong_9_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__tag_memfunc_reset();
}

static void G__setup_memfuncG__ulonglong(void) {
   /* G__ulonglong */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__longifLN_G__ulonglong));
   G__memfunc_setup("G__ulonglong",1242,G__G__ulonglong_G__ulonglong_0_0,105,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"l - - 0 0 l",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("G__ulonglong",1242,G__G__ulonglong_G__ulonglong_1_0,105,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator long",1340,G__G__ulonglong_operatorsPlong_3_0,108,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator int",1239,G__G__ulonglong_operatorsPint_4_0,105,-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__G__ulonglong_operatorpLpL_5_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator++",962,G__G__ulonglong_operatorpLpL_6_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"i - - 0 - dmy",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__G__ulonglong_operatormImI_7_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator--",966,G__G__ulonglong_operatormImI_8_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,1,1,1,0,"i - - 0 - dmy",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__G__ulonglong_operatoreQ_9_0,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"l - - 10 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator=",937,G__G__ulonglong_operatoreQ_0_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator+=",980,G__G__ulonglong_operatorpLeQ_1_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator-=",982,G__G__ulonglong_operatormIeQ_2_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator*=",979,G__G__ulonglong_operatormUeQ_3_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator&=",975,G__G__ulonglong_operatoraNeQ_4_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator|=",1061,G__G__ulonglong_operatoroReQ_5_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator<<=",1057,G__G__ulonglong_operatorlElEeQ_6_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   G__memfunc_setup("operator>>=",1061,G__G__ulonglong_operatorgRgReQ_7_1,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,1,1,1,1,0,"u 'G__ulonglong' - 11 - x",(char*)NULL,(void*)NULL,0);
   // automatic destructor
   G__memfunc_setup("~G__ulonglong",1368,G__G__ulonglong_wAG__ulonglong_8_1,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL,(void*)NULL,0);
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

   G__memfunc_setup("operator+",919,G___operatorpL_2_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator-",921,G___operatormI_3_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator*",918,G___operatormU_4_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator/",923,G___operatordI_5_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator%",913,G___operatorpE_6_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&",914,G___operatoraN_7_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator|",1000,G___operatoroR_8_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_9_3,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_0_4,117,G__get_linked_tagnum(&G__longifLN_G__longlong),-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&&",952,G___operatoraNaN_1_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator||",1124,G___operatoroRoR_2_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<",936,G___operatorlE_3_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>",938,G___operatorgR_4_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<=",997,G___operatorlEeQ_5_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>=",999,G___operatorgReQ_6_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator!=",970,G___operatornOeQ_7_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator==",998,G___operatoreQeQ_8_4,105,-1,-1,0,2,1,1,0,
"u 'G__longlong' - 11 - a u 'G__longlong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_9_4,117,G__get_linked_tagnum(&G__longifLN_ostream),-1,1,2,1,1,0,
"u 'ostream' - 1 - ost u 'G__longlong' - 11 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_0_5,117,G__get_linked_tagnum(&G__longifLN_istream),-1,1,2,1,1,0,
"u 'istream' - 1 - ist u 'G__longlong' - 1 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("G__ateval",898,G___G__ateval_1_5,105,-1,-1,0,1,1,1,0,"u 'G__longlong' - 11 - a",(char*)NULL
#ifndef G__ateval
,(void*)(int (*)(const G__longlong&))G__ateval,0);
#else
,(void*)NULL,0);
#endif
   G__memfunc_setup("operator+",919,G___operatorpL_2_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator-",921,G___operatormI_3_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator*",918,G___operatormU_4_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator/",923,G___operatordI_5_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator%",913,G___operatorpE_6_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&",914,G___operatoraN_7_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator|",1000,G___operatoroR_8_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_9_5,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_0_6,117,G__get_linked_tagnum(&G__longifLN_G__ulonglong),-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator&&",952,G___operatoraNaN_1_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator||",1124,G___operatoroRoR_2_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<",936,G___operatorlE_3_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>",938,G___operatorgR_4_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<=",997,G___operatorlEeQ_5_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>=",999,G___operatorgReQ_6_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator!=",970,G___operatornOeQ_7_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator==",998,G___operatoreQeQ_8_6,105,-1,-1,0,2,1,1,0,
"u 'G__ulonglong' - 11 - a u 'G__ulonglong' - 11 - b",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator<<",996,G___operatorlElE_9_6,117,G__get_linked_tagnum(&G__longifLN_ostream),-1,1,2,1,1,0,
"u 'ostream' - 1 - ost u 'G__ulonglong' - 11 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("operator>>",1000,G___operatorgRgR_0_7,117,G__get_linked_tagnum(&G__longifLN_istream),-1,1,2,1,1,0,
"u 'istream' - 1 - ist u 'G__ulonglong' - 1 - a",(char*)NULL
,(void*)NULL,0);
   G__memfunc_setup("G__ateval",898,G___G__ateval_1_7,105,-1,-1,0,1,1,1,0,"u 'G__ulonglong' - 11 - a",(char*)NULL
#ifndef G__ateval
,(void*)(int (*)(const G__ulonglong&))G__ateval,0);
#else
,(void*)NULL,0);
#endif

   G__resetifuncposition();
}

extern "C" void G__cpp_setup_funclongif() {
  G__cpp_setup_func0();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__longifLN_ostream = { "ostream" , 99 , -1 };
G__linked_taginfo G__longifLN_istream = { "istream" , 99 , -1 };
G__linked_taginfo G__longifLN_G__longlong = { "G__longlong" , 99 , -1 };
G__linked_taginfo G__longifLN_G__ulonglong = { "G__ulonglong" , 99 , -1 };

/* Reset class/struct taginfo */
extern "C" void G__cpp_reset_tagtablelongif() {
  G__longifLN_ostream.tagnum = -1 ;
  G__longifLN_istream.tagnum = -1 ;
  G__longifLN_G__longlong.tagnum = -1 ;
  G__longifLN_G__ulonglong.tagnum = -1 ;
}


extern "C" void G__cpp_setup_tagtablelongif() {

   /* Setting up class,struct,union tag entry */
   G__tagtable_setup(G__get_linked_tagnum(&G__longifLN_G__longlong),sizeof(G__longlong),-1,3840,(char*)NULL,G__setup_memvarG__longlong,G__setup_memfuncG__longlong);
   G__tagtable_setup(G__get_linked_tagnum(&G__longifLN_G__ulonglong),sizeof(G__ulonglong),-1,3840,(char*)NULL,G__setup_memvarG__ulonglong,G__setup_memfuncG__ulonglong);
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
