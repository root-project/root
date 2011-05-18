/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file inherit.c
 ************************************************************************
 * Description:
 *  Class inheritance 
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

/**************************************************************************
* G__inheritclass
*
*  Recursively inherit base class
*
**************************************************************************/
void G__inheritclass(int to_tagnum,int from_tagnum,char baseaccess)
{
  int i,basen;
  long offset;
  struct G__inheritance *to_base,*from_base;
  int isvirtualbase;

  if(-1==to_tagnum || -1==from_tagnum) return;

  if(G__NOLINK==G__globalcomp && 
     G__CPPLINK==G__struct.iscpplink[from_tagnum] &&
     G__CPPLINK!=G__struct.iscpplink[to_tagnum]) {
    int warn = 1;
#ifdef G__ROOT
    if (!strcmp(G__fulltagname(from_tagnum,1), "TSelector")) warn = 0;
#endif
    if(
       G__dispmsg>=G__DISPSTRICT
       && warn) {
      G__fprinterr(G__serr,
                   "Warning: Interpreted class %s derived from"
                   ,G__fulltagname(to_tagnum,1));
      G__fprinterr(G__serr,
                   " precompiled class %s",G__fulltagname(from_tagnum,1));
      G__printlinenum();
      G__fprinterr(G__serr,"!!!There are some limitations regarding compiled/interpreted class inheritance\n");
    }
  }

  to_base = G__struct.baseclass[to_tagnum];
  from_base = G__struct.baseclass[from_tagnum];

  if(!to_base || !from_base) return;

  offset = to_base->herit[to_base->basen]->baseoffset; /* just to simplify */

  /****************************************************
  * copy virtual offset 
  ****************************************************/
  /* Bug fix for multiple inheritance, if virtual offset is already
   * set, don't overwrite.  */
  if(-1 != G__struct.virtual_offset[from_tagnum] &&
     -1 == G__struct.virtual_offset[to_tagnum]) {
#ifdef G__VIRTUALBASE
    if(to_base->herit[to_base->basen]->property&G__ISVIRTUALBASE) {
      G__struct.virtual_offset[to_tagnum] 
         = offset+G__struct.virtual_offset[from_tagnum]+G__DOUBLEALLOC;
    }
    else {
      G__struct.virtual_offset[to_tagnum] 
         = offset+G__struct.virtual_offset[from_tagnum];
    }
#else
    G__struct.virtual_offset[to_tagnum] 
      =offset+G__struct.virtual_offset[from_tagnum];
#endif
  }

  G__struct.isabstract[to_tagnum]+=G__struct.isabstract[from_tagnum];
  G__struct.funcs[to_tagnum] |= (G__struct.funcs[from_tagnum]&0xf0);

  /****************************************************
  *  copy grand base class info 
  ****************************************************/
  isvirtualbase = (to_base->herit[to_base->basen]->property&G__ISVIRTUALBASE); 
  if(to_base->herit[to_base->basen]->property&G__ISVIRTUALBASE) {
    isvirtualbase |= G__ISINDIRECTVIRTUALBASE;
  }
  basen=to_base->basen;
  for(i=0;i<from_base->basen;i++) {
    ++basen;
    to_base->herit[basen]->basetagnum = from_base->herit[i]->basetagnum;
    to_base->herit[basen]->baseoffset = offset+from_base->herit[i]->baseoffset;
    to_base->herit[basen]->property 
      = ((from_base->herit[i]->property&(G__ISVIRTUALBASE|G__ISINDIRECTVIRTUALBASE)) 
          | isvirtualbase);
    if(from_base->herit[i]->baseaccess>=G__PRIVATE) 
      to_base->herit[basen]->baseaccess=G__GRANDPRIVATE;
    else if(G__PRIVATE==baseaccess)
      to_base->herit[basen]->baseaccess=G__PRIVATE;
    else if(G__PROTECTED==baseaccess&&G__PUBLIC==from_base->herit[i]->baseaccess)
      to_base->herit[basen]->baseaccess=G__PROTECTED;
    else
      to_base->herit[basen]->baseaccess=from_base->herit[i]->baseaccess;
  }
  to_base->basen=basen+1;

}

/**************************************************************************
* G__baseconstructorwp
*
*  Read constructor arguments and
*  Recursively call base class constructor
*
**************************************************************************/
int G__baseconstructorwp()
{
  int c;
  G__FastAllocString buf(G__ONELINE);
  int n=0;
  struct G__baseparam *pbaseparamin = (struct G__baseparam*)NULL;
  struct G__baseparam *pbaseparam = pbaseparamin;
  
  /*  X::X(int a,int b) : base1(a), base2(b) { }
   *                   ^
   */
  c=G__fignorestream(":{");
  if(':'==c) c=',';
  
  while(','==c) {
    c=G__fgetstream_newtemplate(buf, 0, "({,"); /* case 3) */
    if('('==c) {
      if(pbaseparamin) {
        pbaseparam->next
          = (struct G__baseparam*)malloc(sizeof(struct G__baseparam));
        pbaseparam=pbaseparam->next;
      }
      else {
        pbaseparamin
          = (struct G__baseparam*)malloc(sizeof(struct G__baseparam));
        pbaseparam=pbaseparamin;
      }
      pbaseparam->next = (struct G__baseparam*)NULL;
      pbaseparam->name = (char*)NULL;
      pbaseparam->param = (char*)NULL;
      pbaseparam->name=(char*)malloc(strlen(buf)+1);
      strcpy(pbaseparam->name,buf); // Okay, we allocated enough space
      c=G__fgetstream_newtemplate(buf, 0, ")");
      pbaseparam->param=(char*)malloc(strlen(buf)+1);
      strcpy(pbaseparam->param,buf); // Okay, we allocated enough space
      ++n;
      c=G__fgetstream(buf, 0, ",{");
    }
  }
  
  G__baseconstructor(n,pbaseparamin);
  
  pbaseparam = pbaseparamin;
  while(pbaseparam) {
    struct G__baseparam *pb = pbaseparam->next;
    free((void*)pbaseparam->name);
    free((void*)pbaseparam->param);
    free((void*)pbaseparam);
    pbaseparam=pb;
  }
  
  fseek(G__ifile.fp,-1,SEEK_CUR);
  if(G__dispsource) G__disp_mask=1;
  return(0);
}

#ifdef G__VIRTUALBASE
/**************************************************************************
* struct and global object for virtual base class address list
**************************************************************************/
struct G__vbaseaddrlist {
  int tagnum;
  long vbaseaddr;
  struct G__vbaseaddrlist *next;
};

static struct G__vbaseaddrlist *G__pvbaseaddrlist 
  = (struct G__vbaseaddrlist*)NULL;
static int G__toplevelinstantiation=1;

/**************************************************************************
* G__storevbaseaddrlist()
**************************************************************************/
static struct G__vbaseaddrlist* G__storevbaseaddrlist()
{
  struct G__vbaseaddrlist *temp;
  temp = G__pvbaseaddrlist;
  G__pvbaseaddrlist = (struct G__vbaseaddrlist*)NULL;
  return(temp);
}

/**************************************************************************
* G__freevbaseaddrlist()
**************************************************************************/
static void G__freevbaseaddrlist(G__vbaseaddrlist *pvbaseaddrlist)
{
  if(pvbaseaddrlist) {
    if(pvbaseaddrlist->next) G__freevbaseaddrlist(pvbaseaddrlist->next);
    free((void*)pvbaseaddrlist);
  }
}

/**************************************************************************
* G__restorevbaseaddrlist()
**************************************************************************/
static void G__restorevbaseaddrlist(G__vbaseaddrlist *pvbaseaddrlist)
{
  G__freevbaseaddrlist(G__pvbaseaddrlist);
  G__pvbaseaddrlist = pvbaseaddrlist;
}

/**************************************************************************
* G__setvbaseaddrlist()
*
* class B : virtual public A { };
* class C : virtual public A { };
* class D : public B, public C { };
*
* ----AAAABBBB----aaaaCCCCDDDD
*   8          -x
*  vos
*
**************************************************************************/
static void G__setvbaseaddrlist(int tagnum,long pobject,long baseoffset)
{
  struct G__vbaseaddrlist *pvbaseaddrlist;
  struct G__vbaseaddrlist *last=(struct G__vbaseaddrlist*)NULL;
  long vbaseosaddr;
  vbaseosaddr = pobject+baseoffset;

  pvbaseaddrlist = G__pvbaseaddrlist;
  while(pvbaseaddrlist) {
    if(pvbaseaddrlist->tagnum == tagnum) {
      /* *(long*)vbaseosaddr = pvbaseaddrlist->vbaseaddr - pobject; */
      *(long*)vbaseosaddr = pvbaseaddrlist->vbaseaddr - vbaseosaddr;
      return;
    }
    last = pvbaseaddrlist;
    pvbaseaddrlist = pvbaseaddrlist->next;
  }
  if(last) {
    last->next
      = (struct G__vbaseaddrlist*)malloc(sizeof(struct G__vbaseaddrlist));
    pvbaseaddrlist = last->next;
  }
  else {
    G__pvbaseaddrlist
      = (struct G__vbaseaddrlist*)malloc(sizeof(struct G__vbaseaddrlist));
    pvbaseaddrlist = G__pvbaseaddrlist;
  }
  pvbaseaddrlist->tagnum = tagnum;
  pvbaseaddrlist->vbaseaddr = vbaseosaddr + G__DOUBLEALLOC;
  pvbaseaddrlist->next = (struct G__vbaseaddrlist*)NULL;
  /* *(long*)vbaseosaddr = pvbaseaddrlist->vbaseaddr - pobject ; */
  *(long*)vbaseosaddr = pvbaseaddrlist->vbaseaddr - vbaseosaddr ;
}
#endif

/**************************************************************************
* G__baseconstructor
*
*  Recursively call base class constructor
*
**************************************************************************/
int G__baseconstructor(int n, G__baseparam *pbaseparamin)
{
  struct G__var_array *mem;
  struct G__inheritance *baseclass;
  int store_tagnum;
  long store_struct_offset;
  int i;
  struct G__baseparam *pbaseparam = pbaseparamin;
  char *tagname,*memname;
  int flag;
  G__FastAllocString construct(G__ONELINE);
  int size;
  long store_globalvarpointer;
  int donen=0;
  long addr,lval;
  double dval;
  G__int64 llval;
  G__uint64 ullval;
  long double ldval;
#ifdef G__VIRTUALBASE
  int store_toplevelinstantiation;
  struct G__vbaseaddrlist *store_pvbaseaddrlist=NULL;
#endif
  
  /* store current tag information */
  store_tagnum=G__tagnum;
  store_struct_offset = G__store_struct_offset;
  store_globalvarpointer = G__globalvarpointer;
  
#ifdef G__VIRTUALBASE
  if(G__toplevelinstantiation) {
    store_pvbaseaddrlist = G__storevbaseaddrlist();
  }
  store_toplevelinstantiation=G__toplevelinstantiation;
  G__toplevelinstantiation=0;
#endif
  
  /****************************************************************
   * base classes
   ****************************************************************/
  if(-1==store_tagnum) return(0);
  baseclass=G__struct.baseclass[store_tagnum];
  if(!baseclass) return(0);
  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
      G__tagnum = baseclass->herit[i]->basetagnum;
#define G__OLDIMPLEMENTATION1606
#ifdef G__VIRTUALBASE
      if(baseclass->herit[i]->property&G__ISVIRTUALBASE) {
        long vbaseosaddr;
        vbaseosaddr = store_struct_offset+baseclass->herit[i]->baseoffset;
        G__setvbaseaddrlist(G__tagnum,store_struct_offset
                            ,baseclass->herit[i]->baseoffset);
        /*
        if(baseclass->herit[i]->baseoffset+G__DOUBLEALLOC==(*(long*)vbaseosaddr)) {
          G__store_struct_offset=store_struct_offset+(*(long*)vbaseosaddr);
        }
        */
        if(G__DOUBLEALLOC==(*(long*)vbaseosaddr)) {
          G__store_struct_offset=vbaseosaddr+(*(long*)vbaseosaddr);
        }
        else {
          G__store_struct_offset=vbaseosaddr+G__DOUBLEALLOC;
          if(-1 != G__struct.virtual_offset[G__tagnum]) {
            *(long*)(G__store_struct_offset+G__struct.virtual_offset[G__tagnum])
              = store_tagnum;
          }
          continue;
        }
      }
      else {
        G__store_struct_offset=store_struct_offset+baseclass->herit[i]->baseoffset;
      }
#else
      G__store_struct_offset=store_struct_offset+baseclass->herit[i]->baseoffset;
#endif

      /* search for constructor argument */ 
      flag=0;
      tagname = G__struct.name[G__tagnum];
      if(donen<n) {
        pbaseparam = pbaseparamin;
        while(pbaseparam) {
          if(strcmp(pbaseparam->name,tagname)==0) {
            flag=1;
            ++donen;
            break;
          }
          pbaseparam=pbaseparam->next;
        }
      }
      if (flag) {
         construct.Format("%s(%s)", tagname, pbaseparam->param);
      }
      else {
         construct.Format("%s()", tagname);
      }
      if (G__dispsource) {
         G__fprinterr(
              G__serr
            , "\n!!!Calling base class constructor %s  %s:%d\n"
            , construct()
            , __FILE__
            , __LINE__
         );
      }
      if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
         // c++ compiled class
         G__globalvarpointer = G__store_struct_offset;
      }
      else {
         G__globalvarpointer = G__PVOID;
      }
      {
         int tmp = 0;
         G__getfunction(construct, &tmp , G__TRYCONSTRUCTOR);
      }
      /* Set virtual_offset to every base classes for possible multiple
       * inheritance. */
      if(-1 != G__struct.virtual_offset[G__tagnum]) {
        *(long*)(G__store_struct_offset+G__struct.virtual_offset[G__tagnum])
          = store_tagnum;
      }
    } /* end of if ISDIRECTINHERIT */
    else { /* !ISDIREDCTINHERIT , bug fix for multiple inheritance */
      if(0==(baseclass->herit[i]->property&G__ISVIRTUALBASE)) {
        G__tagnum = baseclass->herit[i]->basetagnum;
        if(-1 != G__struct.virtual_offset[G__tagnum]) {
          G__store_struct_offset=store_struct_offset+baseclass->herit[i]->baseoffset;
          *(long*)(G__store_struct_offset+G__struct.virtual_offset[G__tagnum])
            = store_tagnum;
        }
      }
    }
  }
  G__globalvarpointer = store_globalvarpointer;

#ifdef G__VIRTUALBASE
  if(store_toplevelinstantiation) {
    G__restorevbaseaddrlist(store_pvbaseaddrlist);
  }
  G__toplevelinstantiation=1;
#endif
  
  /****************************************************************
   * class members
   ****************************************************************/
  G__incsetup_memvar(store_tagnum);
  mem=G__struct.memvar[store_tagnum];

  while(mem) {
    for(i=0;i<mem->allvar;i++) {
      if('u'==mem->type[i] && 
#ifndef G__NEWINHERIT
         0==mem->isinherit[i] &&
#endif
         'e'!=G__struct.type[mem->p_tagtable[i]] &&
         G__LOCALSTATIC!=mem->statictype[i]) {
        
        G__tagnum=mem->p_tagtable[i];
        G__store_struct_offset = store_struct_offset+mem->p[i];
        
        flag=0;
        memname=mem->varnamebuf[i];
        if(donen<n) {
          pbaseparam = pbaseparamin;
          while(pbaseparam) {
            if(strcmp(pbaseparam->name ,memname)==0) {
              flag=1;
              ++donen;
              break;
            }
            pbaseparam=pbaseparam->next;
          }
        }
        if(flag) {
          if(G__PARAREFERENCE==mem->reftype[i]) {
#ifndef G__OLDIMPLEMENTATION945
            if(G__NOLINK!=G__globalcomp) 
#endif
              {
            if(
               '\0'==pbaseparam->param[0]
               ) {
              G__fprinterr(G__serr,"Error: No initializer for reference %s "
                      ,memname);
              G__genericerror((char*)NULL);
            }
            else {
              G__genericerror("Limitation: initialization of reference member not implemented");
            }
              }
            continue;
          }
          construct.Format("%s(%s)" ,G__struct.name[G__tagnum]
                  ,pbaseparam->param);
        }
        else {
           construct.Format("%s()" ,G__struct.name[G__tagnum]);
          if(G__PARAREFERENCE==mem->reftype[i]) {
#ifndef G__OLDIMPLEMENTATION945
            if(G__NOLINK!=G__globalcomp) 
#endif
              {
            G__fprinterr(G__serr,"Error: No initializer for reference %s "
                    ,memname);
            G__genericerror((char*)NULL);
              }
            continue;
          }
        }
        if (G__dispsource) {
           G__fprinterr(
                G__serr
              , "\n!!!Calling class member constructor %s  %s:%d\n"
              , construct()
              , __FILE__
              , __LINE__
           );
        }
        long linear_index = mem->varlabel[i][1] /* number of elements */;
        if (linear_index) {
          --linear_index;
        }
        size = G__struct.size[G__tagnum];
        for (; linear_index >= 0; --linear_index) {
          if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
            // C++ compiled
            G__globalvarpointer = G__store_struct_offset;
          }
          else {
            G__globalvarpointer = G__PVOID;
          }
          int known = 0;
          G__getfunction(construct, &known, G__TRYCONSTRUCTOR);
          G__store_struct_offset += size;
        }
      }
      else if ((donen < n) && (mem->statictype[i] != G__LOCALSTATIC)) {
        flag = 0;
        memname = mem->varnamebuf[i];
        pbaseparam = pbaseparamin;
        while (pbaseparam) {
          if (!strcmp(pbaseparam->name, memname)) {
            flag = 1;
            ++donen;
            break;
          }
          pbaseparam = pbaseparam->next;
        }
        if (flag) {
          if(G__PARAREFERENCE==mem->reftype[i]) {
#ifndef G__OLDIMPLEMENTATION945
            if(G__NOLINK!=G__globalcomp) 
#endif
              {
            if(
               '\0'==pbaseparam->param[0]
               ) {
              G__fprinterr(G__serr,"Error: No initializer for reference %s "
                      ,memname);
              G__genericerror((char*)NULL);
            }
            else {
              G__genericerror("Limitation: initialization of reference member not implemented");
            }
              }
            continue;
          }
          else {
            long local_store_globalvarpointer = G__globalvarpointer;
            G__globalvarpointer = G__PVOID;
            addr = store_struct_offset+mem->p[i];
            if(isupper(mem->type[i])) {
              lval = G__int(G__getexpr(pbaseparam->param));
              *(long*)addr = lval;
            }
            else {
              switch(mem->type[i]) {
              case 'b':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(unsigned char*)addr = (unsigned char)lval;
                break;
              case 'c':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(char*)addr = (char)lval;
                break;
              case 'r':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(unsigned short*)addr = (unsigned short)lval;
                break;
              case 's':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(short*)addr = (short)lval;
                break;
              case 'h':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(unsigned int*)addr = lval;
                break;
              case 'i':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(int*)addr = (int)lval;
                break;
              case 'k':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(unsigned long*)addr = lval;
                break;
              case 'l':
                lval = G__int(G__getexpr(pbaseparam->param));
                *(long*)addr = lval;
                break;
              case 'f':
                dval = G__double(G__getexpr(pbaseparam->param));
                *(float*)addr = (float)dval;
                break;
              case 'd':
                dval = G__double(G__getexpr(pbaseparam->param));
                *(double*)addr = dval;
                break;
              case 'g':
                lval = G__int(G__getexpr(pbaseparam->param))?1:0;
#ifdef G__BOOL4BYTE
                *(int*)addr = (int)lval;
#else // G__BOOL4BYTE
                *(unsigned char*)addr = (unsigned char)lval;
#endif // G__BOOL4BYTE
                break;
              case 'n':
                llval = G__Longlong(G__getexpr(pbaseparam->param));
                *(G__int64*)addr = llval;
                break;
              case 'm':
                ullval = G__ULonglong(G__getexpr(pbaseparam->param));
                *(G__uint64*)addr = ullval;
                break;
              case 'q':
                ldval = G__Longdouble(G__getexpr(pbaseparam->param));
                *(long double*)addr = ldval;
                break;
              default:
                G__genericerror("Error: Illegal type in member initialization");
                break;
              }
            } /* if(isupper) else */
            G__globalvarpointer = local_store_globalvarpointer;
          } /* if(reftype) else */
        }
      }
    }
    mem = mem->next;
  }
  G__globalvarpointer = store_globalvarpointer;
#ifdef G__VIRTUALBASE
  G__toplevelinstantiation = store_toplevelinstantiation;
#endif
  G__tagnum = store_tagnum;
  G__store_struct_offset = store_struct_offset;
  // assign virtual_identity if contains virtual function.
  if ((G__struct.virtual_offset[G__tagnum] != -1) && !G__xrefflag) {
    *((long*) (G__store_struct_offset + G__struct.virtual_offset[G__tagnum])) = G__tagnum;
  }
  return 0;
}

/**************************************************************************
* G__basedestructor
*
*  Recursively call base class destructor
*
**************************************************************************/
int G__basedestructor()
{
  struct G__var_array *mem;
  struct G__inheritance *baseclass;
  int store_tagnum;
  long store_struct_offset;
  int i,j;
  G__FastAllocString destruct(G__ONELINE);
  long store_globalvarpointer;
  long store_addstros=0;

  /* store current tag information */
  store_tagnum=G__tagnum;
  store_struct_offset = G__store_struct_offset;
  store_globalvarpointer = G__globalvarpointer;
  
  /****************************************************************
   * class members
   ****************************************************************/
  G__incsetup_memvar(store_tagnum);
  mem=G__struct.memvar[store_tagnum];
  G__basedestructrc(mem);

  /****************************************************************
   * base classes
   ****************************************************************/
  baseclass=G__struct.baseclass[store_tagnum];
  for(i=baseclass->basen-1;i>=0;i--) {
    if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
      G__tagnum = baseclass->herit[i]->basetagnum;
#ifdef G__VIRTUALBASE
      if(baseclass->herit[i]->property&G__ISVIRTUALBASE) {
        long vbaseosaddr;
        vbaseosaddr = store_struct_offset+baseclass->herit[i]->baseoffset;
        /*
        if(baseclass->herit[i]->baseoffset+G__DOUBLEALLOC==(*(long*)vbaseosaddr)) {
          G__store_struct_offset=store_struct_offset+(*(long*)vbaseosaddr);
        }
        */
        if(G__DOUBLEALLOC==(*(long*)vbaseosaddr)) {
          G__store_struct_offset=vbaseosaddr+(*(long*)vbaseosaddr);
          if(G__asm_noverflow) {
            store_addstros=baseclass->herit[i]->baseoffset+(*(long*)vbaseosaddr);
          }
        }
        else {
          continue;
        }
      }
      else {
        G__store_struct_offset=store_struct_offset+baseclass->herit[i]->baseoffset;
        if(G__asm_noverflow) {
          store_addstros=baseclass->herit[i]->baseoffset;
        }
      }
#else
      G__store_struct_offset=store_struct_offset+baseclass->herit[i]->baseoffset;
#endif
      if(G__asm_noverflow) G__gen_addstros(store_addstros);
      /* avoid recursive and infinite virtual destructor call 
       * let the base class object pretend like its own class object */
      if(-1!=G__struct.virtual_offset[G__tagnum]) 
        *(long*)(G__store_struct_offset+G__struct.virtual_offset[G__tagnum])
          = G__tagnum;
      destruct.Format("~%s()",G__struct.name[G__tagnum]);
      if (G__dispsource) {
         G__fprinterr(
              G__serr
            , "\n!!!Calling base class destructor %s  %s:%d\n"
            , destruct()
            , __FILE__
            , __LINE__
         );
      }
      j=0;
      if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
        G__globalvarpointer = G__store_struct_offset;
      }
      else G__globalvarpointer = G__PVOID;
      G__getfunction(destruct,&j ,G__TRYDESTRUCTOR);
      if(G__asm_noverflow) G__gen_addstros(-store_addstros);
    }
  }
  G__globalvarpointer = store_globalvarpointer;

  /* finish up */
  G__tagnum = store_tagnum;
  G__store_struct_offset = store_struct_offset;
  return(0);
}

/**************************************************************************
* G__basedestructrc
*
*  calling desructors for member objects
**************************************************************************/
int G__basedestructrc(G__var_array *mem)
{
  G__FastAllocString destruct(G__ONELINE);
  if (!mem) {
    return 1;
  }
  long store_globalvarpointer = G__globalvarpointer;
  if (mem->next) {
    G__basedestructrc(mem->next);
  }
  // store current tag information
  long store_struct_offset = G__store_struct_offset;
  for (int i = mem->allvar - 1; i >= 0; --i) {
    if (
      (mem->type[i] == 'u') && 
#ifndef G__NEWINHERIT
      !mem->isinherit[i] &&
#endif
      (G__struct.type[mem->p_tagtable[i]] != 'e') &&
      (mem->statictype[i] != G__LOCALSTATIC) &&
      (G__PARAREFERENCE != mem->reftype[i])
    ) {
      G__tagnum = mem->p_tagtable[i];
      G__store_struct_offset = store_struct_offset + mem->p[i];
      destruct.Format("~%s()", G__struct.name[G__tagnum]);
      long linear_index = mem->varlabel[i][1] /* number of elements */;
      if (linear_index) {
        --linear_index;
      }
      int size = G__struct.size[G__tagnum];
      if (G__asm_noverflow) {
        G__gen_addstros(mem->p[i] + (linear_index * size));
      }
      G__store_struct_offset += linear_index * size;
      for (int known = 1; known && (linear_index >= 0); --linear_index) {
        if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
          // C++ compiled
          G__globalvarpointer = G__store_struct_offset;
        }
        else {
          G__globalvarpointer = G__PVOID;
        }
        // avoid recursive and infinite virtual destructor call
        if (G__struct.virtual_offset[G__tagnum] != -1) 
          *((long*) (G__store_struct_offset + G__struct.virtual_offset[G__tagnum])) = G__tagnum;
        if (G__dispsource) {
           G__fprinterr(
                G__serr
              , "\n!!!Calling class member destructor %s  %s:%d\n"
              , destruct()
              , __FILE__
              , __LINE__
           );
        }
        G__getfunction(destruct, &known, G__TRYDESTRUCTOR);
        G__store_struct_offset -= size;
        if (linear_index && G__asm_noverflow) {
          G__gen_addstros(-size);
        }
      }
      G__globalvarpointer = G__PVOID;
      if (G__asm_noverflow) {
        G__gen_addstros(-mem->p[i]);
      }
    }
    else if (
      (G__security & G__SECURE_GARBAGECOLLECTION) && 
      !G__no_exec_compile &&
      isupper(mem->type[i])
    ) {
      long linear_index = mem->varlabel[i][1] /* number of elements */;
      if (linear_index) {
        --linear_index;
      }
      long address = 0;
      for (; linear_index >= 0; --linear_index) {
        address = G__store_struct_offset + mem->p[i] + (linear_index * G__LONGALLOC);
        if (*((long*) address)) {
          G__del_refcount((void*) (*((long*) address)), (void**) address);
        }
      }
    }
  }
  G__globalvarpointer = store_globalvarpointer;
  G__store_struct_offset = store_struct_offset;
  return 0;
}


/**************************************************************************
* G__ispublicbase()
*
* check if derivedtagnum is derived from basetagnum. 
* If public base or reference from member function return offset
* else return -1
* Used in standard pointer conversion
**************************************************************************/
long G__ispublicbase(int basetagnum,int derivedtagnum
#ifdef G__VIRTUALBASE
                    ,long pobject
#endif
                    )
{
  struct G__inheritance *derived;
  int i,n;

  if(0>derivedtagnum) return(-1);
  if(basetagnum==derivedtagnum) return(0);
  derived = G__struct.baseclass[derivedtagnum];
  if(derived==0) return -1;

  n = derived->basen;

  for(i=0;i<n;i++) {
    if(basetagnum == derived->herit[i]->basetagnum) {
      if(derived->herit[i]->baseaccess==G__PUBLIC ||
         (G__exec_memberfunc && G__tagnum==derivedtagnum &&
          G__GRANDPRIVATE!=derived->herit[i]->baseaccess)) {
#ifdef G__VIRTUALBASE
        if(derived->herit[i]->property&G__ISVIRTUALBASE) {
          return(G__getvirtualbaseoffset(pobject,derivedtagnum,derived,i));
        }
        else {
          return(derived->herit[i]->baseoffset);
        }
#else
        return(derived->herit[i]->baseoffset);
#endif
      }
    }
  }

  return(-1);
}

/**************************************************************************
* G__isanybase()
*
* check if derivedtagnum is derived from basetagnum. If true return offset
* to the base object. If faulse, return -1.
* Used in cast operatotion
**************************************************************************/
long G__isanybase(int basetagnum,int derivedtagnum
#ifdef G__VIRTUALBASE
                    ,long pobject
#endif
                 )
{
  struct G__inheritance *derived;
  int i,n;

  if (0 > derivedtagnum) {
    for (i = 0; i < G__globalusingnamespace.basen; i++) {
      if (G__globalusingnamespace.herit[i]->basetagnum == basetagnum)
        return 0;
    }
    return -1;
  }
  if(basetagnum==derivedtagnum) return(0);
  derived = G__struct.baseclass[derivedtagnum];
  n = derived ? derived->basen : -1;

  for(i=0;i<n;i++) {
    if(basetagnum == derived->herit[i]->basetagnum) {
#ifdef G__VIRTUALBASE
      if(derived->herit[i]->property&G__ISVIRTUALBASE) {
        return(G__getvirtualbaseoffset(pobject,derivedtagnum,derived,i));
      }
      else {
        return(derived->herit[i]->baseoffset);
      }
#else
      return(derived->herit[i]->baseoffset);
#endif
    }
  }

  return(-1);
}


/**************************************************************************
* G__find_virtualoffset()
*
*  Used in G__interpret_func to subtract offset for calling virtual function
*
**************************************************************************/
long G__find_virtualoffset(long virtualtag
#ifdef G__VIRTUALBASE
                          , long pobject
#endif
)

{
  int i;
  struct G__inheritance *baseclass;
  
  if(0>virtualtag) return(0);
  baseclass = G__struct.baseclass[virtualtag];
  for(i=0;i<baseclass->basen;i++) {
    if(G__tagnum==baseclass->herit[i]->basetagnum) {
      if(baseclass->herit[i]->property&G__ISVIRTUALBASE) {
#ifdef G__VIRTUALBASE
         if(G__CPPLINK==G__struct.iscpplink[virtualtag]) {
            long (*f) G__P((long));
            f = (long (*) G__P((long)))(baseclass->herit[i]->baseoffset);
            return((*f)(pobject));
         }
#endif
        return(baseclass->herit[i]->baseoffset+G__DOUBLEALLOC);
      }
      else {
        return(baseclass->herit[i]->baseoffset);
      }
    }
  }
  return(0);
}

#ifdef G__VIRTUALBASE
/**************************************************************************
* G__getvirtualbaseoffset()
**************************************************************************/
long G__getvirtualbaseoffset(long pobject,int tagnum
                             ,G__inheritance *baseclass,int basen)
{
  long (*f) G__P((long));
  if(pobject==G__STATICRESOLUTION) return(0);
  if(!pobject || G__no_exec_compile
     || -1==pobject || 1==pobject
     ) {
    if(!G__cintv6) G__abortbytecode();
    return(0);
  }
  if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
    f = (long (*) G__P((long)))(baseclass->herit[basen]->baseoffset);
    return((*f)(pobject));
  }
  else {
    /* return((*(long*)(pobject+baseclass->baseoffset[basen]))); */
    return(baseclass->herit[basen]->baseoffset
           +(*(long*)(pobject+baseclass->herit[basen]->baseoffset)));
  }
}
#endif

/***********************************************************************
* G__publicinheritance()
***********************************************************************/
long G__publicinheritance(G__value *val1,G__value *val2)
{
  long lresult;
  if('U'==val1->type && 'U'==val2->type) {
    if(-1!=(lresult=G__ispublicbase(val1->tagnum,val2->tagnum,val2->obj.i))) {
      val2->tagnum = val1->tagnum;
      val2->obj.i += lresult;
      return(lresult);
    }
    else if(-1!=(lresult=G__ispublicbase(val2->tagnum,val1->tagnum
                                         ,val1->obj.i))) {
      val1->tagnum = val2->tagnum;
      val1->obj.i += lresult;
      return(-lresult);
    }
  }
  return 0;
}

} /* extern "C" */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
