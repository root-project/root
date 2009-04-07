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
#include "Dict.h"

using namespace Cint::Internal;

/**************************************************************************
* G__inheritclass
*
*  Recursively inherit base class
*
**************************************************************************/
void Cint::Internal::G__inheritclass(int to_tagnum, int from_tagnum, char baseaccess)
{
   size_t i;
   char *offset;
   struct G__inheritance *to_base, *from_base;
   int isvirtualbase;

   if (-1 == to_tagnum || -1 == from_tagnum) return;

   if (G__NOLINK == G__globalcomp &&
         G__CPPLINK == G__struct.iscpplink[from_tagnum] &&
         G__CPPLINK != G__struct.iscpplink[to_tagnum]) {
      int warn = 1;
#ifdef G__ROOT
      if (!strcmp(G__fulltagname(from_tagnum, 1), "TSelector")) warn = 0;
#endif
      if (
         G__dispmsg >= G__DISPSTRICT
         && warn) {
         G__fprinterr(G__serr,
                      "Warning: Interpreted class %s derived from"
                      , G__fulltagname(to_tagnum, 1));
         G__fprinterr(G__serr,
                      " precompiled class %s", G__fulltagname(from_tagnum, 1));
         G__printlinenum();
         G__fprinterr(G__serr, "!!!There are some limitations regarding compiled/interpreted class inheritance\n");
      }
   }

   to_base = G__struct.baseclass[to_tagnum];
   from_base = G__struct.baseclass[from_tagnum];

   if (!to_base || !from_base) return;

   size_t idx = to_base->vec.size() - 1;
   offset = to_base->vec[idx].baseoffset; /* just to simplify */

   /****************************************************
   * copy virtual offset
   ****************************************************/
   /* Bug fix for multiple inheritance, if virtual offset is already
    * set, don't overwrite.  */
   if (G__PVOID != G__struct.virtual_offset[from_tagnum] &&
         G__PVOID == G__struct.virtual_offset[to_tagnum]) {
#ifdef G__VIRTUALBASE
      if (to_base->vec[idx].property&G__ISVIRTUALBASE) {
         G__struct.virtual_offset[to_tagnum]
         = offset + (size_t)G__struct.virtual_offset[from_tagnum] + G__DOUBLEALLOC;
      }
      else {
         G__struct.virtual_offset[to_tagnum]
         = offset + (size_t)G__struct.virtual_offset[from_tagnum];
      }
#else
      G__struct.virtual_offset[to_tagnum]
      = offset + (size_t)G__struct.virtual_offset[from_tagnum];
#endif
   }

   //fprintf(stderr, "G__inheritclass: Incrementing abstract count for '%s' by: %d because of '%s'\n", G__Dict::GetDict().GetScope(to_tagnum).Name(Reflex::SCOPED).c_str(), G__struct.isabstract[from_tagnum], G__Dict::GetDict().GetScope(from_tagnum).Name(Reflex::SCOPED).c_str());
   G__struct.isabstract[to_tagnum] += G__struct.isabstract[from_tagnum];
   G__struct.funcs[to_tagnum] |= (G__struct.funcs[from_tagnum] & 0xf0);

   /****************************************************
   *  copy grand base class info
   ****************************************************/
   isvirtualbase = (to_base->vec[idx].property & G__ISVIRTUALBASE);
   if (to_base->vec[idx].property&G__ISVIRTUALBASE) {
      isvirtualbase |= G__ISINDIRECTVIRTUALBASE;
   }
   for (i = 0;i < from_base->vec.size();i++) {
      G__inheritance::G__Entry entry(from_base->vec[i].basetagnum,
                                     offset + (size_t)from_base->vec[i].baseoffset,
                                     G__PUBLIC,
                                     ((from_base->vec[i].property & (G__ISVIRTUALBASE | G__ISINDIRECTVIRTUALBASE)) | isvirtualbase));
      if (from_base->vec[i].baseaccess >= G__PRIVATE)
         entry.baseaccess = G__GRANDPRIVATE;
      else if (G__PRIVATE == baseaccess)
         entry.baseaccess = G__PRIVATE;
      else if (G__PROTECTED == baseaccess && G__PUBLIC == from_base->vec[i].baseaccess)
         entry.baseaccess = G__PROTECTED;
      else
         entry.baseaccess = from_base->vec[i].baseaccess;
      to_base->vec.push_back(entry);
   }
}

/**************************************************************************
* G__baseconstructorwp
*
*  Read constructor arguments and
*  Recursively call base class constructor
*
**************************************************************************/
int Cint::Internal::G__baseconstructorwp()
{
   int c;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   int n = 0;
   struct G__baseparam *pbaseparamin = (struct G__baseparam*)NULL;
   struct G__baseparam *pbaseparam = pbaseparamin;

   /*  X::X(int a,int b) : base1(a), base2(b) { }
    *                   ^
    */
   c = G__fignorestream(":{");
   if (':' == c) c = ',';

   while (',' == c) {
      c = G__fgetstream_newtemplate(buf, "({,"); /* case 3) */
      if ('(' == c) {
         if (pbaseparamin) {
            pbaseparam->next
            = (struct G__baseparam*)malloc(sizeof(struct G__baseparam));
            pbaseparam = pbaseparam->next;
         }
         else {
            pbaseparamin
            = (struct G__baseparam*)malloc(sizeof(struct G__baseparam));
            pbaseparam = pbaseparamin;
         }
         pbaseparam->next = (struct G__baseparam*)NULL;
         pbaseparam->name = (char*)NULL;
         pbaseparam->param = (char*)NULL;
         pbaseparam->name = (char*)malloc(strlen(buf) + 1);
         strcpy(pbaseparam->name, buf);
         c = G__fgetstream_newtemplate(buf, ")");
         pbaseparam->param = (char*)malloc(strlen(buf) + 1);
         strcpy(pbaseparam->param, buf);
         ++n;
         c = G__fgetstream(buf, ",{");
      }
   }

   G__baseconstructor(n, pbaseparamin);

   pbaseparam = pbaseparamin;
   while (pbaseparam) {
      struct G__baseparam *pb = pbaseparam->next;
      free((void*)pbaseparam->name);
      free((void*)pbaseparam->param);
      free((void*)pbaseparam);
      pbaseparam = pb;
   }

   fseek(G__ifile.fp, -1, SEEK_CUR);
   if (G__dispsource) G__disp_mask = 1;
   return(0);
}

#ifdef G__VIRTUALBASE
/**************************************************************************
* struct and global object for virtual base class address list
**************************************************************************/
struct G__vbaseaddrlist
{
   int tagnum;
   long vbaseaddr;
   struct G__vbaseaddrlist *next;
};

static struct G__vbaseaddrlist* G__pvbaseaddrlist = 0;
static int G__toplevelinstantiation = 1;

/**************************************************************************
* G__storevbaseaddrlist()
**************************************************************************/
static G__vbaseaddrlist* G__storevbaseaddrlist()
{
   G__vbaseaddrlist* temp = G__pvbaseaddrlist;
   G__pvbaseaddrlist = 0;
   return temp;
}

/**************************************************************************
* G__freevbaseaddrlist()
**************************************************************************/
static void G__freevbaseaddrlist(G__vbaseaddrlist* pvbaseaddrlist)
{
   if (pvbaseaddrlist) {
      if (pvbaseaddrlist->next) G__freevbaseaddrlist(pvbaseaddrlist->next);
      free((void*)pvbaseaddrlist);
   }
}

/**************************************************************************
* G__restorevbaseaddrlist()
**************************************************************************/
static void G__restorevbaseaddrlist(G__vbaseaddrlist* pvbaseaddrlist)
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
static void G__setvbaseaddrlist(int tagnum, char* pobject, char* baseoffset)
{
   G__vbaseaddrlist* last = 0;
   char* vbaseosaddr = pobject + (size_t) baseoffset;
   G__vbaseaddrlist* pvbaseaddrlist = G__pvbaseaddrlist;
   while (pvbaseaddrlist) {
      if (pvbaseaddrlist->tagnum == tagnum) {
         *(long*)vbaseosaddr = pvbaseaddrlist->vbaseaddr - (size_t)vbaseosaddr;
         return;
      }
      last = pvbaseaddrlist;
      pvbaseaddrlist = pvbaseaddrlist->next;
   }
   if (last) {
      last->next = (G__vbaseaddrlist*) malloc(sizeof(G__vbaseaddrlist));
      pvbaseaddrlist = last->next;
   }
   else {
      G__pvbaseaddrlist = (G__vbaseaddrlist*) malloc(sizeof(G__vbaseaddrlist));
      pvbaseaddrlist = G__pvbaseaddrlist;
   }
   pvbaseaddrlist->tagnum = tagnum;
   pvbaseaddrlist->vbaseaddr = (long)(vbaseosaddr + G__DOUBLEALLOC);
   pvbaseaddrlist->next = 0;
   *(long*)vbaseosaddr = (long)(pvbaseaddrlist->vbaseaddr - (size_t)vbaseosaddr);
}
#endif

/**************************************************************************
* G__baseconstructor
*
*  Recursively call base class constructor
*
**************************************************************************/
int Cint::Internal::G__baseconstructor(int n, G__baseparam* pbaseparamin)
{
   struct G__inheritance *baseclass;
   ::Reflex::Scope store_tagnum = G__tagnum;
   char *store_struct_offset;
   struct G__baseparam *pbaseparam = pbaseparamin;
   char *tagname;
   int flag;
   G__StrBuf construct_sb(G__ONELINE);
   char *construct = construct_sb;
   int p_inc, size;
   char *store_globalvarpointer;
   int donen = 0;
   char *addr;
   long lval;
   double dval;
   G__int64 llval;
   G__uint64 ullval;
   long double ldval;
#ifdef G__VIRTUALBASE
   int store_toplevelinstantiation;
   struct G__vbaseaddrlist *store_pvbaseaddrlist = NULL;
#endif

   /* store current tag information */
   store_struct_offset = G__store_struct_offset;
   store_globalvarpointer = G__globalvarpointer;

#ifdef G__VIRTUALBASE
   if (G__toplevelinstantiation) {
      store_pvbaseaddrlist = G__storevbaseaddrlist();
   }
   store_toplevelinstantiation = G__toplevelinstantiation;
   G__toplevelinstantiation = 0;
#endif

   /****************************************************************
    * base classes
    ****************************************************************/
   if (!store_tagnum) return(0);
   baseclass = G__struct.baseclass[G__get_tagnum(store_tagnum)];
   if (!baseclass) return(0);
   for (size_t bi = 0; bi < baseclass->vec.size(); ++bi) {
      if (baseclass->vec[bi].property&G__ISDIRECTINHERIT) {
         G__tagnum = G__Dict::GetDict().GetScope(baseclass->vec[bi].basetagnum);
#define G__OLDIMPLEMENTATION1606
#ifdef G__VIRTUALBASE
         if (baseclass->vec[bi].property&G__ISVIRTUALBASE) {
            char *vbaseosaddr;
            vbaseosaddr = store_struct_offset + (size_t)baseclass->vec[bi].baseoffset;
            G__setvbaseaddrlist(G__get_tagnum(G__tagnum), store_struct_offset
                                , baseclass->vec[bi].baseoffset);
            /*
            if(baseclass->vec[bi].baseoffset+G__DOUBLEALLOC==(*(long*)vbaseosaddr)) {
              G__store_struct_offset=store_struct_offset+(*(long*)vbaseosaddr);
            }
            */
            if (G__DOUBLEALLOC == (*(long*)vbaseosaddr)) {
               G__store_struct_offset = vbaseosaddr + (*(long*)vbaseosaddr);
            }
            else {
               G__store_struct_offset = vbaseosaddr + G__DOUBLEALLOC;
               if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)]) {
                  *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
                  = G__get_tagnum(store_tagnum);
               }
               continue;
            }
         }
         else {
            G__store_struct_offset = store_struct_offset + (size_t)baseclass->vec[bi].baseoffset;
         }
#else
         G__store_struct_offset = store_struct_offset + baseclass->vec[bi].baseoffset;
#endif

         /* search for constructor argument */
         flag = 0;
         tagname = G__struct.name[G__get_tagnum(G__tagnum)];
         if (donen < n) {
            pbaseparam = pbaseparamin;
            while (pbaseparam) {
               if (strcmp(pbaseparam->name, tagname) == 0) {
                  flag = 1;
                  ++donen;
                  break;
               }
               pbaseparam = pbaseparam->next;
            }
         }
         if (flag) sprintf(construct, "%s(%s)" , tagname, pbaseparam->param);
         else sprintf(construct, "%s()", tagname);
         if (G__dispsource) {
            G__fprinterr(G__serr, "\n!!!Calling base class constructor %s", construct);
         }
         if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(G__tagnum)]) { /* C++ compiled class */
            G__globalvarpointer = G__store_struct_offset;
         }
         else G__globalvarpointer = G__PVOID;
         {
            int tmp = 0;
            G__getfunction(construct, &tmp , G__TRYCONSTRUCTOR);
         }
         /* Set virtual_offset to every base classes for possible multiple
          * inheritance. */
         if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)]) {
            *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
            = G__get_tagnum(store_tagnum);
         }
      } /* end of if ISDIRECTINHERIT */
      else { /* !ISDIREDCTINHERIT , bug fix for multiple inheritance */
         if (0 == (baseclass->vec[bi].property&G__ISVIRTUALBASE)) {
            G__tagnum = G__Dict::GetDict().GetScope(baseclass->vec[bi].basetagnum);
            if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)]) {
               G__store_struct_offset = store_struct_offset + (size_t)baseclass->vec[bi].baseoffset;
               *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
               = G__get_tagnum(store_tagnum);
            }
         }
      }
   }
   G__globalvarpointer = store_globalvarpointer;

#ifdef G__VIRTUALBASE
   if (store_toplevelinstantiation) {
      G__restorevbaseaddrlist(store_pvbaseaddrlist);
   }
   G__toplevelinstantiation = 1;
#endif

   /****************************************************************
    * class members
    ****************************************************************/
   G__incsetup_memvar((store_tagnum));

   for (unsigned int di = 0; di < store_tagnum.DataMemberSize(); ++di) {
      ::Reflex::Member mem = store_tagnum.DataMemberAt(di);
      if (
         (G__get_type(mem.TypeOf()) == 'u') &&
#ifndef G__NEWINHERIT
         !mem->isinherit[di] &&
#endif
         !mem.TypeOf().RawType().IsEnum() &&
         (G__get_properties(mem)->statictype != G__LOCALSTATIC)
      ) {

         G__set_G__tagnum(mem.TypeOf().RawType());
         G__store_struct_offset = store_struct_offset + (size_t)G__get_offset(mem);

         flag = 0;
         if (donen < n) {
            pbaseparam = pbaseparamin;
            while (pbaseparam) {
               if (mem.Name() == pbaseparam->name) {
                  flag = 1;
                  ++donen;
                  break;
               }
               pbaseparam = pbaseparam->next;
            }
         }
         if (flag) {
            if (mem.TypeOf().FinalType().IsReference()) {
#ifndef G__OLDIMPLEMENTATION945
               if (G__NOLINK != G__globalcomp)
#endif
               {
                  if (
                     '\0' == pbaseparam->param[0]
                  ) {
                     G__fprinterr(G__serr, "Error: No initializer for reference %s "
                     , mem.Name().c_str());
                     G__genericerror((char*)NULL);
                  }
                  else {
                     G__genericerror("Limitation: initialization of reference member not implemented");
                  }
               }
               continue;
            }
            sprintf(construct, "%s(%s)" , G__tagnum.Name().c_str()
                    , pbaseparam->param);
         }
         else {
            sprintf(construct, "%s()" , G__tagnum.Name().c_str());
            if (mem.TypeOf().FinalType().IsReference()) {
#ifndef G__OLDIMPLEMENTATION945
               if (G__NOLINK != G__globalcomp)
#endif
               {
                  G__fprinterr(G__serr, "Error: No initializer for reference %s "
                  , mem.Name().c_str());
                  G__genericerror((char*)NULL);
               }
               continue;
            }
         }
         if (G__dispsource) {
            G__fprinterr(G__serr, "\n!!!Calling class member constructor %s", construct);
         }
         p_inc = G__get_varlabel(mem, 1);
         if (p_inc) {
            --p_inc;
         }
         size = G__struct.size[G__get_tagnum(G__tagnum)];
         do {
            if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(G__tagnum)]) { /* C++ compiled */
               G__globalvarpointer = G__store_struct_offset;
            }
            else G__globalvarpointer = G__PVOID;
            {
               int tmp = 0;
               G__getfunction(construct, &tmp , G__TRYCONSTRUCTOR);
            }
            G__store_struct_offset += size;
            --p_inc;
         }
         while (p_inc >= 0) ;
      } /* if('u') */

      else if (donen < n && (G__get_properties(mem)->statictype != G__LOCALSTATIC)) {
         flag = 0;
         pbaseparam = pbaseparamin;
         while (pbaseparam) {
            if (mem.Name() == pbaseparam->name) {
               flag = 1;
               ++donen;
               break;
            }
            pbaseparam = pbaseparam->next;
         }
         if (flag) {
            if (mem.TypeOf().FinalType().IsReference()) {
#ifndef G__OLDIMPLEMENTATION945
               if (G__NOLINK != G__globalcomp)
#endif
               {
                  if (
                     '\0' == pbaseparam->param[0]
                  ) {
                     G__fprinterr(G__serr, "Error: No initializer for reference %s "
                     , mem.Name().c_str());
                     G__genericerror((char*)NULL);
                  }
                  else {
                     G__genericerror("Limitation: initialization of reference member not implemented");
                  }
               }
               continue;
            }
            else {
               char *local_store_globalvarpointer = G__globalvarpointer;
               G__globalvarpointer = G__PVOID;
               addr = store_struct_offset + (size_t)G__get_offset(mem);
               char type = G__get_type(mem.TypeOf());
               if (isupper(type)) {
                  lval = G__int(G__getexpr(pbaseparam->param));
                  *(char**)addr = (char*)lval;
               }
               else {
                  switch (type) {
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
                        *(int*)addr = lval;
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
                        lval = G__int(G__getexpr(pbaseparam->param)) ? 1 : 0;
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
         } /* if(flag) */
      } /* else if(!LOCALSTATIC) */

   } /* for(di) */
   G__globalvarpointer = store_globalvarpointer;
#ifdef G__VIRTUALBASE
   G__toplevelinstantiation = store_toplevelinstantiation;
#endif

   /* restore derived tagnum */
   G__tagnum = store_tagnum;
   G__store_struct_offset = store_struct_offset;

   /* assign virtual_identity if contains virtual
   * function.  */
   if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)]
         /* && 0==G__no_exec_compile  << this one is correct */
         && 0 == G__xrefflag
      ) {
      *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
      = G__get_tagnum(G__tagnum);
   }
   return(0);
}

/**************************************************************************
* G__basedestructor
*
*  Recursively call base class destructor
*
**************************************************************************/
int Cint::Internal::G__basedestructor()
{
   struct G__inheritance *baseclass;
   ::Reflex::Scope store_tagnum = G__tagnum;
   char *store_struct_offset;
   int i, j;
   G__StrBuf destruct_sb(G__ONELINE);
   char *destruct = destruct_sb;
   char *store_globalvarpointer;
   char *store_addstros = 0;

   /* store current tag information */
   store_struct_offset = G__store_struct_offset;
   store_globalvarpointer = G__globalvarpointer;

   /****************************************************************
    * class members
    ****************************************************************/
   G__incsetup_memvar((store_tagnum));
   G__basedestructrc(store_tagnum);

   /****************************************************************
    * base classes
    ****************************************************************/
   baseclass = G__struct.baseclass[G__get_tagnum(store_tagnum)];
   for (i = baseclass->vec.size() - 1;i >= 0;i--) {
      if (baseclass->vec[i].property&G__ISDIRECTINHERIT) {
         G__tagnum = G__Dict::GetDict().GetScope(baseclass->vec[i].basetagnum);
#ifdef G__VIRTUALBASE
         if (baseclass->vec[i].property&G__ISVIRTUALBASE) {
            long vbaseosaddr;
            vbaseosaddr = (long)(store_struct_offset + (size_t)baseclass->vec[i].baseoffset);
            /*
            if(baseclass->vec[i].baseoffset+G__DOUBLEALLOC==(*(long*)vbaseosaddr)) {
              G__store_struct_offset=store_struct_offset+(*(long*)vbaseosaddr);
            }
            */
            if (G__DOUBLEALLOC == (*(long*)vbaseosaddr)) {
               G__store_struct_offset = (char*)(vbaseosaddr + (*(long*)vbaseosaddr));
               if (G__asm_noverflow) {
                  store_addstros = baseclass->vec[i].baseoffset + (*(long*)vbaseosaddr);
               }
            }
            else {
               continue;
            }
         }
         else {
            G__store_struct_offset = store_struct_offset + (size_t)baseclass->vec[i].baseoffset;
            if (G__asm_noverflow) {
               store_addstros = baseclass->vec[i].baseoffset;
            }
         }
#else
         G__store_struct_offset = store_struct_offset + baseclass->vec[i].baseoffset;
#endif
         if (G__asm_noverflow) G__gen_addstros((long)store_addstros);
         /* avoid recursive and infinite virtual destructor call
          * let the base class object pretend like its own class object */
         if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
            *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
            = G__get_tagnum(G__tagnum);
         sprintf(destruct, "~%s()", G__struct.name[G__get_tagnum(G__tagnum)]);
         if (G__dispsource)
            G__fprinterr(G__serr, "\n!!!Calling base class destructor %s", destruct);
         j = 0;
         if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
            G__globalvarpointer = G__store_struct_offset;
         }
         else G__globalvarpointer = G__PVOID;
         G__getfunction(destruct, &j , G__TRYDESTRUCTOR);
         if (G__asm_noverflow) G__gen_addstros(-(long)store_addstros);
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
int Cint::Internal::G__basedestructrc(const ::Reflex::Type& type)
{
   /* int store_tagnum; */
   char *store_struct_offset;
   int j;
   G__StrBuf destruct_sb(G__ONELINE);
   char *destruct = destruct_sb;
   int p_inc, size;
   char *store_globalvarpointer;
   char *address;

   if (!type) return(1);

   store_globalvarpointer = G__globalvarpointer;

   /* store current tag information */
   /* store_tagnum=G__tagnum; */
   store_struct_offset = G__store_struct_offset;

   for (::Reflex::Reverse_Member_Iterator mem = type.DataMember_RBegin();
         mem != type.DataMember_REnd();
         ++mem) {
      if ('u' == G__get_type(mem->TypeOf()) &&
#ifndef G__NEWINHERIT
            0 == mem->isinherit[i] &&
#endif
            !mem->TypeOf().RawType().IsEnum() &&
            (G__get_properties(*mem)->statictype != G__LOCALSTATIC) &&
            !mem->TypeOf().FinalType().IsReference()
         ) {

         G__set_G__tagnum(mem->TypeOf().RawType());
         G__store_struct_offset = store_struct_offset + (size_t)G__get_offset(*mem);
         sprintf(destruct, "~%s()", G__tagnum.Name().c_str());
         p_inc = G__get_varlabel(*mem, 1);
         if (p_inc) {
            --p_inc;
         }
         size = G__struct.size[G__get_tagnum(G__tagnum)];
         if (G__asm_noverflow) {
            if (0 == p_inc) G__gen_addstros((size_t)G__get_offset(*mem));
            else         G__gen_addstros((size_t)G__get_offset(*mem) + size*p_inc);
         }

         j = 0;
         G__store_struct_offset += size * p_inc;
         do {
            if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(G__tagnum)]) { /* C++ compiled */
               G__globalvarpointer = G__store_struct_offset;
            }
            else G__globalvarpointer = G__PVOID;
            /* avoid recursive and infinite virtual destructor call */
            if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
               *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
               = G__get_tagnum(G__tagnum);
            if (G__dispsource) {
               G__fprinterr(G__serr, "\n!!!Calling class member destructor %s" , destruct);
            }
            G__getfunction(destruct, &j, G__TRYDESTRUCTOR);
            G__store_struct_offset -= size;
            if (p_inc && G__asm_noverflow) G__gen_addstros(-size);
            --p_inc;
         }
         while (p_inc >= 0 && j) ;
         G__globalvarpointer = G__PVOID;
         if (G__asm_noverflow) G__gen_addstros(-(long)G__get_offset(*mem));
      }
      else if (G__security&G__SECURE_GARBAGECOLLECTION &&
               (!G__no_exec_compile) &&
               isupper(G__get_type(mem->TypeOf()))) {
         j = G__get_varlabel(*mem, 1);
         do {
            --j;
            address = G__store_struct_offset + (size_t)G__get_offset(*mem) + G__LONGALLOC * j;
            if (*(long*)address) {
               G__del_refcount((void*)(*((long*)address)) , (void**)address);
            }
         }
         while (j);
      }
   }
   G__globalvarpointer = store_globalvarpointer;
   G__store_struct_offset = store_struct_offset;
   return(0);
}

/**************************************************************************
* G__ispublicbase()
*
* check if derivedtagnum is derived from basetagnum.
* If public base or reference from member function return offset
* else return -1
* Used in standard pointer conversion
**************************************************************************/
long Cint::Internal::G__ispublicbase(const ::Reflex::Type& basetagnum, const ::Reflex::Type& derivedtagnum, void* pobject)
{
   return G__ispublicbase(G__get_tagnum(basetagnum.RawType()), G__get_tagnum(derivedtagnum.RawType()), pobject);
}

long Cint::Internal::G__ispublicbase(const ::Reflex::Scope& basetagnum, const ::Reflex::Scope& derivedtagnum, void* pobject)
{
   return G__ispublicbase(G__get_tagnum(basetagnum), G__get_tagnum(derivedtagnum), pobject);
}

long Cint::Internal::G__ispublicbase(
   int basetagnum,
   int derivedtagnum
#ifdef G__VIRTUALBASE
   , void *pobject
#endif // G__VIRTUALBASE
   )
{
   struct G__inheritance *derived;
   int i, n;
   if (0 > derivedtagnum) return -1;
   if (0 > basetagnum) return -1;
   if (basetagnum == derivedtagnum) return 0;
   derived = G__struct.baseclass[derivedtagnum];
   if (derived == 0) return -1;
   n = derived->vec.size();
   for (i = 0;i < n;i++) {
      if (basetagnum == derived->vec[i].basetagnum) {
         if (derived->vec[i].baseaccess == G__PUBLIC ||
               (G__exec_memberfunc && G__tagnum == G__Dict::GetDict().GetScope(derivedtagnum) &&
                G__GRANDPRIVATE != derived->vec[i].baseaccess)) {
#ifdef G__VIRTUALBASE
            if (derived->vec[i].property&G__ISVIRTUALBASE) {
               return G__getvirtualbaseoffset((char*)pobject, derivedtagnum, derived, i);
            }
            else {
               return (long) derived->vec[i].baseoffset;
            }
#else // G__VIRTUALBASE
            return derived->vec[i].baseoffset;
#endif // G__VIRTUALBASE
            // --
         }
      }
   }
   return -1;
}

/**************************************************************************
* G__isanybase()
*
* check if derivedtagnum is derived from basetagnum. If true return offset
* to the base object. If false, return -1.
* Used in cast operatotion
**************************************************************************/
extern "C" int G__isanybase(
   int basetagnum,
   int derivedtagnum
#ifdef G__VIRTUALBASE
   , long pobject
#endif // G__VIRTUALBASE
   )
{
   struct G__inheritance *derived;
   int i;
   int n;

   if (0 > derivedtagnum) {
      for (i = 0; i < (int) G__globalusingnamespace.vec.size(); i++) {
         if (G__globalusingnamespace.vec[i].basetagnum == basetagnum)
            return 0;
      }
      return -1;
   }
   if (basetagnum == derivedtagnum) return(0);
   derived = G__struct.baseclass[derivedtagnum];
   n = derived ? (int) derived->vec.size() : -1;

   for (i = 0;i < n;i++) {
      if (basetagnum == derived->vec[i].basetagnum) {
#ifdef G__VIRTUALBASE
         if (derived->vec[i].property&G__ISVIRTUALBASE) {
            return(G__getvirtualbaseoffset((void*)pobject, derivedtagnum, derived, i));
         }
         else {
            return(long)(derived->vec[i].baseoffset);
         }
#else
         return(derived->vec[i].baseoffset);
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
long Cint::Internal::G__find_virtualoffset(int virtualtag)
{
   size_t i;
   struct G__inheritance *baseclass;

   if (0 > virtualtag) return(0);
   baseclass = G__struct.baseclass[virtualtag];
   for (i = 0;i < baseclass->vec.size();i++) {
      if (G__tagnum == G__Dict::GetDict().GetScope(baseclass->vec[i].basetagnum)) {
         if (baseclass->vec[i].property&G__ISVIRTUALBASE) {
            return(long)(baseclass->vec[i].baseoffset + G__DOUBLEALLOC);
         }
         else {
            return(long)(baseclass->vec[i].baseoffset);
         }
      }
   }
   return(0);
}

#ifdef G__VIRTUALBASE
/**************************************************************************
* G__getvirtualbaseoffset()
**************************************************************************/
long Cint::Internal::G__getvirtualbaseoffset(void* i_pobject, int tagnum, G__inheritance* baseclass, int basen)
{
   char *pobject = (char*)i_pobject;
   long(*f)(long);
   if (pobject == (char*)G__STATICRESOLUTION) return(0);
   if (!pobject || G__no_exec_compile
         || G__PVOID == pobject || ((char*)1) == pobject
      ) {
      G__abortbytecode();
      return(0);
   }
   if (G__CPPLINK == G__struct.iscpplink[tagnum]) {
      f = (long(*)(long))(baseclass->vec[basen].baseoffset);
      return((*f)((long)pobject));
   }
   else {
      /* return((*(long*)(pobject+baseclass->vec[basen].baseoffset))); */
      return(long)(baseclass->vec[basen].baseoffset
                   + (*(long*)(pobject + (size_t)baseclass->vec[basen].baseoffset)));
   }
}
#endif // G__VIRTUALBASE

/***********************************************************************
* G__publicinheritance()
***********************************************************************/
int Cint::Internal::G__publicinheritance(G__value* val1, G__value* val2)
{
   long lresult;
   if ('U' == G__get_type(*val1) && 'U' == G__get_type(*val2)) {
      if (-1 != (lresult = G__ispublicbase(G__value_typenum(*val1), G__value_typenum(*val2), (void*)val2->obj.i))) {
         G__value_typenum(*val2) = G__value_typenum(*val1);
         val2->obj.i += lresult;
         return(lresult);
      }
      else if (-1 != (lresult = G__ispublicbase(G__value_typenum(*val2), G__value_typenum(*val1)
                                , (void*)val1->obj.i))) {
         G__value_typenum(*val1) = G__value_typenum(*val2);
         val1->obj.i += lresult;
         return(-lresult);
      }
   }
   return 0;
}

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
