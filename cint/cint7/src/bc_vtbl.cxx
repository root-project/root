#if 0
/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_vtbl.cxx
 ************************************************************************
 * Description:
 *  virtual table generator
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_vtbl.h"
#include "Api.h"
#include <cstdio>
#include <algorithm>
using namespace std;

namespace Cint {
   namespace Bytecode {
      using namespace ::Cint::Internal;
   
/***********************************************************************
* G__Vtabledata
***********************************************************************/
////////////////////////////////////////////////////////////////////
void G__Vtabledata::disp(FILE *fp) {
  fprintf(fp,"%s offset=%d "
          ,m_ifn.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str()
          ,m_offset);
}
////////////////////////////////////////////////////////////////////

/***********************************************************************
* G__Vtableoffset
***********************************************************************/
////////////////////////////////////////////////////////////////////
void G__Vtbloffset::disp(FILE *fp) {
  fprintf(fp,"base=%s offset=%d "
          ,m_basetagnum.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str()
          ,m_vtbloffset);
}
////////////////////////////////////////////////////////////////////

/***********************************************************************
* G__Vtable
***********************************************************************/
int G__Vtable::addbase(const Reflex::Scope& basetagnum,int vtbloffset) {
  for(vector<G__Vtbloffset>::iterator i=m_vtbloffset.begin();
      i!=m_vtbloffset.end();++i) {
    if((*i).m_basetagnum==basetagnum) return(0);
  }
  G__Vtbloffset x;
  x.m_basetagnum = basetagnum;
  x.m_vtbloffset = vtbloffset;
  m_vtbloffset.push_back(x);
  return(1);
}
////////////////////////////////////////////////////////////////////
G__Vtabledata* G__Vtable::resolve(int index,const Reflex::Scope& basetagnum) {
  int vtbloffset=0;
  for(vector<G__Vtbloffset>::iterator i=m_vtbloffset.begin();
      i!=m_vtbloffset.end();++i) {
    if((*i).m_basetagnum==basetagnum) {
      vtbloffset = (*i).m_vtbloffset;
      break;
    }
  }
  return(&m_vtbl[index+vtbloffset]);
}

////////////////////////////////////////////////////////////////////
void G__Vtable::disp(FILE *fp) {
  for(vector<G__Vtabledata>::iterator i=m_vtbl.begin();i!=m_vtbl.end();++i) {
    (*i).disp(fp);
  }
  fprintf(fp,"\n");
  for(vector<G__Vtbloffset>::iterator j=m_vtbloffset.begin();
      j!=m_vtbloffset.end();++j) {
    (*j).disp(fp);
  }
  fprintf(fp,"\n");
}

/***********************************************************************
* G__bc_make_vtbl() 
***********************************************************************/
void G__bc_make_vtbl(const Reflex::Scope& tagnum) {
  if(G__NOLINK!=G__globalcomp) return;

  G__ClassInfo cls(G__get_tagnum(tagnum));
  vector<int> nextbaseoffset; // temporary baseoffset buffer
  G__Vtable *pvtbl = new G__Vtable; // vtable stored in G__struct

  // check if there is base class with virtual function
  G__BaseClassInfo bas(cls);
  while(bas.Next()) { // iterate only direct base classes
    if(bas.ClassProperty()&G__CLS_HASVIRTUAL) {
      int basetagnum = bas.Tagnum();
      G__Vtable *pbasevtbl = (G__Vtable*)G__struct.vtable[basetagnum];
      // copy vtbloffset table with basetagnum
      int pbasesize=pbasevtbl->m_vtbloffset.size();
      for(int i=0;i<pbasesize;++i) {
         pvtbl->addbase(pbasevtbl->m_vtbloffset[i].m_basetagnum
                      ,pbasevtbl->m_vtbloffset[i].m_vtbloffset
                          +pvtbl->m_vtbl.size());
      }
      // copy vtbl
      copy(pbasevtbl->m_vtbl.begin(),pbasevtbl->m_vtbl.end()
           ,back_inserter(pvtbl->m_vtbl));
      // calculate baseoffset addition for each vtbl entry (temporary)
      while(nextbaseoffset.size()<pvtbl->m_vtbl.size()) {
        nextbaseoffset.push_back(bas.Offset());
      }
    }
  }
  int isbase=pvtbl->m_vtbl.size();

  if(!isbase) pvtbl->addbase(tagnum,0);

  // check each member function and create or override vtbl entry
  G__MethodInfo met(cls);
  while(met.Next()) { 
    int done=0;
    if(isbase) {
      // search for overridden function in base class
      for(int i=0;i<isbase;++i) {
         if(G__function_signature_match(met.ReflexFunction()
                                        ,pvtbl->m_vtbl[i].GetFunction()
                                        ,true
                                        ,G__EXACT,0)) {
          met.SetVtblIndex(G__get_funcproperties(pvtbl->m_vtbl[i].GetFunction())->entry.vtblindex);
          met.SetVtblIndex(G__get_funcproperties(pvtbl->m_vtbl[i].GetFunction())->entry.vtblbasetagnum);
          met.SetIsVirtual(1);
          // override function
          //            A
          //  *           B       << origin of virtual function
          //  *         C C C     << offset=B.Offset()
          //          D
          //  *       E E E E E   << offset=C.Offset()+vtbloffset
          pvtbl->m_vtbl[i].SetIfn(met.ReflexFunction());
          pvtbl->m_vtbl[i].SetOffset(pvtbl->m_vtbl[i].GetOffset()
                                     +nextbaseoffset[i]); 
          done=1;
          //break; // base class method can override virtual function from
                   // multiple inheritances.
        }
      }
    }
    if(met.Property()&G__BIT_ISVIRTUAL && !done) { // create new vtbl entry
      // Set virtualindex and basetagnum information to the virtual function
      met.SetVtblIndex(pvtbl->m_vtbl.size());
      met.SetVtblBasetagnum(G__get_tagnum(pvtbl->m_vtbloffset[0].m_basetagnum));
      //met.SetIsVirtual(1); // do nothing, isvirtual is already set
      // offset is always 0 for the first vfunc decl
      pvtbl->addvfunc(met.ReflexFunction(),0);
    }
  }

  // allocate and copy virtual table in G__struct table
  if(pvtbl->m_vtbl.size()) {
    G__struct.vtable[G__get_tagnum(tagnum)]=(void*)pvtbl;
    G__get_properties(tagnum)->vtable = pvtbl;
  }
  else {
    G__struct.vtable[G__get_tagnum(tagnum)]=(G__Vtabledata*)NULL;
    G__get_properties(tagnum)->vtable = 0;
  }
}

/***********************************************************************
* G__bc_make_defaultctor() 
***********************************************************************/
void G__bc_make_defaultctor(const Reflex::Scope& tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(G__get_tagnum(tagnum));

  // Judge if implicit default constructor is needed. Return if not.
  //if(cls.Property()&G__BIT_ISABSTRACT) return;//make protected ctor
  if(cls.FuncFlag()&(G__HAS_CONSTRUCTOR|G__HAS_XCONSTRUCTOR)) return;
  G__MethodInfo m=cls.GetDefaultConstructor();
  if(m.IsValid()) return;

  G__BaseClassInfo bas(cls);
  while(bas.Next()) {
    m = bas.GetDefaultConstructor();
    if(!m.IsValid() || (m.Property()&G__BIT_ISPRIVATE)) return;
  }

  G__DataMemberInfo dat(cls);
  while(dat.Next()) {
    G__TypeInfo *typ = dat.Type();
    if((typ->Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT))
       && 0==(typ->Property()&G__BIT_ISPOINTER)) {
      m = typ->GetDefaultConstructor();
      if(!m.IsValid() || (m.Property()&G__BIT_ISPRIVATE)) return;
    }
  }

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"!!! Generating default constructor %s()\n"
                              ,cls.Name());
#endif

#pragma message(FIXME("The return type of A::A() is _not_ A&!"))
  string fname(tagnum.Name());
  string rtntype(tagnum.Name(Reflex::SCOPED));
  rtntype.append("&");
  G__MethodInfo met = cls.AddMethod(rtntype.c_str(),fname.c_str(),"");
  Reflex::Member func = met.ReflexFunction();
#pragma message(FIXME("Can't change the access for an existing member! Need to be able to unload"))
  //if(tagnum.IsAbstract()) func->access[ifn]=G__PROTECTED;

  G__functionscope compiler;
  compiler.compile_implicitdefaultctor(func);
}

/***********************************************************************
* G__bc_make_copyctor() 
***********************************************************************/
void G__bc_make_copyctor(const Reflex::Scope& tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(G__get_tagnum(tagnum));

  // Judge if implicit copy constructor is needed. Return if not.
  //if(cls.Property()&G__BIT_ISABSTRACT) return;//make protected ctor
  G__MethodInfo m=cls.GetCopyConstructor();
  if(m.IsValid()) return;

  G__BaseClassInfo bas(cls);
  while(bas.Next()) {
    m = bas.GetCopyConstructor();
    if(!m.IsValid() || (m.Property()&G__BIT_ISPRIVATE)) return;
  }

  G__DataMemberInfo dat(cls);
  while(dat.Next()) {
    G__TypeInfo *typ = dat.Type();
    if(typ->Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) {
      m = typ->GetCopyConstructor();
      if(!m.IsValid() || (m.Property()&G__BIT_ISPRIVATE)) return;
    }
  }

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"!!! Generating copy constructor %s(const %s&)\n"
                              ,cls.Name(),cls.Name());
#endif

#pragma message(FIXME("The return type of A::A() is _not_ A&!"))
  string fname(tagnum.Name());
  string rtntype(tagnum.Name(Reflex::SCOPED));
  rtntype.append("&");
  string arg = "const "; arg.append(fname.c_str()); arg.append("&");
  G__MethodInfo met = cls.AddMethod(rtntype.c_str(),fname.c_str(),arg.c_str());
  Reflex::Member func = met.ReflexFunction();
#pragma message(FIXME("Workaround to set which const flag to signal what?!"))
  //func->para_isconst[ifn][0] = G__CONSTVAR; // workaround to set const flag
#pragma message(FIXME("Can't change the access for an existing member! Need to be able to unload"))
  //if(cls.Property()&G__BIT_ISABSTRACT) ifunc->access[ifn]=G__PROTECTED;

  G__functionscope compiler;
  compiler.compile_implicitcopyctor(func);
}

/***********************************************************************
* G__bc_make_assignopr() 
***********************************************************************/
void G__bc_make_assignopr(const Reflex::Scope& tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(G__get_tagnum(tagnum));

  // Judge if implicit assignment operator is needed. Return if not.
  G__MethodInfo m=cls.GetAssignOperator();
  if(m.IsValid()) return;

  G__BaseClassInfo bas(cls);
  while(bas.Next()) {
    m = bas.GetAssignOperator();
    if(!m.IsValid() || (m.Property()&G__BIT_ISPRIVATE)) return;
  }

  G__DataMemberInfo dat(cls);
  while(dat.Next()) {
    G__TypeInfo *typ = dat.Type();
    if(typ->Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) {
      m = typ->GetAssignOperator();
      if(!m.IsValid() || (m.Property()&G__BIT_ISPRIVATE)) return;
    }
  }

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"!!! Generating implicit %s::operator=\n",cls.Name());
#endif

  string rtntype(tagnum.Name(Reflex::SCOPED)); rtntype.append("&");
  string arg = "const "; arg.append(rtntype);
  G__MethodInfo met = cls.AddMethod(rtntype.c_str(),"operator=",arg.c_str());
  Reflex::Member func = met.ReflexFunction();

  G__functionscope compiler;
  compiler.compile_implicitassign(func);
}
/***********************************************************************
* G__bc_make_dtor() 
***********************************************************************/
void G__bc_make_dtor(const Reflex::Scope& tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(G__get_tagnum(tagnum));

  // Judge if implicit default constructor is needed. Return if not.
  //if(cls.Property()&G__BIT_ISABSTRACT) return;//make protected ctor
  if(cls.FuncFlag()&(G__HAS_DESTRUCTOR)) return;
  G__MethodInfo m=cls.GetDestructor();
  if(m.IsValid()) return;

  int flag=0;

  G__BaseClassInfo bas(cls);
  while(bas.Next()) {
    m = bas.GetDestructor();
    if((m.Property()&G__BIT_ISPRIVATE)) return;
    if(m.IsValid()) ++flag;
  }

  G__DataMemberInfo dat(cls);
  while(dat.Next()) {
    G__TypeInfo *typ = dat.Type();
    if(typ->Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) {
      m = typ->GetDestructor();
      if((m.Property()&G__BIT_ISPRIVATE)) return;
      if(m.IsValid()) ++flag;
    }
  }

  if(!flag) return;

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"!!! Generating destructor %s()\n"
                              ,cls.Name());
#endif

#pragma message(FIXME("Why don't we go via cls.AddMethod() like for the c'tors?"))
#if 0
  // the first entry is reserved for dtor
  struct G__ifunc_table* ifunc = G__struct.memfunc[tagnum];
  int ifn = 0;

  // set function name and hash
  string fname("~");
  fname.append(tagnum.Name());
  G__savestring(&ifunc->funcname[ifn],(char*)fname.c_str());
  int tmp;
  G__hash(ifunc->funcname[ifn],ifunc->hash[ifn],tmp);

  ifunc->type[ifn] = 'y'; // void 
  // miscellaneous flags
  ifunc->isexplicit[ifn] = 0;
  ifunc->iscpp[ifn] = 1;
  ifunc->ansi[ifn] = 1;
  ifunc->busy[ifn] = 0;
  ifunc->friendtag[ifn] = (struct G__friendtag*)NULL;
  ifunc->globalcomp[ifn] = G__NOLINK;
  ifunc->comment[ifn].p.com = (char*)NULL;
  ifunc->comment[ifn].filenum = -1;

  if(cls.Property()&G__BIT_ISABSTRACT) ifunc->access[ifn]=G__PROTECTED;

  G__functionscope compiler;
  compiler.compile_implicitdtor(ifunc,ifn);
#endif
}

/***********************************************************************
* G__bc_struct() 
***********************************************************************/
void G__bc_struct(const Reflex::Scope& tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
#if ENABLE_CPP_EXCEPTIONS
  try {
#endif //ENABLE_CPP_EXCEPTIONS
    G__bc_make_vtbl(tagnum);
    G__bc_make_defaultctor(tagnum); 
    G__bc_make_copyctor(tagnum);
    G__bc_make_assignopr(tagnum);
    G__bc_make_dtor(tagnum); // need defaultdtor if base or member has dtor
#if ENABLE_CPP_EXCEPTIONS
  }
  // TODO
  catch(...) {
  }
#endif //ENABLE_CPP_EXCEPTIONS
}

/***********************************************************************
* G__bc_delete_vtbl() 
***********************************************************************/
void G__bc_delete_vtbl(const Reflex::Scope& tagnum) {
  G__Vtable *&pvtbl = (G__Vtable*&)G__get_properties(tagnum)->vtable;
  if(pvtbl) delete pvtbl;
  pvtbl = NULL;

  pvtbl = (G__Vtable*)G__struct.vtable[G__get_tagnum(tagnum)];
  if(pvtbl) delete pvtbl;
  pvtbl = NULL;
}
/***********************************************************************
* G__bc_disp_vtbl() 
***********************************************************************/
void G__bc_disp_vtbl(FILE* fp,const Reflex::Scope& tagnum) {
  G__Vtable *pvtbl = (G__Vtable*)G__get_properties(tagnum)->vtable;
  if(pvtbl) pvtbl->disp(fp);
}

   } // namespace Bytecode
} // namespace Cint

//////////////////////////////////////////////////////////////////////////

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


#endif // 0
