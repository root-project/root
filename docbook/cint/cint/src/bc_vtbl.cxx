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
#include <iterator>
using namespace std;

/***********************************************************************
* G__Vtabledata
***********************************************************************/
////////////////////////////////////////////////////////////////////
void G__Vtabledata::disp(FILE *fp) {
  G__ifunc_table_internal* ifunc = G__get_ifunc_internal(m_ifunc);
  fprintf(fp,"%s::%s offset=%d "
	  ,G__struct.name[ifunc->tagnum]
	  ,ifunc->funcname[m_ifn]
	  ,m_offset);
}
////////////////////////////////////////////////////////////////////

/***********************************************************************
* G__Vtableoffset
***********************************************************************/
////////////////////////////////////////////////////////////////////
void G__Vtbloffset::disp(FILE *fp) {
  fprintf(fp,"base=%s offset=%d "
	  ,G__struct.name[m_basetagnum]
	  ,m_vtbloffset);
}
////////////////////////////////////////////////////////////////////

/***********************************************************************
* G__Vtable
***********************************************************************/
int G__Vtable::addbase(int basetagnum,int vtbloffset) {
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
G__Vtabledata* G__Vtable::resolve(int index,int basetagnum) {
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
* G__function_signature_match()
***********************************************************************/
extern "C" int G__function_signature_match(struct G__ifunc_table* iref1
					,int ifn1
					,struct G__ifunc_table* iref2
					,int ifn2
					,int mask
					,int /*matchmode*/) {
  G__ifunc_table_internal* ifunc1 = G__get_ifunc_internal(iref1);
  G__ifunc_table_internal* ifunc2 = G__get_ifunc_internal(iref2);
  if(ifunc1->hash[ifn1]!=ifunc2->hash[ifn2] ||
      strcmp(ifunc1->funcname[ifn1],ifunc2->funcname[ifn2]) != 0 ||
      (ifunc1->para_nu[ifn1]!=ifunc2->para_nu[ifn2] && 
       ifunc1->para_nu[ifn1]>=0 && ifunc2->para_nu[ifn2]>=0)
	 || ((ifunc1->isconst[ifn1]&mask) /* 1798 */
	     !=(ifunc2->isconst[ifn2]&mask))) return(0);
      
  int paran;
  if(ifunc1->para_nu[ifn1]>=0 && ifunc2->para_nu[ifn2]>=0)
    paran=ifunc1->para_nu[ifn1];
  else
    paran = 0;

  int j;
  for(j=0;j<paran;j++) {
   if(ifunc1->param[ifn1][j]->type !=ifunc2->param[ifn2][j]->type ||
     ifunc1->param[ifn1][j]->p_tagtable !=ifunc2->param[ifn2][j]->p_tagtable
     || ifunc1->param[ifn1][j]->reftype !=ifunc2->param[ifn2][j]->reftype
     || ifunc1->param[ifn1][j]->isconst !=ifunc2->param[ifn2][j]->isconst) {
     return(0);
   }
 }
 return(1);
}

/***********************************************************************
* G__bc_make_vtbl() 
***********************************************************************/
void G__bc_make_vtbl(int tagnum) {
  if(G__NOLINK!=G__globalcomp) return;

  G__ClassInfo cls(tagnum);
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
        if(G__function_signature_match((struct G__ifunc_table*)met.Handle()
					,met.Index()
					,pvtbl->m_vtbl[i].GetIfunc()
					,pvtbl->m_vtbl[i].GetIfn()
					,0xffff
					,G__EXACT)) {
          G__ifunc_table_internal* ifunc = G__get_ifunc_internal(pvtbl->m_vtbl[i].GetIfunc());
          met.SetVtblIndex(ifunc->vtblindex[pvtbl->m_vtbl[i].GetIfn()]);
          met.SetVtblBasetagnum(ifunc->vtblbasetagnum[pvtbl->m_vtbl[i].GetIfn()]);
          met.SetIsVirtual(1);
          // override function
          //            A
          //  *           B       << origin of virtual function
          //  *         C C C     << offset=B.Offset()
          //          D
          //  *       E E E E E   << offset=C.Offset()+vtbloffset
          pvtbl->m_vtbl[i].SetIfunc((struct G__ifunc_table*)met.Handle());
          pvtbl->m_vtbl[i].SetIfn(met.Index());
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
      met.SetVtblBasetagnum(pvtbl->m_vtbloffset[0].m_basetagnum);
      //met.SetIsVirtual(1); // do nothing, isvirtual is already set
      // offset is always 0 for the first vfunc decl
      pvtbl->addvfunc((struct G__ifunc_table*)met.Handle(),met.Index(),0);
    }
  }

  // allocate and copy virtual table in G__struct table
  if(pvtbl->m_vtbl.size()) {
    G__struct.vtable[tagnum]=(void*)pvtbl;
  }
  else {
    G__struct.vtable[tagnum]=(G__Vtabledata*)NULL;
    delete pvtbl;
  }
}

/***********************************************************************
* G__bc_make_defaultctor() 
***********************************************************************/
void G__bc_make_defaultctor(int tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(tagnum);

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

  string rtntype(G__struct.name[tagnum]); rtntype.append("&");
  string fname(G__struct.name[tagnum]);
  string arg = ""; 
  G__MethodInfo met = cls.AddMethod(rtntype.c_str(),fname.c_str(),"");
  struct G__ifunc_table_internal* ifunc = G__get_ifunc_internal((struct G__ifunc_table*)met.Handle());
  int ifn = met.Index();
  if(cls.Property()&G__BIT_ISABSTRACT) ifunc->access[ifn]=G__PROTECTED;

  G__functionscope* compiler = new G__functionscope;
  compiler->compile_implicitdefaultctor(ifunc,ifn);
  delete compiler;
}

/***********************************************************************
* G__bc_make_copyctor() 
***********************************************************************/
void G__bc_make_copyctor(int tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(tagnum);

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

  string rtntype(G__struct.name[tagnum]); rtntype.append("&");
  string fname(G__struct.name[tagnum]);
  string arg = "const "; arg.append(G__struct.name[tagnum]); arg.append("&");
  G__MethodInfo met = cls.AddMethod(rtntype.c_str(),fname.c_str(),arg.c_str());
  struct G__ifunc_table_internal* ifunc = G__get_ifunc_internal((struct G__ifunc_table*)met.Handle());
  int ifn = met.Index();
  ifunc->param[ifn][0]->isconst = G__CONSTVAR; // workaround to set const flag
  if(cls.Property()&G__BIT_ISABSTRACT) ifunc->access[ifn]=G__PROTECTED;

  G__functionscope* compiler = new G__functionscope;
  compiler->compile_implicitcopyctor(ifunc,ifn);
  delete compiler;
}

/***********************************************************************
* G__bc_make_assignopr() 
***********************************************************************/
void G__bc_make_assignopr(int tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(tagnum);

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

  string rtntype(G__struct.name[tagnum]); rtntype.append("&");
  string arg = "const "; arg.append(G__struct.name[tagnum]); arg.append("&");
  G__MethodInfo met = cls.AddMethod(rtntype.c_str(),"operator=",arg.c_str());
  struct G__ifunc_table* ifunc = (struct G__ifunc_table*)met.Handle();
  int ifn = met.Index();

  G__functionscope* compiler = new G__functionscope;
  compiler->compile_implicitassign(G__get_ifunc_internal(ifunc),ifn);
  delete compiler;
}
/***********************************************************************
* G__bc_make_dtor() 
***********************************************************************/
void G__bc_make_dtor(int tagnum) {
  if(G__NOLINK!=G__globalcomp) return;
  G__ClassInfo cls(tagnum);

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

  // the first entry is reserved for dtor
  struct G__ifunc_table_internal* ifunc = G__struct.memfunc[tagnum];
  int ifn = 0;

  // set function name and hash
  string fname("~");
  fname.append(G__struct.name[tagnum]);
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

  G__functionscope* compiler = new G__functionscope;
  compiler->compile_implicitdtor(ifunc,ifn);
  delete compiler;
}

/***********************************************************************
* G__bc_struct() 
***********************************************************************/
extern "C" void G__bc_struct(int tagnum) {
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
extern "C" void G__bc_delete_vtbl(int tagnum) {
  G__Vtable *pvtbl = (G__Vtable*)G__struct.vtable[tagnum];
  if(pvtbl) delete pvtbl;
  G__struct.vtable[tagnum] = (void*)NULL;
}
/***********************************************************************
* G__bc_disp_vtbl() 
***********************************************************************/
extern "C" void G__bc_disp_vtbl(FILE* fp,int tagnum) {
  G__Vtable *pvtbl = (G__Vtable*)G__struct.vtable[tagnum];
  if(pvtbl) pvtbl->disp(fp);
}

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


