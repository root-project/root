#if 0
/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_debug.cxx
 ************************************************************************
 * Description:
 *  debugging features
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_debug.h"

using namespace ::Cint::Internal;
using namespace ::Cint::Bytecode;

/***********************************************************************
 * G__bc_funccall
 ***********************************************************************/
////////////////////////////////////////////////////////////////
struct G__input_file Cint::Bytecode::G__bc_funccall::getifile() const {
  struct G__input_file ifile;
  ifile.str = 0;
  // ifile.pos = 0;
  ifile.vindex = 0;

  if(!m_bytecode) {
    ifile=G__ifile;
  }
  else {
    ifile.filenum = G__get_funcproperties(m_bytecode->ifunc)->entry.filenum;
    ifile.fp = G__srcfile[ifile.filenum].fp;
    ifile.line_number = m_line_number;
    strcpy(ifile.name,G__srcfile[ifile.filenum].filename);
  }

  return(ifile);
}


////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__bc_funccall::setstackenv(struct G__view* pview) const {
  // todo, need some review
  pview->file = getifile();
  if(!m_bytecode) {
    pview->var_local = G__p_local;
    pview->struct_offset = G__store_struct_offset;
    pview->tagnum = G__tagnum;
    pview->exec_memberfunc = G__exec_memberfunc;
    pview->localmem = 0;
    return(0);
  }
  else {
    pview->var_local = m_bytecode->frame;
    pview->struct_offset = m_struct_offset;
    pview->tagnum = m_bytecode->ifunc.DeclaringScope();
    pview->exec_memberfunc=!pview->tagnum.IsTopScope(); 
    pview->localmem = m_localmem;
    return(1);
  }
}

////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__bc_funccall::disp(FILE* fout) const {
  // todo, need some review
  if(!m_bytecode)  return(0);
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  const Reflex::Member& func = m_bytecode->ifunc;
  int filenum = G__get_funcproperties(func)->entry.filenum;
  struct G__param* libp=m_libp;

  sprintf(msg,"%s::",func.DeclaringScope().Name(Reflex::SCOPED).c_str());
  if(G__more(fout,msg)) return(1);

  // function name
  sprintf(msg,"%s(",func.Name().c_str());
  if(G__more(fout,msg)) return(1);

  // function parameter
  for(int temp1=0;temp1<libp->paran;temp1++) {
    if(temp1) {
      sprintf(msg,",");
      if(G__more(fout,msg)) return(1);
    }
    G__valuemonitor(libp->para[temp1],msg);
    if(G__more(fout,msg)) return(1);
  }
  if(-1!=filenum) {
    sprintf(msg,") [%s:%d]\n" 
            ,G__stripfilename(G__srcfile[filenum].filename)
            ,m_line_number);
    if(G__more(fout,msg)) return(1);
  }
  else {
    if(G__more(fout,") [entry]\n")) return(1);
  }

  return(0);
}


/***********************************************************************
 * G__bc_funccallstack
 ***********************************************************************/
////////////////////////////////////////////////////////////////
Cint::Bytecode::G__bc_funccallstack::G__bc_funccallstack() { 
  //m_funccallstack.push_front(G__bc_funccall()); 
}

////////////////////////////////////////////////////////////////
Cint::Bytecode::G__bc_funccallstack::~G__bc_funccallstack() { 
  // do nothing
}

////////////////////////////////////////////////////////////////
G__bc_funccall& Cint::Bytecode::G__bc_funccallstack::getStackPosition(int i) {
  if(0==m_funccallstack.size()) return(m_staticenv);
  if(i<0 || i>=(int)m_funccallstack.size()) {
    // error, stack isn't that deep
    G__fprinterr(G__serr,"!!!Function call stack isn't that deep!!!\n");
    return(m_staticenv);
  }
  return(m_funccallstack[i]);
}

////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__bc_funccallstack::setstackenv(int i,struct G__view* pview) {
  return(getStackPosition(i).setstackenv(pview));
}

////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__bc_funccallstack::disp(FILE* fout) const {
  //deque<G__bc_funccall>::iterator i;
  char msg[100];
  for(int i=0;i<(int)m_funccallstack.size();++i) {
    sprintf(msg,"%d ",i);
    if(G__more(fout,msg)) return(1);
    if(m_funccallstack[i].disp(fout)) return(1);
  }
  return(0);
}

////////////////////////////////////////////////////////////////

/***********************************************************************
 * static objects
 ***********************************************************************/
Cint::Bytecode::G__bc_funccallstack Cint::Bytecode::G__bc_funccallstack_obj;

/***********************************************************************
 * C function wrappers
 ***********************************************************************/

////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__bc_setdebugview(int i,struct G__view* pview) {
  return(G__bc_funccallstack_obj.setstackenv(i,pview));
}

////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__bc_showstack(FILE* fout) {
  return(G__bc_funccallstack_obj.disp(fout));
}

////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__bc_setlinenum(int line) {
  G__bc_funccallstack_obj.setlinenum(line);
}
#endif // 0
