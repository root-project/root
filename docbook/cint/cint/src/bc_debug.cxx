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

/***********************************************************************
 * G__bc_funccall
 ***********************************************************************/
////////////////////////////////////////////////////////////////
struct G__input_file G__bc_funccall::getifile() const {
  struct G__input_file ifile;
  ifile.str = 0;
  ifile.pos = 0;
  ifile.vindex = 0;

  if(!m_bytecode) {
    ifile=G__ifile;
  }
  else {
    struct G__ifunc_table_internal *ifunc = m_bytecode->ifunc;
    int ifn = m_bytecode->ifn;
    ifile.filenum = ifunc->pentry[ifn]->filenum;
    ifile.fp = G__srcfile[ifile.filenum].fp;
    ifile.line_number = m_line_number;
    strncpy(ifile.name,G__srcfile[ifile.filenum].filename, sizeof(ifile.name) - 1);
  }

  return(ifile);
}


////////////////////////////////////////////////////////////////
int G__bc_funccall::setstackenv(struct G__view* pview) const {
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
    struct G__ifunc_table_internal *ifunc = m_bytecode->ifunc;
    //int ifn = m_bytecode->ifn;
    pview->var_local = m_bytecode->var;
    pview->struct_offset = m_struct_offset;
    pview->tagnum = ifunc->tagnum;
    pview->exec_memberfunc=(-1!=ifunc->tagnum)?1:0; 
    pview->localmem = m_localmem;
    return(1);
  }
}

////////////////////////////////////////////////////////////////
int G__bc_funccall::disp(FILE* fout) const {
  // todo, need some review
  if(!m_bytecode)  return(0);
  G__FastAllocString msg(G__LONGLINE);
  struct G__ifunc_table_internal *ifunc = m_bytecode->ifunc;
  int ifn = m_bytecode->ifn;
  int tagnum=ifunc->tagnum;
  int filenum = ifunc->pentry[ifn]->filenum;
  struct G__param* libp=m_libp;

  // class name if member function
  if(-1!=tagnum) {
     msg.Format("%s::",G__struct.name[tagnum]);
     if(G__more(fout,msg())) return(1);
  }

  // function name
  msg.Format("%s(",ifunc->funcname[ifn]);
  if(G__more(fout,msg())) return(1);

  // function parameter
  for(int temp1=0;temp1<libp->paran;temp1++) {
    if(temp1) {
      msg = ",";
      if(G__more(fout,msg())) return(1);
    }
    G__valuemonitor(libp->para[temp1],msg);
    if(G__more(fout,msg())) return(1);
  }
  if(-1!=filenum) {
     msg.Format(") [%s:%d]\n" 
	    ,G__stripfilename(G__srcfile[filenum].filename)
	    ,m_line_number);
     if(G__more(fout,msg())) return(1);
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
G__bc_funccallstack::G__bc_funccallstack() { 
  //m_funccallstack.push_front(G__bc_funccall()); 
}

////////////////////////////////////////////////////////////////
G__bc_funccallstack::~G__bc_funccallstack() { 
  // do nothing
}

////////////////////////////////////////////////////////////////
G__bc_funccall& G__bc_funccallstack::getStackPosition(int i) {
  if(0==m_funccallstack.size()) return(m_staticenv);
  if(i<0 || i>=(int)m_funccallstack.size()) {
    // error, stack isn't that deep
    G__fprinterr(G__serr,"!!!Function call stack isn't that deep!!!\n");
    return(m_staticenv);
  }
  return(m_funccallstack[i]);
}

////////////////////////////////////////////////////////////////
int G__bc_funccallstack::setstackenv(int i,struct G__view* pview) {
  return(getStackPosition(i).setstackenv(pview));
}

////////////////////////////////////////////////////////////////
int G__bc_funccallstack::disp(FILE* fout) const {
  //deque<G__bc_funccall>::iterator i;
  G__FastAllocString msg(100);
  for(int i=0;i<(int)m_funccallstack.size();++i) {
     msg.Format("%d ",i);
     if(G__more(fout,msg())) return(1);
     if(m_funccallstack[i].disp(fout)) return(1);
  }
  return(0);
}

////////////////////////////////////////////////////////////////

/***********************************************************************
 * static objects
 ***********************************************************************/
G__bc_funccallstack G__bc_funccallstack_obj;

/***********************************************************************
 * C function wrappers
 ***********************************************************************/

////////////////////////////////////////////////////////////////
extern "C" int G__bc_setdebugview(int i,struct G__view* pview) {
  return(G__bc_funccallstack_obj.setstackenv(i,pview));
}

////////////////////////////////////////////////////////////////
extern "C" int G__bc_showstack(FILE* fout) {
  return(G__bc_funccallstack_obj.disp(fout));
}

////////////////////////////////////////////////////////////////
extern "C" void G__bc_setlinenum(int line) {
  G__bc_funccallstack_obj.setlinenum(line);
}

