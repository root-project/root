/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_reader.cxx
 ************************************************************************
 * Description:
 *  Source file reader class
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_reader.h"

const string G__endmark("+-*/<>.!#%&|~^()[]{}:;,= \t\n\r\f");


/***********************************************************************
 * G__fstream
 ***********************************************************************/
//////////////////////////////////////////////////////////////////////
void G__fstream::Init(G__input_file& ifile) {
  //m_fp = ifile.fp;
  //m_linenum = ifile.line_number;
  G__ifile.fp=ifile.fp; 
  G__ifile.filenum=ifile.filenum; 
  G__ifile.line_number=ifile.line_number; 
  G__strlcpy(G__ifile.name,ifile.name,G__MAXFILENAME);
}

//////////////////////////////////////////////////////////////////////
void G__fstream::storepos(int c) {
  m_fp = G__ifile.fp;
  m_linenum = G__ifile.line_number;
  if(m_fp) fgetpos(m_fp,&m_pos);
  m_c = c;
}

//////////////////////////////////////////////////////////////////////
int G__fstream::rewindpos() {
  G__ifile.fp = m_fp;
  G__ifile.line_number = m_linenum;
  if(m_fp) fsetpos(m_fp,&m_pos);
  return(m_c);
}
//////////////////////////////////////////////////////////////////////


