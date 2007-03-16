/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//$Id: rflx_gensrc.h,v 1.8 2006/07/26 13:00:35 axel Exp $

#ifndef RFLX_GENSRC_H
#define RFLX_GENSRC_H 1

#include "Api.h"
#include "Shadow.h"
#include <sstream>
#include <string>
#include <vector>
#include <map>


class rflx_gensrc {

 private:

  typedef std::vector<std::string>          TypeVec;
  typedef std::map<std::string,std::string> TypeMap;
  typedef std::vector<std::string>          ClassVec;

 public:

  rflx_gensrc(const std::string & dictfile,
	      const std::string & sourcefile) : 
    m_typeNum(0),
    m_typeVec(TypeVec()),
    m_typeMap(TypeMap()),
    m_dictfile(dictfile),
    m_sourcefile(sourcefile),
    m_classNames(ClassVec()),
    m_shadowClassNames(ClassVec()),
    m_shadowMaker(m_sh, "ROOT::Reflex"),
    m_split(false),
    ind(indentation()) {} 

  void gen_header();
  void gen_typedicts();
  void gen_classdicts();
  void gen_freefundicts();
  void gen_freevardicts();
  void gen_dictinstances();
  void gen_decl(char type, int num = 0, const std::string & clname = "", const std::string & fclname = "");
  void gen_parTypesNames(std::string & retParTypes, std::string & parNames, G__MethodInfo & mi);
  void gen_classdictdefs(G__ClassInfo & ci);
  void gen_baseclassdefs(G__ClassInfo & ci);
  void gen_datamemberdefs(G__ClassInfo & ci);
  void gen_functionmemberdefs(G__ClassInfo & ci);
  void gen_classdictdecls(std::ostringstream & s, G__ClassInfo & ci);
  std::string gen_type(G__ClassInfo & tn);
  std::string gen_type(G__TypeInfo & tn);
  int gen_stubfuncdecl_header(std::ostringstream & s, G__MethodInfo & fm, const std::string & objcaststr, int argNum = -1);
  void gen_stubfuncdecl_params(std::ostringstream & s, G__MethodInfo & fm, int argNum = -1);
  void gen_stubfuncdecl_trailer(std::ostringstream & s, G__MethodInfo & fm, int argNum = -1);
  void gen_typedefdicts();
  void gen_enumdicts();
  void gen_file();

 private:

  struct indentation {
    int m_len;
    std::string operator() () { return std::string(m_len, ' '); }
    void operator++ ()        { m_len += 2;                     }
#if _MSC_VER<1300
    void operator-- ()        { if (m_len>1) m_len -= 2; else m_len = 0; }
#else
    void operator-- ()        { m_len = std::max(0, m_len - 2); }
#endif
    void clear()              { m_len = 0;                      }
    int get()                 { return m_len;                   }
    void set(int val)         { m_len = val;                    }
  };

  std::ostringstream m_hd;
  std::ostringstream m_sh;
  std::ostringstream m_td;
  std::ostringstream m_cd;
  std::ostringstream m_cds;
  std::ostringstream m_ff;
  std::ostringstream m_fv;
  std::ostringstream m_di;

  int          m_typeNum;
  TypeVec      m_typeVec;
  TypeMap      m_typeMap;
  std::string  m_dictfile;
  std::string  m_sourcefile;
  ClassVec     m_classNames;
  ClassVec     m_shadowClassNames;
  G__ShadowMaker m_shadowMaker;

  bool         m_split;

  indentation  ind;

};

#endif // RFLX_GENSRC_H
