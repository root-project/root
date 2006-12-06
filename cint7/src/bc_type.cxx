/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_type.cxx
 ************************************************************************
 * Description:
 *  type information interface
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_type.h"

namespace Cint {
   namespace Bytecode {
      using namespace ::Cint::Internal;

/*************************************************************************
 * class G__type 
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::clear() {
  reflexInfo->type = 0;
  tagnum = -1;
  reflexInfo->typenum = ::ROOT::Reflex::Type();
  reftype = 0;
  isconst = 0;
  m_static = 0;
  m_type = 0;
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::append(const std::string& token,int c) {
  if(token=="static")        append_static();
  else if(token=="unsigned") append_unsigned();
  else if(token=="long")     append_long();
  else if(token=="int")      append_int();
  else if(token=="short")    append_short();
  else if(token=="char")     append_char();
  else if(token=="double")   append_double();
  else if(token=="float")    append_float();
  else if(token=="void")     append_void();
  else if(token=="FILE")     append_FILE();
  else if(token=="bool")     append_bool();
  else if(token=="") {
    if(c=='(') {
      // type (*p)(args);
      // type (*p)[2][3];
      // type (*p[3][4])[2][3];
      return(0)  ;
    }
  }
  else if(token=="volatile") ;
  else if(token=="mutable")  ;
  else if(token=="register") ;
  else if(token=="const") {
    if(Ispointer()) isconst|=G__PCONSTVAR;
    else isconst|=G__CONSTVAR;
  }
  else if(token=="class")    m_type='c';
  else if(token=="struct")   m_type='s';
  else if(token=="union")    m_type='u';
  else if(token=="enum")     m_type='e';
  else {
    if( (-1!=tagnum) || reflexInfo->typenum) return(0); // already got type
#ifdef __GNUC__
#else
#pragma message(FIXME("what's the enclosing scope == context for token?!"))
#endif
    ::ROOT::Reflex::Type ltypenum = ::ROOT::Reflex::Scope().LookupType(token); // legacy
    if(ltypenum) {
      reflexInfo->typenum = ltypenum;
      tagnum = G__get_tagnum(reflexInfo->typenum.FinalType());
      reflexInfo->type = G__get_type(reflexInfo->typenum);
      reftype = G__get_reftype(reflexInfo->typenum);
      isconst |= G__get_isconst(reflexInfo->typenum);
    }
    else {
      int ltagnum = G__defined_tagname(token.c_str(),1);
      if(-1!=ltagnum) {
        tagnum = ltagnum;
        if(m_type && G__struct.type[tagnum]!=m_type) {
          //error;
        }
        switch(G__struct.type[tagnum]) {
        case 'e': reflexInfo->type = 'i'; break;
        default:  reflexInfo->type = 'u'; break;
        }
        reftype = 0;
        isconst |= 0;
      }
      else {
        // this is not a part of type name
        // TODO, template class instantiation has to be implemented here???
        return(0);
      }
    }
  }

  if(c=='*')      incplevel();
  else if(c=='&') increflevel();

  return(1);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_unsigned() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0: // unsigned
    reflexInfo->type = 'h'; // unsigned
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_long() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> long
    reflexInfo->type = 'l'; 
    break;
  case 'h': // unsigned -> long
    reflexInfo->type = 'k'; 
    break;
  case 'l': // long -> long
    {
      reflexInfo->type = 'n'; 
    }
    break;
  case 'k': // unsigned long -> long
    {
      reflexInfo->type = 'm'; 
    }
    break;
  //case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  //case 'k': //unsigned long
  //case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long double
  case 'm': //long double
    //case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_int() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> int
    reflexInfo->type = 'i'; 
    break;
  case 'h': // unsigned -> int
    reflexInfo->type = 'h'; 
    break;
  case 'l': // long -> int
    reflexInfo->type = 'l';
    break;
  case 's': // short -> int
    reflexInfo->type = 's';
    break;
  case 'k': // unsigned long -> int
    reflexInfo->type = 'k';
    break;
  case 'r': // unsigned short -> int
    reflexInfo->type = 'r';
    break;
  case 'n':
    reflexInfo->type = 'n';
    break;
  case 'm':
    reflexInfo->type = 'm';
    break;
  case 'u': // long long -> int
            // unsigned long long -> int
    break;
  //case 'l': //long
  case 'i': //int
  //case 's': //short
  case 'c': //char
  //case 'k': //unsigned long
  //case 'h': //unsigned int
  //case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  //case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'q'://long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_short() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> short
    reflexInfo->type = 's'; 
    break;
  case 'h': // unsigned -> short
    reflexInfo->type = 'r'; 
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  //case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_char() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> char
    reflexInfo->type = 'c'; 
    break;
  case 'h': // unsigned -> char
    reflexInfo->type = 'b'; 
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  //case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_double() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> double
    reflexInfo->type = 'd'; 
    break;
  case 'l': // long -> double
    {
      reflexInfo->type = 'q';
    }
    break;
  //case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_float() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> float
    reflexInfo->type = 'f'; 
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_void() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> void
    reflexInfo->type = 'y'; 
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_FILE() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> FILE
    reflexInfo->type = 'e'; 
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_bool() {
  if(tagnum!=1) { 
    //error;
  }
  switch(reflexInfo->type) {
  case 0:   // -> bool
    reflexInfo->type = 'g'; 
    break;
  case 'l': //long
  case 'i': //int
  case 's': //short
  case 'c': //char
  case 'k': //unsigned long
  case 'h': //unsigned int
  case 'r': //unsigned short
  case 'b': //unsigned char
  case 'd': //double
  case 'f': //float
  case 'u': //struct/class
  case 'e': //FILE
  case 'y': //void
  case 'g': //bool
  case 'n': //long long
  case 'm': //unsigned long long
  case 'q': //long double
  default:
    // error??
    break;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::incplevel() {
  if(islower(reflexInfo->type)) reflexInfo->type=toupper(reflexInfo->type);
  else {
    if(reftype==0) reftype = G__PARAP2P;
    else if(reftype==G__PARAREFERENCE) reftype=G__PARAREFP2P;
    else ++reftype;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::decplevel() {
  if(islower(reflexInfo->type)) {
    // error
  }
  else {
    if(reftype==0) reflexInfo->type=tolower(reflexInfo->type);
    else if(reftype==G__PARAREFERENCE) reflexInfo->type=tolower(reflexInfo->type);
    else if(reftype==G__PARAP2P) reftype=G__PARANORMAL;
    else if(reftype==G__PARAREFP2P) reftype=G__PARAREFERENCE;
    else --reftype;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::increflevel() {
  if(reftype==0) reftype = G__PARAREFERENCE;
  else if(reftype==G__PARAREFERENCE) reftype=G__PARAREFERENCE;
  else reftype+=G__PARAREF;
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::decreflevel() {
  if(reftype==0) {
    // error
  }
  else if(reftype==G__PARAREFERENCE) reftype=G__PARANORMAL;
  else if(reftype>G__PARAREF) reftype-=G__PARAREF;
  else {
    // error
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::nextdecl() {
  reflexInfo->type = tolower(reflexInfo->type);
  reftype=G__PARANORMAL;
  isconst &= G__CONSTVAR;
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::Ispointer() const {
  if(!reflexInfo->type || islower(reflexInfo->type)) return(0);
  else if(reftype==0) return(1);
  else if(reftype<G__PARAREF) return(reftype);
  else return(reftype-G__PARAREF);
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::Isreference() const {
  if(reftype==G__PARAREFERENCE || reftype>G__PARAREF) return(1);
  else return(0);
}

/////////////////////////////////////////////////////////////////////////
long G__TypeReader::Property()  {
  return((m_static?G__BIT_ISSTATIC:0)|G__TypeInfo::Property());
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::Init(G__value& x) {
  //G__TypeInfo::Init(x); // This can also do the job except for m_static
  reflexInfo->type = x.type;
  reflexInfo->typenum = G__value_typenum(x);
  tagnum = x.tagnum;
  if(reflexInfo->type!='d'&&reflexInfo->type!='f') reftype = x.obj.reftype.reftype;
  else                     reftype = G__PARANORMAL;
  isconst = x.isconst;
  m_static = 0;
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::Init(G__TypeInfo& x) {
  //G__TypeInfo::Init(x); // This can also do the job except for m_static
  reflexInfo->type = x.Type();
  reflexInfo->typenum = x.ReflexType();
  tagnum = x.Tagnum();
  if(reflexInfo->type!='d'&&reflexInfo->type!='f') reftype = x.Reftype();
  else                     reftype = G__PARANORMAL;
  isconst = x.Isconst();
  m_static = 0;
}

/////////////////////////////////////////////////////////////////////////
G__value G__TypeReader::GetValue() const {
  G__value x;
  x.type = reflexInfo->type;
  G__value_typenum(x) = reflexInfo->typenum;
  x.tagnum = tagnum;
  if(reflexInfo->type!='d'&&reflexInfo->type!='f') x.obj.reftype.reftype = reftype;
  x.isconst = (G__SIGNEDCHAR_T)isconst;
  return(x);
}

   } // namespace Bytecode
} // namespace Cint
