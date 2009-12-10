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

/*************************************************************************
 * class G__type 
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::clear() {
  type = 0;
  tagnum = -1;
  typenum = -1;
  reftype = 0;
  isconst = 0;
  m_static = 0;
  m_type = 0;
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::append(const string& token,int c) {
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
    if(-1!=tagnum || -1!=typenum) return(0); // already got type
    int ltypenum = G__defined_typename(token.c_str()); // legacy
    if(-1!=ltypenum) {
      typenum = ltypenum;
      tagnum = G__newtype.tagnum[typenum];
      type = G__newtype.type[typenum];
      reftype = G__newtype.reftype[typenum];
      isconst |= G__newtype.isconst[typenum];
    }
    else {
      int ltagnum = G__defined_tagname(token.c_str(),1);
      if(-1!=ltagnum) {
        tagnum = ltagnum;
        if(m_type && G__struct.type[tagnum]!=m_type) {
          //error;
        }
        switch(G__struct.type[tagnum]) {
        case 'e': type = 'i'; break;
        default:  type = 'u'; break;
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
  switch(type) {
  case 0: // unsigned
    type = 'h'; // unsigned
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
  switch(type) {
  case 0:   // -> long
    type = 'l'; 
    break;
  case 'h': // unsigned -> long
    type = 'k'; 
    break;
  case 'l': // long -> long
    {
      type = 'n'; 
    }
    break;
  case 'k': // unsigned long -> long
    {
      type = 'm'; 
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
  switch(type) {
  case 0:   // -> int
    type = 'i'; 
    break;
  case 'h': // unsigned -> int
    type = 'h'; 
    break;
  case 'l': // long -> int
    type = 'l';
    break;
  case 's': // short -> int
    type = 's';
    break;
  case 'k': // unsigned long -> int
    type = 'k';
    break;
  case 'r': // unsigned short -> int
    type = 'r';
    break;
  case 'n':
    type = 'n';
    break;
  case 'm':
    type = 'm';
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
  switch(type) {
  case 0:   // -> short
    type = 's'; 
    break;
  case 'h': // unsigned -> short
    type = 'r'; 
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
  switch(type) {
  case 0:   // -> char
    type = 'c'; 
    break;
  case 'h': // unsigned -> char
    type = 'b'; 
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
  switch(type) {
  case 0:   // -> double
    type = 'd'; 
    break;
  case 'l': // long -> double
    {
      type = 'q';
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
  switch(type) {
  case 0:   // -> float
    type = 'f'; 
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
  switch(type) {
  case 0:   // -> void
    type = 'y'; 
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
  switch(type) {
  case 0:   // -> FILE
    type = 'e'; 
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
  switch(type) {
  case 0:   // -> bool
    type = 'g'; 
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
  if(islower(type)) type=toupper(type);
  else {
    if(reftype==0) reftype = G__PARAP2P;
    else if(reftype==G__PARAREFERENCE) reftype=G__PARAREFP2P;
    else ++reftype;
  }
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::decplevel() {
  if(islower(type)) {
    // error
  }
  else {
    if(reftype==0) type=tolower(type);
    else if(reftype==G__PARAREFERENCE) type=tolower(type);
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
  type = tolower(type);
  reftype=G__PARANORMAL;
  isconst &= G__CONSTVAR;
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::Ispointer() const {
  if(!type || islower(type)) return(0);
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
  type = x.type;
  typenum = x.typenum;
  tagnum = x.tagnum;
  if(type!='d'&&type!='f') reftype = x.obj.reftype.reftype;
  else                     reftype = G__PARANORMAL;
  isconst = x.isconst;
  m_static = 0;
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::Init(G__TypeInfo& x) {
  //G__TypeInfo::Init(x); // This can also do the job except for m_static
  type = x.Type();
  typenum = x.Typenum();
  tagnum = x.Tagnum();
  if(type!='d'&&type!='f') reftype = x.Reftype();
  else                     reftype = G__PARANORMAL;
  isconst = x.Isconst();
  m_static = 0;
}

/////////////////////////////////////////////////////////////////////////
G__value G__TypeReader::GetValue() const {
  G__value x = G__null;
  x.type = type;
  x.typenum = typenum;
  x.tagnum = tagnum;
  if(type!='d'&&type!='f') x.obj.reftype.reftype = reftype;
  x.isconst = (G__SIGNEDCHAR_T)isconst;
  return(x);
}

