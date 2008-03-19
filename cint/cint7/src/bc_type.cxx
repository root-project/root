#if 0
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
#include "Reflex/Tools.h"

namespace Cint {
   namespace Bytecode {
      using namespace ::Cint::Internal;

/*************************************************************************
 * class G__type 
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::clear() {
  typeiter = -1;
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
     // also properly transforms "char*" into "char*const"
     typenum = Reflex::ConstBuilder(typenum);
  }
  else if(token=="class")    m_type='c';
  else if(token=="struct")   m_type='s';
  else if(token=="union")    m_type='u';
  else if(token=="enum")     m_type='e';
  else {
    if(typenum) return 0; // already got type
#ifdef __GNUC__
#else
#pragma message(FIXME("what's the enclosing scope == context for token?!"))
#endif
    ::Reflex::Type ltypenum = ::Reflex::Scope::GlobalScope().LookupType(token); // legacy
    if(ltypenum) {
      typenum = ltypenum;
    } 
#ifdef __GNUC__
#else
#pragma message(FIXME("Type being a struct is handled - but what about enum -> type='i'?"))
#endif
    /*
    else {
      int ltagnum = G__defined_tagname(token.c_str(),1);
      if(ltagnum) {
        tagnum = ltagnum;
        if(m_type && G__struct.type[tagnum]!=m_type) {
          //error;
        }
        switch(G__struct.type[tagnum]) {
        case 'e': type = 'i'; break; <======== ???
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
    } */
  }

  if(c=='*')      incplevel();
  else if(c=='&') increflevel();

  return(1);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_unsigned() {
   if (!typenum)
      typenum = Reflex::Type::ByName("unsigned");
   else
      // error??
      ;
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_long() {
   bool wasConst = typenum.IsConst();
   if (!typenum)
      typenum = Reflex::Type::ByName("long");
   else
   switch(Reflex::Tools::FundamentalType(typenum)) {
   case Reflex::kUNSIGNED_INT: // unsigned -> long
      typenum = Reflex::Type::ByName("long");
      break;
   case Reflex::kLONG_INT: // long -> long
      typenum = Reflex::Type::ByName("long long");
      break;
   case Reflex::kUNSIGNED_LONG_INT: // unsigned long -> long
      typenum = Reflex::Type::ByName("unsigned long long"); 
      break;
   default:
      // error??
    return;
   }
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_int() {
  bool wasConst = typenum.IsConst();
  if (!typenum)   // -> int
    typenum = Reflex::Type::ByName("int");
  else
  switch(Reflex::Tools::FundamentalType(typenum)) {
   case Reflex::kUNSIGNED_INT: // unsigned -> int
   case Reflex::kLONG_INT: // long -> int
   case Reflex::kSHORT_INT: // short -> int
   case Reflex::kUNSIGNED_LONG_INT: // unsigned long -> int
   case Reflex::kUNSIGNED_SHORT_INT: // unsigned short -> int
   case Reflex::kLONGLONG: // long long -> int VALID???
   case Reflex::kULONGLONG: // unsigned long long -> int VALID???
      break;
   default:
      // error??
    return;
   }
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_short() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> short
    typenum = Reflex::Type::ByName("short");
  else
  if (Reflex::Tools::FundamentalType(typenum) == Reflex::kUNSIGNED_INT)
     typenum = Reflex::Type::ByName("unsigned short");
  else return; // error??
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_char() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> char
    typenum = Reflex::Type::ByName("char");
  else
  if (Reflex::Tools::FundamentalType(typenum) == Reflex::kUNSIGNED_INT)
     typenum = Reflex::Type::ByName("unsigned char");
  else return; // error??
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_double() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> double
    typenum = Reflex::Type::ByName("double");
  else 
  if (Reflex::Tools::FundamentalType(typenum) == Reflex::kLONG_INT)
     typenum = Reflex::Type::ByName("long double");
  else return; // error??
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_float() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> float
    typenum = Reflex::Type::ByName("float");
  else return; //error?
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_void() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> void
    typenum = Reflex::Type::ByName("void");
  else return; //error?
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_FILE() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> FILE
    typenum = Reflex::Type::ByName("FILE");
  else return; //error?
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::append_bool() {
  bool wasConst = typenum.IsConst();
  if (!typenum) // -> bool
    typenum = Reflex::Type::ByName("bool");
  else return; //error?
   if (wasConst)
      typenum = Reflex::ConstBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::incplevel() {
   typenum = Reflex::PointerBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::decplevel() {
   assert(typenum.FinalType().IsPointer());
   typenum = G__deref(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::increflevel() {
   typenum = Reflex::ReferenceBuilder(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::decreflevel() {
   assert(typenum.FinalType().IsReference());
   typenum = G__deref(typenum);
}

/////////////////////////////////////////////////////////////////////////
void G__TypeReader::nextdecl() {
   // int ***a[] -> int
   while (typenum != typenum.RawType() 
      // const int *** const a[], ... -> const int
      && !(typenum.ToType() == typenum.RawType() && typenum.IsConst()))
      typenum = typenum.ToType();
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::Ispointer() const {
   return typenum.IsPointer();
}

/////////////////////////////////////////////////////////////////////////
int G__TypeReader::Isreference() const {
   return typenum.IsReference();
}

/////////////////////////////////////////////////////////////////////////
long G__TypeReader::Property()  {
  return((m_static?G__BIT_ISSTATIC:0)|G__TypeInfo::Property());
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::Init(G__value& x) {
  //G__TypeInfo::Init(x); // This can also do the job except for m_static
  typenum = G__value_typenum(x);
#pragma message(FIXME("Here's that strange 'd','f' reftype handling again..."))
/*
  if(type!='d'&&type!='f') reftype = x.obj.reftype.reftype;
  else                     reftype = G__PARANORMAL; */
  m_static = 0;
}
/////////////////////////////////////////////////////////////////////////
void G__TypeReader::Init(G__TypeInfo& x) {
  //G__TypeInfo::Init(x); // This can also do the job except for m_static
  typenum = x.ReflexType();
#pragma message(FIXME("Here's that strange 'd','f' reftype handling again..."))
/*
  if(type!='d'&&type!='f') reftype = x.Reftype();
  else                     reftype = G__PARANORMAL; */
  m_static = 0;
}

/////////////////////////////////////////////////////////////////////////
G__value G__TypeReader::GetValue() const {
  G__value x;
  G__value_typenum(x) = typenum;
#pragma message(FIXME("Here's that strange 'd','f' reftype handling again..."))
  /*if(type!='d'&&type!='f') x.obj.reftype.reftype = reftype;*/
  return(x);
}

   } // namespace Bytecode
} // namespace Cint
#endif // 0
