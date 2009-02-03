/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file cast.c
 ************************************************************************
 * Description:
 *  Type casting
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Dict.h"
#include "common.h"
#include "value.h"

#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

// Static functions.
static void G__SlideString(char* str, unsigned int slide);
static void G__castclass(G__value* result3, int tagnum, int castflag, int* ptype, int reftype, int isconst);

// External functions.
namespace Cint {
namespace Internal {
void G__asm_cast(G__value* buf, const Reflex::Type totype);
G__value G__castvalue(char* casttype, G__value result3, int bc);
G__value G__castvalue(char* casttype, G__value result3);
} // namespace Internal
} // namespace Cint

// Functions in the C interface.
extern "C" long G__int_cast(G__value buf);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__SlideString(char* str, unsigned int slide)
{
   // Do the equivalent of strcpy(str,str+slide);
   unsigned int i = 0;
   while (str[i+slide]) {
      str[i] = str[i+slide];
      ++i;
   }
   str[i] = 0;
}

//______________________________________________________________________________
static void G__castclass(G__value* result3, int tagnum, int castflag, int* ptype, int reftype, int isconst)
{
   // FIXME: Describe this function!
   int offset = 0;
   Reflex::Type valueRawType = G__value_typenum(*result3).RawType();
   if (valueRawType.IsClass() ||  valueRawType.IsStruct()) {
#ifdef G__VIRTUALBASE
      if (-1 != (offset = G__isanybase(tagnum, G__get_tagnum(valueRawType), result3->obj.i)))
         offset = offset;
      else if (0 == castflag && 0 == G__oprovld &&
               (G__SECURE_MARGINAL_CAST&G__security) &&
               (islower(*ptype) || islower(G__get_type(*result3))) &&
               0 == reftype &&
               'i' != G__get_type(*result3) && 'l' != G__get_type(*result3) && 'Y' != G__get_type(*result3)) {
         G__genericerror("Error: illegal type cast (1)");
         return;
      }
      else {
         ::Reflex::Scope store_tagnum = G__tagnum;
         G__set_G__tagnum(*result3);
         offset = -G__find_virtualoffset(tagnum);
         G__tagnum = store_tagnum;
      }
#else // G__VIRTUALBASE
      if (-1 != (offset = G__isanybase(tagnum, result3->tagnum)))
         offset = offset;
      else if (-1 != (offset = G__isanybase(result3->tagnum, tagnum)))
         offset = -offset;
      else
         offset = 0;
#endif // G__VIRTUALBASE
   }
   else {
      ::Reflex::Type raw(G__value_typenum(*result3).RawType());
      if (raw &&
            (raw.Name() == "longlong" ||
             raw.Name() == "ulonglong" ||
             raw.Name() == "long double" ||
             raw.Name() == "G__longlong" ||
             raw.Name() == "G__ulonglong" ||
             raw.Name() == "G__longdouble")) {
         char com[G__ONELINE], buf2[G__ONELINE];
         int known = 0;
         ::Reflex::Type store_typenum = G__value_typenum(*result3);
         // Note Cint 5 code was reseting the typedef value (result3.typenum)
         // So maybe we should have
         // while (G__value_typenum(*result3)).IsTypedef())
         //   G__value_typenum(*result3) = G__value_typenum(*result3).ToType();
         G__valuemonitor(*result3, buf2);
         sprintf(com, "%s(%s)", G__fulltagname(tagnum, 1), buf2);
         *result3 = G__getfunction(com, &known, G__TRYNORMAL);
         G__value_typenum(*result3) = store_typenum;
         return;
      }
      else if (0 == castflag && 0 == G__oprovld &&
               (G__SECURE_MARGINAL_CAST&G__security) &&
               'i' != G__get_type(*result3) && 'l' != G__get_type(*result3) && 'Y' != G__get_type(*result3)) {
         G__genericerror("Error: illegal type cast (2)");
         return;
      }
   }
   Reflex::Type resultType = G__Dict::GetDict().GetType(tagnum);
   if (isconst) {
      resultType = Reflex::Type(resultType, Reflex::CONST);
   }
   if (castflag) {

      resultType = ::Reflex::PointerBuilder(resultType);
   }

   G__value_typenum(*result3) = resultType;

   *ptype = 'u' + castflag;
   result3->obj.i += offset;
}

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
void Cint::Internal::G__asm_cast(G__value* buf, const Reflex::Type totype)
{
   // FIXME: Describe this function!
   char type = G__get_type(totype);
   char reftype = G__get_reftype(totype);
   switch (type) {
      case 'd':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letdouble(buf, (char)type , (double)G__double(*buf));
         break;
      case 'f':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letdouble(buf, (char)type , (float)G__double(*buf));
         break;
      case 'b':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned char)G__int_cast(*buf));
         break;
      case 'c':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (char)G__int_cast(*buf));
         break;
      case 'r':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned short)G__int_cast(*buf));
         break;
      case 's':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (short)G__int_cast(*buf));
         break;
      case 'h':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned int)G__int_cast(*buf));
         break;
      case 'i':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (int)G__int_cast(*buf));
         break;
      case 'k':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned long)G__int_cast(*buf));
         break;
      case 'l':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (long)G__int_cast(*buf));
         break;
      case 'g':
         if (type != G__get_type(*buf)) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned char)(G__int_cast(*buf) ? 1 : 0));
         break;
      case 'U': {
            int offset = G__ispublicbase(G__value_typenum(*buf).RawType(), totype.RawType(), (void*)buf->obj.i);
            if (offset != -1) {
               buf->obj.i += offset;
            }
         }
         G__value_typenum(*buf) = totype;
         buf->ref = buf->obj.i;
         break;
      case 'u':
         if (G__PARAREFERENCE == reftype) {
            int offset = G__ispublicbase(G__value_typenum(*buf).RawType(), totype.RawType(), (void*)buf->obj.i);
            if (offset != -1) {
               buf->obj.i += offset;
               buf->ref += offset;
            }
         }
         G__value_typenum(*buf) = totype;
         buf->ref = buf->obj.i;
         break;
      default:
         G__letint(buf, (char)type , G__int(*buf));
         buf->ref = buf->obj.i;
         break;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__this_adjustment(const Reflex::Member ifunc)
{
   // FIXME: Describe this function!
   G__store_struct_offset += G__get_funcproperties(ifunc)->entry.ptradjust;
}

//______________________________________________________________________________
G__value Cint::Internal::G__castvalue_bc(char* casttype, G__value result3, int bc)
{
   // FIXME: Describe this function!
   int lenitem, castflag, type;
   int tagnum;
   long offset;
   int reftype = G__PARANORMAL;
   int isconst = 0;
   char hasstar = 0;
   G__value store_result;
   store_result = result3;

   /* Questionable condition */
   G__CHECK(G__SECURE_CASTING
            , !G__oprovld && !G__cppconstruct && !G__castcheckoff
            , return(G__null));
#ifdef G__SECURITY
   G__castcheckoff = 0;
#endif

   /* ignore volatile */
   if (strncmp(casttype, "volatile", 8) == 0) {
      G__SlideString(casttype, 8);
   }
   else if (strncmp(casttype, "mutable", 7) == 0) {
      G__SlideString(casttype, 7);
   }
   else if (strncmp(casttype, "typename", 8) == 0) {
      G__SlideString(casttype, 8);
   }
   if (casttype[0] == ' ') strcpy(casttype, casttype + 1);
   while (strncmp(casttype, "const ", 6) == 0) {
      isconst = 1;
      G__SlideString(casttype, 6);
   }
#ifndef G__OLDIMPLEMENTATION1857
   if (strstr(casttype, " const")) {
      char *px = strstr(casttype, " const");
      isconst = 1;
      if (strncmp(px, " const *", 8) == 0 || strncmp(px, " const &", 8) == 0) {
         while (*(px + 7)) {
            *px = *(px + 7);
            ++px;
         }
         *px = 0;
      }
      else {
         while (*(px + 6)) {
            *px = *(px + 6);
            ++px;
         }
         *px = 0;
      }
   }
#endif
   if (strncmp(casttype, "const", 5) == 0) {
      for (lenitem = strlen(casttype) - 1;
            lenitem >= 5 && (casttype[lenitem] == '*' || casttype[lenitem] == '&');
            lenitem--) {}
      if (lenitem >= 5) {
         lenitem++;
         hasstar = casttype[lenitem];
         casttype[lenitem] = '\0';
      }
      if (-1 == G__defined_tagname(casttype, 2) && !G__find_typedef(casttype)) {
         isconst = 1;
         if (hasstar) casttype[lenitem] = hasstar;
         G__SlideString(casttype, 5);
      }
      else if (hasstar) casttype[lenitem] = hasstar;
   }
   if (isspace(casttype[0])) G__SlideString(casttype, 1);
   lenitem = strlen(casttype);
   castflag = 0;

   if (casttype[lenitem-1] == '&') {
      casttype[--lenitem] = '\0';
      reftype = G__PARAREFERENCE;
   }

   /* check if pointer */
   while (casttype[lenitem-1] == '*') {
      if ((G__security&G__SECURE_CAST2P) && !G__oprovld && !G__cppconstruct && !G__castcheckoff) {
         if (isupper(G__get_type(result3)) && (G__value_typenum(result3).ToType().IsClass() || G__value_typenum(result3).ToType().IsStruct())) {
            /* allow casting between public base-derived */
            casttype[lenitem-1] = '\0';
            tagnum = G__defined_tagname(casttype, 2);
            if (-1 != tagnum) {
               if (
#ifdef G__VIRTUALBASE
                  - 1 != (offset = G__ispublicbase(tagnum, G__get_tagnum(G__value_typenum(result3).RawType()), (void*)result3.obj.i))
#else // G__VIRTUALBASE
                  - 1 != (offset = G__ispublicbase(tagnum, result3.tagnum))
#endif // G__VIRTUALBASE
               ) {
                  result3.obj.i += offset;
                  if (isconst) {
                     G__replace_rawtype(G__value_typenum(result3), Reflex::Type(G__Dict::GetDict().GetType(tagnum), Reflex::CONST));
                  }
                  else {
                     G__replace_rawtype(G__value_typenum(result3), G__Dict::GetDict().GetType(tagnum));
                  }
                  return(result3);
               }
               else {
                  ::Reflex::Scope store_tagnum = G__tagnum;
                  G__set_G__tagnum(result3);
                  offset = G__find_virtualoffset(tagnum);
                  G__tagnum = store_tagnum;
                  if (offset) {
                     result3.obj.i -= offset;
                     if (isconst) {
                        G__replace_rawtype(G__value_typenum(result3), Reflex::Type(G__Dict::GetDict().GetType(tagnum), Reflex::CONST));
                     }
                     else {
                        G__replace_rawtype(G__value_typenum(result3), G__Dict::GetDict().GetType(tagnum));
                     }
                     return(result3);
                  }
               }
            }
            G__CHECK(G__SECURE_CAST2P, 1, return(G__null));
         }
         else {
            G__CHECK(G__SECURE_CAST2P, 1, return(G__null));
         }
      }

      if (0 == castflag) castflag = 'A' -'a';
      else if (G__PARANORMAL == reftype) reftype = G__PARAP2P;
      else ++reftype;
      casttype[--lenitem] = '\0';
      while (lenitem > 1 && isspace(casttype[lenitem-1])) casttype[--lenitem] = 0;
   }

   // this part will be a problem if (type&*)
   if (casttype[lenitem-1] == '&') {
      casttype[--lenitem] = '\0';
      reftype = G__PARAREFERENCE;
   }

   bool isUnsigned = !strncmp(casttype, "unsigned ", 9);
   if (isUnsigned) {
      casttype += 9;
      lenitem -= 9;
   }
   else if (!strncmp(casttype, "signed ", 7)) {
      casttype += 7;
      lenitem -= 7;
   }
   int numLong = 0;
   while (!strncmp(casttype, "long ", 5)) {
      ++numLong;
      casttype += 5;
      lenitem -= 5;
   }
   if (!isUnsigned) {
      isUnsigned = !strncmp(casttype, "unsigned ", 9);
      if (isUnsigned) {
         casttype += 9;
         lenitem -= 9;
      }
      else if (!strncmp(casttype, "signed ", 7)) {
         casttype += 7;
         lenitem -= 7;
      }
   }

   type = '\0';
   switch (lenitem) {
      case 3:
         if (strcmp(casttype, "int") == 0) {
            type = 'i';
            break;
         }
         break;
      case 4:
         if (strcmp(casttype, "char") == 0) {
            type = 'c';
            break;
         }
         if (strcmp(casttype, "long") == 0) {
            type = 'l';
            break;
         }
         if (strcmp(casttype, "FILE") == 0) {
            type = 'e';
            break;
         }
         if (strcmp(casttype, "void") == 0) {
            type = 'y';
            break;
         }
         if (strcmp(casttype, "bool") == 0) {
            type = 'g';
            break;
         }
         break;
      case 5:
         if (strcmp(casttype, "short") == 0) {
            type = 's';
            break;
         }
         if (strcmp(casttype, "float") == 0) {
            type = 'f';
            break;
         }
         break;
      case 6:
         if (strcmp(casttype, "double") == 0) {
            type = 'd';
            break;
         }
         break;
      case 8:
         if (strcmp(casttype, "unsigned") == 0) {
            type = 'h';
            break;
         }
         if (strcmp(casttype, "longlong") == 0) {
            type = 'n';
            break;
         }
         break;
      case 9:
         if (strcmp(casttype, "signedint") == 0) {
            type = 'i';
            break;
         }
         break;
      case 10:
         if (strcmp(casttype, "longdouble") == 0) {
            type = 'q';
            break;
         }
         if (strcmp(casttype, "signedchar") == 0) {
            type = 'c';
            break;
         }
         if (strcmp(casttype, "signedlong") == 0) {
            type = 'l';
            break;
         }
         break;
      case 11:
         if (strcmp(casttype, "unsignedint") == 0) {
            type = 'h';
            break;
         }
         if (strcmp(casttype, "signedshort") == 0) {
            type = 's';
            break;
         }
         break;
      case 12:
         if (strcmp(casttype, "unsignedchar") == 0) {
            type = 'b';
            break;
         }
         if (strcmp(casttype, "unsignedlong") == 0) {
            type = 'k';
            break;
         }
         break;
      case 13:
         if (strcmp(casttype, "unsignedshort") == 0) {
            type = 'r';
            break;
         }
         break;
      case 14:
         if (strcmp(casttype, "signedlonglong") == 0) {
            type = 'n';
            break;
         }
         break;
      case 16:
         if (strcmp(casttype, "unsignedlonglong") == 0) {
            type = 'm';
            break;
         }
         break;
   }

   if (type) {
      while (numLong) {
         switch (type) {
            case 'i':
               type = 'l';
               break;
            case 'l':
               type = 'n';
               break;
            case 'd':
               type = 'q';
               break;
         }
         --numLong;
      }
      type += castflag;
      if (isUnsigned)
         --type;
      G__value_typenum(result3) = G__get_from_type(type, 1, isconst);
   }

   if (type && 'u' == G__get_type(store_result)) {
      G__fundamental_conversion_operator(type, -1,::Reflex::Type(), reftype, isconst, &store_result);
      return store_result;
   }
   else if ('u' == G__get_type(store_result) && strcmp(casttype, "bool") == 0) {
      G__fundamental_conversion_operator(type, G__defined_tagname("bool", 2), ::Reflex::Type(), reftype, isconst, &store_result);
      return store_result;
   }

   if (type == '\0') {
      if (strncmp(casttype, "struct", 6) == 0) {
         if (isspace(casttype[6])) tagnum = G__defined_tagname(casttype + 7, 0);
         else tagnum = G__defined_tagname(casttype + 6, 0);
         G__castclass(&result3, tagnum, castflag, &type, reftype, isconst); // FIXME: Should we use store_result here
      }
      else if (strncmp(casttype, "class", 5) == 0) {
         if (isspace(casttype[5])) tagnum = G__defined_tagname(casttype + 6, 0);
         else tagnum = G__defined_tagname(casttype + 5, 0);
         G__castclass(&result3, tagnum, castflag, &type, reftype, isconst); // FIXME: Should we use store_result here?
      }
      else if (strncmp(casttype, "union", 5) == 0) {
         if (isspace(casttype[5])) tagnum = G__defined_tagname(casttype + 6, 0);
         else tagnum = G__defined_tagname(casttype + 5, 0);
         // Note Cint 5 code was reseting the typedef value (result3.typenum)
         type = 'u' + castflag;
      }
      else if (strncmp(casttype, "enum", 4) == 0) {
         if (isspace(casttype[4])) tagnum = G__defined_tagname(casttype + 5, 0);
         else tagnum = G__defined_tagname(casttype + 4, 0);
         // Note Cint 5 code was reseting the typedef value (result3.typenum)
         type = 'i' + castflag;
      }

      if (type == '\0' && strstr(casttype, "(*)")) {
         // pointer to function casted to void*
#ifndef G__OLDIMPLEMENTATION2191
         type = '1';
#else // G__OLDIMPLEMENTATION2191
         type = 'Q';
#endif // G__OLDIMPLEMENTATION2191
         // Note Cint 5 code was reseting the typedef value (result3.typenum)
      }

      if (type == '\0') {
         int store_var_type = G__var_type;
         ::Reflex::Type typenum = G__find_typedef(casttype);
         G__var_type = store_var_type;
         if (!typenum) {
            tagnum = G__defined_tagname(casttype, 0);
            if (tagnum == -1) type = 'Y'; /* checked */
            else G__castclass(&result3, tagnum, castflag, &type, reftype, isconst); // FIXME: Should we use store_result here?
         }
         else {
            tagnum = G__get_tagnum(typenum);
            {
               // Accumulate the typedef pointer information
               // into a local version of reftype
               // to pass down to G__castclass
               type = G__get_type(typenum);
               int local_reftype;
               if (islower(type))
                  type = type + castflag;
               else {
                  local_reftype = G__get_reftype(typenum);
                  switch (local_reftype) {
                     case G__PARANORMAL:
                     case G__PARAREFERENCE:
                        local_reftype = G__PARAP2P;
                        break;
                     default:
                        local_reftype = local_reftype;
                        break;
                  }
               }
               if (G__get_type(typenum) == 'u' && tagnum != -1) {
                  if (
                     G__struct.type[tagnum] == 'e'
                  ) {
                     G__value_typenum(result3) = typenum;
                     type = 'i' + castflag;
                  }
                  else {
                     G__castclass(&result3, tagnum, castflag, &lenitem, reftype, isconst); // FIXME: Should we use store_result here?
                  }
               }
               else if ('u' == G__get_type(store_result)) {
                  G__fundamental_conversion_operator(type, -1, G__value_typenum(result3), reftype, isconst, &store_result);
                  return store_result;
               }
               else {
                  G__value_typenum(result3) = typenum;
               }
            }
            if (castflag) {
               G__value_typenum(result3) = ::Reflex::PointerBuilder(typenum);
            }
         }
      }
      G__var_type = 'p';
   }

   for (int i = 2; i <= reftype; ++i) {
      G__value_typenum(result3) = ::Reflex::PointerBuilder(G__value_typenum(result3));
   }

#ifdef G__ASM
   if (G__asm_noverflow
      // --
#ifndef G__OLDIMPLEMENTATION1073
      && 0 == G__oprovld
#endif // G__OLDIMPLEMENTATION1073
      // --
   ) {
      //
      // CAST
      //
      if (bc) {
#ifdef G__ASM_DBG
         if (G__asm_dbg && G__asm_noverflow) {
            G__fprinterr(G__serr, "%3x,%3x: CAST to '%s'  %s:%d\n", G__asm_cp, G__asm_dt, G__value_typenum(result3).Name(::Reflex::SCOPED).c_str(), __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__CAST;
         ::Reflex::Type castto = G__value_typenum(result3);
         *(reinterpret_cast<Reflex::Type*>(&G__asm_inst[G__asm_cp+1])) = castto;
         G__inc_cp_asm(5, 0);
      }
   }
#endif // G__ASM

   if (G__security_error) return(result3);

   if (type != G__get_type(store_result)) result3.ref = 0; /* questionable */

   switch (type) {
      case 'd':
         G__letdouble(&result3, type , (double)G__double(store_result));
         break;
      case 'f':
         G__letdouble(&result3, type , (float)G__double(store_result));
         break;
      case 'b':
         G__setvalue(&result3, (unsigned char)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 'c':
         G__setvalue(&result3, (char)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 'r':
         G__setvalue(&result3, (unsigned short)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 's':
         G__setvalue(&result3, (short)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 'h':
         G__setvalue(&result3, (unsigned int)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 'i':
         G__setvalue(&result3, (int)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 'k':
         G__setvalue(&result3, (unsigned long)G__int_cast(store_result));
         G__value_typenum(result3) = G__get_from_type(type, 0);
         break;
      case 'l':
         G__letint(&result3, type , (long)G__int_cast(store_result));
         break;
      case 'm':
         G__letULonglong(&result3, type , G__ULonglong(store_result));
         break;
      case 'n':
         G__letLonglong(&result3, type , G__Longlong(store_result));
         break;
      case 'q':
         G__letLongdouble(&result3, type , G__Longdouble(store_result));
         break;
      case 'g':
         // --
#ifdef G__BOOL4BYTE
         G__letint(&result3, type , (int)(G__int_cast(store_result) ? 1 : 0));
#else // G__BOOL4BYTE
         G__letint(&result3, type , (unsigned char)(G__int_cast(store_result) ? 1 : 0));
#endif // G__BOOL4BYTE
         break;
      default: {
            G__letpointer(&result3, G__int(result3), G__value_typenum(result3));
            if (islower(type)) result3.ref = result3.obj.i;
         }
         break;
   }
   return result3;
}

//______________________________________________________________________________
G__value Cint::Internal::G__castvalue(char* casttype, G__value result3)
{
   return G__castvalue_bc(casttype, result3, 1);
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
extern "C" long G__int_cast(G__value buf)
{
   return G__convertT<long>(&buf);
}

