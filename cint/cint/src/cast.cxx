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

#include "common.h"
#include "value.h"

extern "C" {

// Static functions.
static void G__SlideString(char* str, unsigned int slide);
static void G__castclass(G__value* result3, int tagnum, int castflag, int* ptype, int reftype);

// External functions.
void G__asm_cast(int type, G__value* buf, int tagnum, int reftype);
void G__this_adjustment(G__ifunc_table_internal* ifunc, int ifn);
G__value G__castvalue_bc(char* casttype, G__value result3, int bc);
G__value G__castvalue(char* casttype, G__value result3);

// Functions in the C interface.
long G__int_cast(G__value buf);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__SlideString(char* str, unsigned int slide)
{
   // -- Do the equivalent of strcpy(str,str+slide);
   unsigned int i = 0;
   while (str[i+slide]) {
      str[i] = str[i+slide];
      ++i;
   }
   str[i] = 0;
}

//______________________________________________________________________________
void* G__dynamiccast(int totagnum, int fromtagnum, void* addr)
{
   // Cast addr pointing to an object of class fromtagnum to the class
   // tottagnum, returning the adjusted object pointer. Similar to
   //    dynamic_cast<totagnum*>((fromtagnum*)add)

   G__value val = G__null;
   val.tagnum = fromtagnum;
   val.obj.i = (long) addr;
   int type = 'C';
   G__castclass(&val, totagnum, 'A'-'a', &type, G__PARANORMAL);
   return (void*) val.obj.i;
}

//______________________________________________________________________________
static void G__castclass(G__value* result3, int tagnum, int castflag, int* ptype, int reftype)
{
   // -- FIXME: Describe this function!
   int offset = 0;
   if (tagnum < 0) {
      result3->obj.i = 0;
      return;
   }
   if (-1 != result3->tagnum) {
#ifdef G__VIRTUALBASE
      if (-1 != (offset = G__isanybase(tagnum, result3->tagnum, result3->obj.i))) {
         (void)offset /* = offset*/;
      } else if (0 == castflag && 0 == G__oprovld &&
               (G__SECURE_MARGINAL_CAST&G__security) &&
               (islower(*ptype) || islower(result3->type)) &&
               0 == reftype &&
               'i' != result3->type && 'l' != result3->type && 'Y' != result3->type) {
         G__genericerror("Error: illegal type cast (1)");
         return;
      }
      else {
         int store_tagnum = G__tagnum;
         G__tagnum = result3->tagnum;
         offset = -G__find_virtualoffset(tagnum, result3->obj.i);
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
   else if (-1 == result3->tagnum && -1 != result3->typenum &&
            (strcmp(G__newtype.name[result3->typenum], "unsigned long long") == 0 ||
             strcmp(G__newtype.name[result3->typenum], "unsigned long long") == 0 ||
             strcmp(G__newtype.name[result3->typenum], "long double") == 0 ||
             (-1 != G__newtype.tagnum[result3->typenum] &&
              (strcmp(G__struct.name[G__newtype.tagnum[result3->typenum]]
                      , "G__longlong") == 0 ||
               strcmp(G__struct.name[G__newtype.tagnum[result3->typenum]]
                      , "G__ulonglong") == 0 ||
               strcmp(G__struct.name[G__newtype.tagnum[result3->typenum]]
                      , "G__longdouble") == 0)))) {
      G__FastAllocString com(G__ONELINE);
      G__FastAllocString buf2(G__ONELINE);
      int known = 0;
      int store_typenum = result3->typenum;
      result3->typenum = -1;
      G__valuemonitor(*result3, buf2);
      com.Format("%s(%s)", G__fulltagname(tagnum, 1), buf2());
      *result3 = G__getfunction(com, &known, G__TRYNORMAL);
      result3->typenum = store_typenum;
      return;
   }
   else if (0 == castflag && 0 == G__oprovld &&
            (G__SECURE_MARGINAL_CAST&G__security) &&
            'i' != result3->type && 'l' != result3->type && 'Y' != result3->type) {
      G__genericerror("Error: illegal type cast (2)");
      return;
   }
   result3->tagnum = tagnum;
   result3->typenum = -1;
   *ptype = 'u' + castflag;
   result3->obj.i += offset;
}

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
void G__asm_cast(int type, G__value* buf, int tagnum, int reftype)
{
   // -- FIXME: Describe this function!
   switch ((char)type) {
      case 'd':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letdouble(buf, (char)type , (double)G__double(*buf));
         break;
      case 'f':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letdouble(buf, (char)type , (float)G__double(*buf));
         break;
      case 'b':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned char)G__int_cast(*buf));
         break;
      case 'c':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (char)G__int_cast(*buf));
         break;
      case 'r':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned short)G__int_cast(*buf));
         break;
      case 's':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (short)G__int_cast(*buf));
         break;
      case 'h':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned int)G__int_cast(*buf));
         break;
      case 'i':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (int)G__int_cast(*buf));
         break;
      case 'k':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned long)G__int_cast(*buf));
         break;
      case 'l':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (long)G__int_cast(*buf));
         break;
      case 'g':
         if (type != buf->type) buf->ref = 0; /* questionable */
         G__letint(buf, (char)type , (unsigned char)(G__int_cast(*buf) ? 1 : 0));
         break;
      case 'U': {
         int offset = G__ispublicbase(buf->tagnum, tagnum, buf->obj.i);
         if (-1 != offset) buf->obj.i += offset;
         break;
      }
      case 'u':
         if (G__PARAREFERENCE == reftype) {
            int offset = G__ispublicbase(buf->tagnum, tagnum, buf->obj.i);
            if (-1 != offset) {
               buf->obj.i += offset;
               buf->ref += offset;
            }
         }
         break;
      default:
         G__letint(buf, (char)type , G__int(*buf));
         buf->ref = buf->obj.i;
         break;
   }
}

//______________________________________________________________________________
void G__this_adjustment(G__ifunc_table_internal* ifunc, int ifn)
{
   // -- FIXME: Describe this function!
   if (ifunc && ifunc->pentry[ifn]) {
      G__store_struct_offset += ifunc->pentry[ifn]->ptradjust;
   }
}

//______________________________________________________________________________
G__value G__castvalue_bc(char* casttype, G__value result3, int bc)
{
   // -- FIXME: Describe this function!
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
#endif // G__SECURITY
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
   if (casttype[0] == ' ') G__SlideString(casttype, 1);
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
#endif // G__OLDIMPLEMENTATION1857
   if (strncmp(casttype, "const", 5) == 0) {
      for (lenitem = strlen(casttype) - 1;
            lenitem >= 5 && (casttype[lenitem] == '*' || casttype[lenitem] == '&');
         lenitem--) {}
      if (lenitem >= 5) {
         lenitem++;
         hasstar = casttype[lenitem];
         casttype[lenitem] = '\0';
      }
      if (-1 == G__defined_tagname(casttype, 2) && -1 == G__defined_typename(casttype)) {
         isconst = 1;
         if (hasstar) casttype[lenitem] = hasstar;
         G__SlideString(casttype, 5);
      }
      else if (hasstar) casttype[lenitem] = hasstar;
   }
   /* since we have the information let's return it */
   result3.isconst = isconst;
   if (isspace(casttype[0])) G__SlideString(casttype, 1);
   lenitem = strlen(casttype);
   castflag = 0;
   if (casttype[lenitem-1] == '&') {
      casttype[--lenitem] = '\0';
      reftype = G__PARAREFERENCE;
   }
   /* check if pointer */
   while (casttype[lenitem-1] == '*') {
      if ((G__security&G__SECURE_CAST2P) &&
            !G__oprovld && !G__cppconstruct && !G__castcheckoff) {
         if (isupper(result3.type) && (-1 != result3.tagnum)) {
            /* allow casting between public base-derived */
            casttype[lenitem-1] = '\0';
            tagnum = G__defined_tagname(casttype, 2);
            if (tagnum != -1) {
               // --
#ifdef G__VIRTUALBASE
               offset = G__ispublicbase(tagnum, result3.tagnum, result3.obj.i);
#else // G__VIRTUALBASE
               offset = G__ispublicbase(tagnum, result3.tagnum);
#endif // G__VIRTUALBASE
               if (offset != -1) {
                  result3.obj.i += offset;
                  result3.tagnum = tagnum;
                  result3.typenum = -1;
                  return(result3);
               }
               else {
                  int store_tagnum = G__tagnum;
                  G__tagnum = result3.tagnum;
#ifdef G__VIRTUALBASE
                  offset = G__find_virtualoffset(tagnum, result3.obj.i);

#else // G__VIRTUALBASE
                  offset = G__find_virtualoffset(tagnum);
#endif // G__VIRTUALBASE
                  G__tagnum = store_tagnum;
                  if (offset) {
                     result3.obj.i -= offset;
                     result3.tagnum = tagnum;
                     result3.typenum = -1;
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
   /* this part will be a problem if (type&*) */
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
         if (strcmp(casttype, "long long") == 0) {
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
         if (strcmp(casttype, "long double") == 0) {
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
         if (strcmp(casttype, "unsigned int") == 0) {
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
         if (strcmp(casttype, "unsigned long") == 0) {
            type = 'k';
            break;
         }
         break;
      case 13:
         if (strcmp(casttype, "unsigned short") == 0) {
            type = 'r';
            break;
         }
         break;
      case 14:
         if (strcmp(casttype, "signedlong long") == 0) {
            type = 'n';
            break;
         }
         break;
      case 16:
         if (strcmp(casttype, "unsigned long long") == 0) {
            type = 'm';
            break;
         }
         break;
   }
   if (type) {
      result3.tagnum = -1;
      result3.typenum = -1;

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
   }
   if (type && 'u' == store_result.type) {
      G__FastAllocString ttt(G__ONELINE);
      G__fundamental_conversion_operator(type , -1 , -1 , reftype, isconst
                                         , &store_result, ttt);
      return(store_result);
   }
   else if ('u' == store_result.type && strcmp(casttype, "bool") == 0) {
      G__FastAllocString ttt(G__ONELINE);
      G__fundamental_conversion_operator(type , G__defined_tagname("bool", 2)
                                         , -1 , reftype, isconst
                                         , &store_result, ttt);
      return(store_result);
   }
   if (type == '\0') {
      if (strncmp(casttype, "struct", 6) == 0) {
         if (isspace(casttype[6])) tagnum = G__defined_tagname(casttype + 7, 0);
         else tagnum = G__defined_tagname(casttype + 6, 0);
         if (tagnum >= 0) {
            G__castclass(&result3, tagnum, castflag, &type, reftype);
         } else {
            G__fprinterr(G__serr, "unknown %s, cannot cast\n", casttype);
         }
      }
      else if (strncmp(casttype, "class", 5) == 0) {
         if (isspace(casttype[5])) tagnum = G__defined_tagname(casttype + 6, 0);
         else tagnum = G__defined_tagname(casttype + 5, 0);
         if (tagnum >= 0) {
            G__castclass(&result3, tagnum, castflag, &type, reftype);
         } else {
            G__fprinterr(G__serr, "unknown %s, cannot cast\n", casttype);
         }
      }
      else if (strncmp(casttype, "union", 5) == 0) {
         if (isspace(casttype[5])) tagnum = G__defined_tagname(casttype + 6, 0);
         else tagnum = G__defined_tagname(casttype + 5, 0);
         if (tagnum < 0) {
            G__fprinterr(G__serr, "unknown %s, cannot cast\n", casttype);
         }
         result3.typenum = -1;
         type = 'u' + castflag;
      }
      else if (strncmp(casttype, "enum", 4) == 0) {
         if (isspace(casttype[4])) tagnum = G__defined_tagname(casttype + 5, 0);
         else tagnum = G__defined_tagname(casttype + 4, 0);
         if (tagnum < 0) {
            G__fprinterr(G__serr, "unknown %s, cannot cast\n", casttype);
         }
         result3.typenum = -1;
         type = 'i' + castflag;
      }

      if (type == '\0' && strstr(casttype, "(*)")) {
         /* pointer to function casted to void* */
#ifndef G__OLDIMPLEMENTATION2191
         type = '1';
#else // G__OLDIMPLEMENTATION2191
         type = 'Q';
#endif // G__OLDIMPLEMENTATION2191
         result3.tagnum = -1;
         result3.typenum = -1;
      }
      if (type == '\0') {
         int store_var_type = G__var_type;
         result3.typenum = G__defined_typename(casttype);
         G__var_type = store_var_type;
         if (result3.typenum == -1) {
            tagnum = G__defined_tagname(casttype, 0);
            if (tagnum == -1) type = 'Y'; /* checked */
            else G__castclass(&result3, tagnum, castflag, &type, reftype);
         }
         else {
            tagnum = G__newtype.tagnum[result3.typenum];
            if (islower(G__newtype.type[result3.typenum]))
               type = G__newtype.type[result3.typenum] + castflag;
            else {
               type = G__newtype.type[result3.typenum];
               switch (G__newtype.reftype[result3.typenum]) {
                  case G__PARANORMAL:
                  case G__PARAREFERENCE:
                     reftype = G__PARAP2P;
                     break;
                  default:
                     reftype = G__newtype.reftype[result3.typenum] + 1;
                     break;
               }
            }
            if (tagnum != -1) {
               if (
                  G__struct.type[tagnum] == 'e'
               ) {
                  type = 'i' + castflag;
               }
               else {
                  G__castclass(&result3, tagnum, castflag, &lenitem, reftype);
               }
            }
            else if ('u' == store_result.type) {
               G__FastAllocString ttt(G__ONELINE);
               G__fundamental_conversion_operator(type, -1, result3.typenum
                                                  , reftype, isconst
                                                  , &store_result, ttt);
               return(store_result);
            }
         }
      }
      G__var_type = 'p';
   }
#ifdef G__ASM
   if (G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION1073
         && 0 == G__oprovld
#endif // G__OLDIMPLEMENTATION1073
      ) {
         // --
      /*********************************************************
       * CAST
       *********************************************************/
#ifdef G__ASM_DBG
      if (G__asm_dbg && G__asm_noverflow) {
         G__fprinterr(G__serr, "%3x: CAST to %c\n", G__asm_cp, type);
      }
#endif // G__ASM_DBG
      if (bc) {
         G__asm_inst[G__asm_cp] = G__CAST;
         G__asm_inst[G__asm_cp+1] = type;
         G__asm_inst[G__asm_cp+2] = result3.typenum;
         G__asm_inst[G__asm_cp+3] = result3.tagnum;
         G__asm_inst[G__asm_cp+4] = reftype;
         G__inc_cp_asm(5, 0);
      }
   }
#endif // G__ASM
   if (G__security_error) return(result3);
   if (type != result3.type) result3.ref = 0; /* questionable */
   result3.isconst = isconst;
   switch (type) {
      case 'd':
         G__letdouble(&result3, type , (double)G__double(result3));
         break;
      case 'f':
         G__letdouble(&result3, type , (float)G__double(result3));
         break;
      case 'b':
         //G__letint(&result3,type ,(unsigned char)G__int_cast(result3));
         result3.obj.uch = (unsigned char)G__int_cast(result3);
         result3.type = type;
         break;
      case 'c':
         //G__letint(&result3,type ,(char)G__int_cast(result3));
         result3.obj.ch = (char)G__int_cast(result3);
         result3.type = type;
         break;
      case 'r':
         //G__letint(&result3,type ,(unsigned short)G__int_cast(result3));
         result3.obj.ush = (unsigned short)G__int_cast(result3);
         result3.type = type;
         break;
      case 's':
         //G__letint(&result3,type ,(short)G__int_cast(result3));
         result3.obj.sh = (short)G__int_cast(result3);
         result3.type = type;
         break;
      case 'h':
         //G__letint(&result3,type ,(unsigned int)G__int_cast(result3));
         result3.obj.uin = (unsigned int)G__int_cast(result3);
         result3.type = type;
         break;
      case 'i':
         //G__letint(&result3,type ,(int)G__int_cast(result3));
         result3.obj.in = (int)G__int_cast(result3);
         result3.type = type;
         break;
      case 'k':
         //G__letint(&result3,type ,(unsigned long)G__int_cast(result3));
         result3.obj.ulo = (unsigned long)G__int_cast(result3);
         result3.type = type;
         break;
      case 'l':
         G__letint(&result3, type , (long)G__int_cast(result3));
         break;
      case 'm':
         G__letULonglong(&result3, type , G__ULonglong(result3));
         break;
      case 'n':
         G__letLonglong(&result3, type , G__Longlong(result3));
         break;
      case 'q':
         G__letLongdouble(&result3, type , G__Longdouble(result3));
         break;
      case 'g':
#ifdef G__BOOL4BYTE
         G__letint(&result3, type , (int)(G__int_cast(result3) ? 1 : 0));
#else // G__BOOL4BYTE
         G__letint(&result3, type , (unsigned char)(G__int_cast(result3) ? 1 : 0));
#endif // G__BOOL4BYTE
         break;
      default:
         G__letint(&result3, type, G__int(result3));
         if (islower(type)) result3.ref = result3.obj.i;
         result3.obj.reftype.reftype = reftype;
         break;
   }
   return result3;
}

//______________________________________________________________________________
G__value G__castvalue(char* casttype, G__value result3)
{
   // -- FIXME: Describe this function!
   return G__castvalue_bc(casttype, result3, 1);
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
long G__int_cast(G__value buf)
{
   // -- FIXME: Describe this function!
   return G__convertT<long>(&buf);
}

} // extern "C"

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
