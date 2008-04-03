/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file val2a.c
 ************************************************************************
 * Description:
 *  G__value to ASCII expression
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Dict.h"
#include "value.h"

#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

//______________________________________________________________________________
char* Cint::Internal::G__valuemonitor(G__value buf, char* temp)
{
   char temp2[G__ONELINE];
   if (G__value_typenum(buf).IsTypedef()) {
      switch (G__get_type(G__value_typenum(buf))) {
         case 'd':
         case 'f':
            /* typedef can be local to a class */
            if (buf.obj.d < 0.0)
               sprintf(temp, "(%s)(%.17e)", G__value_typenum(buf).Name(::Reflex::SCOPED).c_str()
                       , G__convertT<double>(&buf));
            else
               sprintf(temp, "(%s)%.17e", G__value_typenum(buf).Name(::Reflex::SCOPED).c_str()
                       , G__convertT<double>(&buf));
            break;
         case 'b':
            if (G__in_pause)
               sprintf(temp, "(unsigned char)%u", G__convertT<unsigned char>(&buf));
            else
               sprintf(temp, "(unsignedchar)%u", G__convertT<unsigned char>(&buf));
            break;
         case 'r':
            if (G__in_pause)
               sprintf(temp, "(unsigned short)%u", G__convertT<unsigned short>(&buf));
            else
               sprintf(temp, "(unsignedshort)%u", G__convertT<unsigned short>(&buf));
            break;
         case 'h':
            if (G__in_pause)
               sprintf(temp, "(unsigned int)%u", G__convertT<unsigned int>(&buf));
            else
               sprintf(temp, "(unsignedint)%u", G__convertT<unsigned int>(&buf));
            break;
         case 'k':
            if (G__in_pause)
               sprintf(temp, "(unsigned long)%lu", G__convertT<unsigned long>(&buf));
            else
               sprintf(temp, "(unsignedlong)%lu,%#08lX", G__convertT<unsigned long>(&buf), G__convertT<unsigned long>(&buf));
            break;
         default:
            if (islower(G__get_type(G__value_typenum(buf)))) {
               if ('u' == G__get_type(G__value_typenum(buf)) && G__value_typenum(buf) &&
                     (G__value_typenum(buf).IsClass() || G__value_typenum(buf).IsStruct())) {
                  ::Reflex::Type type = G__value_typenum(buf);
                  if (type.Name() == "G__longlong" ||
                        type.Name() == "G__ulonglong" ||
                        type.Name() == "G__longdouble") {
                     if (G__in_pause) {
                        char llbuf[100];
                        sprintf(temp, "(%s)", type.Name(::Reflex::SCOPED).c_str());
                        if (type.RawType().Name() == "G__longlong") {
                           sprintf(llbuf
                                   , "G__printformatll((char*)(%ld),\"%%lld\",(void*)(%ld))"
                                   , (long)llbuf, buf.obj.i);
                           G__getitem(llbuf);
                           strcat(temp, llbuf);
                        }
                        else if (type.RawType().Name() == "G__ulonglong") {
                           sprintf(llbuf
                                   , "G__printformatull((char*)(%ld),\"%%llu\",(void*)(%ld))"
                                   , (long)llbuf, buf.obj.i);
                           G__getitem(llbuf);
                           strcat(temp, llbuf);
                        }
                        else if (type.RawType().Name() == "G__longdouble") {
                           sprintf(llbuf
                                   , "G__printformatld((char*)(%ld),\"%%LG\",(void*)(%ld))"
                                   , (long)llbuf, buf.obj.i);
                           G__getitem(llbuf);
                           strcat(temp, llbuf);
                        }
                     }
                     else
                        G__setiparseobject(&buf, temp);
                  }
                  else {
                     sprintf(temp, "(class %s)%ld", type.Name(::Reflex::SCOPED).c_str(),buf.obj.i);
                  }
               }
               else
#if defined(G__WIN32)
                  if (G__get_type(G__value_typenum(buf)) == 'n' && buf.obj.ll < 0)
                     sprintf(temp, "(%s)(%I64d)"
                             , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                              , G__get_typenum(G__value_typenum(buf))
                                              , 0, 0)
                             , buf.obj.ll);
                  else if (G__get_type(G__value_typenum(buf)) == 'm' || G__get_type(G__value_typenum(buf)) == 'n')
                     sprintf(temp, "(%s)%I64u"
                             , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                              , G__get_typenum(G__value_typenum(buf))
                                              , 0, 0)
                             , buf.obj.ull);

                  else
#else
                  if (G__get_type(G__value_typenum(buf)) == 'n' && buf.obj.ll < 0)
                     sprintf(temp, "(%s)(%lld)"
                             , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                              , G__get_typenum(G__value_typenum(buf))
                                              , 0, 0)
                             , buf.obj.ll);
                  else if (G__get_type(G__value_typenum(buf)) == 'm' || G__get_type(G__value_typenum(buf)) == 'n')
                     sprintf(temp, "(%s)%llu"
                             , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                              , G__get_typenum(G__value_typenum(buf))
                                              , 0, 0)
                             , buf.obj.ull);

                  else
#endif
                     if (buf.obj.i < 0)
                        sprintf(temp, "(%s)(%ld)"
                                , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                                 , G__get_typenum(G__value_typenum(buf))
                                                 , G__get_reftype(G__value_typenum(buf)), 0)
                                , buf.obj.i);
                     else
                        sprintf(temp, "(%s)%ld"
                                , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                                 , G__get_typenum(G__value_typenum(buf))
                                                 , G__get_reftype(G__value_typenum(buf)), 0)
                                , buf.obj.i);
            }
            else {
               if ('C' == G__get_type(G__value_typenum(buf)) && G__in_pause && buf.obj.i > 0x10000 &&
                     G__PARANORMAL == G__get_reftype(G__value_typenum(buf)))
                  sprintf(temp, "(%s 0x%lx)\"%s\""
                          , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                           , G__get_typenum(G__value_typenum(buf))
                                           , G__get_reftype(G__value_typenum(buf)), 0)
                          , buf.obj.i, (char*)buf.obj.i);
               else
                  sprintf(temp, "(%s)0x%lx"
                          , G__type2string(G__get_type(G__value_typenum(buf)) , G__get_tagnum(G__value_typenum(buf))
                                           , G__get_typenum(G__value_typenum(buf))
                                           , G__get_reftype(G__value_typenum(buf)), 0)
                          , buf.obj.i);
            }
      }
      return(temp);
   }
   switch (G__get_type(G__value_typenum(buf))) {
      case '\0':
         sprintf(temp, "NULL");
         break;
      case 'b':
         if (G__in_pause)
            sprintf(temp, "(unsigned char)%u", G__convertT<unsigned char>(&buf));
         else
            sprintf(temp, "(unsignedchar)%u", G__convertT<unsigned char>(&buf));
         break;
      case 'B':
         if (G__in_pause)
            sprintf(temp, "(unsigned char*)0x%lx", buf.obj.i);
         else
            sprintf(temp, "(unsignedchar*)0x%lx", buf.obj.i);
         break;
      case 'T':
      case 'C':
         if (buf.obj.i != 0) {
            if (G__in_pause && G__PARANORMAL == G__get_reftype(G__value_typenum(buf))) {
               if (strlen((char*)buf.obj.i) > G__ONELINE - 25) {
                  strncpy(temp2, (char*)buf.obj.i, G__ONELINE - 25);
                  temp2[G__ONELINE-25] = 0;
                  sprintf(temp, "(char* 0x%lx)\"%s\"...", buf.obj.i, temp2);
               }
               else {
                  G__add_quotation((char*)buf.obj.i, temp2);
                  sprintf(temp, "(char* 0x%lx)%s", buf.obj.i, temp2);
               }
            }
            else {
               sprintf(temp, "(char*)0x%lx", buf.obj.i);
            }
         }
         else {
            if (G__in_pause)
               sprintf(temp, "(char* 0x0)\"\"");
            else
               sprintf(temp, "(char*)0x0");
         }
         break;
      case 'c':
         G__charaddquote(temp2, G__convertT<char>(&buf));
         if (G__in_pause)
            sprintf(temp, "(char %d)%s", G__convertT<char>(&buf), temp2);
         else
            sprintf(temp, "(char)%d", G__convertT<char>(&buf));
         break;
      case 'r':
         if (G__in_pause)
            sprintf(temp, "(unsigned short)%u", G__convertT<unsigned short>(&buf));
         else
            sprintf(temp, "(unsignedshort)%u", G__convertT<unsigned short>(&buf));
         break;
      case 'R':
         if (G__in_pause)
            sprintf(temp, "(unsigned short*)0x%lx", buf.obj.i);
         else
            sprintf(temp, "(unsignedshort*)0x%lx", buf.obj.i);
         break;
      case 's':
         if (buf.obj.i < 0)
            sprintf(temp, "(short)(%d)", G__convertT<short>(&buf));
         else
            sprintf(temp, "(short)%d", G__convertT<short>(&buf));
         break;
      case 'S':
         sprintf(temp, "(short*)0x%lx", buf.obj.i);
         break;
      case 'h':
         if (G__in_pause)
            sprintf(temp, "(unsigned int)%u", G__convertT<unsigned int>(&buf));
         else
            sprintf(temp, "(unsignedint)%u", G__convertT<unsigned int>(&buf));
         break;
      case 'H':
         if (G__in_pause)
            sprintf(temp, "(unsigned int*)0x%lx", buf.obj.i);
         else
            sprintf(temp, "(unsignedint*)0x%lx", buf.obj.i);
         break;
      case 'i':
         if (G__value_typenum(buf)) {
            if (G__value_typenum(buf).RawType().IsEnum()) {
               if (buf.obj.i < 0)
                  sprintf(temp, "(enum %s)(%d)", G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), G__convertT<int>(&buf));
               else
                  sprintf(temp, "(enum %s)%d", G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), G__convertT<int>(&buf));
            }
            else {
               if (buf.obj.i < 0)
                  sprintf(temp, "(int)(%d)", G__convertT<int>(&buf));
               else
                  sprintf(temp, "(int)%d", G__convertT<int>(&buf));
            }
         }
         else {
            if (buf.obj.i < 0)
               sprintf(temp, "(int)(%d)", G__convertT<int>(&buf));
            else
               sprintf(temp, "(int)%d", G__convertT<int>(&buf));
         }
         break;
      case 'I':
         if (G__value_typenum(buf)) {
            if (G__value_typenum(buf).RawType().IsEnum()) {
               sprintf(temp, "(enum %s*)0x%lx", G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), buf.obj.i);
            }
            else {
               sprintf(temp, "(int*)0x%lx", buf.obj.i);
            }
         }
         else {
            sprintf(temp, "(int*)0x%lx", buf.obj.i);
         }
         break;
#if defined(G__WIN32)
      case 'n':
         if (buf.obj.ll < 0)
            sprintf(temp, "(long long)(%I64d)", buf.obj.ll);
         else
            sprintf(temp, "(long long)%I64d", buf.obj.ll);
         break;
      case 'm':
         sprintf(temp, "(unsigned long long)%I64u", buf.obj.ull);
         break;
#else
      case 'n':
         if (buf.obj.ll < 0)
            sprintf(temp, "(long long)(%lld)", buf.obj.ll);
         else
            sprintf(temp, "(long long)%lld", buf.obj.ll);
         break;
      case 'm':
         sprintf(temp, "(unsigned long long)%llu", buf.obj.ull);
         break;
#endif
      case 'q':
         if (buf.obj.ld < 0)
            sprintf(temp, "(long double)(%Lg)", buf.obj.ld);
         else
            sprintf(temp, "(long double)%Lg", buf.obj.ld);
         break;
      case 'g':
         sprintf(temp, "(bool)%d", G__convertT<bool>(&buf));
         break;
      case 'k':
         if (G__in_pause)
            sprintf(temp, "(unsigned long)%lu", G__convertT<unsigned long>(&buf));
         else
            sprintf(temp, "(unsignedlong)%lu", G__convertT<unsigned long>(&buf));
         break;
      case 'K':
         if (G__in_pause)
            sprintf(temp, "(unsigned long*)0x%lx", buf.obj.i);
         else
            sprintf(temp, "(unsignedlong*)0x%lx", buf.obj.i);
         break;
      case 'l':
         if (buf.obj.i < 0)
            sprintf(temp, "(long)(%ld)", buf.obj.i);
         else
            sprintf(temp, "(long)%ld", buf.obj.i);
         break;
      case 'L':
         sprintf(temp, "(long*)0x%lx", buf.obj.i);
         break;
      case 'y':
         if (buf.obj.i < 0)
            sprintf(temp, "(void)(%ld)", buf.obj.i);
         else
            sprintf(temp, "(void)%ld", buf.obj.i);
         break;
#ifndef G__OLDIMPLEMENTATION2191
      case '1':
#else
      case 'Q':
#endif
      case 'Y':
         sprintf(temp, "(void*)0x%lx", buf.obj.i);
         break;
      case 'E':
         sprintf(temp, "(FILE*)0x%lx", buf.obj.i);
         break;
      case 'd':
         if (buf.obj.d < 0.0)
            sprintf(temp, "(double)(%.17e)", buf.obj.d);
         else
            sprintf(temp, "(double)%.17e", buf.obj.d);
         break;
      case 'D':
         sprintf(temp, "(double*)0x%lx", buf.obj.i);
         break;
      case 'f':
         if (buf.obj.d < 0.0)
            sprintf(temp, "(float)(%.17e)", buf.obj.d);
         else
            sprintf(temp, "(float)%.17e", buf.obj.d);
         break;
      case 'F':
         sprintf(temp, "(float*)0x%lx", buf.obj.i);
         break;
      case 'u':
         {
            ::Reflex::Type type(G__value_typenum(buf).RawType());
            switch (G__get_tagtype(type)) {
               case 's':
                  if (buf.obj.i < 0) {
                     sprintf(temp, "(struct %s)(%ld)", G__value_typenum(buf).Name(::Reflex::SCOPED).c_str(), buf.obj.i);
                  }
                  else {
                     sprintf(temp, "(struct %s)%ld", G__value_typenum(buf).Name(::Reflex::SCOPED).c_str(), buf.obj.i);
                  }
                  break;
               case 'c':
                  if (
                     (type.Name() == "G__longlong") ||
                     (type.Name() == "G__ulonglong") ||
                     (type.Name() == "G__longdouble")
                  ) {
                     if (G__in_pause) {
                        char llbuf[100];
                        sprintf(temp, "(%s)", G__type2string(G__get_type(type), G__get_tagnum(type), G__get_typenum(G__value_typenum(buf)), G__get_reftype(G__value_typenum(buf)), 0));
                        if (type.Name() == "G__longlong") {
                           sprintf(llbuf, "G__printformatll((char*)(%ld),\"%%lld\",(void*)(%ld))", (long) llbuf, buf.obj.i);
                           G__getitem(llbuf);
                           strcat(temp, llbuf);
                        }
                        else if (type.Name() == "G__ulonglong") {
                           sprintf(llbuf, "G__printformatull((char*)(%ld),\"%%llu\",(void*)(%ld))", (long) llbuf, buf.obj.i);
                           G__getitem(llbuf);
                           strcat(temp, llbuf);
                        }
                        else if (type.Name() == "G__longdouble") {
                           sprintf(llbuf, "G__printformatld((char*)(%ld),\"%%LG\",(void*)(%ld))", (long) llbuf, buf.obj.i);
                           G__getitem(llbuf);
                           strcat(temp, llbuf);
                        }
                     }
                     else
                        G__setiparseobject(&buf, temp);
                  }
                  else {
                     if (buf.obj.i < 0) {
                        sprintf(temp, "(class %s)(%ld)", type.Name(::Reflex::SCOPED).c_str(), buf.obj.i);
                     }
                     else {
                        sprintf(temp, "(class %s)%ld", type.Name(::Reflex::SCOPED).c_str(), buf.obj.i);
                     }
                  }
                  break;
               case 'u':
                  if (buf.obj.i < 0) {
                     sprintf(temp, "(union %s)(%ld)", type.Name(::Reflex::SCOPED).c_str(), buf.obj.i);
                  }
                  else {
                     sprintf(temp, "(union %s)%ld", type.Name(::Reflex::SCOPED).c_str(), buf.obj.i);
                  }
                  break;
               case 'e':
                  sprintf(temp, "(enum %s)%d", type.Name(::Reflex::SCOPED).c_str(), (int) buf.obj.i);
                  break;
               default:
                  if (buf.obj.i < 0) {
                     sprintf(temp, "(unknown %s)(%ld)", type.Name().c_str(), buf.obj.i);
                  }
                  else {
                     sprintf(temp, "(unknown %s)%ld" , type.Name().c_str(), buf.obj.i);
                  }
                  break;
            }
         }
         break;
      case 'U':
         switch (G__get_tagtype(G__value_typenum(buf))) {
            case 's':
               sprintf(temp, "(struct %s*)0x%lx" , G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), buf.obj.i);
               break;
            case 'c':
               sprintf(temp, "(class %s*)0x%lx" , G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), buf.obj.i);
               break;
            case 'u':
               sprintf(temp, "(union %s*)0x%lx" , G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), buf.obj.i);
               break;
            case 'e':
               sprintf(temp, "(enum %s*)0x%lx" , G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), buf.obj.i);
               break;
            default:
               sprintf(temp, "(unknown %s*)0x%lx", G__value_typenum(buf).RawType().Name(::Reflex::SCOPED).c_str(), buf.obj.i);
               break;
         }
         break;
      case 'w':
         G__logicstring(buf, 1, temp2);
         sprintf(temp, "(logic)0b%s", temp2);
         break;
      default:
         if (buf.obj.i < 0) {
            sprintf(temp, "(unknown)(%ld)", buf.obj.i);
         }
         else {
            sprintf(temp, "(unknown)%ld", buf.obj.i);
         }
         break;
   }
   if (isupper(G__get_type(G__value_typenum(buf)))) {
      char sbuf[G__ONELINE];
      char* p = strchr(temp, '*');
      switch (G__get_reftype(G__value_typenum(buf))) {
         case G__PARAP2P:
            strcpy(sbuf, p);
            strcpy(p + 1, sbuf);
            break;
         case G__PARAP2P2P:
            strcpy(sbuf, p);
            *(p + 1) = '*';
            strcpy(p + 2, sbuf);
            break;
         case G__PARANORMAL:
            break;
         default:
            strcpy(sbuf, p);
            for (int i = G__PARAP2P - 1; i < G__get_reftype(G__value_typenum(buf)); ++i) {
               *(p + i) = '*';
            }
            strcpy(p + G__get_reftype(G__value_typenum(buf)) - G__PARAP2P + 1, sbuf);
            break;
      }
   }
   return temp;
}

//______________________________________________________________________________
char* Cint::Internal::G__access2string(int caccess)
{
   switch (caccess) {
      case G__PRIVATE:
         return("private:");
      case G__PROTECTED:
         return("protected:");
      case G__PUBLIC:
         return("public:");
   }
   return("");
}

//______________________________________________________________________________
char* Cint::Internal::G__tagtype2string(int tagtype)
{
   switch (tagtype) {
      case 'c':
         return("class");
      case 's':
         return("struct");
      case 'e':
         return("enum");
      case 'u':
         return("union");
      case 'n':
         return("namespace");
      case  0 :
         return("(unknown)");
   }
   G__genericerror("Internal error: Unexpected tagtype G__tagtype2string()");
   return("");
}

//______________________________________________________________________________
extern "C" char* G__fulltagname(int tagnum, int mask_dollar)
{
   // -- Return full tagname, if mask_dollar=1, $ for the typedef class is omitted.
#ifndef G__OLDIMPLEMENTATION1823
   static char string[G__LONGLINE];
#else // G__OLDIMPLEMENTATION1823
   static char string[G__ONELINE];
#endif // G__OLDIMPLEMENTATION1823
   int p_tagnum[G__MAXBASE];
   int pt;
   int len = 0;
   int os;
   if (-1 == tagnum || G__struct.alltag == 0) {
      return "";
   }
   /* enclosed class scope , need to go backwards */
   p_tagnum[0] = G__struct.parent_tagnum[tagnum];
   for (pt = 0; (p_tagnum[pt] > 0) && (pt < (G__MAXBASE - 1)); pt++) {
      p_tagnum[pt+1] = G__struct.parent_tagnum[p_tagnum[pt]];
   }
   if (pt == G__MAXBASE) {
      return "";
   }
   while (pt) {
      --pt;
      os = mask_dollar && ('$' == G__struct.name[p_tagnum[pt]][0] && '$' != G__struct.name[p_tagnum[pt]][ strlen(G__struct.name[p_tagnum[pt]]) - 1]);
#define G__OLDIMPLEMENTATION1503
#ifndef G__OLDIMPLEMENTATION1503
      if (G__struct.defaulttypenum[p_tagnum[pt]]) {
         sprintf(string + len, "%s::", G__struct.defaulttypenum[p_tagnum[pt]].Name().c_str());
      }
      else {
         sprintf(string + len, "%s::", G__struct.name[p_tagnum[pt]] + os);
      }
#else // G__OLDIMPLEMENTATION1503
      sprintf(string + len, "%s::", G__struct.name[p_tagnum[pt]] + os);
#endif // G__OLDIMPLEMENTATION1503
      len = strlen(string);
   }
   os = mask_dollar && ('$' == G__struct.name[tagnum][0] && '$' != G__struct.name[tagnum][strlen(G__struct.name[tagnum]) - 1]);
#ifndef G__OLDIMPLEMENTATION1503
   if (G__struct.defaulttypenum[tagnum]) {
      sprintf(string + len, "%s", G__struct.defaulttypenum[tagnum].Name().c_str());
   }
   else {
      sprintf(string + len, "%s", G__struct.name[tagnum] + os);
   }
#else // G__OLDIMPLEMENTATION1503
   sprintf(string + len, "%s", G__struct.name[tagnum] + os);
#endif // G__OLDIMPLEMENTATION1503
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
   {
      char *ptr = strstr(string, "long long");
      if (ptr) {
         memcpy(ptr, " __int64 ", strlen(" __int64 "));
      }
   }
#endif // _MSC_VER && _MSC_VER < 1310
   return string;
}

//______________________________________________________________________________
extern "C" char* G__type2string(int type, int tagnum, int typenum_in, int reftype, int isconst)
{
   static char stringbuf[G__LONGLINE];
   char* string = stringbuf;
   int ref = G__REF(reftype);
   reftype = G__PLVL(reftype);
   int len;
   int i;
   ::Reflex::Type typenum = G__Dict::GetDict().GetTypedef(typenum_in);
   if ((isconst & G__CONSTVAR) && (!typenum || !(isconst & G__get_isconst(typenum.ToType())))) {
      strcpy(string, "const ");
      string += 6;
   }
   // Handle G__longlong, G__ulonglong, and G__longdouble early
   if (!typenum && (tagnum != -1)) {
      char* ss = G__struct.name[tagnum];
      if (!strcmp(ss, "G__longlong") && !G__defined_macro("G__LONGLONGTMP")) {
         strcpy(stringbuf, "long long");
         return stringbuf;
      }
      if (!strcmp(ss, "G__ulonglong") && !G__defined_macro("G__LONGLONGTMP")) {
         strcpy(stringbuf, "unsigned long long");
         return stringbuf;
      }
      if (!strcmp(ss, "G__longdouble") && !G__defined_macro("G__LONGLONGTMP")) {
         strcpy(stringbuf, "long double");
         return stringbuf;
      }
   }
#ifndef G__OLDIMPLEMENTATION1503
   if (
      !typenum &&
      (tagnum != 0) &&
      (G__struct.defaulttypenum[tagnum] != -1) &&
      (G__get_type(G__Dict::GetDict().GetTypedef(G__struct.defaulttypenum[tagnum])) == 'u')
   ) {
      typenum = G__Dict::GetDict().GetTypedef(G__struct.defaulttypenum[tagnum]);
   }
#endif // G__OLDIMPLEMENTATION1503
   //
   //  Handle base type.
   //
   if (typenum) { // typedef
      strcpy(string, typenum.Name(::Reflex::SCOPED).c_str());
      if (G__get_nindex(typenum)) { // We have array bounds.
         int pointlevel = 0;
         if (isupper(type)) {
            pointlevel = 1;
         }
         switch (reftype) {
            case G__PARANORMAL:
            case G__PARAREFERENCE:
               break;
            default:
               pointlevel = reftype;
               break;
         }
         pointlevel -= G__get_nindex(typenum);
         switch (pointlevel) {
            case 0:
               type = tolower(type);
               if (reftype != G__PARAREFERENCE) {
                  reftype = G__PARANORMAL;
               }
               break;
            case 1:
               type = toupper(type);
               if (reftype != G__PARAREFERENCE) {
                  reftype = G__PARANORMAL;
               }
               break;
            default:
               if (pointlevel > 0) {
                  type = toupper(type);
                  reftype = pointlevel;
               }
               break;
         }
      }
      if (isupper(G__get_type(typenum))) { // We are a pointer.
         switch (G__get_reftype(typenum)) {
            case G__PARANORMAL:
            case G__PARAREFERENCE:
               if (isupper(type)) {
                  switch (reftype) {
                     case G__PARAREFERENCE:
                     case G__PARANORMAL:
                        type = tolower(type);
                        break;
                     case G__PARAP2P:
                        reftype = G__PARANORMAL;
                        break;
                     default:
                        --reftype;
                        break;
                  }
               }
               else {
                  G__type2string(type, tagnum, -1, reftype, isconst); // Note: The return buffer is static, so stringbuf is set above.
                  goto endof_type2string;
               }
               break;
            default:
               // --
#ifndef G__OLDIMPLEMENTATION2191
               if (type == '1') {
                  switch (reftype) {
                     case G__PARAREFERENCE:
                     case G__PARANORMAL:
                        type = tolower(type);
                        break;
                     case G__PARAP2P:
                        reftype = G__PARANORMAL;
                        break;
                     default:
                        --reftype;
                        break;
                  }
               }
#else // G__OLDIMPLEMENTATION2191
               if (type == 'Q') {
                  if (isupper(type)) {
                     switch (reftype) {
                        case G__PARAREFERENCE:
                        case G__PARANORMAL:
                           type = tolower(type);
                           break;
                        case G__PARAP2P:
                           reftype = G__PARANORMAL;
                           break;
                        default:
                           --reftype;
                           break;
                     }
                  }
               }
#endif // G__OLDIMPLEMENTATION2191
               else {
                  if (islower(type) || G__get_reftype(typenum.ToType()) > reftype) {
                     G__type2string(type, tagnum, -1, reftype, isconst);
                     goto endof_type2string;
                  }
                  else if (G__get_reftype(typenum.ToType()) == reftype) {
                     reftype = G__PARANORMAL;
                     type = tolower(type);
                  }
                  else if (G__get_reftype(typenum.ToType()) + 1 == reftype) {
                     reftype = G__PARANORMAL;
                  }
                  else {
                     reftype = G__PARAP2P + reftype - G__get_reftype(typenum.ToType()) - 2;
                  }
               }
               break;
         }
      }
   }
   else if (tagnum != -1) { // class/struct/union/enum/namespace
      if (tagnum >= G__struct.alltag || !G__struct.name[tagnum]) {
         strcpy(stringbuf, "(invalid_class)");
         return stringbuf;
      }
      if (G__struct.name[tagnum][0] == '$') { // unnamed class/struct/union/enum/namespace
         len = 0;
         if (!G__struct.name[tagnum][1]) { // name is only '$', must be unnamed enum
            G__ASSERT(G__struct.type[tagnum] == 'e');
            strcpy(string, "enum ");
            len = 5;
         }
      }
      else { // nothing special, normal named struct
         if ((G__globalcomp == G__CPPLINK) || G__iscpp) {
            len = 0;
         }
         else {
            switch (G__struct.type[tagnum]) {
               case 'e':
                  strcpy(string, "enum ");
                  break;
               case 'c':
                  strcpy(string, "class ");
                  break;
               case 's':
                  strcpy(string, "struct ");
                  break;
               case 'u':
                  strcpy(string, "union ");
                  break;
               case 'n':
                  sprintf(string, "namespace ");
                  break;
               case 'a':
                  strcpy(string, "");
                  break;
               case 0:
                  sprintf(string, "(unknown) ");
                  break;
            }
            len = strlen(string);
         }
      }
      sprintf(string + len, "%s", G__fulltagname(tagnum, 1));
   }
   else { // fundamental type
      switch (tolower(type)) {
         case 'b':
            strcpy(string, "unsigned char");
            break;
         case 'c':
            strcpy(string, "char");
            break;
         case 'r':
            strcpy(string, "unsigned short");
            break;
         case 's':
            strcpy(string, "short");
            break;
         case 'h':
            strcpy(string, "unsigned int");
            break;
         case 'i':
            strcpy(string, "int");
            break;
         case 'k':
            strcpy(string, "unsigned long");
            break;
         case 'l':
            strcpy(string, "long");
            break;
         case 'g':
            strcpy(string, "bool");
            break;
         case 'n':
            strcpy(string, "long long");
            break;
         case 'm':
            strcpy(string, "unsigned long long");
            break;
         case 'q':
            strcpy(string, "long double");
            break;
         case 'f':
            strcpy(string, "float");
            break;
         case 'd':
            strcpy(string, "double");
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case '1':
#else // G__OLDIMPLEMENTATION2191
         case 'q':
#endif // G__OLDIMPLEMENTATION2191
         case 'y':
            strcpy(string, "void");
            break;
         case 'e':
            strcpy(string, "FILE");
            break;
         case 'u':
            strcpy(string, "enum");
            break;
         case 't':
#ifndef G__OLDIMPLEMENTATION2191
         case 'j':
#else
         case 'm':
#endif
         case 'p':
            sprintf(string, "#define");
            return string;
         case 'o':
            string[0] = '\0';
            return string;
         case 'a':
            strcpy(string, "G__p2memfunc");
            type = tolower(type);
            break;
         default:
            strcpy(string, "(unknown)");
            break;
      }
   }
   //
   //  Handle pointer and reference parts of type.
   //
   if ((type != 'q') && (type != 'a')) {
      // Take care of the first pointer level.
      if (isupper(type)) {
         if ((isconst & G__PCONSTVAR) && (reftype == G__PARANORMAL)) {
            strcpy(string + strlen(string), " *const");
         }
         else {
            strcpy(string + strlen(string), "*");
         }
      }
      // Handle the second and greater pointer levels,
      // and possibly a reference with zero or one pointer.
      switch (reftype) {
         case G__PARANORMAL:
            break;
         case G__PARAREFERENCE:
            if (!typenum || G__get_reftype(typenum.ToType()) != G__PARAREFERENCE) {
               if ((isconst & G__PCONSTVAR) && !(isconst & G__CONSTVAR)) {
                  strcpy(string + strlen(string), " const&");
               }
               else {
                  strcpy(string + strlen(string), "&");
               }
            }
            break;
         case G__PARAP2P:
            if (isconst & G__PCONSTVAR) {
               strcpy(string + strlen(string), " *const");
            }
            else {
               strcpy(string + strlen(string), "*");
            }
            break;
         case G__PARAP2P2P:
            if (isconst & G__PCONSTVAR) {
               strcpy(string + strlen(string), " **const");
            }
            else {
               strcpy(string + strlen(string), "**");
            }
            break;
         default:
            if ((reftype > 10) || (reftype < 0)) { // workaround
               break;
            }
            if (isconst & G__PCONSTVAR) {
               strcpy(string + strlen(string), " ");
            }
            for (i = G__PARAP2P; i <= reftype; ++i) {
               strcpy(string + strlen(string), "*");
            }
            if (isconst & G__PCONSTVAR) {
               strcpy(string + strlen(string), " const");
            }
            break;
      }
   }
   endof_type2string:
   // Handle a reference to two or more pointer levels.
   if (ref) { // We are a reference to a pointer to xxx.
      strcat(stringbuf, "&");
   }
   if (strlen(stringbuf) >= sizeof(stringbuf)) {
      G__fprinterr(G__serr, "Error in G__type2string: string length (%d) greater than buffer length (%d)!", strlen(stringbuf), sizeof(stringbuf));
      G__genericerror(0);
   }
   return stringbuf;
}

//______________________________________________________________________________
int Cint::Internal::G__val2pointer(G__value* result7)
{
   if (0 == result7->ref) {
      G__genericerror("Error: incorrect use of referencing operator '&'");
      return(1);
   }

   G__value_typenum(*result7) = ::Reflex::PointerBuilder(G__value_typenum(*result7));
   result7->obj.i = result7->ref;
   result7->ref = 0;

#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      G__fprinterr(G__serr, "%3x: TOPNTR\n", G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__TOPNTR;
      G__inc_cp_asm(1, 0);
   }
#endif

   return(0);
}

//______________________________________________________________________________
char* Cint::Internal::G__getbase(unsigned int expression, int base, int digit, char* result1)
{
   char result[G__MAXNAME];
   int ig18 = 0, ig28 = 0;
   unsigned int onedig, value;  /* bug fix  3 mar 1993 */

   value = expression;

   while ((ig28 < digit) || ((digit == 0) && (value != 0))) {
      onedig = value % base ;
      result[ig28] = G__getdigit(onedig);
      value = (value - onedig) / base;
      ig28++ ;
   }
   ig28-- ;

   /*
   result1[ig18++]='0' ;
   switch(base) {
   case 2:
           result1[ig18++]='b' ;
           break;
   case 4:
           result1[ig18++]='q' ;
           break;
   case 8:
           result1[ig18++]='o' ;
           break;
   case 10:
           result1[ig18++]='d' ;
           break;
   case 16:
           result1[ig18++]='x' ;
           break;
   default:
           result1[ig18++]= base + '0' ;
   }
   */

   while (0 <= ig28) {
      result1[ig18++] = result[ig28--] ;
   }
   if (ig18 == 0) {
      result1[ig18++] = '0';
   }
   result1[ig18] = '\0';

   return(result1);
}

//______________________________________________________________________________
int Cint::Internal::G__getdigit(unsigned int number)
{
   switch (number) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
         return(number + '0');
      case 10:
         return('a');
      case 11:
         return('b');
      case 12:
         return('c');
      case 13:
         return('d');
      case 14:
         return('e');
      case 15:
         return('f');
      default:
         return('x');
   }
   /* return('x'); */
}

//______________________________________________________________________________
G__value Cint::Internal::G__checkBase(char* string, int* known4)
{
   G__value result4;
   int n = 0, nchar, base = 0;
   G__uint64 value = 0, tristate = 0;
   char type;
   int unsign = 0;

   /********************************************
   * default type int, becomes long through strlen check
   ********************************************/
   type = 'i';

   /********************************************
   * count number of characters
   ********************************************/
   nchar = strlen(string);


   /********************************************
   * scan string
   ********************************************/
   while (n < nchar) {

      /********************************************
       * error check, never happen because same
       * condition is already checked in G__getitem()
       * Thus, this condition can be removed
       ********************************************/
      if (string[0] != '0') {
         G__fprinterr(G__serr, "Error: G__checkBase(%s) " , string);
         G__genericerror((char*)NULL);
         return(G__null);
      }
      else {
         /**********************************************
          * check base specification
          **********************************************/
         switch (string[++n]) {
            case 'b':
            case 'B':
               /*******************************
                * binary base doesn't exist in
                * ANSI. Original enhancement
                *******************************/
               base = 2;
               break;
            case 'q':
            case 'Q':
               /******************************
                * quad specifier may not exist
                * in ANSI
                ******************************/
               base = 4;
               break;
            case 'o':
            case 'O':
               /******************************
                * octal specifier may not exist
                * in ANSI
                ******************************/
               base = 8;
               break;
            case 'h':
            case 'H':
               /*******************************
                * 0h and 0H doesn't exist in
                * ANSI. Original enhancement
                *******************************/
            case 'x':
            case 'X':
               base = 16;
               break;
            default:
               /*******************************
                * Octal if other specifiers,
                * namely a digit
                *******************************/
               base = 8;
               --n;
               break;
         }

         /********************************************
          * initialize value
          ********************************************/
         value = 0;
         tristate = 0;

         /********************************************
          * increment scanning pointer
          ********************************************/
         n++;

         /********************************************
          * scan constant expression body
          ********************************************/
         while ((string[n] != ' ') && (string[n] != '\t') && (n < nchar)) {
            switch (string[n]) {
               case '0':
                  /*
                    case 'L':
                    case 'l':
                    */
                  value = value * base;
                  tristate *= base;
                  break;
               case '1':
                  /*
                    case 'H':
                    case 'h':
                    */
                  value = value * base + 1;
                  tristate *= base;
                  break;
               case '2':
                  value = value * base + 2;
                  tristate *= base;
                  break;
               case '3':
                  value = value * base + 3;
                  tristate *= base;
                  break;
               case '4':
                  value = value * base + 4;
                  tristate *= base;
                  break;
               case '5':
                  value = value * base + 5;
                  tristate *= base;
                  break;
               case '6':
                  value = value * base + 6;
                  tristate *= base;
                  break;
               case '7':
                  value = value * base + 7;
                  tristate *= base;
                  break;
               case '8':
                  value = value * base + 8;
                  tristate *= base;
                  break;
               case '9':
                  value = value * base + 9;
                  tristate *= base;
                  break;
               case 'a':
               case 'A':
                  value = value * base + 10;
                  tristate *= base;
                  break;
               case 'b':
               case 'B':
                  value = value * base + 11;
                  tristate *= base;
                  break;
               case 'c':
               case 'C':
                  value = value * base + 12;
                  tristate *= base;
                  break;
               case 'd':
               case 'D':
                  value = value * base + 13;
                  tristate *= base;
                  break;
               case 'e':
               case 'E':
                  value = value * base + 14;
                  tristate *= base;
                  break;
               case 'f':
               case 'F':
                  value = value * base + 15;
                  tristate *= base;
                  break;
               case 'l':
               case 'L':
                  /********************************************
                   * long
                   ********************************************/
                  type = 'l';
                  break;
               case 'u':
               case 'U':
                  /********************************************
                   * unsigned
                   ********************************************/
                  unsign = 1;
                  break;
#ifdef G__NEVER
               case 's':
               case 'S':
                  /********************************************
                   * short, may not exist in ANSI
                   * and doesn't work fine. 
                   * shoud be done as casting
                   ********************************************/
                  type = 's';
                  break;
#endif
               case 'x':
               case 'X':
                  /***************************************
                   * Not ANSI
                   * enhancement for tristate logic expression
                   ***************************************/
                  value *= base;
                  tristate = tristate * base + (base - 1);
                  break;
               case 'z':
               case 'Z':
                  /***************************************
                   * Not ANSI
                   * enhancement for tristate logic expression
                   ***************************************/
                  value = value * base + (base - 1);
                  tristate = tristate * base + (base - 1);
                  break;
               default:
                  value = value * base;
                  G__fprinterr(G__serr, "Error: unexpected character in expression %s "
                               , string);
                  G__genericerror((char*)NULL);
                  break;
            }
            n++;
         }
      }
   }
   *known4 = 1;

   /*******************************************************
   * store constant value and type to result4
   *******************************************************/

   // determine whether int is enough to hold value
   // standard says:
   // non-decimals literals have smallest type they fit in,
   // precedence int, uint, long, ulong,...
   if (type == 'i') {
      if (value > (G__uint64(-1ll)) / 2)
         type = 'm'; // ull
      else if (value > ULONG_MAX)
         type = 'n'; // ll
      else if (value > LONG_MAX)
         type = 'k'; // ul
      else if (value > UINT_MAX)
         type = 'l'; // l
      else if (value > INT_MAX)
         type = 'h'; // u
   }

   if (type == 'i' || type == 'n' || type == 'l')
      type = type - unsign ;
   if (type == 'm')
      G__letULonglong(&result4, type, value);
   else if (type == 'n')
      G__letLonglong(&result4, type, value);
   else
      G__letint(&result4, type, (long)value);

   /*******************************************************
   * Not ANSI
   * if tristate logic , specify it
   *******************************************************/
   if ((base == 2) || (tristate != 0)) {
      *(&result4.obj.i + 1) = tristate;
      G__value_typenum(result4) = G__get_from_type('w', 0);
   }

   return(result4);
}

//______________________________________________________________________________
int Cint::Internal::G__isfloat(char* string, int* type)
{
   int ig17 = 0;
   int c;
   int result = 0, unsign = 0 ;
   unsigned int len = 0;
   static unsigned int lenmaxint = 0;
   static unsigned int lenmaxuint = 0;
   static unsigned int lenmaxlong = 0;
   static unsigned int lenmaxulong = 0;

   if (!lenmaxint) {
      int maxint = INT_MAX;
      unsigned int maxuint = UINT_MAX;

      while (maxint /= 10) ++lenmaxint;
      ++lenmaxint;
      while (maxuint /= 10) ++lenmaxuint;
      ++lenmaxuint;

      long maxlong = LONG_MAX;
      unsigned long maxulong = ULONG_MAX;

      while (maxlong /= 10) ++lenmaxlong;
      ++lenmaxlong;
      while (maxulong /= 10) ++lenmaxulong;
      ++lenmaxulong;
   }

   /*************************************************************
    * default type is int
    *************************************************************/
   *type = 'i';

   /**************************************************************
    * check type of constant expression
    **************************************************************/
   while ((c = string[ig17++]) != '\0') {
      switch (c) {
         case '.':
         case 'e':
         case 'E':
            /******************************
             * double
             ******************************/
            result = 1;
            *type = 'd';
            break;
         case 'f':
         case 'F':
            /******************************
             * float
             ******************************/
            result = 1;
            *type = 'f';
            break;
         case 'l':
         case 'L':
            /******************************
             * long
             ******************************/
#ifndef G__OLDIMPLEMENTATION1874
            if ('l' == *type) *type = 'n';
            else           *type = 'l';
#else
            *type = 'l';
#endif
            break;
#ifdef G__NEVER
         case 's':
         case 'S':
            /******************************
             * short, This may not be
             * included in ANSI
             * and does't work fine.
             * should be done by casting
             ******************************/
            *type = 's';
            break;
#endif
         case 'u':
         case 'U':
            /******************************
             * unsigned
             ******************************/
            unsign = 1;
            break;
         case '0':
         case '1':
         case '2':
         case '3':
         case '4':
         case '5':
         case '6':
         case '7':
         case '8':
         case '9':
            ++len;
            break;
         case '+':
         case '-':
            break;
         default:
            G__fprinterr(G__serr, "Warning: Illegal numerical expression %s", string);
            G__printlinenum();
            break;
      }
   }

   // determine whether unsigned int is enough to hold value
   unsigned int lenmax = unsign ? lenmaxuint : lenmaxint;
   unsigned int lenmaxl = unsign ? lenmaxulong : lenmaxlong;
   if (*type == 'i')
      if (len > lenmax)
         if (len > lenmaxl)
            *type = 'n';
         else
            *type = 'l';
      else if (len == lenmax) {
         long l = atol(string);
         if (!unsign && (l > INT_MAX || l < INT_MIN)
               || unsign && l > (long)UINT_MAX)
            *type = 'l';
      }
      else if (len == lenmaxl) {
         if (unsign) {
            G__uint64 l = G__expr_strtoull(string, 0, 10);
            if (l > ULONG_MAX)
               *type = 'n'; // unsign adjusted below
            else
               *type = 'l';
         }
         else {
            G__int64 l = G__expr_strtoll(string, 0, 10);
            if (l > LONG_MAX || l < LONG_MIN)
               *type = 'n';
            else
               *type = 'l';
         }
      }


   /**************************************************************
    * check illegal type specification
    **************************************************************/
   if (unsign) {
      switch (*type) {
         case 'd':
         case 'f':
            G__fprinterr(G__serr,
                         "Error: unsigned can not be specified for float or double %s "
                         , string);
            G__genericerror((char*)NULL);
            break;
         default:
            *type = *type - unsign ;
            break;
      }
   }


   /**************************************************************
    * return 1 if float or double
    **************************************************************/
   return(result);
}

//______________________________________________________________________________
int Cint::Internal::G__isoperator(int c)
{
   switch (c) {
      case '+':
      case '-':
      case '*':
      case '/':
      case '@':
      case '&':
      case '%':
      case '|':
      case '^':
      case '>':
      case '<':
      case '=':
      case '~':
      case '!':
#ifdef G__NEVER
      case '.':
      case '(':
      case ')':
      case '[':
      case ']':
#endif
         return(1);
      default:
         return(0);
   }
}

//______________________________________________________________________________
int Cint::Internal::G__isexponent(char* expression4, int lenexpr)
{
   // -- Identify power and operator.


   int c;
   int flag = 0;

   G__ASSERT(lenexpr > 1); /* must be guaranteed in G__getexpr() */

   if (toupper(expression4[--lenexpr]) == 'E') {
      while (isdigit(c = expression4[--lenexpr]) || '.' == c) {
         if (lenexpr < 1) return(1);
         if ('.' != c) flag = 1;
      }
      if (flag && (G__isoperator(c) || c == ')')) {
         return(1);
      }
      else {
         return(0);
      }
   }
   else {
      switch (expression4[lenexpr]) {
         case '*':
         case '/':
         case '%':
         case '@':
            return(1);
      }
      return(0);
   }
}

//______________________________________________________________________________
int Cint::Internal::G__isvalue(char* temp)
{
   // -- Identify char pointer and string.
   if ((isdigit(temp[0])) || ((temp[0] == '-') && (isdigit(temp[1])))) {
      return(1);
   }
   else {
      return(0);
   }
}

//______________________________________________________________________________
extern "C" G__value G__string2type_body(const char* typenamin, int noerror)
{
   std::string typenam(typenamin);
   std::string temp;
   int len;
   int plevel = 0;
   int rlevel = 0;
   int flag;
   G__value result;
   int isconst = 0;

   result = G__null;

   if (strncmp(typenam.c_str(), "volatile ", 9) == 0) {
      typenam.replace(0, 9, "");
   }
   else
      if (strncmp(typenam.c_str(), "volatile", 8) == 0) {
         typenam.replace(0, 8, "");
      }

   if (strncmp(typenam.c_str(), "const ", 6) == 0) {
      typenam.replace(0, 6, "");
      isconst = 1;
   }
   else
      if (strncmp(typenam.c_str(), "const", 5) == 0 &&
            -1 == G__defined_tagname(typenam.c_str(), 2) && !G__find_typedef(typenam.c_str())) {
         typenam.replace(0, 5, "");
         isconst = 1;
      }

   len = typenam.length();
   do {
      switch (typenam[len-1]) {
         case '*':
            ++plevel;
            typenam.erase(typenam.end() - 1);
            flag = 1;
            break;
         case '&':
            ++rlevel;
            typenam.erase(typenam.end() - 1);
            flag = 1;
            break;
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            typenam.erase(typenam.end() - 1);
            flag = 1;
            break;
         default:
            flag = 0;
            break;
      }
   }
   while (flag);


   switch (len) {
      case 8:
         if (strcmp(typenam.c_str(), "longlong") == 0) {
            typenam = "long long";
         }
         break;
      case 9:
         if (strcmp(typenam.c_str(), "__int64") == 0) {
            typenam = "long long";
         }
         break;
      case 10:
         if (strcmp(typenam.c_str(), "longdouble") == 0) {
            typenam = "long double";
         }
         break;
      case 11:
         if (strcmp(typenam.c_str(), "unsignedint") == 0) {
            typenam = "unsigned int";
         }
         break;
      case 12:
         if (strcmp(typenam.c_str(), "unsignedchar") == 0) {
            typenam = "unsigned char";
            break;
         }
         if (strcmp(typenam.c_str(), "unsignedlong") == 0) {
            typenam = "unsigned long";
            break;
         }
         break;
      case 13:
         if (strcmp(typenam.c_str(), "unsignedshort") == 0) {
            typenam = "unsigned short";
            break;
         }
         break;
      case 16:
         if (strcmp(typenam.c_str(), "unsignedlonglong") == 0) {
            typenam = "unsigned long long";
            break;
         }
         break;
      case 18:
         if ((strcmp(typenam.c_str(), "unsigned __in64") == 0)) {
            typenam = "unsigned long long";
            break;
         }
         break;
   }

   ::Reflex::Type type(::Reflex::Type::ByName(typenam));

   if (!type) {
      if (strncmp(typenam.c_str(), "struct", 6) == 0) {
         type = G__Dict::GetDict().GetType(G__defined_tagname(typenam.c_str() + 6, 0));
      }
      else if (strncmp(typenam.c_str(), "class", 5) == 0) {
         type = G__Dict::GetDict().GetType(G__defined_tagname(typenam.c_str() + 5, 0));
      }
      else if (strncmp(typenam.c_str(), "union", 5) == 0) {
         type = G__Dict::GetDict().GetType(G__defined_tagname(typenam.c_str() + 5, 0));
      }
      else if (strncmp(typenam.c_str(), "enum", 4) == 0) {
         type = G__Dict::GetDict().GetType(G__defined_tagname(typenam.c_str() + 4, 0));
      }
   }

   if (!type) {
      type = G__find_typedef(typenam.c_str());
      if (!type) {
         type = G__Dict::GetDict().GetType(G__defined_tagname(typenam.c_str(), noerror));
      }
   }

   if (type) {
      while (plevel) {
         type = ::Reflex::PointerBuilder(type);
         --plevel;
      }
      if (rlevel) {
         type = ::Reflex::Type(type, ::Reflex::REFERENCE, Reflex::Type::APPEND);
      }
      result.obj.i = isconst; /* borrowing space of the value */
      if (isconst) {
         type = ::Reflex::Type(type, ::Reflex::CONST, Reflex::Type::APPEND);
      }
      G__value_typenum(result) = type;
   }

   return(result);
}

//______________________________________________________________________________
extern "C" G__value G__string2type(const char* typenamin)
{
   int store_var_type = G__var_type;
   G__value buf = G__string2type_body(typenamin, 0);
   G__var_type = store_var_type;
   return(buf);
}

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
