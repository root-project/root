/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//$Id: Shadow.cxx,v 1.14 2007/03/15 17:59:30 axel Exp $

#include "Shadow.h"
#include "common.h"
#include "global.h"
#include <ostream>
#include <string>
#include <list>

bool Cint::G__ShadowMaker::fgVetoShadow = false;

Cint::G__ShadowMaker::G__ShadowMaker(std::ostream& out, const char* nsprefix,
                                     bool(*needShadowClass)(G__ClassInfo &cl) /*=Cint::G__ShadowMaker::NeedShadowClass*/,
                                     bool(*needTypedefShadow)(G__ClassInfo &cl) /*=0*/):
      fOut(out), fNSPrefix(nsprefix), fMaxCachedNeedShadow(-1), fNeedShadowClass(needShadowClass), fNeedTypedefShadow(needTypedefShadow)
{
   // Typical constructor.
   
   memset(fCacheNeedShadow, 0, sizeof(fCacheNeedShadow));
};

//______________________________________________________________________________
void Cint::G__ShadowMaker::UpdateCachedNeedShadow() 
{
   // Set or extend the cache of the value of 'need shadow'.
   
   G__ClassInfo cl;
   // loop over all classes, deciding whether they need a shadow by themselves
   cl.Init(fMaxCachedNeedShadow);
   
   while (cl.Next()) {
      fCacheNeedShadow[cl.Tagnum()] = cl.IsValid()
                                      && (cl.Property() & (G__BIT_ISCLASS | G__BIT_ISSTRUCT))
                                      && fNeedShadowClass(cl);
   }

   // loop over all classes again, and for nested classes also generate shadow if parent needs shadow
   cl.Init(fMaxCachedNeedShadow);
   while (cl.Next()) {
      if (fCacheNeedShadow[cl.Tagnum()]) continue;
      G__ClassInfo encClass = cl.EnclosingClass();
      if (!encClass.IsValid() || !fCacheNeedShadow[encClass.Tagnum()]) continue;
      fCacheNeedShadow[cl.Tagnum()] = 1;
   }

   // loop over all classes again, and for nested classes needing shadow set
   // parent to "dummy shadow" (i.e. only providing the enclosing scope)
   // if parent doesn't need shadow
   cl.Init(fMaxCachedNeedShadow);
   while (cl.Next()) {
      if (!fCacheNeedShadow[cl.Tagnum()]) continue;
      G__ClassInfo encClass = cl.EnclosingClass();
      if (!encClass.IsValid() || !(encClass.Property() & G__BIT_ISCLASS))
         continue;
      int tagnum = encClass.Tagnum();
      if (!fCacheNeedShadow[tagnum])
         fCacheNeedShadow[tagnum] = 2;
   }
   
   fMaxCachedNeedShadow = G__struct.alltag - 1;
}

//______________________________________________________________________________
char Cint::G__ShadowMaker::NeedShadowCached(int tagnum) 
{ 
   if (tagnum >= G__struct.alltag) return 0;
   
   if (tagnum > fMaxCachedNeedShadow) {
      UpdateCachedNeedShadow();
   }
   return fCacheNeedShadow[tagnum]; 
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::VetoShadow(bool veto /*=true*/)
{
   // Set whether the shadow should be written or not.
   
   fgVetoShadow = veto;
}

//______________________________________________________________________________
bool Cint::G__ShadowMaker::NeedShadowClass(G__ClassInfo& cl)
{
   // Return whether cl needs a shadow class.

   // no streamer, no shadow
   if (cl.RootFlag() == G__NOSTREAMER) return false;
   if (IsStdPair(cl)) return true;
   if (IsSTLCont(cl.Name())) return false;
   if (strcmp(cl.Name(), "string") == 0) return false;
   if (strcmp(cl.Name(),"complex<float>") == 0 || strcmp(cl.Name(),"complex<double>") == 0) return true;

   return (cl.FileName() == 0 || strncmp(cl.FileName(), "prec_stl", 8) != 0);
}

//______________________________________________________________________________
int Cint::G__ShadowMaker::WriteNamespaceHeader(G__ClassInfo &cl)
{
   // Write all the necessary opening part of the namespace and
   // return the number of closing brackets needed
   // For example for Space1::Space2
   // we write: namespace Space1 { namespace Space2 {
   // and return 2.

   int closing_brackets = 0;
   G__ClassInfo namespace_obj = cl.EnclosingSpace();
   //fprintf(stderr,"DEBUG: in WriteNamespaceHeader for %s with %s\n",
   //    cl.Fullname(),namespace_obj.Fullname());
   if (namespace_obj.Property() & G__BIT_ISNAMESPACE) {
      closing_brackets = WriteNamespaceHeader(namespace_obj);
      for (int indent = 0; indent < closing_brackets; ++indent) {
         fOut << "   ";
      }
      fOut << "      namespace " << namespace_obj.Name() << " {" << std::endl;
      closing_brackets++;
   }

   return closing_brackets;
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::GetFullyQualifiedName(G__ClassInfo &cl, std::string &fullyQualifiedName)
{
   GetFullyQualifiedName(cl.Fullname(), fullyQualifiedName);
   const char *qual = fullyQualifiedName.c_str();
   if (!strncmp(qual, "::vector", strlen("::vector"))
         || !strncmp(qual, "::list", strlen("::list"))
         || !strncmp(qual, "::deque", strlen("::deque"))
         || !strncmp(qual, "::map", strlen("::map"))
         || !strncmp(qual, "::multimap", strlen("::multimap"))
         || !strncmp(qual, "::set", strlen("::set"))
         || !strncmp(qual, "::multiset", strlen("::multiset"))
         || !strncmp(qual, "::allocator", strlen("::allocator"))
         || !strncmp(qual, "::pair", strlen("::pair"))
      ) {

      fullyQualifiedName.erase(0, 2);

   }
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::GetFullyQualifiedName(G__TypeInfo &type, std::string &fullyQualifiedName)
{

   const char *s = type.TmpltName();

   std::string typeName;
   if (s) typeName = s;

   if (typeName == "string") {

      fullyQualifiedName = type.TrueName();

   } else if (typeName == "vector"
              || typeName == "list"
              || typeName == "deque"
              || typeName == "map"
              || typeName == "multimap"
              || typeName == "set"
              || typeName == "multiset"
              || typeName == "allocator"
              || typeName == "pair"
             ) {

      GetFullyQualifiedName(type.Name(), fullyQualifiedName);
      const char *qual = fullyQualifiedName.c_str();
      if (!strncmp(qual, "::vector", strlen("::vector"))
            || !strncmp(qual, "::list", strlen("::list"))
            || !strncmp(qual, "::deque", strlen("::deque"))
            || !strncmp(qual, "::map", strlen("::map"))
            || !strncmp(qual, "::multimap", strlen("::multimap"))
            || !strncmp(qual, "::set", strlen("::set"))
            || !strncmp(qual, "::multiset", strlen("::multiset"))
            || !strncmp(qual, "::allocator", strlen("::allocator"))
            || !strncmp(qual, "::pair", strlen("::pair"))
         ) {

         fullyQualifiedName.erase(0, 2);

      }

   } else if (type.Property() & (G__BIT_ISCLASS | G__BIT_ISSTRUCT
                                 | G__BIT_ISENUM | G__BIT_ISUNION))  {

      GetFullyQualifiedName(type.TrueName(), fullyQualifiedName);

   } else {

      fullyQualifiedName = type.TrueName();

   }
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::GetFullyQualifiedName(const char *originalName, std::string &fullyQualifiedName)
{
   //fprintf(stderr,"qualifying %s\n",originalName);
   std::string subQualifiedName = "";

   fullyQualifiedName = "::";

   std::string name = originalName;
   G__ClassInfo arg;

   int len = name.length();
   if (!len) {
      fullyQualifiedName = "";
      return;
   }

   int nesting = 0;
   const char *current, *next;
   current = next = 0;
   current = &(name[0]);
   next = &(name[0]);
   for (int c = 0; c < len; c++) {
      switch (name[c]) {
         case '<':
            if (nesting == 0) {
               name[c] = 0;
               current = next;
               if (c + 1 < len) next = &(name[c+1]);
               else next = 0;
               fullyQualifiedName += current;
               fullyQualifiedName += "< ";
               //fprintf(stderr,"will copy1: %s ...accu: %s\n",current,fullyQualifiedName.c_str());
            }
            nesting++;
            break;
         case '>':
            nesting--;
            if (nesting == 0) {
               name[c] = 0;
               current = next;
               if (c + 1 < len) next = &(name[c+1]);
               else next = 0;
               arg.Init(current);
               if (strlen(current) && arg.IsValid()) {
                  GetFullyQualifiedName(arg, subQualifiedName);
                  fullyQualifiedName += subQualifiedName;
               } else {
                  fullyQualifiedName += current;
               }
               fullyQualifiedName += " >";
               //fprintf(stderr,"will copy2: %s ...accu: %s\n",current,fullyQualifiedName.c_str());
            }
            break;
         case ',':
            if (nesting == 1) {
               name[c] = 0;
               current = next;
               if (c + 1 < len) next = &(name[c+1]);
               else next = 0;
               arg.Init(current);
               if (strlen(current) && arg.IsValid()) {
                  GetFullyQualifiedName(arg, subQualifiedName);
                  fullyQualifiedName += subQualifiedName;
               } else {
                  fullyQualifiedName += current;
               }
               fullyQualifiedName += ", ";
               //fprintf(stderr,"will copy3: %s ...accu: %s\n",current,fullyQualifiedName.c_str());
            }
            break;
         case ' ':
         case '&':
         case '*':
            if (nesting == 1) {
               char keep = name[c];
               name[c] = 0;
               current = next;
               if (c + 1 < len) next = &(name[c+1]);
               else next = 0;
               arg.Init(current);
               if (strlen(current) && arg.IsValid()) {
                  GetFullyQualifiedName(arg, subQualifiedName);
                  fullyQualifiedName += subQualifiedName;
               } else {
                  fullyQualifiedName += current;
               }
               fullyQualifiedName += keep;
               //fprintf(stderr,"will copy4: %s ...accu: %s\n",current,fullyQualifiedName.c_str());
               //fprintf(stderr,"have current %p, &name[0] %p for name %s \n",
               //        current,&(name[0]),name.c_str());
            }
            break;
      }
   }
   //fprintf(stderr,"preCalculated: %s\n",fullyQualifiedName.c_str());
   if (current == &(name[0])) {
      fullyQualifiedName += name;
   } else if (next) {
      for (int i = (next - &(name[0])); i < len; i++) {
         fullyQualifiedName += name[i];
      }
   }
   //fprintf(stderr,"Calculated: %s\n",fullyQualifiedName.c_str());
}

//______________________________________________________________________________
bool Cint::G__ShadowMaker::IsSTLCont(const char *type)
{
   // Just checking whether it contains one of
   // "any","vector","list","deque","map","multimap","set","multiset"
   // with a "<" afterwards.

   if (!type) return false;

   if (!strncmp(type, "std::", 5)) type += 5;
   const char* templateChar = strchr(type, '<');
   size_t posTemplate = templateChar - type;
   if (!templateChar || !posTemplate) return false;

   switch (posTemplate) {
      case 3:
         return !(strncmp(type, "map", 3) && strncmp(type, "set", 3)
                  && strncmp(type, "any", 3));
      case 4:
         return !(strncmp(type, "list", 4));
      case 5:
         return !(strncmp(type, "deque", 5));
      case 6:
         return !(strncmp(type, "vector", 6));
      case 8:
         return !(strncmp(type, "multimap", 8) && strncmp(type, "multiset", 8));
   }
   return false;
}

//______________________________________________________________________________
bool Cint::G__ShadowMaker::IsStdPair(G__ClassInfo &cl)
{
   // Is this an std pair

   return (strncmp(cl.Name(), "pair<", strlen("pair<")) == 0
           && cl.FileName() != 0 && strncmp(cl.FileName(), "prec_stl", 8) == 0);
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::GetFullShadowNameRecurse(G__ClassInfo &cl, std::string &fullname)
{

   if (fullname.empty()) {
      G__ClassInfo space = cl.EnclosingClass();
      if (!space.IsValid())
         space = cl.EnclosingSpace();
      if (space.IsValid())
         GetFullShadowNameRecurse(space, fullname);
      else {
         fullname = "::";
         if (!fNSPrefix.empty())
            fullname += fNSPrefix + "::";
         fullname += "Shadow::";
      }
   }

   if (NeedShadowCached(cl.Tagnum())) {
      fullname += G__map_cpp_name((char*)cl.Name());
      fullname += "::";
   } else {
      fullname += cl.Name();
      //if (cl.Property() & G__BIT_ISCLASS)
      fullname += "::";
      //else
      //   fullname += "__";
   }
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::GetFullShadowName(G__ClassInfo &cl, std::string &fullname)
{
   GetFullShadowNameRecurse(cl, fullname);
   size_t len = fullname.length();
   if (len > 1)
      fullname.erase(fullname.length() - 2);
}

//______________________________________________________________________________
std::string Cint::G__ShadowMaker::GetNonConstTypeName(G__DataMemberInfo &m, bool fullyQualified /*= false*/)
{
   // Return the type of the data member, without ANY const keyword

   if (m.Property() & (G__BIT_ISCONSTANT | G__BIT_ISPCONSTANT)) {
      std::string full;
      G__TypeInfo* type = m.Type();
      const char *typeName = 0;
      if (fullyQualified) {
         GetFullyQualifiedName(*(m.Type()), full);
         typeName = full.c_str();
      } else {
         typeName = type->Name();
      }
      static const char *constwd = "const";
      const char *s;
      int lev = 0;
      std::string ret;
      for (s = typeName;*s;s++) {
         if (*s == '<') lev++;
         if (*s == '>') lev--;
         if (lev == 0 && strncmp(constwd, s, strlen(constwd)) == 0) {
            const char *after = s + strlen(constwd);
            if (strspn(after, "&* ") >= 1 || *after == 0) {
               s += strlen(constwd) - 1;
               continue;
            }
         }
         ret += *s;
      }
      return ret;
   } else {
      if (fullyQualified) {
         std::string typeName;
         GetFullyQualifiedName(*(m.Type()), typeName);
         return typeName;
      } else {
         return m.Type()->Name();
      }
   }
}

//______________________________________________________________________________
void Cint::G__ShadowMaker::WriteShadowClass(G__ClassInfo &cl, int level /*=0*/)
{
   // This function writes or make available a class named fNSPrefix::Shadow::ClassName
   // for which all data member are the same as the one in the class but are
   // all public.

   if (!NeedShadowCached(cl.Tagnum())) return;

   // if nested class let the enclosing class generate the shadow
   G__ClassInfo encClass = cl.EnclosingClass();
   if (!level && encClass.IsValid() && NeedShadowCached(encClass.Tagnum()))
      return;

   // Here we copy the shadow only if !TypedefShadow(cl) class does not have a ClassDef
   // in it.
   std::string classname = G__map_cpp_name((char*)cl.Name());

   int closing_brackets = 0;
   if (!level) closing_brackets = WriteNamespaceHeader(cl);
   if (closing_brackets) fOut << std::endl;
   std::string indent;
   for (int iIndent = 0; iIndent < level; ++iIndent) indent += "   ";

   std::string fullname;
   GetFullyQualifiedName(cl, fullname);

   if (NeedShadowCached(cl.Tagnum()) != 2
         && fNeedTypedefShadow  && fNeedTypedefShadow(cl)) {
      fOut << indent << "      typedef " << fullname << " " << classname << ";" << std::endl;
   } else {
      fOut << indent << "      #if !(defined(R__ACCESS_IN_SYMBOL) || defined(R__USE_SHADOW_CLASS))" << std::endl;
      fOut << indent << "      typedef " << fullname << " " << classname << ";" << std::endl
      << indent << "      #else" << std::endl;

      if (cl.Property() & G__BIT_ISENUM)
         fOut << indent << "      enum " << classname << " ";
      else if (cl.Property() & G__BIT_ISUNION)
         fOut << indent << "      union " << classname << " ";
      else
         fOut << indent << "      class " << classname << " ";

      // Write ShowMembers for base class(es) when they have the ShowMember() method
      G__BaseClassInfo b(cl);
      bool first = true;
      while (b.Next()) {
         if ((b.Property() & G__BIT_ISVIRTUALBASE) &&
               !(b.Property() & G__BIT_ISDIRECTINHERIT)) {
            // CINT duplicates the remote virtual base class in the list scanned
            // by G__BaseClassInfo, we need to skip them.
            continue;
         }
         if (first) {
            fOut << " : ";
            first = false;
         } else {
            fOut << ", ";
         }
         if (b.Property() & G__BIT_ISVIRTUALBASE)
            fOut << " virtual";
         if (b.Property() & G__BIT_ISPRIVATE)
            fOut << " private ";
         else if (b.Property() & G__BIT_ISPROTECTED)
            fOut << " protected ";
         else if (b.Property() & G__BIT_ISPUBLIC)
            fOut << " public ";
         else
            fOut << " UNKNOWN inheritance ";

         std::string type_name;
         GetFullyQualifiedName(b, type_name);
         fOut << type_name;
      }
      fOut << " {" << std::endl;
      if (!(cl.Property() & G__BIT_ISENUM)) {
         fOut << indent << "         public:" << std::endl
         << indent << "         //friend XX;" << std::endl;
      }

      G__ClassInfo clContained;
      clContained.Init();
      while (clContained.Next()) {
         if (clContained.IsValid() &&
               clContained.EnclosingClass().IsValid() &&
               clContained.EnclosingClass().Tagnum() == cl.Tagnum()
               && strcmp(clContained.Name(), "$"))
            WriteShadowClass(clContained, level + 1);
      }

      // Loop over all types, generating shadows for types contained in this class
      G__TypedefInfo tdContained;
      tdContained.Init();
      while (tdContained.Next()) {
         if (tdContained.IsValid() &&
               tdContained.EnclosingClassOfTypedef().IsValid() &&
               tdContained.EnclosingClassOfTypedef().Tagnum() == cl.Tagnum()) {
            std::string nsprefixOriginal("::");

            std::string typedefedTypename = tdContained.Name();
            //remove leading scopes, as we want to have it defined locally
            size_t posLastScope = typedefedTypename.rfind("::");
            if (posLastScope != std::string::npos && posLastScope > 0)
               typedefedTypename.erase(0, posLastScope + 2);

            std::string typenameOriginal;
            if (tdContained.Fullname()) {
               typenameOriginal = tdContained.TrueName();
               // don't generate dummy typedefs for template default params
               if (!typedefedTypename.empty()
                     && typedefedTypename[typedefedTypename.length()-1] == '>')
                  continue;
               // Cint doesn't put the FQI as the template arg's type,
               // so they end up in the shadow namespace, as in
               // typedef ::SomeType<AnotherType> SomeTypeDefInShadow;
               // AnotherType should really be ::AnotherType.
               // We need to fix that, by putting all but built-in template
               // arg types into the global namespace

               size_t posTemplArg = typenameOriginal.find('<');
               while (posTemplArg != std::string::npos) {
                  int tlevel = 0;
                  size_t posArgEnd = posTemplArg;
                  do {
                     posArgEnd = typenameOriginal.find_first_of("<,>", posArgEnd + 1);
                     if (posArgEnd !=  std::string::npos) {
                        if (typenameOriginal[posArgEnd] == '<')
                           ++tlevel;
                        if (typenameOriginal[posArgEnd] == '>') {
                           --tlevel;
                           if (tlevel == -1) break;
                        }
                        if (typenameOriginal[posArgEnd] == ',' && tlevel == 0)
                           break;
                     }
                  } while (posArgEnd != std::string::npos);

                  if (posArgEnd == std::string::npos) {
                     printf("WARNING: error extracting template aruments for %s!",
                            typenameOriginal.c_str());
                     break;
                  }

                  //char endChar = typenameOriginal[posArgEnd];

                  std::string arg = typenameOriginal.substr(posTemplArg + 1, posArgEnd - posTemplArg - 1);
                  std::string::size_type posRef = 0;
                  int templateLevel = 0;
                  size_t lenArg = arg.length();
                  for (; posRef < lenArg; ++posRef) {
                     switch (arg[posRef]) {
                     case '<':
                        if (posRef + 1 < lenArg && arg[posRef + 1] != '<') {
                           ++templateLevel;
                        } else ++posRef;
                        break;
                     case '>':
                        if (posRef + 1 < lenArg && arg[posRef + 1] != '>') {
                           --templateLevel;
                        } else if (posRef > 8) {
                           std::string::size_type posOp = posRef - 1;
                           while (posOp && isspace(arg[posOp])) --posOp;
                           if (posOp > 8 && !arg.compare(posOp - 8, 8, "operator"))
                              // it's the operator >>
                              ++posRef; 
                           else {
                              --templateLevel;
                           }
                        } else {
                           --templateLevel;
                        }
                        break;
                     case '*':
                     case '&':
                        if (!templateLevel) {
                           arg.erase(posRef, 1);
                           --lenArg;
                           --posRef;
                        }
                        break;
                     }
                  }

                  // if the type is not defined (i.e. it's fundamental)
                  // and if we don't have a real shadoew for it, use the
                  // global scope's type.
                  int tagname = G__defined_tagname(arg.c_str(), 1);

                  if (tagname != -1) {
                     G__ClassInfo ciArg(tagname);
                     // templates or function without shadow
                     // need to have explicit namespace
                     if (arg.find('<') != std::string::npos ||
                           (NeedShadowCached(tagname) != 1
                            && ((ciArg.Property() & G__BIT_ISCLASS)
                                || (ciArg.Property() & G__BIT_ISSTRUCT)))) {
                        // replace "pair" by "std::pair"
                        if (arg == "pair") arg.insert(0, "std::");
                        // we don't have a (full) shadow for this guy
                        // need space before "::" to prevent "<:"
                        typenameOriginal.insert(posTemplArg + 1, " ::");
                        posArgEnd += 3;
                     }
                  }
                  posTemplArg = typenameOriginal.find_first_of("<,", posTemplArg + 1);
               }

               if (IsSTLCont(tdContained.Fullname()) || IsStdPair(tdContained))
                  nsprefixOriginal = "::std::";
            }

            if (!tdContained.Fullname()) {
               // func ptr, make typedefedTypename contain the full typedef
               // and make the func be void (func)() - the exact signature is
               // irrelevant for the shadow, and we can't extract the params from Cint.
               typenameOriginal =  "void* (*";
               typenameOriginal += typedefedTypename + ")()";
               typedefedTypename = "";
               nsprefixOriginal = "";
            }

            // convert T<A> into T< ::A> to ensure that we pick up a non-shadow A.
            // Major problem here: "::int" doesn't exist, so revert those.
            size_t posArg = typenameOriginal.find_first_of("<,");
            size_t lenType = typenameOriginal.length();
            while (posArg != std::string::npos) {
               ++posArg;
               size_t lenArg = 0;
               do {
                  lenArg = 0;
                  while (isspace(typenameOriginal[posArg]))
                     ++posArg;
                  while (lenType > posArg + lenArg) {
                     char c = typenameOriginal[posArg + lenArg];
                     if ((lenArg && isalnum(c))
                         || (!lenArg
                             && ((c >= 'A' && c <= 'Z')
                                 || (c >= 'a' && c <= 'z'))
                             )
                         || c == '_')
                        ++lenArg;
                     else break;
                  }
               } while (lenArg == 5
                        && !typenameOriginal.compare(posArg, lenArg, "const")
                        && lenType > (posArg += 5) );
               bool builtinType = false;
               if (lenArg) {
                  switch (lenArg) {
                  case 3: builtinType = !typenameOriginal.compare(posArg, lenArg, "int");
                     break;
                  case 4: builtinType = !typenameOriginal.compare(posArg, lenArg, "long")
                        || !typenameOriginal.compare(posArg, lenArg, "char")
                        || !typenameOriginal.compare(posArg, lenArg, "void")
                        || !typenameOriginal.compare(posArg, lenArg, "bool");
                     break;
                  case 5: builtinType = !typenameOriginal.compare(posArg, lenArg, "short")
                        || !typenameOriginal.compare(posArg, lenArg, "float");
                     break;
                  case 6: builtinType = !typenameOriginal.compare(posArg, lenArg, "double");
                     break;
                  case 8: builtinType = !typenameOriginal.compare(posArg, lenArg, "unsigned");
                     break;
                  default:;
                  }
                  if (!builtinType) {
                     typenameOriginal.insert(posArg, " ::");
                     lenArg += 3;
                     lenType += 3;
                  }
               }
               posArg = typenameOriginal.find_first_of("<,", posArg + lenArg + 1);
            }

            fOut << indent << "         typedef "
            << nsprefixOriginal << typenameOriginal << " "
            << typedefedTypename << ";" << std::endl;
         }
      }

      // Figure out if there are virtual function and write a dummy one if needed
      G__MethodInfo methods(cl);
      while (methods.Next()) {
         // fprintf(stderr,"%s::%s has property 0x%x\n",cl.Fullname(),methods.Name(),methods.Property());
         if (methods.Property() &
               (G__BIT_ISVIRTUALBASE | G__BIT_ISVIRTUAL | G__BIT_ISPUREVIRTUAL)) {
            fOut << indent << "         // To force the creation of a virtual table, throw just in case." << std::endl
            << indent << "         virtual ~" << classname << "() throw() {};" << std::endl;
            break;
         }
      }

      if (NeedShadowCached(cl.Tagnum()) == 1) {
         // Write data members
         G__DataMemberInfo d(cl);
         while (d.Next()) {
            // fprintf(stderr,"%s %s %ld\n",d.Type()->Name(),d.Name(),d.Property());
            if (d.Property() & G__BIT_ISSTATIC) { // a static member
               continue;
            }
            if (d.Property() & G__BIT_ISUSINGVARIABLE) { // a member brought in via a using statement
               continue;
            }
            if (strcmp("G__virtualinfo", d.Name()) == 0) continue;

            std::string type_name = GetNonConstTypeName(d, true); // .Type()->Name();

            if (((d.Type()->Property() & G__BIT_ISENUM) &&
                 (type_name.length() == 0 || type_name == "enum")) ||
                  type_name.find("::") == type_name.length() - 2) {
               // We have unamed enums, let's fake it:
               fOut << indent << "         enum {kDummy} " << d.Name();
            } else {
               if (d.Type()->Name() && strstr(d.Type()->Name(), "(*)")) {
                  // func ptr
                  std::string funcptr_name = d.Type()->Name();
                  size_t posName = funcptr_name.find("(*)");
                  funcptr_name.insert(posName + 2, d.Name());
                  fOut << indent << "         " << " " << funcptr_name;
               } else {
                  // Keep original "time_t", as it's not "long" on all platforms
                  if (!strcmp(d.Type()->Name(), "time_t"))
                     type_name = "time_t";
                  // for p2memfunc, take original type instead of Cint's para-type
                  if (d.Type()->Type() == 'a' && d.Type()->Name()) {
                     if (d.Type()->TrueName()
                           && !strcmp(d.Type()->TrueName(), "G__p2memfunc")) {
                        fOut << indent << "         "
                        << "void (" << cl.Name() << "::*" << d.Name() << ")()";
                        type_name = "";
                     } else
                        type_name = d.Type()->Name();
                  }
                  if (!type_name.empty()) {
                     // Replace 'long long' and 'unsigned long long' by 'Long64_t' and 'ULong64_t'
                     const char* ulonglong_s = "unsigned long long";
                     const char* longlong_s  = ulonglong_s + 9;
                     const unsigned int ulonglong_len = 18;
                     const unsigned int longlong_len  = 9;

                     int pos = 0;
                     while ((pos = type_name.find(ulonglong_s, pos)) >= 0) {
                        type_name.replace(pos, ulonglong_len, "G__uint64");
                     }
                     pos = 0;
                     while ((pos = type_name.find(longlong_s, pos)) >= 0) {
                        type_name.replace(pos, longlong_len, "G__int64");
                     }

                     if (type_name[type_name.length()-1] == '&') {
                        type_name[type_name.length()-1] = '*';
                     }
                     // if type is contained in cl, remove sope (so it points to nested shadow)
                     if (!strncmp(type_name.c_str(), fullname.c_str(), fullname.length())
                           && type_name[fullname.length()] == ':')
                        type_name.erase(0, fullname.length() + 2);

                     fOut << indent << "         " << type_name << " " << d.Name();
                  }
               }
            }

            for (int dim = 0; dim < d.ArrayDim(); dim++) {
               fOut << "[" << d.MaxIndex(dim) << "]";
            }
            fOut << "; //" << d.Title() << std::endl;
         }
      } // if NeedShadowCached(cl.Tagnum())==1 (i.e. real shadow class, not just enclosing scope)
      fOut << indent << "      };" << std::endl;

      fOut << indent << "      #endif" << std::endl;
   }
   if (closing_brackets) fOut << indent << "      ";
   for (int brack = 0; brack < closing_brackets; brack++) {
      fOut << "} ";
   }
   fOut << std::endl;
}

void Cint::G__ShadowMaker::WriteAllShadowClasses()
{
   if (fgVetoShadow) return;

   // In some case, WriteShadowClass will induce (via calls to G__ClassInfo) template instantiation,
   // if the a function has a default value, we do not want to execute it.
   // Setting G__globalcomp to something else then G__NOLINK is the only way 
   // to accomplish this.
   int store_G__globalcomp = G__globalcomp;
   G__globalcomp = 7; // Intentionally not a valid value.

   fOut << "// START OF SHADOWS" << std::endl << std::endl;

   std::string remainingNSPrefix = fNSPrefix;
   std::string ind;
   std::list<std::string> namespaceParts;
   while (!remainingNSPrefix.empty()) {
      std::string::size_type posNextNS = remainingNSPrefix.find("::");
      if (posNextNS == std::string::npos)
         posNextNS = remainingNSPrefix.length();
      std::string ns = remainingNSPrefix.substr(0, posNextNS);
      if (!ns.empty()) {
         fOut << ind << "namespace " << ns << " {" << std::endl;
         namespaceParts.push_back(ns);
         ind += "   ";
      }
      remainingNSPrefix.erase(0, posNextNS + 2);
   }
   fOut << ind << "namespace Shadow {" << std::endl;

   G__ClassInfo cl;
   cl.Init();
   while (cl.Next()) {
      if ((cl.Property() & (G__BIT_ISCLASS | G__BIT_ISSTRUCT)) && cl.Linkage() == G__CPPLINK) {
         // Write Code for initialization object
         WriteShadowClass(cl);
      }
   }

   fOut << ind << "} // of namespace Shadow" << std::endl;
   while (namespaceParts.size()) {
      ind.erase(0, 3);
      fOut << ind << "} // of namespace "
      << namespaceParts.back() << std::endl;
      namespaceParts.pop_back();
   }
   fOut << "// END OF SHADOWS" << std::endl << std::endl;
   
   G__globalcomp = store_G__globalcomp;
}
