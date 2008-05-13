// @(#)root/io:$Id$
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMakeProject                                                         //
//                                                                      //
// Helper class implementing the TFile::MakeProject.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <ctype.h>
#include "TMakeProject.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TROOT.h"

//______________________________________________________________________________
void TMakeProject::AddUniqueStatement(FILE *fp, const char *statement, char *inclist)
{
   // Add an include statement, if it has not already been added.

   if (!strstr(inclist, statement)) {
      strcat(inclist, statement);
      fprintf(fp, statement);
   }
}

//______________________________________________________________________________
void TMakeProject::AddInclude(FILE *fp, const char *header, Bool_t system, char *inclist)
{
   // Add an include statement, if it has not already been added.

   TString what;
   if (system) {
      what.Form("#include <%s>\n", header);
   } else {
      what.Form("#include \"%s\"\n", header);
   }
   AddUniqueStatement(fp, what.Data(), inclist);
}

//______________________________________________________________________________
TString TMakeProject::GetHeaderName(const char *name, Bool_t includeNested)
{
   //Return the header name containing the description of name
   TString result;
   Int_t len = strlen(name);
   Int_t nest = 0;
   for (Int_t i = 0; i < len; ++i) {
      switch (name[i]) {
         case '<':
            ++nest;
            result.Append('_');
            break;
         case '>':
            --nest;
            result.Append('_');
            break;
         case ':':
            if (nest == 0 && name[i+1] == ':') {
               TString nsname(name, i);
               TClass *cl = gROOT->GetClass(nsname);
               if (!includeNested && cl && cl->Size() != 0) {
                  // The requested class is actually nested inside
                  // the class whose name we already 'copied' to
                  // result.  The declaration will be in the same
                  // header file as the outer class.
                  if (strcmp(name + strlen(name) - 2, ".h") == 0) {
                     result.Append(".h");
                  }
                  return result;
               }
            }
            result.Append('_');
            break;
         case ',':
         case '*':
         case '[':
         case ']':
         case ' ':
         case '(':
         case ')':
            result.Append('_');
            break;
         case '/':
         case '\\':
         default:
            result.Append(name[i]);
      }
   }
   return result;
}

//______________________________________________________________________________
UInt_t TMakeProject::GenerateClassPrefix(FILE *fp, const char *clname, Bool_t top, TString &protoname,
      UInt_t *numberOfClasses, Bool_t implementEmptyClass)
{

   // First open the namespace (if any)
   Int_t numberOfNamespaces = 0;
   const char *fullname = clname;

   Bool_t istemplate = kFALSE;
   if (strchr(clname, ':')) {
      // We might have a namespace in front of the classname.
      Int_t len = strlen(clname);
      const char *name = clname;
      UInt_t nest = 0;
      for (Int_t cur = 0; cur < len; ++cur) {
         switch (clname[cur]) {
            case '<':
               ++nest;
               istemplate = kTRUE;
               break;
            case '>':
               --nest;
               break;
            case ':': {
                  if (nest == 0 && clname[cur+1] == ':') {
                     // We have a scope
                     TString nsname(clname, cur);
                     TClass *cl = gROOT->GetClass(nsname);
                     if (top) {
                        if (cl == 0 || (cl && cl->Size() == 0)) {
                           TString last(name, cur - (name - clname));
                           if ((numberOfClasses == 0 || *numberOfClasses == 0) && strchr(last.Data(), '<') == 0) {
                              fprintf(fp, "namespace %s {\n", last.Data());
                              ++numberOfNamespaces;
                           } else {
                              TString headername(GetHeaderName(last));
                              fprintf(fp, "#ifndef %s_h\n", headername.Data());
                              fprintf(fp, "#define %s_h\n", headername.Data());
                              GenerateClassPrefix(fp, last.Data(), top, protoname, 0);
                              fprintf(fp, "{\n");
                              fprintf(fp, "public:\n");
                              if (numberOfClasses) ++(*numberOfClasses);
                              istemplate = kFALSE;
                           }
                           name = clname + cur + 2;
                        }
                     } else {
                        name = clname + cur + 2;
                     }
                  }
                  break;
               }
         }
      }
      clname = name;
   } else {
      istemplate = strstr(clname, "<") != 0;
   }

   protoname = clname;

   if (implementEmptyClass) {
      TString headername(GetHeaderName(fullname));
      fprintf(fp, "#ifndef %s_h\n", headername.Data());
      fprintf(fp, "#define %s_h\n", headername.Data());
   }
   if (istemplate) {
      std::vector<const char*> argtype;

      Ssiz_t pos = protoname.First('<');
      UInt_t nparam = 1;
      if (pos != kNPOS) {
         if (isdigit(protoname[pos+1])) {
            argtype.push_back("int");
         } else {
            argtype.push_back("typename");
         }
         UInt_t nest = 0;
         for (Ssiz_t i = pos; i < protoname.Length(); ++i) {
            switch (protoname[i]) {
               case '<':
                  ++nest;
                  break;
               case '>':
                  --nest;
                  break;
               case ',':
                  if (nest == 1) {
                     if (isdigit(protoname[i+1])) {
                        argtype.push_back("int");
                     } else {
                        argtype.push_back("typename");
                     }
                     ++nparam;
                  }
                  break;
            }
         }
         protoname.Remove(pos);
      }
      // Forward declaration of template.
      fprintf(fp, "template <");
      for (UInt_t p = 0; p < nparam; ++p) {
         fprintf(fp, "%s T%d", argtype[p], p);
         if (p != (nparam - 1)) fprintf(fp, ", ");
      }
      fprintf(fp, "> class %s;\n", protoname.Data());
      fprintf(fp, "template <> ");
   }

   if (implementEmptyClass) {
      if (istemplate) {
         fprintf(fp, "class %s", clname);
         fprintf(fp, " {\n");
         if (numberOfClasses) ++(*numberOfClasses);
         fprintf(fp, "public:\n");
         fprintf(fp, "operator int() { return 0; };\n");
      } else {
         fprintf(fp, "enum %s { kDefault_%s };\n", clname, clname);
         // The nesting space of this class may not be #pragma declared (and without
         // the dictionary is broken), so for now skip those
         if (strchr(fullname, ':') == 0) {
            // yes this is too aggressive, this needs to be fixed properly by moving the #pragma out of band.
            fprintf(fp, Form("#ifdef __MAKECINT__\n#pragma link C++ class %s+;\n#endif\n", fullname));
         }
         fprintf(fp, "#endif\n");
      }
   } else {
      fprintf(fp, "class %s", clname);
   }
   return numberOfNamespaces;
}

//______________________________________________________________________________
UInt_t TMakeProject::GenerateEmptyNestedClass(FILE *fp, const char *topclass, const char *clname)
{
   // Look at clname and generate any 'empty' nested classes that might be used
   // as template parameter.

   UInt_t tlen = strlen(topclass);
   UInt_t len = strlen(clname);
   UInt_t nest = 0;
   UInt_t last = 0;

   for (UInt_t i = 0; i < len; ++i) {
      switch (clname[i]) {
            case '<':
               ++nest;
               if (nest == 1) last = i + 1;
               break;
            case '>':
               --nest; /* intentional fall throught to the next case */
            case ',':
               if ((clname[i] == ',' && nest == 1) || (clname[i] == '>' && nest == 0)) {
                  TString incName(clname + last, i - last);
                  incName = TClassEdit::ShortType(incName.Data(), 1);
                  if (clname[i] == '>' && nest == 1) incName.Append(">");
                  Int_t stlType;
                  if (isdigit(incName[0])) {
                     // Not a class name, nothing to do.
                  } else if ((stlType = TClassEdit::IsSTLCont(incName))) {
                     TMakeProject::GenerateEmptyNestedClass( fp, topclass, incName );
                  } else if (TClassEdit::IsStdClass(incName)) {
                     // Do nothing.
                  } else {
                     TClass *cl = gROOT->GetClass(incName);
                     if (cl) {
                        // We have a cl (and hence a streamerInfo and hence we are not empty,
                        // so we have nothing to do.
                        //if (cl->GetClassInfo()) {
                        //} else {
                        //}
                     } else if (incName.Length() && incName[0] != ' ' && gROOT->GetType(incName) == 0) {
                        if (strchr(incName,'<')) {
                           TMakeProject::GenerateEmptyNestedClass( fp, topclass, incName );
                        }
                        if (strncmp(topclass,incName,tlen)==0 && incName[(Ssiz_t)tlen+1]==':' && strchr(incName.Data()+tlen+2,':')==0) {
                           Bool_t istemplate = kFALSE;
                           if (istemplate) {
                              fprintf(fp, "   class %s", incName.Data()+tlen+3);
                              fprintf(fp, " {\n");
                              fprintf(fp, "public:\n");
                              fprintf(fp, "operator int() { return 0; };\n");
                              fprintf(fp, "};\n");
                           } else {
                              fprintf(fp, "   enum %s { kDefault_%s };\n", incName.Data()+tlen+2, incName.Data()+tlen+2);
                           }
                        }
                     }
                  }
                  last = i + 1;
               }
      }
   }
   return 0;
}

//______________________________________________________________________________
UInt_t TMakeProject::GenerateForwardDeclaration(FILE *fp, const char *clname, char *inclist, Bool_t implementEmptyClass)
{
   // Insert a (complete) forward declaration for the class 'clname'

   UInt_t ninc = 0;

   if (strchr(clname, '<')) {
      ninc += GenerateIncludeForTemplate(fp, clname, inclist, kTRUE);
   }
   TString protoname;
   UInt_t numberOfClasses = 0;
   UInt_t numberOfNamespaces = GenerateClassPrefix(fp, clname, kTRUE, protoname, &numberOfClasses, implementEmptyClass);

   fprintf(fp, ";\n");
   for (UInt_t i = 0;i < numberOfClasses;++i) {
      fprintf(fp, "}; // end of class.\n");
      fprintf(fp, "#endif\n");
   }
   for (UInt_t i = 0;i < numberOfNamespaces;++i) {
      fprintf(fp, "} // end of namespace.\n");
   }

   return ninc;
}

//______________________________________________________________________________
UInt_t TMakeProject::GenerateIncludeForTemplate(FILE *fp, const char *clname, char *inclist, Bool_t forward)
{
   // Add to the header file, the #include needed for the argument of
   // this template.

   UInt_t ninc = 0;
   UInt_t len = strlen(clname);
   UInt_t nest = 0;
   UInt_t last = 0;


   for (UInt_t i = 0; i < len; ++i) {
      switch (clname[i]) {
         case '<':
            ++nest;
            if (nest == 1) last = i + 1;
            break;
         case '>':
            --nest; /* intentional fall throught to the next case */
         case ',':
            if ((clname[i] == ',' && nest == 1) || (clname[i] == '>' && nest == 0)) {
               TString incName(clname + last, i - last);
               incName = TClassEdit::ShortType(incName.Data(), 1);
               if (clname[i] == '>' && nest == 1) incName.Append(">");
               Int_t stlType;
               if (isdigit(incName[0])) {
                  // Not a class name, nothing to do.
               } else if ((stlType = TClassEdit::IsSTLCont(incName))) {
                  const char *what = "";
                  switch (stlType)  {
                     case TClassEdit::kVector:
                        what = "vector";
                        break;
                     case TClassEdit::kList:
                        what = "list";
                        break;
                     case TClassEdit::kDeque:
                        what = "deque";
                        break;
                     case TClassEdit::kMap:
                        what = "map";
                        break;
                     case TClassEdit::kMultiMap:
                        what = "map";
                        break;
                     case TClassEdit::kSet:
                        what = "set";
                        break;
                     case TClassEdit::kMultiSet:
                        what = "set";
                        break;
                  }
                  AddInclude(fp, what, kTRUE, inclist);
                  fprintf(fp, "namespace std {} using namespace std;\n");
                  ninc += GenerateIncludeForTemplate(fp, incName, inclist, forward);
               } else if (strncmp(incName.Data(), "pair<", strlen("pair<")) == 0) {
                  AddInclude(fp, "utility", kTRUE, inclist);
                  ninc += GenerateIncludeForTemplate(fp, incName, inclist, forward);
               } else if (TClassEdit::IsStdClass(incName)) {
                  // Do nothing.
               } else {
                  TClass *cl = gROOT->GetClass(incName);
                  if (!forward && cl) {
                     if (cl->GetClassInfo()) {
                        // We have the real dictionary for this class.

                        const char *include = cl->GetDeclFileName();
                        if (include && strlen(include) != 0) {

                           if (strncmp(include, "include/", 8) == 0) {
                              include += 8;
                           }
                           if (strncmp(include, "include\\", 9) == 0) {
                              include += 9;
                           }
                           TMakeProject::AddInclude(fp, include, kFALSE, inclist);
                        }
                        GenerateIncludeForTemplate(fp, incName, inclist, forward);
                     } else {
                        incName = GetHeaderName(incName);
                        incName.Append(".h");
                        AddInclude(fp, incName, kFALSE, inclist);
                     }
                  } else if (incName.Length() && incName[0] != ' ' && gROOT->GetType(incName) == 0) {
                     GenerateForwardDeclaration(fp, incName, inclist, !cl);
                  }
               }
               last = i + 1;
            }
      }
   }

   Int_t stlType = TClassEdit::IsSTLCont(clname);
   if (stlType) {
      std::vector<std::string> inside;
      int nestedLoc;
      TClassEdit::GetSplit(clname, inside, nestedLoc);
      Int_t stlkind =  TClassEdit::STLKind(inside[0].c_str());
      TClass *key = TClass::GetClass(inside[1].c_str());
      if (key) {
         std::string what;
         switch (stlkind)  {
            case TClassEdit::kMap:
            case TClassEdit::kMultiMap: {
                  what = "pair<";
                  what += inside[1];
                  what += ",";
                  what += inside[2];
                  what += " >";
                  AddUniqueStatement(fp, Form("#ifdef __MAKECINT__\n#pragma link C++ class %s+;\n#endif\n", what.c_str()), inclist);
                  break;
               }
         }
      }
   }

   return ninc;
}

