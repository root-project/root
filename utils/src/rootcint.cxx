// @(#)root/utils:$Name:  $:$Id: rootcint.cxx,v 1.5 2000/09/01 06:23:14 brun Exp $
// Author: Fons Rademakers   13/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rootcint                                                             //
//                                                                      //
// This program generates the Streamer(), TBuffer &operator>>() and     //
// ShowMembers() methods for ROOT classes, i.e. classes using the       //
// ClassDef and ClassImp macros.                                        //
// In addition rootcint can also generate the CINT dictionaries         //
// needed in order to get access to ones classes via the interpreter.   //
//                                                                      //
// Rootcint can be used like:                                           //
//                                                                      //
//   rootcint TAttAxis.h[+][-][!] ... [LinkDef.h] > AxisDict.cxx        //
//                                                                      //
// or                                                                   //
//                                                                      //
//  rootcint [-f] axisDict.cxx [-c] TAttAxis.h[+][-][!] ... [LinkDef.h] //
//                                                                      //
// The difference between the two is that in the first case only the    //
// Streamer() and ShowMembers() methods are generated while in the      //
// latter case a complete compileable file is generated (including      //
// the include statements). The first method also allows the            //
// output to be appended to an already existing file (using >>).        //
// The optional - behind the include file name tells rootcint to not    //
// generate the Streamer() method. A custom method must be provided     //
// by the user in that case. For the ! and + options see below.         //
// When using option -c also the interpreter method interface stubs     //
// will be written to the output file (AxisDict.cxx in the above case). //
// By default the output file will not be overwritten if it exists.     //
// Use the -f (force) option to overwite the output file. The output    //
// file must have one of the following extensions: .cxx, .C, .cpp,      //
// .cc, .cp.                                                            //
//                                                                      //
// Before specifying the first header file one can also add include     //
// file directories to be searched and preprocessor defines, like:      //
//   -I$MYPROJECT/include -DDebug=1                                     //
//                                                                      //
// The (optional) file LinkDef.h looks like:                            //
//                                                                      //
// #ifdef __CINT__                                                      //
//                                                                      //
// #pragma link off all globals;                                        //
// #pragma link off all classes;                                        //
// #pragma link off all functions;                                      //
//                                                                      //
// #pragma link C++ class TAxis;                                        //
// #pragma link C++ class TAttAxis-;                                    //
// #pragma link C++ class TArrayC-!;                                    //
// #pragma link C++ class AliEvent+;                                    //
//                                                                      //
// #pragma link C++ function StrDup;                                    //
// #pragma link C++ function operator+(const TString&,const TString&);  //
//                                                                      //
// #pragma link C++ global gROOT;                                       //
// #pragma link C++ global gEnv;                                        //
//                                                                      //
// #pragma link C++ enum EMessageTypes;                                 //
//                                                                      //
// #endif                                                               //
//                                                                      //
// This file tells rootcint for which classes the method interface      //
// stubs should be generated. A trailing - in the class name tells      //
// rootcint to not generate the Streamer() method. This is necessary    //
// for those classes that need a customized Streamer() method.          //
// A trailing ! in the class name tells rootcint to not generate the    //
// operator>>(TBuffer &b, MyClass *&obj) function. This is necessary to //
// be able to write pointers to objects of classes not inheriting from  //
// TObject. See for an example the source of the TArrayF class.         //
// A trailing + in the class name tells rootcint to generate a          //
// Streamer() with extra byte count information. This adds one int to   //
// each object in the output buffer, but it allows for powerful error   //
// correction in case a Streamer() method is out of sync compared to    //
// the data on the file. The + option is mutual exclusive with both     //
// the - and ! options.                                                 //
// When this linkdef file is not specified a default version exporting  //
// the classes with the names equal to the include files minus the .h   //
// is generated.                                                        //
//                                                                      //
// *** IMPORTANT ***                                                    //
// 1) LinkDef.h must be the last argument on the rootcint command line. //
// 2) Note that the LinkDef file name MUST contain the string:          //
//    LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h is fine    //
//    just like, mylinkdef.h. Linkdef.h is case sensitive.              //
//                                                                      //
// ----------- historical ---------                                     //
//                                                                      //
// Note that the file rootcint.C is constructed in such a way that it   //
// can also be interpreted by CINT. The above two statements become in  //
// that case:                                                           //
//                                                                      //
// cint -I$ROOTSYS/include +V TAttAxis.h TAxis.h LinkDef.h rootcint.C \ //
//                            TAttAxis.h TAxis.h > AxisGen.C            //
//                                                                      //
// or                                                                   //
//                                                                      //
// cint -I$ROOTSYS/include +V TAttAxis.h TAxis.h LinkDef.h rootcint.C \ //
//                            AxisGen.C TAttAxis.h TAxis.h              //
//                                                                      //
// The +V and -I$ROOTSYS/include options are added to the list of       //
// arguments in the compiled version of rootcint.                       //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "Api.h"

extern "C" {
   void  G__setothermain(int othermain);
   void  G__setglobalcomp(int globalcomp);
   int   G__main(int argc, char **argv);
   void  G__exit(int rtn);
   struct G__includepath *G__getipathentry();
}

const char *help =
"\n"
"This program generates the Streamer(), TBuffer &operator>>() and\n"
"ShowMembers() methods for ROOT classes, i.e. classes using the\n"
"ClassDef and ClassImp macros.\n"
"In addition rootcint can also generate the CINT dictionaries\n"
"needed in order to get access to ones classes via the interpreter.\n"
"\n"
"Rootcint can be used like:\n"
"\n"
"  rootcint TAttAxis.h[+][-][!] ... [LinkDef.h] > AxisDict.cxx\n"
"\n"
"or\n"
"\n"
"  rootcint [-f] AxisDict.cxx [-c] TAttAxis.h[+][-][!] ... [LinkDef.h]\n"
"\n"
"The difference between the two is that in the first case only the\n"
"Streamer() and ShowMembers() methods are generated while in the\n"
"latter case a complete compileable file is generated (including\n"
"the include statements). The first method also allows the\n"
"output to be appended to an already existing file (using >>).\n"
"The optional - behind the include file name tells rootcint\n"
"to not generate the Streamer() method. A custom method must be\n"
"provided by the user in that case. For the ! and + options see below.\n"
"When using option -c also the interpreter method interface stubs\n"
"will be written to the output file (AxisDict.cxx in the above case).\n"
"By default the output file will not be overwritten if it exists.\n"
"Use the -f (force) option to overwite the output file. The output\n"
"file must have one of the following extensions: .cxx, .C, .cpp,\n"
".cc, .cp.\n"
"\n"
"Before specifying the first header file one can also add include\n"
"file directories to be searched and preprocessor defines, like:\n"
"   -I$../include -DDebug\n"
"\n"
"The (optional) file LinkDef.h looks like:\n"
"\n"
"#ifdef __CINT__\n"
"\n"
"#pragma link off all globals;\n"
"#pragma link off all classes;\n"
"#pragma link off all functions;\n"
"\n"
"#pragma link C++ class TAxis;\n"
"#pragma link C++ class TAttAxis-;\n"
"#pragma link C++ class TArrayC-!;\n"
"#pragma link C++ class AliEvent+;\n"
"\n"
"#pragma link C++ function StrDup;\n"
"#pragma link C++ function operator+(const TString&,const TString&);\n"
"\n"
"#pragma link C++ global gROOT;\n"
"#pragma link C++ global gEnv;\n"
"\n"
"#pragma link C++ enum EMessageTypes;\n"
"\n"
"#endif\n"
"\n"
"This file tells rootcint for which classes the method interface\n"
"stubs should be generated. A trailing - in the class name tells\n"
"rootcint to not generate the Streamer() method. This is necessary\n"
"for those classes that need a customized Streamer() method.\n"
"A trailing ! in the class name tells rootcint to not generate the\n"
"operator>>(TBuffer &b, MyClass *&obj) method. This is necessary to\n"
"be able to write pointers to objects of classes not inheriting from\n"
"TObject. See for an example the source of the TArrayF class.\n"
"A trailing + in the class name tells rootcint to generate a\n"
"Streamer() with extra byte count information. This adds one int to\n"
"each object in the output buffer, but it allows for powerful error\n"
"correction in case a Streamer() method is out of sync compared to\n"
"the data on the file. The + option is mutual exclusive with both\n"
"the - and ! options.\n"
"When this linkdef file is not specified a default version exporting\n"
"the classes with the names equal to the include files minus the .h\n"
"is generated.\n"
"\n"
"*** IMPORTANT ***\n"
"1) LinkDef.h must be the last argument on the rootcint command line.\n"
"2) Note that the LinkDef file name MUST contain the string:\n"
"   LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h is fine,\n"
"   just like mylinkdef.h. Linkdef.h is case sensitive.\n";

#else
#include <ertti.h>
#endif

#ifdef __MWERKS__
#include <console.h>
#endif

#include <time.h>

char *autold = "G__autoLinkDef.h";

FILE *fp;

//______________________________________________________________________________
int IsSTLContainer(G__DataMemberInfo &m)
{
   // Is this an STL container?

   const char *s = m.Type()->TmpltName();
   if (!s) return 0;
   char type[512];
   strcpy(type, s);

   if (!strcmp(type, "vector")   || !strcmp(type, "list")     ||
       !strcmp(type, "deque")    ||
       !strcmp(type, "map")      || !strcmp(type, "set")      ||
       !strcmp(type, "multimap") || !strcmp(type, "multiset"))
      return 1;
   return 0;
}

//______________________________________________________________________________
G__TypeInfo &TemplateArg(G__DataMemberInfo &m, int count = 0)
{
   // Returns template argument. When count = 0 return first argument,
   // 1 second, etc.

   static G__TypeInfo ti;
   char arg[512], *s;

   strcpy(arg, m.Type()->TmpltArg());
   s = strtok(arg, ",");
   for (int i = 0; i < count; i++)
      s = strtok(0, ",");

   ti.Init(s);

   return ti;
}

//______________________________________________________________________________
void WriteStringOperators(FILE *fd)
{
   // Write static ANSI C++ string to TBuffer operators.

   fprintf(fd, "//_______________________________________");
   fprintf(fd, "_______________________________________\n");
   fprintf(fd, "static TBuffer &operator>>(TBuffer &b, string &s)\n{\n");
   fprintf(fd, "   // Reading string object.\n\n");
   fprintf(fd, "   Assert(b.IsReading());\n");
   fprintf(fd, "   char ch;\n");
   fprintf(fd, "   do {\n");
   fprintf(fd, "      b >> ch;\n");
   fprintf(fd, "      if (ch) s.append(1, ch);\n");
   fprintf(fd, "   } while (ch != 0);\n");
   fprintf(fd, "   return b;\n");
   fprintf(fd, "}\n");
   fprintf(fd, "//_______________________________________");
   fprintf(fd, "_______________________________________\n");
   fprintf(fd, "static TBuffer &operator<<(TBuffer &b, string s)\n{\n");
   fprintf(fd, "   // Writing string object.\n\n");
   fprintf(fd, "   Assert(b.IsWriting());\n");
   fprintf(fd, "   b.WriteString(s.data());\n");
   fprintf(fd, "   return b;\n");
   fprintf(fd, "}\n");
}

//______________________________________________________________________________
int STLStringStreamer(G__DataMemberInfo &m, int rwmode)
{
   // Create Streamer code for a standard string object. Returns 1 if data
   // member was a standard string and if Streamer code has been created,
   // 0 otherwise.

   if (!strcmp(m.Type()->Name(), "string") ||
       !strcmp(m.Type()->Name(), "string*")) {
      if (rwmode == 0) {
         // create read mode
         if ((m.Property() & G__BIT_ISPOINTER) &&
             (m.Property() & G__BIT_ISARRAY)) {

         } else if (m.Property() & G__BIT_ISARRAY) {

         } else {
            printf("      { TString R__str; R__str.Streamer(R__b); ");
            if (m.Property() & G__BIT_ISPOINTER)
               printf("(*(%s = new string)) = R__str.Data(); }\n", m.Name());
            else
               printf("%s = R__str.Data(); }\n", m.Name());
         }
      } else {
         // create write mode
         if (m.Property() & G__BIT_ISPOINTER)
            printf("      { R__b.WriteString(%s->data());\n", m.Name());
         else
            printf("      { R__b.WriteString(%s.data());\n", m.Name());
      }
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int STLContainerStreamer(G__DataMemberInfo &m, int rwmode)
{
   // Create Streamer code for an STL container. Returns 1 if data member
   // was an STL container and if Streamer code has been created, 0 otherwise.

   if (m.Type()->IsTmplt() && IsSTLContainer(m)) {
      const char *stlc = m.Type()->TmpltName();
      if (!strcmp(stlc, "vector") || !strcmp(stlc, "list") ||
          !strcmp(stlc, "deque")) {
         if (rwmode == 0) {
            // create read code
            fprintf(fp, "      {\n");
            if (m.Property() & G__BIT_ISPOINTER)
               fprintf(fp, "         %s = new %s;\n", m.Name(), m.Type()->Name());
            fprintf(fp, "         int R__i, R__n;\n");
            fprintf(fp, "         R__b >> R__n;\n");
            fprintf(fp, "         for (R__i = 0; R__i < R__n; R__i++) {\n");
            const char *s = TemplateArg(m).Name();
            if (!strncmp(s, "const ", 6)) s += 6;
            fprintf(fp, "            %s R__t;\n", s);
            if ((TemplateArg(m).Property() & G__BIT_ISPOINTER) ||
                (TemplateArg(m).Property() & G__BIT_ISFUNDAMENTAL) ||
                (TemplateArg(m).Property() & G__BIT_ISENUM)) {
               if (TemplateArg(m).Property() & G__BIT_ISENUM)
                  fprintf(fp, "            R__b >> (Int_t&)R__t;\n");
               else
                  fprintf(fp, "            R__b >> R__t;\n");
            } else {
               if (TemplateArg(m).HasMethod("Streamer"))
                  fprintf(fp, "            R__t.Streamer(R__b);\n");
               else {
                  fprintf(stderr, "*** Datamember %s::%s: template arg %s has no Streamer()"
                          " method (need manual intervention)\n",
                          m.MemberOf()->Name(), m.Name(), TemplateArg(m).Name());
                  fprintf(fp, "            //R__t.Streamer(R__b);\n");
               }
            }
            if (m.Property() & G__BIT_ISPOINTER)
               fprintf(fp, "            %s->push_back(R__t);\n", m.Name());
            else
               fprintf(fp, "            %s.push_back(R__t);\n", m.Name());
            fprintf(fp, "         }\n");
            fprintf(fp, "      }\n");
         } else {
            // create write code
            fprintf(fp, "      {\n");
            if (m.Property() & G__BIT_ISPOINTER)
               fprintf(fp, "         R__b << %s->size();\n", m.Name());
            else
               fprintf(fp, "         R__b << %s.size();\n", m.Name());
            fprintf(fp, "         %s<%s>::iterator R__k;\n", stlc, TemplateArg(m).Name());
            if (m.Property() & G__BIT_ISPOINTER)
               fprintf(fp, "         for (R__k = %s->begin(); R__k != %s->end(); ++R__k)\n",
                       m.Name(), m.Name());
            else
               fprintf(fp, "         for (R__k = %s.begin(); R__k != %s.end(); ++R__k)\n",
                       m.Name(), m.Name());
            if ((TemplateArg(m).Property() & G__BIT_ISPOINTER) ||
                (TemplateArg(m).Property() & G__BIT_ISFUNDAMENTAL) ||
                (TemplateArg(m).Property() & G__BIT_ISENUM)) {
               if (TemplateArg(m).Property() & G__BIT_ISENUM)
                  fprintf(fp, "            R__b << (Int_t)*R__k;\n");
               else
                  fprintf(fp, "            R__b << *R__k;\n");
            } else {
               if (TemplateArg(m).HasMethod("Streamer"))
                  fprintf(fp, "            (*R__k).Streamer(R__b);\n");
               else
                  fprintf(fp, "            //(*R__k).Streamer(R__b);\n");
            }
            fprintf(fp, "      }\n");
         }
      }
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int PointerToPointer(G__DataMemberInfo &m)
{
   if (strstr(m.Type()->Name(), "**")) return 1;
   return 0;
}

//______________________________________________________________________________
void WriteArrayDimensions(int dim)
{
   for (int i = 0; i < dim-1; i++)
      fprintf(fp, "[0]");
}

//______________________________________________________________________________
void WriteInputOperator(G__ClassInfo &cl)
{
   fprintf(fp, "//_______________________________________");
   fprintf(fp, "_______________________________________\n");

   G__ClassInfo space = cl.EnclosingClass();
   char space_prefix[256] = "";
   if (space.Property() & G__BIT_ISNAMESPACE)
      sprintf(space_prefix,"%s::",space.Fullname());

   if (cl.IsTmplt()) {
      // Produce specialisation for templates:
      fprintf(fp, "template <> TBuffer &%soperator>><%s >"
              "(TBuffer &buf, %s *&obj)\n{\n", space_prefix, cl.TmpltArg(), cl.Fullname());
   } else {
      fprintf(fp, "TBuffer &%soperator>>(TBuffer &buf, %s *&obj)\n{\n",
              space_prefix, cl.Fullname() );
   }
   fprintf(fp, "   // Read a pointer to an object of class %s.\n\n", cl.Fullname());

   if (cl.IsBase("TObject") || !strcmp(cl.Fullname(), "TObject")) {
      fprintf(fp, "   obj = (%s *) buf.ReadObject(%s::Class());\n", cl.Fullname(),
              cl.Fullname());
   } else {
      fprintf(fp, "   ::Error(\"%s::operator>>\", \"objects not inheriting"
                  " from TObject need a specialized operator>>"
                  " function\"); if (obj) { }\n", cl.Fullname());
   }
   fprintf(fp, "   return buf;\n}\n\n");
}

//______________________________________________________________________________
void WriteClassName(G__ClassInfo &cl, int tmplt = 0)
{
   // Write the code to set the class name and the initialization object.

   fprintf(fp, "//_______________________________________");
   fprintf(fp, "_______________________________________\n");
   fprintf(fp, "const char *%s::Class_Name()\n{\n", cl.Fullname());
   fprintf(fp, "   // Return the class name for %s.\n", cl.Fullname());
   fprintf(fp, "   return \"%s\";\n}\n\n", cl.Fullname());
   if (!tmplt) {
      fprintf(fp, "// Static variable to hold initialization object\n");
      fprintf(fp, "static %s::R__Init __gR__Init%s;\n\n",
              cl.Fullname(), G__map_cpp_name((char *)cl.Fullname()));
   } else {
      fprintf(fp, "// Static variable to hold initialization object\n");
      fprintf(fp, "static R__Init%s __gR__Init%s%s;\n\n", cl.Name(),
              cl.TmpltName(), G__map_cpp_name((char *)cl.TmpltArg()));
   }
}

//______________________________________________________________________________
const char *ShortTypeName (const char *typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // You need to use the result immediately before it is being overwritten.

   static char t[64];
   char *s;
   if (!strstr(typeDesc, "(*)(") && (s = (char*)strchr(typeDesc, ' ')))
      strcpy(t, s+1);
   else
      strcpy(t, typeDesc);

   int l = strlen(t);
   while (l > 0 && t[l-1] == '*')
      t[--l] = 0;

   return t;
}

//______________________________________________________________________________
const char *GrabIndex(G__DataMemberInfo &member, int printError)
{
   // GrabIndex return a static string (so use it or copy it immediatly, do not
   // call GrabIndex twice in the same expression) containing the size of the
   // array data member.
   // In case of error, or if the size is not specified, GrabIndex returns 0.

   int error;
   char *where = 0;

   const char *index = member.ValidArrayIndex(&error, &where);
   if (index==0 && printError) {
      char *errorstring;
      switch (error) {
         case G__DataMemberInfo::NOT_INT:
            errorstring = "is not an integer";
            break;
         case G__DataMemberInfo::NOT_DEF:
            errorstring = "has not been defined before the array";
            break;
         case G__DataMemberInfo::IS_PRIVATE:
            errorstring = "is a private member of a parent class";
            break;
         case G__DataMemberInfo::UNKNOWN:
            errorstring = "is not known";
            break;
         default:
            errorstring = "UNKNOWN ERROR!!!!";
      }

      if (where==0) {
         fprintf(stderr,"*** Datamember %s::%s: no size indication!\n",
                 member.MemberOf()->Name(), member.Name());
      } else {
         fprintf(stderr,"*** Datamember %s::%s: size of array (%s) %s!\n",
                   member.MemberOf()->Name(), member.Name(), where, errorstring);
      }
   }
   return index;
}


//______________________________________________________________________________
void WriteStreamer(G__ClassInfo &cl)
{
   fprintf(fp, "//_______________________________________");
   fprintf(fp, "_______________________________________\n");
   fprintf(fp, "void %s::Streamer(TBuffer &R__b)\n{\n", cl.Fullname());
   fprintf(fp, "   // Stream an object of class %s.\n\n", cl.Fullname());

   // In case of VersionID<=0 write dummy streamer only calling
   // its base class Streamer(s). If no base class(es) let Streamer
   // print error message, i.e. this Streamer should never have been called.
   char a[80];
   int version;
   sprintf(a, "%s::Class_Version()", cl.Fullname());
   version = (int)G__int(G__calc(a));
   if (version <= 0) {
      G__BaseClassInfo b(cl);

      int basestreamer = 0;
      while (b.Next())
         if (b.HasMethod("Streamer")) {
            fprintf(fp, "   %s::Streamer(R__b);\n", b.Name());
            basestreamer++;
         }
      if (!basestreamer) {
         fprintf(fp, "   ::Error(\"%s::Streamer\", \"version id <=0 in ClassDef,"
                 " dummy Streamer() called\"); if (R__b.IsReading()) { }\n", cl.Fullname());
      }
      fprintf(fp, "}\n\n");
      return;
   }

   // see if we should generate Streamer with byte count code
   int ubc = 0;
   if ((cl.RootFlag() & G__USEBYTECOUNT)) ubc = 1;

   // loop twice: first time write reading code, second time writing code
   for (int i = 0; i < 2; i++) {

      int decli = 0;

      if (i == 0) {
         if (ubc) fprintf(fp, "   UInt_t R__s, R__c;\n");
         fprintf(fp, "   if (R__b.IsReading()) {\n");
         if (ubc)
            fprintf(fp, "      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }\n");
         else
            fprintf(fp, "      Version_t R__v = R__b.ReadVersion(); if (R__v) { }\n");
      } else {
         if (ubc) fprintf(fp, "      R__b.CheckByteCount(R__s, R__c, %s::IsA());\n", cl.Fullname());
         fprintf(fp, "   } else {\n");
         if (ubc)
            fprintf(fp, "      R__c = R__b.WriteVersion(%s::IsA(), kTRUE);\n",cl.Fullname());
         else
            fprintf(fp, "      R__b.WriteVersion(%s::IsA());\n",cl.Fullname());
      }

      // Stream base class(es) when they have the Streamer() method
      G__BaseClassInfo b(cl);

      while (b.Next())
         if (b.HasMethod("Streamer"))
            fprintf(fp, "      %s::Streamer(R__b);\n", b.Name());

      // Stream data members
      G__DataMemberInfo m(cl);

      while (m.Next()) {

         // we skip:
         //  - static members
         //  - members with an ! as first character in the title (comment) field
         //  - the member G__virtualinfo inserted by the CINT RTTI system

         if (!(m.Property() & G__BIT_ISSTATIC) &&
             strncmp(m.Title(), "!", 1)        &&
             strcmp(m.Name(), "G__virtualinfo")) {

            // fundamental type: short, int, long, etc....
            if (((m.Type())->Property() & G__BIT_ISFUNDAMENTAL) ||
                ((m.Type())->Property() & G__BIT_ISENUM)) {
               if (m.Property() & G__BIT_ISARRAY &&
                   m.Property() & G__BIT_ISPOINTER) {
                  int s = 1;
                  for (int dim = 0; dim < m.ArrayDim(); dim++)
                     s *= m.MaxIndex(dim);
                  if (!decli) {
                     fprintf(fp, "      int R__i;\n");
                     decli = 1;
                  }
                  fprintf(fp, "      for (R__i = 0; R__i < %d; R__i++)\n", s);
                 if (i == 0) {
                     fprintf(stderr,"*** Datamember %s::%s: array of pointers to fundamental type (need manual intervention)\n", cl.Fullname(), m.Name());
                     fprintf(fp, "         ;//R__b.ReadArray(%s);\n", m.Name());
                  } else {
                     fprintf(fp, "         ;//R__b.WriteArray(%s, __COUNTER__);\n", m.Name());
                  }
               } else if (m.Property() & G__BIT_ISPOINTER) {
                  const char *indexvar = GrabIndex(m, i==0);
                  if (indexvar==0) {
                     if (i == 0) {
                        fprintf(stderr,"*** Datamember %s::%s: pointer to fundamental type (need manual intervention)\n", cl.Fullname(), m.Name());
                        fprintf(fp, "      //R__b.ReadArray(%s);\n", m.Name());
                     } else {
                        fprintf(fp, "      //R__b.WriteArray(%s, __COUNTER__);\n", m.Name());
                     }
                  } else {
                     if (i == 0) {
                        fprintf(fp, "      delete []%s; \n",m.Name());
                        fprintf(fp, "      %s = new %s[%s]; \n",
                                m.Name(),ShortTypeName(m.Type()->Name()),indexvar);
                        fprintf(fp, "      R__b.ReadFastArray(%s,%s); \n",
                                m.Name(),indexvar);
                     } else {
                        fprintf(fp, "      R__b.WriteFastArray(%s,%s); \n",
                                m.Name(),indexvar);
                     }
                  }
               } else if (m.Property() & G__BIT_ISARRAY) {
                  if (i == 0) {
                     if (m.ArrayDim() > 1) {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           fprintf(fp, "      R__b.ReadStaticArray((Int_t*)%s);\n", m.Name());
                        else
                           fprintf(fp, "      R__b.ReadStaticArray((%s*)%s);\n", m.Type()->TrueName(), m.Name());
                     } else {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           fprintf(fp, "      R__b.ReadStaticArray((Int_t*)%s);\n", m.Name());
                        else
                           fprintf(fp, "      R__b.ReadStaticArray(%s);\n", m.Name());
                      }
                  } else {
                     int s = 1;
                     for (int dim = 0; dim < m.ArrayDim(); dim++)
                        s *= m.MaxIndex(dim);
                     if (m.ArrayDim() > 1) {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           fprintf(fp, "      R__b.WriteArray((Int_t*)%s, %d);\n", m.Name(), s);
                        else
                           fprintf(fp, "      R__b.WriteArray((%s*)%s, %d);\n", m.Type()->TrueName(), m.Name(), s);
                     } else {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           fprintf(fp, "      R__b.WriteArray((Int_t*)%s, %d);\n", m.Name(), s);
                        else
                           fprintf(fp, "      R__b.WriteArray(%s, %d);\n", m.Name(), s);
                     }
                  }
               } else if ((m.Type())->Property() & G__BIT_ISENUM) {
                  if (i == 0)
                     fprintf(fp, "      R__b >> (Int_t&)%s;\n", m.Name());
                  else
                     fprintf(fp, "      R__b << (Int_t)%s;\n", m.Name());
               } else {
                  if (i == 0)
                     fprintf(fp, "      R__b >> %s;\n", m.Name());
                  else
                     fprintf(fp, "      R__b << %s;\n", m.Name());
               }
            } else {
               // we have an object...

               // check if object is a standard string
               if (STLStringStreamer(m, i))
                  continue;

               // check if object is an STL container
               if (STLContainerStreamer(m, i))
                  continue;

               // handle any other type of objects
               if (m.Property() & G__BIT_ISARRAY &&
                   m.Property() & G__BIT_ISPOINTER) {
                  int s = 1;
                  for (int dim = 0; dim < m.ArrayDim(); dim++)
                     s *= m.MaxIndex(dim);
                  if (!decli) {
                     fprintf(fp, "      int R__i;\n");
                     decli = 1;
                  }
                  fprintf(fp, "      for (R__i = 0; R__i < %d; R__i++)\n", s);
                  if (i == 0)
                     fprintf(fp, "         R__b >> %s", m.Name());
                  else {
                     if (m.Type()->IsBase("TObject") && m.Type()->IsBase("TArray"))
                        fprintf(fp, "         R__b << (TObject*)%s", m.Name());
                     else
                        fprintf(fp, "         R__b << %s", m.Name());
                  }
                  WriteArrayDimensions(m.ArrayDim());
                  fprintf(fp, "[R__i];\n");
               } else if (m.Property() & G__BIT_ISPOINTER) {
                  // This is always good. However, in case of a pointer
                  // to an object that is guarenteed to be there and not
                  // being referenced by other objects we could use
                  //     xx->Streamer(b);
                  // Optimize this with control statement in title.
                  if (PointerToPointer(m)) {
                     if (i == 0) {
                        fprintf(stderr,"*** Datamember %s::%s: pointer to pointer (need manual intervention)\n", cl.Fullname(), m.Name());
                        fprintf(fp, "      //R__b.ReadArray(%s);\n", m.Name());
                     } else {
                        fprintf(fp, "      //R__b.WriteArray(%s, __COUNTER__);\n", m.Name());
                     }
                  } else {
                     if (strstr(m.Type()->Name(), "TClonesArray")) {
                        fprintf(fp, "      %s->Streamer(R__b);\n", m.Name());
                     } else {
                        if (i == 0)
                           fprintf(fp, "      R__b >> %s;\n", m.Name());
                        else {
                           if (m.Type()->IsBase("TObject") && m.Type()->IsBase("TArray"))
                              fprintf(fp, "      R__b << (TObject*)%s;\n", m.Name());
                           else
                              fprintf(fp, "      R__b << %s;\n", m.Name());
                        }
                     }
                  }
               } else if (m.Property() & G__BIT_ISARRAY) {
                  int s = 1;
                  for (int dim = 0; dim < m.ArrayDim(); dim++)
                     s *= m.MaxIndex(dim);
                  if (!decli) {
                     fprintf(fp, "      int R__i;\n");
                     decli = 1;
                  }
                  fprintf(fp, "      for (R__i = 0; R__i < %d; R__i++)\n", s);
                  fprintf(fp, "         %s", m.Name());
                  WriteArrayDimensions(m.ArrayDim());
                  fprintf(fp, "[R__i].Streamer(R__b);\n");
               } else {
                  if ((m.Type())->HasMethod("Streamer"))
                     fprintf(fp, "      %s.Streamer(R__b);\n", m.Name());
                  else {
                     if (i == 0)
                        fprintf(stderr, "*** Datamember %s::%s: object has no Streamer() method (need manual intervention)\n",
                                cl.Fullname(), m.Name());
                     fprintf(fp, "      //%s.Streamer(R__b);\n", m.Name());
                  }
               }
            }
         }
      }
   }
   if (ubc) fprintf(fp, "      R__b.SetByteCount(R__c, kTRUE);\n");
   fprintf(fp, "   }\n");
   fprintf(fp, "}\n\n");
}

//______________________________________________________________________________
void WriteShowMembers(G__ClassInfo &cl)
{
   fprintf(fp, "//_______________________________________");
   fprintf(fp, "_______________________________________\n");
   fprintf(fp, "void %s::ShowMembers(TMemberInspector &R__insp, char *R__parent)\n{\n", cl.Fullname());
   fprintf(fp, "   // Inspect the data members of an object of class %s.\n\n", cl.Fullname());
   fprintf(fp, "   TClass *R__cl  = %s::IsA();\n", cl.Fullname());
   fprintf(fp, "   Int_t   R__ncp = strlen(R__parent);\n");
   fprintf(fp, "   if (R__ncp || R__cl || R__insp.IsA()) { }\n");

   // Inspect data members
   G__DataMemberInfo m(cl);
   char cdim[12], cvar[64];

   while (m.Next()) {

      // we skip:
      //  - static members
      //  - the member G__virtualinfo inserted by the CINT RTTI system

      if (!(m.Property() & G__BIT_ISSTATIC) &&
          strcmp(m.Name(), "G__virtualinfo")) {

         // fundamental type: short, int, long, etc....
         if (((m.Type())->Property() & G__BIT_ISFUNDAMENTAL) ||
             ((m.Type())->Property() & G__BIT_ISENUM)) {
            if (m.Property() & G__BIT_ISARRAY &&
                m.Property() & G__BIT_ISPOINTER) {
               sprintf(cvar, "*%s", m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"%s\", &%s);\n",
                       cvar, m.Name());
            } else if (m.Property() & G__BIT_ISPOINTER) {
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"*%s\", &%s);\n",
                       m.Name(), m.Name());
            } else if (m.Property() & G__BIT_ISARRAY) {
               sprintf(cvar, "%s", m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"%s\", %s);\n",
                       cvar, m.Name());
            } else {
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"%s\", &%s);\n",
                       m.Name(), m.Name());
            }
         } else {
            // we have an object
            if (m.Property() & G__BIT_ISARRAY &&
                m.Property() & G__BIT_ISPOINTER) {
               sprintf(cvar, "*%s", m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"%s\", &%s);\n", cvar,
                       m.Name());
            } else if (m.Property() & G__BIT_ISPOINTER) {
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"*%s\", &%s);\n",
                       m.Name(), m.Name());
            } else if (m.Property() & G__BIT_ISARRAY) {
               sprintf(cvar, "%s", m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               fprintf(fp, "   R__insp.Inspect(R__cl, R__parent, \"%s\", %s);\n",
                       cvar, m.Name());
            } else {
               if ((m.Type())->HasMethod("ShowMembers"))
                  fprintf(fp, "   %s.ShowMembers(R__insp, strcat(R__parent,\"%s.\")); R__parent[R__ncp] = 0;\n",
                          m.Name(), m.Name());
            }
         }
      }
   }

   // Write ShowMembers for base class(es) when they have the ShowMember() method
   G__BaseClassInfo b(cl);

   while (b.Next())
      if (b.HasMethod("ShowMembers"))
         fprintf(fp, "   %s::ShowMembers(R__insp, R__parent);\n", b.Name());

   fprintf(fp, "}\n\n");
}

//______________________________________________________________________________
void GenerateLinkdef(int *argc, char **argv, int iv)
{
   FILE *fl = fopen(autold, "w");

   fprintf(fl, "#ifdef __CINT__\n\n");
   fprintf(fl, "#pragma link off all globals;\n");
   fprintf(fl, "#pragma link off all classes;\n");
   fprintf(fl, "#pragma link off all functions;\n\n");

   for (int i = iv; i < *argc; i++) {
      char *s, trail[3];
      int   nostr = 0, noinp = 0, bcnt = 0, l = strlen(argv[i])-1;
      for (int j = 0; j < 3; j++) {
         if (argv[i][l] == '-') {
            argv[i][l] = '\0';
            nostr = 1;
            l--;
         }
         if (argv[i][l] == '!') {
            argv[i][l] = '\0';
            noinp = 1;
            l--;
         }
         if (argv[i][l] == '+') {
            argv[i][l] = '\0';
            bcnt = 1;
            l--;
         }
      }
      if (nostr || noinp) {
         trail[0] = 0;
         if (nostr) strcat(trail, "-");
         if (noinp) strcat(trail, "!");
      }
      if (bcnt) {
         strcpy(trail, "+");
         if (nostr || noinp)
            fprintf(stderr, "option + mutual exclusive with either - or !\n");
      }
      char *cls = strrchr(argv[i], '/');
      if (!cls) cls = strrchr(argv[i], '\\');
      if (cls)
         cls++;
      else
         cls = argv[i];
      if ((s = strrchr(cls, '.'))) *s = '\0';
      if (nostr || noinp || bcnt)
         fprintf(fl, "#pragma link C++ class %s%s;\n", cls, trail);
      else
         fprintf(fl, "#pragma link C++ class %s;\n", cls);
      if (s) *s = '.';
   }
   argv[(*argc)++] = autold;

   fprintf(fl, "\n#endif\n");
   fclose(fl);
}

//______________________________________________________________________________
const char *Which(const char *fname)
{
   // Find file name in path specified via -I statements to CINT.
   // Can be only called after G__main(). Return pointer to static
   // space containing full pathname or 0 in case file not found.

   static char pname[1024];
   FILE *fp = 0;

   sprintf(pname, "%s", fname);
#ifdef WIN32
   fp = fopen(pname, "rb");
#else
   fp = fopen(pname, "r");
#endif
   if (fp) {
      fclose(fp);
      return pname;
   }

   struct G__includepath *ipath = G__getipathentry();

   while (!fp && ipath->pathname) {
#ifdef WIN32
      sprintf(pname, "%s\\%s", ipath->pathname, fname);
      fp = fopen(pname, "rb");
#else
      sprintf(pname, "%s/%s", ipath->pathname, fname);
      fp = fopen(pname, "r");
#endif
      ipath = ipath->next;
   }
   if (fp) {
      fclose(fp);
      return pname;
   }
   return 0;
}

//______________________________________________________________________________
char *StrDup(const char *str)
{
   // Duplicate the string str. The returned string has to be deleted by
   // the user.

   if (!str) return 0;

   char *s = new char[strlen(str)+1];
   if (s) strcpy(s, str);

   return s;
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
#ifdef __MWERKS__
   argc = ccommand(&argv);
#endif

   if (argc < 2) {
      fprintf(stderr,
      "Usage: %s [-f] [out.cxx] [-c] file1.h[+][-][!] file2.h[+][-][!]...[LinkDef.h]\n",
              argv[0]);
      fprintf(stderr, "For more extensive help type: %s -?\n", argv[0]);
      return 1;
   }

   char dictname[256];
   int i, ic, ifl, force;
   int icc = 0;
   int use_preprocessor = 0;

   if (!strcmp(argv[1], "-f")) {
      force = 1;
      ic    = 2;
   } else {
      force = 0;
      ic    = 1;
   }

   if (strstr(argv[ic],".C")  || strstr(argv[ic],".cpp") ||
       strstr(argv[ic],".cp") || strstr(argv[ic],".cxx") ||
       strstr(argv[ic],".cc")) {
      if ((fp = fopen(argv[ic], "r")) != 0) {
         fclose(fp);
         if (!force) {
            fprintf(stderr, "%s: output file %s already exists\n", argv[0], argv[ic]);
            return 1;
         }
      }
      fp = fopen(argv[ic], "w");
      if (fp) fclose(fp);    // make sure file is created and empty
      ifl = ic;
      ic++;

      // remove possible pathname to get the dictionary name
      strcpy(dictname, argv[ifl]);
      char *p = strrchr(dictname, '/');
      if (!p)
         p = dictname;
      else
         p++;
      strcpy(dictname, p);
   } else if (!strcmp(argv[1], "-?")) {
      fprintf(stderr, "%s\n", help);
      return 1;
   } else {
      fp = stdout;
      ic = 1;
      if (force) ic = 2;
      ifl = 0;
   }

#ifndef __CINT__
   int   argcc, iv, il;
   char  path1[128], path2[128];
   char *argvv[500];

   path1[0] = path2[0] = 0;

#ifndef ROOTINCDIR
# ifndef ROOTBUILD
   if (getenv("ROOTSYS")) {
#  ifdef __MWERKS__
      sprintf(path1,"-I%s:include", getenv("ROOTSYS"));
      sprintf(path2,"-I%s:src", getenv("ROOTSYS"));
#  else
      sprintf(path1,"-I%s/include", getenv("ROOTSYS"));
      sprintf(path2,"-I%s/src", getenv("ROOTSYS"));
#  endif
   } else {
      fprintf(stderr, "%s: env var ROOTSYS not defined\n", argv[0]);
      return 1;
   }
# else
   sprintf(path1,"-Iinclude");
# endif
#else
   sprintf(path1,"-I%s", ROOTINCDIR);
#endif

   argvv[0] = argv[0];
   argcc = 1;

   if (!strcmp(argv[ic], "-c")) {
      icc++;
      if (ifl) {
         char *s;
         ic++;
         argvv[argcc++] = "-q0";
         argvv[argcc++] = "-n";
         argvv[argcc] = (char *)calloc(strlen(argv[ifl])+1, 1);
         strcpy(argvv[argcc], argv[ifl]); argcc++;
         argvv[argcc++] = "-N";
         s = strchr(dictname,'.');
         argvv[argcc] = (char *)calloc(strlen(dictname), 1);
         strncpy(argvv[argcc], dictname, s-dictname); argcc++;

         while (*argv[ic] == '-' || *argv[ic] == '+') {
            argvv[argcc++] = argv[ic++];
         }

         argvv[argcc++] = path1;
         if (strlen(path2))
            argvv[argcc++] = path2;
#ifdef __hpux
         argvv[argcc++] = "-I/usr/include/X11R5";
#endif
         argvv[argcc++] = "-DTRUE=1";
         argvv[argcc++] = "-DFALSE=0";
         argvv[argcc++] = "-Dexternalref=extern";
         argvv[argcc++] = "-DSYSV";
         argvv[argcc++] = "-D__MAKECINT__";
         argvv[argcc++] = "-V";        // include info on private members
         argvv[argcc++] = "-c-1";
         argvv[argcc++] = "+V";        // turn on class comment mode
         argvv[argcc++] = "TROOT.h";
         argvv[argcc++] = "TMemberInspector.h";
      } else {
         fprintf(stderr, "%s: option -c can only be used when an output file has been specified\n", argv[0]);
         return 1;
      }
   }

   iv = 0;
   il = 0;
   // If the user request use of a preprocessor we are going to bundle
   // all the files into one so that cint consider then one compilation
   // unit and so that each file that contains code guard is really
   // included only once.
   for (i = 1; i < argc; i++)
      if (strcmp(argv[i], "-p") == 0) use_preprocessor = 1;

   char bundlename[L_tmpnam];
   FILE *bundle = 0;
   if (use_preprocessor) {
      tmpnam(bundlename);
      if (strlen(bundlename) < (L_tmpnam-3)) strcat(bundlename,".C");
      bundle = fopen(bundlename, "w");
      if (bundle==0) {
         fprintf(stderr,"%s: failed to open %s, usage of external preprocessor by CINT is not optimal\n",
                 argv[0], bundlename);
         use_preprocessor = 0;
      }
   }
   for (i = ic; i < argc; i++) {
      if (!iv && *argv[i] != '-' && *argv[i] != '+') {
         if (!icc) {
            argvv[argcc++] = path1;
            argvv[argcc++] = path2;
            argvv[argcc++] = "+V";
         }
         iv = argcc;
      }
      if ((strstr(argv[i],"LinkDef") || strstr(argv[i],"Linkdef") ||
           strstr(argv[i],"linkdef")) && strstr(argv[i],".h")) {
         il = i;
         if (i != argc-1) {
            fprintf(stderr, "%s: %s must be last file on command line\n", argv[0], argv[i]);
            return 1;
         }
         if (use_preprocessor) argvv[argcc++] = bundlename;
      }
      if (!strcmp(argv[i], "-c")) {
         fprintf(stderr, "%s: option -c must come directly after the output file\n", argv[0]);
         return 1;
      }
      if (use_preprocessor && *argv[i] != '-' && *argv[i] != '+' && (il==0))
         fprintf(bundle,"#include \"%s\"\n", argv[i]);
      else
         argvv[argcc++] = argv[i];
   }
   if (use_preprocessor) {
      // Since we have not seen a linkdef file, we have not yet added the
      // bundle file to the command line!
      if (!il) argvv[argcc++] = bundlename;
      fclose(bundle);
   }

   if (!il)
      GenerateLinkdef(&argcc, argvv, iv);

   G__setothermain(2);
   if (G__main(argcc, argvv) < 0) {
      fprintf(stderr, "%s: error loading headers...\n", argv[0]);
      return 1;
   }
   G__setglobalcomp(0);  // G__NOLINK

#endif

   // Check if code goes to stdout or cint file, use temporary file
   // for prepending of the rootcint generated code (STK)
   char tname[L_tmpnam];
   if (ifl) {
      tmpnam(tname);
      fp = fopen(tname, "w");
   } else
      fp = stdout;

   time_t t = time(0);
   fprintf(fp, "//\n// File generated by %s at %.24s.\n", argv[0], ctime(&t));
   fprintf(fp, "// Do NOT change. Changes will be lost next time file is generated\n//\n\n");
   fprintf(fp, "#include \"TBuffer.h\"\n");
   fprintf(fp, "#include \"TMemberInspector.h\"\n");
   fprintf(fp, "#include \"TError.h\"\n\n");

   // Loop over all command line arguments and write include statements.
   // Skip options and [G__]LinkDef.h.
   if (ifl && !icc) {
      for (i = ic; i < argc; i++) {
         if (*argv[i] != '-' && *argv[i] != '+' &&
             !strstr(argv[i],"LinkDef.h") && !strstr(argv[i],"Linkdef.h") &&
             !strstr(argv[i],"linkdef.h"))
            fprintf(fp, "#include \"%s\"\n", argv[i]);
      }
      fprintf(fp, "\n");
   }

   // Loop over all classes and create Streamer() & Showmembers() methods
   G__ClassInfo cl;

   // Write all TBuffer &operator>>(...) first to allow template
   // specialisation to occur before template instantiation (STK)
   while (cl.Next()) {
      if ((cl.Property() & G__BIT_ISCLASS) && cl.Linkage() == G__CPPLINK) {
         if (cl.HasMethod("Streamer")) {
            if (!(cl.RootFlag() & G__NOINPUTOPERATOR)) {
               WriteInputOperator(cl);
            } else {
               fprintf(stderr, "Class %s: Do not generate operator>>()\n",
                       cl.Fullname());
            }
         }
      }
   }

   // Open LinkDef file for reading, so that we can process classes
   // in order of appearence in this file (STK)
   FILE *fpld = 0;
   if (!il) {
      // Open auto-generated file
      fpld = fopen(autold, "r");
   } else {
      // Open file specified on command line
      fpld = fopen(Which(argv[il]), "r");
   }
   if (!fpld) {
      fprintf(stderr, "%s: cannot open file %s\n", argv[0], il ? argv[il] : autold);
      if (!il) remove(autold);
      if (ifl) {
         remove(tname);
         remove(argv[ifl]);
      }
      return 1;
   }

   // Keep track of classes processed by reading Linkdef file.
   // When all classes in LinkDef are done, loop over all classes known
   // to CINT output the ones that were not in the LinkDef. This can happen
   // in case "#pragma link C++ defined_in" is used.
   const int kMaxClasses = 1000;
   char *clProcessed[kMaxClasses];
   int   ncls = 0;

   // Read LinkDef file and process valid entries (STK)
   char line[256];
   while (fgets(line, 256, fpld)) {

      // Check if the line contains a "#pragma link C++ class" specification,
      // if so, process the class (STK)
      if ((strcmp(strtok(line, " "), "#pragma") == 0) &&
          (strcmp(strtok(0, " "), "link") == 0) &&
          (strcmp(strtok(0, " "), "C++") == 0) &&
          (strcmp(strtok(0, " " ), "class") == 0)) {

         // Create G__ClassInfo object for this class and process. Be
         // careful with the hardcoded string of trailing options in case
         // these change (STK)
         char *request = strtok(0, " -!+;");
         G__ClassInfo cl(request);
         if (cl.IsValid())
            clProcessed[ncls] = StrDup(cl.Fullname());
         else
            clProcessed[ncls] = StrDup(request);
         ncls++;
         if ((cl.Property() & G__BIT_ISCLASS) && cl.Linkage() == G__CPPLINK) {

            if (cl.HasMethod("Streamer")) {
               if (!(cl.RootFlag() & G__NOSTREAMER))
                  WriteStreamer(cl);
               else
                  fprintf(stderr, "Class %s: Do not generate Streamer() [*** custom streamer ***]\n", cl.Fullname());
            } else {
               fprintf(stderr, "Class %s: Streamer() not declared\n", cl.Fullname());
            }
            if (cl.HasMethod("ShowMembers")) {
               WriteShowMembers(cl);
            } else {
               fprintf(stderr, "Class %s: ShowMembers() not declared\n", cl.Fullname());
            }
            // Write Code for Class_Name() and static variable
            // to hold initialization object (STK)
            if (cl.IsTmplt()) {
               if (cl.HasMethod("Class_Name")) {
                  WriteClassName(cl,1);
               } else {
                  fprintf(stderr, "Class %s: Class_Name() and initialization object"
                          " not declared\n", cl.Fullname());
               }
            } else {
               if (cl.HasMethod("Class_Name")) {
                  WriteClassName(cl);
               }
            }
         }
      }
   }

   // Loop over all classes and create Streamer() & ShowMembers() methods
   // for classes not in clProcessed list (exported via
   // "#pragma link C++ defined_in")
   cl.Init();

   while (cl.Next()) {
      int nxt = 0;
      // skip utility class defined in ClassImp
      if (!strncmp(cl.Fullname(), "R__Init", 7))
         continue;
      for (i = 0; i < ncls; i++)
         if (!strcmp(clProcessed[i], cl.Fullname())) {
            nxt++;
            break;
         }
      if (nxt) continue;

      if ((cl.Property() & G__BIT_ISCLASS) && cl.Linkage() == G__CPPLINK) {
         if (cl.HasMethod("Streamer")) {
            if (!(cl.RootFlag() & G__NOSTREAMER))
               WriteStreamer(cl);
            else
               fprintf(stderr, "Class %s: Do not generate Streamer() [*** custom streamer ***]\n", cl.Fullname());
         } else {
            fprintf(stderr, "Class %s: Streamer() not declared\n", cl.Fullname());
         }
         if (cl.HasMethod("ShowMembers")) {
            WriteShowMembers(cl);
         } else {
            fprintf(stderr, "Class %s: ShowMembers() not declared\n", cl.Fullname());
         }
         // Write Code for Class_Name() and static variable
         // to hold initialization object (STK)
         if (cl.IsTmplt()) {
            if (cl.HasMethod("Class_Name")) {
               WriteClassName(cl,1);
            } else {
               fprintf(stderr, "Class %s: Class_Name() and initialization object"
                       " not declared\n", cl.Fullname());
            }
         } else {
            if (cl.HasMethod("Class_Name")) {
               WriteClassName(cl);
            }
         }
      }
   }

   fclose(fp);
   fclose(fpld);

   if (!il) remove(autold);

   // Append CINT dictionary to file containing Streamers and ShowMembers
   if (ifl) {
      char line[BUFSIZ];
      FILE *fpd = fopen(argv[ifl], "r");
      fp = fopen(tname, "a");

      if (fp && fpd)
         while (fgets(line, BUFSIZ, fpd))
            fprintf(fp, "%s", line);

      if (fp)  fclose(fp);
      if (fpd) fclose(fpd);

      // copy back to dictionary file
      fpd = fopen(argv[ifl], "w");
      fp  = fopen(tname, "r");

      if (fp && fpd) {

         // make name of dict include file "aapDict.cxx" -> "aapDict.h"
         int  nl = 0;
         char inclf[64];
         char *s = strrchr(dictname, '.');
         if (s) *s = 0;
         sprintf(inclf, "%s.h", dictname);
         if (s) *s = '.';

         // during copy put dict include on top and remove later reference
         while (fgets(line, BUFSIZ, fp)) {
            if (!strncmp(line, "#include", 8) && strstr(line, inclf))
               continue;
            fprintf(fpd, "%s", line);
            if (++nl == 4)
               fprintf(fpd, "#include \"%s\"\n", inclf);
         }
      }

      if (fp)  fclose(fp);
      if (fpd) fclose(fpd);
      remove(tname);
   }

   G__setglobalcomp(-1);  // G__CPPLINK
   G__exit(0);

   return 0;
}
