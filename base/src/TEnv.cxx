// @(#)root/base:$Name:  $:$Id: TEnv.cxx,v 1.9 2002/01/25 11:21:57 rdm Exp $
// Author: Fons Rademakers   22/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnv                                                                 //
//                                                                      //
// The TEnv class reads a config file, by default .rootrc. Three types  //
// of .rootrc files are read: global, user and local files. The global  //
// file resides in $ROOTSYS/etc, the user file in ~/ and the local file //
// in the current working directory.                                    //
// The format of the .rootrc file is similar to the .Xdefaults format:  //
//                                                                      //
//   [+]<SystemName>.<RootName|ProgName>.<name>[(type)]:  <value>       //
//                                                                      //
// Where <SystemName> is either Unix, WinNT, MacOS or Vms,              //
// <RootName> the name as given in the TApplication ctor (or "RootApp"  //
// in case no explicit TApplication derived object was created),        //
// <ProgName> the current program name and <name> the resource name,    //
// with optionally a type specification. <value> can be either a        //
// string, an integer, a float/double or a boolean with the values      //
// TRUE, FALSE, ON, OFF, YES, NO, OK, NOT. Booleans will be returned as //
// an integer 0 or 1. The options [+] allows the concatenation of       //
// values to the same resouce name.                                     //
//                                                                      //
// E.g.:                                                                //
//                                                                      //
//   Unix.Rint.Root.DynamicPath: .:$ROOTSYS/lib:~/lib                   //
//   myapp.Root.Debug:  FALSE                                           //
//   TH.Root.Debug: YES                                                 //
//   *.Root.MemStat: 1                                                  //
//                                                                      //
// <SystemName> and <ProgName> or <RootName> may be the wildcard "*".   //
// A # in the first column starts comment line.                         //
//                                                                      //
// For the currently defined resources (and their default values) see   //
// $ROOTSYS/etc/system.rootrc.                                          //
//                                                                      //
// Note that the .rootrc config files contain the config for all ROOT   //
// based applications.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "TEnv.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TOrdCollection.h"
#include "TError.h"

#if defined(R__ANSISTREAM)
#  include <strstream>
#else
#  ifndef R__WIN32
#    include <strstream.h>
#  else
#    include <strstrea.h>
#  endif
#endif


TEnv *gEnv;


static struct BoolNameTable_t {
   const char *fName;
   Int_t       fValue;
} boolNames[]= {
   { "TRUE",  1 },
   { "FALSE", 0 },
   { "ON",    1 },
   { "OFF",   0 },
   { "YES",   1 },
   { "NO",    0 },
   { "OK",    1 },
   { "NOT",   0 },
   { 0, 0 }
};


//---- TEnvParser --------------------------------------------------------------

class TEnvParser {

private:
   FILE    *fIfp;

protected:
   TEnv    *fEnv;

public:
   TEnvParser(TEnv *e, FILE *f) : fIfp(f), fEnv(e) { }
   virtual void KeyValue(const TString&, const TString&, const TString&) { }
   virtual void Char(Int_t) { }
   void Parse();
};

//______________________________________________________________________________
void TEnvParser::Parse()
{
   // Parse a line of the env file and create an entry in the resource
   // dictionary (i.e. add a KeyValue pair).

   TString name, type, value;
   int c, state = 0;

   while ((c = fgetc(fIfp)) != EOF) {
      if (c == '\n') {
         state = 0;
         if (name.Length() > 0) {
            KeyValue(name, value, type);
            name  = "";
            value = "";
            type  = "";
         }
         Char(c);
         continue;
      }
      switch (state) {
      case 0:             // start of line
         switch (c) {
         case ' ':
         case '\t':
            break;
         case '#':
            state = 1;
            break;
         default:
            state = 2;
            break;
         }
         break;

      case 1:             // comment
         break;

      case 2:             // name
         switch (c) {
         case ' ':
         case '\t':
         case ':':
            state = 3;
            break;
         case '(':
            state = 7;
            break;
         default:
            break;
         }
         break;

      case 3:             // ws before value
         if (c != ' ' && c != '\t')
            state = 4;
         break;

      case 4:             // value
         break;

      case 5:             // type
         if (c == ')')
            state = 6;
         break;

      case 6:             // optional ':'
         state = (c == ':') ? 3 : 4;
         break;

      case 7:
         state = (c == ')') ? 6 : 5;
         break;

      }
      switch (state) {
      case 2:
         name.Append(c);
         break;
      case 4:
         value.Append(c);
         break;
      case 5:
         type.Append(c);
         break;
      }
      if (state != 4)
         Char(c);
   }
}

//---- TReadEnvParser ----------------------------------------------------------

class TReadEnvParser : public TEnvParser {

private:
   EEnvLevel fLevel;

public:
   TReadEnvParser(TEnv *e, FILE *f, EEnvLevel l) : TEnvParser(e, f), fLevel(l) { }
   void KeyValue(const TString &name, const TString &value, const TString &type)
      { fEnv->SetValue(name, value, fLevel, type); }
};

//---- TWriteEnvParser ---------------------------------------------------------

class TWriteEnvParser : public TEnvParser {

private:
   FILE *fOfp;

public:
   TWriteEnvParser(TEnv *e, FILE *f, FILE *of) : TEnvParser(e, f), fOfp(of) { }
   void KeyValue(const TString &name, const TString &value, const TString &type);
   void Char(Int_t c) { fputc(c, fOfp); }
};

//______________________________________________________________________________
void TWriteEnvParser::KeyValue(const TString &name, const TString &value,
                               const TString &)
{
   // Write resources out to a new file.

   TEnvRec *er = fEnv->Lookup(name.Data());
   if (er && er->fModified) {
      er->fModified = kFALSE;
      fprintf(fOfp, "%s", er->fValue.Data());
   } else
      fprintf(fOfp, "%s", value.Data());
}


//---- TEnvRec -----------------------------------------------------------------

//______________________________________________________________________________
TEnvRec::TEnvRec(const char *n, const char *v, const char *t, EEnvLevel l)
   : fName(n), fType(t), fLevel(l)
{
   // Ctor of a single resource.

   fValue = ExpandValue(v);
   fModified = (l == kEnvChange);
}

//______________________________________________________________________________
void TEnvRec::ChangeValue(const char *v, const char *, EEnvLevel l,
                          Bool_t append)
{
   // Change the value of a resource.

   if (l != kEnvChange && fLevel == l && !append) {
      // use global Warning() since interpreter might not yet be initialized
      // at this stage (called from TROOT ctor)
      ::Warning("TEnvRec::ChangeValue",
        "duplicate entry <%s=%s> for level %d; ignored", fName.Data(), v, l);
      return;
   }
   if (!append) {
      if (fValue != v) {
         if (l == kEnvChange)
            fModified = kTRUE;
         else {
            fModified = kFALSE;
            fLevel = l;
         }
         fValue = ExpandValue(v);
      }
   } else {
      if (l == kEnvChange)
         fModified = kTRUE;
      fValue += " ";
      fValue += ExpandValue(v);
   }
}

//______________________________________________________________________________
Int_t TEnvRec::Compare(const TObject *op) const
{
   // Comparison function for resources.

   return fName.CompareTo(((TEnvRec*)op)->fName);
}

//______________________________________________________________________________
TString TEnvRec::ExpandValue(const char *value)
{
   // Replace all $(XXX) strings by the value defined in the shell
   // (obtained via TSystem::Getenv()).

   const char *vv;
   char *v, *vorg = StrDup(value);
   v = vorg;

   char *s1, *s2;
   int len = 0;
   while ((s1 = (char*)strstr(v, "$("))) {
      s1 += 2;
      s2 = (char*)strchr(s1, ')');
      if (!s2) {
         len = 0;
         break;
      }
      *s2 = 0;
      vv = gSystem->Getenv(s1);
      if (vv) len += strlen(vv);
      *s2 = ')';
      v = s2 + 1;
   }

   if (!len) {
      delete [] vorg;
      return TString(value);
   }

   v = vorg;
   char *nv = new char[strlen(v) + len];
   *nv = 0;

   while ((s1 = (char*)strstr(v, "$("))) {
      *s1 = 0;
      strcat(nv, v);
      *s1 = '$';
      s1 += 2;
      s2 = (char*)strchr(s1, ')');
      *s2 = 0;
      vv = gSystem->Getenv(s1);
      if (vv) strcat(nv, vv);
      *s2 = ')';
      v = s2 + 1;
   }

   if (*v) strcat(nv, v);

   TString val = nv;
   delete [] nv;
   delete [] vorg;

   return val;
}


//---- TEnv --------------------------------------------------------------------

ClassImp(TEnv)

//______________________________________________________________________________
TEnv::TEnv(const char *name)
{
   // Create a resource table and read the (possibly) three resource files, i.e
   // $ROOTSYS/system<name> (or ROOTETCDIR/system<name>), $HOME/<name> and
   // ./<name>. ROOT always reads ".rootrc" (in TROOT::InitSystem()). You can
   // read additional user defined resource files by creating addtional TEnv
   // object.

   if (strlen(name) == 0)
      fTable = 0;
   else {
      fTable  = new TOrdCollection(1000);
      fRcName = name;

      char sname[128] = "system";
      strcat(sname, name);
#ifdef ROOTETCDIR
      char *s = gSystem->ConcatFileName(ROOTETCDIR, sname);
#else
      char etc[1024];
#ifdef WIN32
      sprintf(etc, "%s\\etc", gRootDir);
#else
      sprintf(etc, "%s/etc", gRootDir);
#endif
      char *s = gSystem->ConcatFileName(etc, sname);
      if (gSystem->AccessPathName(s)) {
         // for backward compatibility check also $ROOTSYS/system<name> if
         // $ROOTSYS/etc/system<name> does not exist
         delete [] s;
         s = gSystem->ConcatFileName(gRootDir, sname);
         if (gSystem->AccessPathName(s)) {
            // for backward compatibility check also $ROOTSYS/<name> if
            // $ROOTSYS/system<name> does not exist
            delete [] s;
            s = gSystem->ConcatFileName(gRootDir, name);
         }
      }
#endif
      ReadFile(s, kEnvGlobal);
      delete [] s;
      s = gSystem->ConcatFileName(gSystem->HomeDirectory(), name);
      ReadFile(s, kEnvUser);
      delete [] s;
      ReadFile(name, kEnvLocal);
   }
}

//______________________________________________________________________________
TEnv::~TEnv()
{
   // Delete the resource table.

   if (fTable) {
      fTable->Delete();
      SafeDelete(fTable);
   }
}

//______________________________________________________________________________
const char *TEnv::Getvalue(const char *name)
{
   // Returns the character value for a named resouce.

   Bool_t haveProgName = kFALSE;
   if (gProgName && strlen(gProgName) > 0)
      haveProgName = kTRUE;

   TEnvRec *er = 0;
   if (haveProgName)
      er = Lookup(Form("%s.%s.%s", gSystem->GetName(), gProgName, name));
   if (er == 0)
      er = Lookup(Form("%s.%s.%s", gSystem->GetName(), gROOT->GetName(), name));
   if (er == 0)
      er = Lookup(Form("%s.*.%s", gSystem->GetName(), name));
   if (er == 0 && haveProgName)
      er = Lookup(Form("%s.%s", gProgName, name));
   if (er == 0)
      er = Lookup(Form("%s.%s", gROOT->GetName(), name));
   if (er == 0)
      er = Lookup(Form("*.*.%s", name));
   if (er == 0)
      er = Lookup(Form("*.%s", name));
   if (er == 0)
      er = Lookup(name);
   if (er == 0)
      return 0;
   return er->fValue.Data();
}

//______________________________________________________________________________
Int_t TEnv::GetValue(const char *name, Int_t dflt)
{
   // Returns the integer value for a resource. If the resource is not found
   // return the dflt value.

   const char *cp = TEnv::Getvalue(name);
   if (cp) {
      char buf2[512], *cp2 = buf2;

      while (isspace((int)*cp))
         cp++;
      if (*cp) {
         BoolNameTable_t *bt;
         if (isdigit((int)*cp) || *cp == '-' || *cp == '+')
            return atoi(cp);
         while (isalpha((int)*cp))
            *cp2++ = toupper((int)*cp++);
         *cp2 = 0;
         for (bt = boolNames; bt->fName; bt++)
            if (strcmp(buf2, bt->fName) == 0)
               return bt->fValue;
      }
   }
   return dflt;
}

//______________________________________________________________________________
Double_t TEnv::GetValue(const char *name, Double_t dflt)
{
   // Returns the dobule value for a resource. If the resource is not found
   // return the dflt value.

   const char *cp = TEnv::Getvalue(name);
   if (cp) {
      char *endptr;
      Double_t val = strtod(cp, &endptr);
      if (val == 0.0 && cp == endptr)
         return dflt;
      return val;
   }
   return dflt;
}

//______________________________________________________________________________
const char *TEnv::GetValue(const char *name, const char *dflt)
{
   // Returns the character value for a named resouce. If the resource is
   // not found the dflt value is returned.

   const char *cp = TEnv::Getvalue(name);
   if (cp)
      return cp;
   return dflt;
}

//______________________________________________________________________________
TEnvRec *TEnv::Lookup(const char *name)
{
   // Loop over all resource records and return the one with name.
   // Return 0 in case name is not in the resoucre table.

   TIter next(fTable);
   TEnvRec *er;

   while ((er = (TEnvRec*) next()))
      if (er->fName == name)
            return er;

   return 0;
}

//______________________________________________________________________________
void TEnv::Print(Option_t *opt) const
{
   // Print all resources or the global, user or local resources separately.

   if (strlen(opt) == 0) {
      PrintEnv();
      return;
   }

   if (!strcmp(opt, "global"))
      PrintEnv(kEnvGlobal);
   if (!strcmp(opt, "user"))
      PrintEnv(kEnvUser);
   if (!strcmp(opt, "local"))
      PrintEnv(kEnvLocal);
}

//______________________________________________________________________________
void TEnv::PrintEnv(EEnvLevel level) const
{
   // Print all resources for a certain level (global, user, local).

   TIter next(fTable);
   TEnvRec *er;
   static const char *lc[] = { "Global", "User", "Local" };

   while ((er = (TEnvRec*) next()))
      if (er->fLevel == level || level == kEnvAll)
         Printf("%-25s %-30s [%s]", Form("%s:", er->fName.Data()),
                er->fValue.Data(), lc[er->fLevel]);
}

//______________________________________________________________________________
void TEnv::ReadFile(const char *fname, EEnvLevel level)
{
   // Read and parse the resource file for a certain level.

   FILE *ifp;
   if ((ifp = fopen(fname, "r"))) {
      TReadEnvParser rp(this, ifp, level);
      rp.Parse();
      fclose(ifp);
   }
}

//______________________________________________________________________________
void TEnv::Save()
{
   // Write the resource files for each level. The new files have the same
   // name as the original files. The old files are renamed to *.bak.

   SaveLevel(kEnvGlobal);
   SaveLevel(kEnvUser);
   SaveLevel(kEnvLocal);
}

//______________________________________________________________________________
void TEnv::SaveLevel(EEnvLevel level)
{
   // Write the resource file for a certain level.

   TIter next(fTable);
   TEnvRec *er;

   while ((er = (TEnvRec*) next()))
      if (er->fLevel == level)
         break;

   if (!er) return;

   TString   rootrcdir;
   FILE     *ifp, *ofp;

   if (level == kEnvGlobal) {
#ifdef ROOTETCDIR
      char sname[128] = "system";
      strcat(sname, fRcName.Data());
      char *s = gSystem->ConcatFileName(ROOTETCDIR, sname);
#else
      char *s = gSystem->ConcatFileName(gRootDir, fRcName.Data());
#endif
      rootrcdir = s;
      delete [] s;
   } else if (level == kEnvUser) {
      char *s = gSystem->ConcatFileName(gSystem->HomeDirectory(), fRcName.Data());
      rootrcdir = s;
      delete [] s;
   } else if (level == kEnvLocal)
      rootrcdir = fRcName;
   else
      return;

   if ((ofp = fopen(Form("%s.new", rootrcdir.Data()), "w"))) {
      ifp = fopen(rootrcdir.Data(), "r");
      if (ifp == 0) {     // try to create file
         ifp = fopen(rootrcdir.Data(), "w");
         if (ifp) {
            fclose(ifp);
            ifp = 0;
         }
      }
      if (ifp || (ifp = fopen(rootrcdir.Data(), "r"))) {
         TWriteEnvParser wp(this, ifp, ofp);
         wp.Parse();

         next.Reset();
         while ((er = (TEnvRec*) next()))
            if (er->fModified) {
               er->fModified = kFALSE;
               fprintf(ofp, "%-40s %s\n", Form("%s:", er->fName.Data()),
                       er->fValue.Data());
            }
         fclose(ifp);
         fclose(ofp);
         gSystem->Rename(rootrcdir.Data(), Form("%s.bak", rootrcdir.Data()));
         gSystem->Rename(Form("%s.new", rootrcdir.Data()), rootrcdir.Data());
         return;
      }
      fclose(ofp);
   }
}

//______________________________________________________________________________
void TEnv::SetValue(const char *name, const char *value, EEnvLevel level,
                    const char *type)
{
   // Set the value of a resource or create a new resource.

   const char *nam = name;
   Bool_t append = kFALSE;
   if (name[0] == '+') {
      nam    = &name[1];
      append = kTRUE;
   }

   TEnvRec *er = Lookup(nam);
   if (er)
      er->ChangeValue(value, type, level, append);
   else
      fTable->Add(new TEnvRec(nam, value, type, level));
}

//______________________________________________________________________________
void TEnv::SetValue(const char *name, EEnvLevel level)
{
   // Set the value of a resource or create a new resource.
   // Use this method to set a resource like, "name=val".
   // If just "name" is given it will be interpreted as "name=1".

   TString buf = name;
   int l = buf.Index("=");
   if (l > 0) {
      TString nm  = buf(0, l);
      TString val = buf(l+1, buf.Length());
      SetValue(nm, val, level);
   } else
      SetValue(name, "1", level);
}

//______________________________________________________________________________
void TEnv::SetValue(const char *name, Int_t value)
{
   // Set or create an integer resource value.

   SetValue(name, Form("%d", value));
}

//______________________________________________________________________________
void TEnv::SetValue(const char *name, double value)
{
   // Set or create a double resource value.

   SetValue(name, Form("%f", value));
}
