// @(#)root/meta:$Name$:$Id$
// Author: Rene Brun   09/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Each ROOT class (see TClass) has a linked list of methods.          //
//  This class describes one single method (member function).           //
//  The method info is obtained via the CINT api. See class TCint.      //
//                                                                      //
//  The method information is used a.o. by the THml class and by the    //
//  TTree class.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClass.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TInterpreter.h"
#include "Strlen.h"
#include "Api.h"
#include "TDataMember.h"


ClassImp(TMethod)

//______________________________________________________________________________
TMethod::TMethod(G__MethodInfo *info, TClass *cl) : TFunction(info)
{
   // Default TMethod ctor. TMethods are constructed in TClass.
   // Comment strings are pre-parsed to find out whether the method is
   // a contxt-menu item.

   fClass = cl;

   if (fInfo) {
      const char *t = fInfo->Title();

      if (t && strstr(t, "*TOGGLE"))
         fMenuItem = kMenuToggle;
      else
      if (t && strstr(t, "*MENU"))
         fMenuItem = kMenuDialog;
      else
      if (t && strstr(t, "*SUBMENU"))
         {
         fMenuItem = kMenuSubMenu;
         }
      else
         fMenuItem = kMenuNoMenu;
   }
}


//______________________________________________________________________________
const char *TMethod::GetCommentString()
{
    // Returns a comment string from the class declaration.

    return fInfo->Title();
}


//______________________________________________________________________________
void TMethod::CreateSignature()
{
   // Using the CINT method arg information create a complete signature string.

   TFunction::CreateSignature();

   if (Property() & kIsConstant) fSignature += " const";
}

//______________________________________________________________________________
TDataMember *TMethod::FindDataMember()
{
   // Tries to guess DataMember from comment string
   // and Method's name <==(only if 1 Argument!).
   // If more then one argument=> returns pointer to the last argument.
   // It also sets MethodArgs' pointers to point to specified data members.
   //
   // The form of comment string defining arguments is:
   // void XXX(Int_t x1, Float_t y2) //*ARGS={x1=>fX1,y2=>fY2}
   // where fX1, fY2 are data fields in the same class.
   // ("pointers" to data members)

   Char_t *argstring = (char*)strstr(GetCommentString(),"*ARGS={");

   // the following statement has been commented (Rene). Not needed
   // it was making troubles in BuildRealData for classes with protected
   // default constructors.
   // if (!(GetClass()->GetListOfRealData())) GetClass()->BuildRealData();

   if (argstring) {

      // if we found any argument-specifying hints  - parse it

      if (!fMethodArgs) return 0;

      char argstr[2048];    // workspace...
      char *ptr1 = 0;
      char *tok  = 0;
      char *ptr2 = 0;
      Int_t i;

      strcpy(argstr,argstring);       //let's move it to "worksapce"  copy

      ptr2 = strtok(argstr,"{}");     //extract the data!
      ptr2 = strtok((char*)0,"{}");

      //extract argument tokens//
      char *tokens[20];
      Int_t cnt       = 0;
      Int_t token_cnt = 0;
      do {
          ptr1 = strtok((char*) (cnt++ ? 0:ptr2),",;"); //extract tokens
                                                         // separated by , or ;
          if (ptr1) {
             tok = new char[strlen(ptr1)+1];
             strcpy(tok,ptr1);
             tokens[token_cnt] = tok;            //store this token.
             token_cnt++;
          }
      } while (ptr1);

      //now let's  parse all argument tokens...
      TClass     *cl = 0;
      TMethodArg *a  = 0;
      TMethodArg *ar = 0;
      TDataMember *member = 0;

      for (i=0; i<token_cnt;i++) {
         cnt = 0;
         ptr1 = strtok(tokens[i],"=>");  //LeftHandedSide=methodarg
         ptr2 = strtok((char*)0,"=>"); //RightHandedSide-points to datamember

         //find the MethodArg
         a      = 0;
         ar     = 0;
         member = 0;
         TIter nextarg(fMethodArgs);     // iterate through all arguments.
         while ((ar = (TMethodArg*)nextarg())) {
            if (!strcmp(ptr1,ar->GetName())) {
                a = ar;
               break;
            }
         }

         //now find the data member
         cl = GetClass()->GetBaseDataMember(ptr2);
         if (cl) {
            member = cl->GetDataMember(ptr2);
            if (a) a->fDataMember = member; //SET THE APROPRIATE FIELD !!!
                                     //We can do it - friend decl. in MethodArg
         }
         delete tokens[i];
      }
      return member; // nothing else to do! We return a pointer to the last
                     // found data member

   // if not found in comment string - try to guess it from name!
   } else {
      if (fMethodArgs)
        if (fMethodArgs->GetSize() != 1) return 0;

      TMethodArg *a = 0;
      if (fMethodArgs) a = (TMethodArg*)(fMethodArgs->First());

      char dataname[64]    = "";
      char basename[64]    = "";
      const char *funcname = GetName();
      if ( strncmp(funcname,"Get",3) == 0 || strncmp(funcname,"Set",3) == 0 )
         sprintf(basename,"%s",funcname+3);
      else if ( strncmp(funcname,"Is",2) == 0 )
         sprintf(basename,"%s",funcname+2);
      else
         return 0;

      sprintf(dataname,"f%s",basename);

      TClass *cl = GetClass()->GetBaseDataMember(dataname);
      if (cl) {
         TDataMember *member   = cl->GetDataMember(dataname);
         if (a) a->fDataMember = member;
         return member;
      } else {
         sprintf(dataname,"fIs%s",basename);  //in case of IsEditable()
                                                        //and fIsEditable
         cl = GetClass()->GetBaseDataMember(dataname);
         if (cl) {
           TDataMember *member = cl->GetDataMember(dataname);
           if (a) a->fDataMember = member;
           return member;
         }
      }
   }

    //if nothing found - return null -pointer:
    return 0;
}


//______________________________________________________________________________
TList *TMethod::GetListOfMethodArgs()
{
   // Returns methodarg list and additionally updates fDataMember in TMethod by
   // calling FindDataMember();

   if (!fMethodArgs){
      TFunction::GetListOfMethodArgs();
      FindDataMember();
   }
   return fMethodArgs;
}
