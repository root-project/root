// @(#)root/base:$Name:  $:$Id: TGenericClassInfo.h,v 1.2 2002/11/01 19:12:09 brun Exp $
// Author: Philippe Canal   23/2/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGenericClassInfo
#define ROOT_TGenericClassInfo

namespace ROOT {

   class TGenericClassInfo {
      // This class in not inlined because it is used is non time critical
      // section (the dictionaries) and inline would lead to too much
      // repetition of the code (once per class!).

      const TInitBehavior  *fAction;
      TClass              *fClass;
      const char          *fClassName;
      const char          *fDeclFileName;
      Int_t                fDeclFileLine;
      VoidFuncPtr_t        fDictionary;
      const type_info     &fInfo;
      const char          *fImplFileName;
      Int_t                fImplFileLine;
      IsAFunc_t            fIsA;
      void                *fShowMembers;
      Int_t                fVersion;
      newFunc_t            fNew;
      newArrFunc_t         fNewArray;
      delFunc_t            fDelete;
      delArrFunc_t         fDeleteArray;
      desFunc_t            fDestructor;
      
   public:
      TGenericClassInfo(const char *fullClassname,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers, VoidFuncPtr_t dictionary,
                       IsAFunc_t isa, Int_t pragmabits);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers,  VoidFuncPtr_t dictionary,
                       IsAFunc_t isa, Int_t pragmabits);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       VoidFuncPtr_t dictionary, 
                       IsAFunc_t isa, Int_t pragmabits);

      void Init(Int_t pragmabits);
      ~TGenericClassInfo();

      const TInitBehavior &GetAction() const;
      TClass              *GetClass();
      const char          *GetClassName() const;
      const char          *GetDeclFileName() const;
      Int_t                GetDeclFileLine() const;
      delFunc_t            GetDelete() const;
      delArrFunc_t         GetDeleteArray() const;
      desFunc_t            GetDestructor() const;
      const char          *GetImplFileName();
      Int_t                GetImplFileLine();
      const type_info     &GetInfo() const;
      IsAFunc_t            GetIsA() const;
      newFunc_t            GetNew() const;
      newArrFunc_t         GetNewArray() const;
      void                *GetShowMembers() const;
      Int_t                GetVersion() const;

      TClass              *IsA(const void *obj);

      void                 SetDelete(delFunc_t deleteFunc);
      void                 SetDeleteArray(delArrFunc_t deleteArrayFunc);
      void                 SetDestructor(desFunc_t destructorFunc);
      void                 SetFromTemplate();
      Int_t                SetImplFile(const char *file, Int_t line);
      void                 SetNew(newFunc_t newFunc);
      void                 SetNewArray(newArrFunc_t newArrayFunc);
      Short_t              SetVersion(Short_t version);
      
   };

}

#endif
