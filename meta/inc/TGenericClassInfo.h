// @(#)root/base:$Name:  $:$Id: RtypesImp.h,v 1.6 2002/05/09 22:56:28 rdm Exp $
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
                       VoidFuncPtr_t dictionary, Int_t pragmabits);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers, VoidFuncPtr_t dictionary, Int_t pragmabits);

      void Init(Int_t pragmabits);
      ~TGenericClassInfo();

      const TInitBehavior &GetAction() const;
      TClass *GetClass();
      const char *GetClassName() const;
      const type_info &GetInfo() const;
      void *GetShowMembers() const;
      Short_t SetVersion(Short_t version);
      void SetFromTemplate();
      Int_t SetImplFile(const char *file, Int_t line);
      const char *GetDeclFileName() const;
      Int_t GetDeclFileLine() const;
      const char *GetImplFileName();
      Int_t GetImplFileLine();
      Int_t GetVersion() const;
      TClass *IsA(const void *obj);
      IsAFunc_t GetIsA() const;
   };

}

#endif
