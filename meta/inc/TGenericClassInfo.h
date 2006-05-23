// @(#)root/base:$Name:  $:$Id: TGenericClassInfo.h,v 1.8 2005/05/27 16:42:58 pcanal Exp $
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

// Forward declarations
class TVirtualIsAProxy;

namespace ROOT {

   class TGenericClassInfo {
      // This class in not inlined because it is used is non time critical
      // section (the dictionaries) and inline would lead to too much
      // repetition of the code (once per class!).

      const TInitBehavior     *fAction;
      TClass                  *fClass;
      const char              *fClassName;
      const char              *fDeclFileName;
      Int_t                    fDeclFileLine;
      VoidFuncPtr_t            fDictionary;
      const type_info         &fInfo;
      const char              *fImplFileName;
      Int_t                    fImplFileLine;
      TVirtualIsAProxy        *fIsA;
      void                    *fShowMembers;
      Int_t                    fVersion;
      NewFunc_t                fNew;
      NewArrFunc_t             fNewArray;
      DelFunc_t                fDelete;
      DelArrFunc_t             fDeleteArray;
      DesFunc_t                fDestructor;
      TClassStreamer          *fStreamer;
      TVirtualCollectionProxy *fCollectionProxy;
      Int_t                    fSizeof;
      
   public:
      TGenericClassInfo(const char *fullClassname,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers, VoidFuncPtr_t dictionary,
                       TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       void *showmembers,  VoidFuncPtr_t dictionary,
                       TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const TInitBehavior *action,
                       VoidFuncPtr_t dictionary, 
                       TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof);

      TGenericClassInfo(const char *fullClassname, Int_t version,
                        const char *declFileName, Int_t declFileLine,
                        const TInitBehavior *action,
                        VoidFuncPtr_t dictionary, Int_t pragmabits);

      void Init(Int_t pragmabits);
      ~TGenericClassInfo();

      const TInitBehavior &GetAction() const;
      TClass              *GetClass();
      const char          *GetClassName() const;
      const char          *GetDeclFileName() const;
      Int_t                GetDeclFileLine() const;
      DelFunc_t            GetDelete() const;
      DelArrFunc_t         GetDeleteArray() const;
      DesFunc_t            GetDestructor() const;
      const char          *GetImplFileName();
      Int_t                GetImplFileLine();
      const type_info     &GetInfo() const;
      TVirtualIsAProxy    *GetIsA() const;
      NewFunc_t            GetNew() const;
      NewArrFunc_t         GetNewArray() const;
      void                *GetShowMembers() const;
      Int_t                GetVersion() const;

      TClass              *IsA(const void *obj);

      Short_t              AdoptStreamer(TClassStreamer*);
      Short_t              AdoptCollectionProxy(TVirtualCollectionProxy*);
      void                 SetDelete(DelFunc_t deleteFunc);
      void                 SetDeleteArray(DelArrFunc_t deleteArrayFunc);
      void                 SetDestructor(DesFunc_t destructorFunc);
      void                 SetFromTemplate();
      Int_t                SetImplFile(const char *file, Int_t line);
      void                 SetNew(NewFunc_t newFunc);
      void                 SetNewArray(NewArrFunc_t newArrayFunc);
      Short_t              SetStreamer(ClassStreamerFunc_t);
      Short_t              SetVersion(Short_t version);

      //   protected:
   private:
      TGenericClassInfo(const TGenericClassInfo&); // Not implemented
      TGenericClassInfo& operator=(const TGenericClassInfo&); // Not implemented
      
   private:
      TGenericClassInfo(); 

   };

}

#endif
