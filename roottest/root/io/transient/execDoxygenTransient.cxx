class MyClass {
public:
   Int_t fPx; ///< Some doxygen comment for persistent data.
   Int_t fPy; //!< Some doxygen comment for persistent data.
   Int_t fPz; /*!< Some doxygen comment for persistent data. */
   Int_t fPa; /**< Some doxygen comment for persistent data. */

   Int_t fCachePx; ///<! Some doxygen comment for transient data.
   Int_t fCachePy; //!<! Some doxygen comment for transient data.
   Int_t fCachePz; /*!<! Some doxygen comment for transient data. */
   Int_t fCachePa; /**<! Some doxygen comment for transient data. */
};

#include "TClass.h"
#include "TList.h"
#include "TDataMember.h"

bool CheckMember(TClass *c, const char *name, bool isPersistent)
{
   TDataMember *d = (TDataMember*)c->GetListOfDataMembers()->FindObject(name);
   if (!d) {
      printf("Error: can not find data member named %s\n",name);
      return false;
   }
   if (d->IsPersistent() != isPersistent) {
      printf("Error: %s is %s when it is expected to be %s\n",
             name,
             d->IsPersistent() ? "persistent" : "transient",
             isPersistent ? "persistent" : "transient");
      d->ls();
      return false;
   }
   //d->ls();
   return true;
}

int execDoxygenTransient() {
   TClass *c = TClass::GetClass("MyClass");
   if (!c) return 1;

   if (!CheckMember(c,"fPx",true)) return 2;
   if (!CheckMember(c,"fPy",true)) return 3;
   if (!CheckMember(c,"fPz",true)) return 4;
   if (!CheckMember(c,"fPa",true)) return 5;
   if (!CheckMember(c,"fCachePx",false)) return 10;
   if (!CheckMember(c,"fCachePy",false)) return 11;
   if (!CheckMember(c,"fCachePz",false)) return 12;
   if (!CheckMember(c,"fCachePa",false)) return 13;

   return 0;
}