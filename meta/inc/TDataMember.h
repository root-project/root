// @(#)root/meta:$Name:  $:$Id: TDataMember.h,v 1.2 2000/09/08 16:07:33 rdm Exp $
// Author: Fons Rademakers   04/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataMember
#define ROOT_TDataMember


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataMember                                                          //
//                                                                      //
// Dictionary interface for a class data member.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif

class TList;
class TClass;
class TDataType;
class G__DataMemberInfo;
class TMethodCall;

class TDataMember : public TDictionary {

private:
   enum { kObjIsPersistent = BIT(2) };

   G__DataMemberInfo  *fInfo;        //pointer to CINT data member info
   TClass             *fClass;       //pointer to the class
   TDataType          *fDataType;    //pointer to data basic type descriptor

   // The following fields allows to access all (even private) datamembers and
   // provide a possibility of having options with names and strings.
   // These options are defined in a comment to a field!
   TMethodCall        *fValueGetter; //method that returns a value;
   TMethodCall        *fValueSetter; //method which sets value;
   TList              *fOptions;     //list of possible values 0=no restrictions

public:
   TDataMember(G__DataMemberInfo *info = 0, TClass *cl = 0);
   virtual       ~TDataMember();
   Int_t          GetArrayDim() const;
   Int_t          GetMaxIndex(Int_t dim) const;
   TClass        *GetClass() const { return fClass; }
   TDataType     *GetDataType() const { return fDataType; } //only for basic type
   const char    *GetName() const;
   Int_t          GetOffset() const;
   const char    *GetTitle() const;
   const char    *GetTypeName() const;
   const char    *GetFullTypeName() const;
   const char    *GetArrayIndex() const;

   TList         *GetOptions() const;
   TMethodCall   *SetterMethod();
   TMethodCall   *GetterMethod();

   Int_t          Compare(const TObject *obj) const;
   ULong_t        Hash() const;
   Bool_t         IsBasic() const;
   Bool_t         IsEnum() const;
   Bool_t         IsaPointer() const;
   Bool_t         IsPersistent() const { return TestBit(kObjIsPersistent); }
   Long_t         Property() const;

   ClassDef(TDataMember,0)  //Dictionary for a class data member
};


// This class implements one option in options list. All Data members are public
// for cenvenience reasons.

class TOptionListItem : public TObject {

public:
   TDataMember     *fDataMember;     //Data member to which this option belongs
   Long_t           fValue;          //Numerical value assigned to option
   Long_t           fValueMaskBit;   //Not used yet: bitmask used when option is a toggle group
   Long_t           fToggleMaskBit;  //Not used yet: bitmask used when toggling value
   char            *fOptName;        //Text assigned to option which appears in option menu
   char            *fOptLabel;       //Text (or enum) value assigned to option.

   TOptionListItem(TDataMember *m,Long_t val, Long_t valmask, Long_t tglmask,
                   const char *name, const char *label);
  ~TOptionListItem();
};

#endif
