// @(#)root/table:$Id$
// Author: Valery Fine   09/08/99  (E-mail: fine@bnl.gov)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTableDescriptor
#define ROOT_TTableDescriptor

#include "TTable.h"
#include "tableDescriptor.h"

class TClass;
//______________________________________________________________________________
//
// TTableDescriptor - run-time descriptor of the TTable object rows.
//______________________________________________________________________________


class TTableDescriptor : public TTable {
private:
   TTableDescriptor &operator=(const TTableDescriptor &dsc); // Intentionally not implemented.
protected:
   friend class TTable;
   TClass  *fRowClass;                  // TClass defining 
                                          // the table row C-structure
   TTableDescriptor *fSecondDescriptor; // shadow descriptor 
                                          // to back TTable::Streamer
   static TString fgCommentsName;        // The name of dataset to keep the comments fields
   virtual void Init(TClass *classPtr);
   static void SetCommentsSetName(const char *name=".comments");

public:

   TTableDescriptor(const TTable *parentTable);
   TTableDescriptor(TClass *classPtr);
   TTableDescriptor(const TTableDescriptor &dsc):TTable(dsc),fRowClass(dsc.fRowClass),fSecondDescriptor(0){}
   virtual ~TTableDescriptor();
   virtual  Int_t  AddAt(const void *c);
   virtual  void   AddAt(const void *c, Int_t i);
   virtual  void   AddAt(const tableDescriptor_st &element, const char *comment,Int_t indx);
   virtual  void   AddAt(TDataSet *dataset,Int_t idx=0);
   TString         CreateLeafList() const;
   void            LearnTable(const TTable *parentTable);
   void            LearnTable(TClass *classPtr);
   const Char_t   *ColumnName(Int_t columnIndex)              const;
   Int_t           ColumnByName(const Char_t *columnName=0)   const;
   UInt_t          NumberOfColumns()                          const;
   const UInt_t   *IndexArray(Int_t columnIndex)              const;
   UInt_t          Offset(Int_t columnIndex)                  const;
   Int_t           Offset(const Char_t *columnName=0)         const;
   UInt_t          ColumnSize(Int_t columnIndex)              const;
   Int_t           ColumnSize(const Char_t *columnName=0)     const;
   UInt_t          TypeSize(Int_t columnIndex)                const;
   Int_t           TypeSize(const Char_t *columnName=0)       const;
   UInt_t          Dimensions(Int_t columnIndex)              const;
   Int_t           Dimensions(const Char_t *columnName=0)     const;
   TTable::EColumnType ColumnType(Int_t columnIndex)          const;
   TTable::EColumnType ColumnType(const Char_t *columnName=0) const;
   TClass         *RowClass() const;
   void            SetOffset(UInt_t offset,Int_t column);
   void            SetSize(UInt_t size,Int_t column);
   void            SetTypeSize(UInt_t size,Int_t column);
   void            SetDimensions(UInt_t dim,Int_t column);
   Int_t           Sizeof() const;
   void            SetColumnType(TTable::EColumnType type,Int_t column);
   virtual Int_t   UpdateOffsets(const TTableDescriptor *newDesciptor);

   static          TTableDescriptor *MakeDescriptor(const char *structName);
   TDataSet       *MakeCommentField(Bool_t createFlag=kTRUE);

//    ClassDefTable(TTableDescriptor,tableDescriptor_st)
protected:                                        
   static  TTableDescriptor *fgColDescriptors;     
   virtual TTableDescriptor *GetDescriptorPointer() const;
   virtual void SetDescriptorPointer(TTableDescriptor *list);
public:                                           
   typedef tableDescriptor_st* iterator;                   
   TTableDescriptor() : TTable("TTableDescriptor",sizeof(tableDescriptor_st)), fRowClass(0), fSecondDescriptor(0) {SetType("tableDescriptor_st");}      
   TTableDescriptor(const char *name) : TTable(name,sizeof(tableDescriptor_st)), fRowClass(0), fSecondDescriptor(0) {SetType("tableDescriptor_st");}     
   TTableDescriptor(Int_t n) : TTable("TTableDescriptor",n,sizeof(tableDescriptor_st)), fRowClass(0), fSecondDescriptor(0) {SetType("tableDescriptor_st");}
   TTableDescriptor(const char *name,Int_t n) : TTable(name,n,sizeof(tableDescriptor_st)), fRowClass(0), fSecondDescriptor(0) {SetType("tableDescriptor_st");}
   tableDescriptor_st *GetTable(Int_t i=0) const { return ((tableDescriptor_st *)GetArray())+i;}                       
   tableDescriptor_st &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }             
   const tableDescriptor_st &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const tableDescriptor_st *)(GetTable(i))); } 
   tableDescriptor_st *begin() const  {                      return GetNRows()? GetTable(0):0;}
   tableDescriptor_st *end()   const  {Long_t i = GetNRows(); return          i? GetTable(i):0;}
   static const char *TableDictionary();
   ClassDef(TTableDescriptor,0) // descrpitor defining the internal layout of TTable objects
};

//______________________________________________________________________________
// inline  TTableDescriptor(const TTableDescriptor &dsc) : TTable(dsc), fRowClass(dsc.fRowClass),fSecondDescriptor(0){}
inline  const Char_t *TTableDescriptor::ColumnName(Int_t column)const {return ((tableDescriptor_st *)At(column))->fColumnName;}
inline  UInt_t  TTableDescriptor::Offset(Int_t column)          const {return ((tableDescriptor_st *)At(column))->fOffset;    }
inline  const UInt_t *TTableDescriptor::IndexArray(Int_t column)const {return ((tableDescriptor_st *)At(column))->fIndexArray;}
inline  UInt_t  TTableDescriptor::NumberOfColumns()             const {return  GetNRows();                                      }
inline  UInt_t  TTableDescriptor::ColumnSize(Int_t column)      const {return ((tableDescriptor_st *)At(column))->fSize;      }
inline  UInt_t  TTableDescriptor::TypeSize(Int_t column)        const {return ((tableDescriptor_st *)At(column))->fTypeSize;  }
inline  UInt_t  TTableDescriptor::Dimensions(Int_t column)      const {return ((tableDescriptor_st *)At(column))->fDimensions;}
inline  TTable::EColumnType TTableDescriptor::ColumnType(Int_t column) const {return EColumnType(((tableDescriptor_st *)At(column))->fType);}
inline  TClass *TTableDescriptor::RowClass() const { return fRowClass;}
inline  void    TTableDescriptor::SetOffset(UInt_t offset,Int_t column)  {((tableDescriptor_st *)At(column))->fOffset     = offset;}
inline  void    TTableDescriptor::SetSize(UInt_t size,Int_t column)      {((tableDescriptor_st *)At(column))->fSize       = size;  }
inline  void    TTableDescriptor::SetTypeSize(UInt_t size,Int_t column)  {((tableDescriptor_st *)At(column))->fTypeSize   = size;  }
inline  void    TTableDescriptor::SetDimensions(UInt_t dim,Int_t column) {((tableDescriptor_st *)At(column))->fDimensions = dim;   }
inline  void    TTableDescriptor::SetColumnType(TTable::EColumnType type,Int_t column) {((tableDescriptor_st *)At(column))->fType = type;  }

#endif
