// @(#)root/star:$Name:  $:$Id: TTableDescriptor.h,v 1.1.1.1 2000/11/27 22:57:13 fisyak Exp $
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

class TTableDescriptor : public TTable {
  protected:
     TClass  *fRowClass;  // TClass defining the table row C-structure
     virtual void Init(TClass *classPtr);

  public:

    TTableDescriptor(const TTable *parentTable);
    TTableDescriptor(TClass *classPtr);
   ~TTableDescriptor();
    TString CreateLeafList() const;
             void        LearnTable(const TTable *parentTable);
             void        LearnTable(TClass *classPtr);
             const Char_t *ColumnName(Int_t columnIndex)        const;
             const Int_t ColumnByName(const Char_t *columnName=0) const;
             UInt_t      NumberOfColumns()                      const;
             const UInt_t *IndexArray(Int_t columnIndex)        const;
             UInt_t      Offset(Int_t columnIndex)              const;
             Int_t       Offset(const Char_t *columnName=0)     const;
             UInt_t      ColumnSize(Int_t columnIndex)          const;
             Int_t       ColumnSize(const Char_t *columnName=0) const;
             UInt_t      TypeSize(Int_t columnIndex)            const;
             Int_t       TypeSize(const Char_t *columnName=0)   const;
             UInt_t      Dimensions(Int_t columnIndex)          const;
             Int_t       Dimensions(const Char_t *columnName=0) const;
             TTable::EColumnType ColumnType(Int_t columnIndex)          const;
             TTable::EColumnType ColumnType(const Char_t *columnName=0) const;
             TClass     *RowClass() const;
             void        SetOffset(UInt_t offset,Int_t column);
             void        SetSize(UInt_t size,Int_t column);
             void        SetTypeSize(UInt_t size,Int_t column);
             void        SetDimensions(UInt_t dim,Int_t column);
             void        SetColumnType(TTable::EColumnType type,Int_t column);
    virtual  Int_t       UpdateOffsets(const TTableDescriptor *newDesciptor);
    ClassDefTable(TTableDescriptor,tableDescriptor_st)
    ClassDef(TTableDescriptor,0)
};

//______________________________________________________________________________
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
