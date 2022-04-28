// Author: Roel Aaij   21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTableHeader
#define ROOT_TGTableHeader

#include "TGTableCell.h"

enum EHeaderType {
   kColumnHeader,
   kRowHeader,
   kTableHeader
};

class TGWindow;

class TGTableHeader : public TGTableCell {

protected:
   EHeaderType fType;        ///< Type of header
   UInt_t      fWidth;       ///< Width for the column
   UInt_t      fHeight;      ///< Height of the row
   Bool_t      fReadOnly;    ///< Cell readonly state
   Bool_t      fEnabled;     ///< Cell enabled state
   Bool_t      fHasOwnLabel; ///< Flag on default or specific label usage

   void        Init();

public:
   TGTableHeader(const TGWindow *p = nullptr, TGTable *table = nullptr,
                 TGString *label = nullptr, UInt_t position = 0,
                 EHeaderType type = kColumnHeader, UInt_t width = 80,
                 UInt_t height = 25, GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   TGTableHeader(const TGWindow *p, TGTable *table, const char *label,
                 UInt_t position, EHeaderType type = kColumnHeader,
                 UInt_t width = 80, UInt_t height = 25,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   virtual ~TGTableHeader();

   void SetWidth(UInt_t width) override;
   void SetHeight(UInt_t height) override;

   void SetLabel(const char *label) override;

   virtual void SetDefaultLabel();
   virtual void SetPosition(UInt_t pos);
   void Resize(UInt_t width, UInt_t height) override;  // Resize width or height
   void Resize(TGDimension newsize) override;          // depending on type
   virtual void Sort(Bool_t order = kSortAscending);
   virtual void UpdatePosition();

   virtual EHeaderType GetType() { return fType; }

   ClassDefOverride(TGTableHeader, 0) // Header for use in TGTable.
};

#endif
