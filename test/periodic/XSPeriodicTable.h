/*
 * $Header$
 * $Log$
 */

#ifndef __XSPERIODIC_TABLE_H
#define __XSPERIODIC_TABLE_H

#include <TGLabel.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGToolTip.h>

/* =========== XSTblElement ============== */
class XSTblElement : public TGButton
{
friend class TGClient;

private:
   Int_t        Z;
   TGLabel     *lZ;
   TGToolTip   *tpZ;
   TGLabel     *lName;
   TGToolTip   *tpName;

public:
   XSTblElement( const TGWindow *p, Int_t z, UInt_t color);
   ~XSTblElement() override;
   void      Layout() override;
   TGDimension   GetDefaultSize() const override
      { return TGDimension(20,20); }

   void   SetState(EButtonState state, Bool_t emit = kFALSE) override;

   Int_t      GetZ()   const { return Z; }
   void   ChangeBackground( ULong_t color ) override;

   //ClassDefOverride(XSTblElement,1)
}; // XSTblElement

//////////////////////////////////////////////////////////////

#define   XSPTBL_ROWS   12
#define XSPTBL_COLS   20

/* ================== XSPeriodicTable ===================== */
class XSPeriodicTable : public TGCompositeFrame
{
private:
   Int_t      width, height;
   TGFrame      *elem[XSPTBL_ROWS][XSPTBL_COLS];

public:
   XSPeriodicTable(const TGWindow *msgWnd, const TGWindow* p,
         UInt_t w, UInt_t h);
        ~XSPeriodicTable() override;

   virtual   void      SelectZ( ULong_t Z );
   void      Layout() override;
     TGDimension   GetDefaultSize() const override
            { return TGDimension(width,height); }

   //ClassDefOverride(XSPeriodicTable,1)
}; // XSPeriodicTable

#endif
