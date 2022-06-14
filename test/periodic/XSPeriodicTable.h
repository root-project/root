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
   ~XSTblElement();
   virtual void      Layout();
   virtual TGDimension   GetDefaultSize() const
      { return TGDimension(20,20); }

   virtual void   SetState(EButtonState state, Bool_t emit = kFALSE);

   Int_t      GetZ()   const { return Z; }
   virtual void   ChangeBackground( ULong_t color );

   //ClassDef(XSTblElement,1)
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
   virtual      ~XSPeriodicTable();

   virtual   void      SelectZ( ULong_t Z );
   virtual void      Layout();
   virtual   TGDimension   GetDefaultSize() const
            { return TGDimension(width,height); }

   //ClassDef(XSPeriodicTable,1)
}; // XSPeriodicTable

#endif
