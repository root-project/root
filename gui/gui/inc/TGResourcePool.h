// @(#)root/gui:$Id$
// Author: Fons Rademakers   19/5/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGResourcePool
#define ROOT_TGResourcePool


#include "TGObject.h"

class TGClient;
class TGFontPool;
class TGFont;
class TGGCPool;
class TGGC;
class TGPicturePool;
class TGPicture;
class TGMimeTypes;


class TGResourcePool : public TGObject {

private:
   Pixel_t          fBackColor;        ///< default background color
   Pixel_t          fForeColor;        ///< default foreground color
   Pixel_t          fHilite;           ///< default highlight color
   Pixel_t          fShadow;           ///< default shadow color
   Pixel_t          fHighLightColor;   ///< highlight color
   Pixel_t          fSelBackColor;     ///< default selection background color
   Pixel_t          fSelForeColor;     ///< default selection foreground color
   Pixel_t          fDocBackColor;     ///< default document background color
   Pixel_t          fDocForeColor;     ///< default document foreground color
   Pixel_t          fTipBackColor;     ///< default tip background color
   Pixel_t          fTipForeColor;     ///< default tip foreground color
   Pixel_t          fWhite;            ///< white color index
   Pixel_t          fBlack;            ///< black color index

   TGFontPool      *fFontPool;         ///< font pool manager

   TGFont          *fDefaultFont;      ///< default font
   TGFont          *fMenuFont;         ///< menu font
   TGFont          *fMenuHiFont;       ///< menu highlight font
   TGFont          *fDocFixedFont;     ///< document fixed font
   TGFont          *fDocPropFont;      ///< document proportional font
   TGFont          *fIconFont;         ///< icon font
   TGFont          *fStatusFont;       ///< status bar font

   TGPicturePool   *fPicturePool;        ///< picture pool manager

   const TGPicture *fDefaultBackPicture;    ///< default background picture
   const TGPicture *fDefaultDocBackPicture; ///< default document background picture

   TGGCPool        *fGCPool;           ///< graphics drawing context pool manager

   TGGC            *fWhiteGC;          ///< white gc
   TGGC            *fBlackGC;          ///< black gc
   TGGC            *fFrameGC;          ///< frame gc
   TGGC            *fBckgndGC;         ///< frame background gc
   TGGC            *fHiliteGC;         ///< frame hilite gc
   TGGC            *fShadowGC;         ///< frame shadow gc
   TGGC            *fFocusGC;          ///< frame focus gc
   TGGC            *fDocGC;            ///< document gc
   TGGC            *fDocbgndGC;        ///< document background gc
   TGGC            *fSelGC;            ///< selection gc
   TGGC            *fSelbgndGC;        ///< selection background gc
   TGGC            *fTipGC;            ///< tooltip gc

   Pixmap_t        fCheckered;         ///< checkered pixmap
   Pixmap_t        fCheckeredBitmap;   ///< checkered bitmap

   Cursor_t        fDefaultCursor;     ///< default cursor
   Cursor_t        fGrabCursor;        ///< grab cursor
   Cursor_t        fTextCursor;        ///< text cursor
   Cursor_t        fWaitCursor;        ///< wait cursor

   Colormap_t      fDefaultColormap;   ///< default colormap

   Atom_t          fClipboardAtom;     ///< handle to clipboard

   TGMimeTypes    *fMimeTypeList;      ///< list of mime types

public:
   TGResourcePool(TGClient *client);
   virtual ~TGResourcePool();

   TGGCPool       *GetGCPool() const { return fGCPool; }
   TGFontPool     *GetFontPool() const { return fFontPool; }
   TGPicturePool  *GetPicturePool() const { return fPicturePool; }

   //--- inline functions:

   // Color values...

   Pixel_t GetWhiteColor()        const { return fWhite; }
   Pixel_t GetBlackColor()        const { return fBlack; }

   Pixel_t GetFrameFgndColor()    const { return fForeColor; }
   Pixel_t GetFrameBgndColor()    const { return fBackColor; }
   Pixel_t GetFrameHiliteColor()  const { return fHilite; }
   Pixel_t GetFrameShadowColor()  const { return fShadow; }

   Pixel_t GetHighLightColor()    const { return fHighLightColor; }

   Pixel_t GetDocumentFgndColor() const { return fDocForeColor; }
   Pixel_t GetDocumentBgndColor() const { return fDocBackColor; }

   Pixel_t GetSelectedFgndColor() const { return fSelForeColor; }
   Pixel_t GetSelectedBgndColor() const { return fSelBackColor; }

   Pixel_t GetTipFgndColor()      const { return fTipForeColor; }
   Pixel_t GetTipBgndColor()      const { return fTipBackColor; }

   // Fonts...

   const TGFont *GetDefaultFont()       const { return fDefaultFont; }
   const TGFont *GetMenuFont()          const { return fMenuFont; }
   const TGFont *GetMenuHiliteFont()    const { return fMenuHiFont; }
   const TGFont *GetDocumentFixedFont() const { return fDocFixedFont; }
   const TGFont *GetDocumentPropFont()  const { return fDocPropFont; }
   const TGFont *GetIconFont()          const { return fIconFont; }
   const TGFont *GetStatusFont()        const { return fStatusFont; }

   // GCs...

   const TGGC *GetWhiteGC()          const { return fWhiteGC; }
   const TGGC *GetBlackGC()          const { return fBlackGC; }

   const TGGC *GetFrameGC()          const { return fFrameGC; }
   const TGGC *GetFrameBckgndGC()    const { return fBckgndGC; }
   const TGGC *GetFrameHiliteGC()    const { return fHiliteGC; }
   const TGGC *GetFrameShadowGC()    const { return fShadowGC; }
   const TGGC *GetFocusHiliteGC()    const { return fFocusGC; }

   const TGGC *GetDocumentGC()       const { return fDocGC; }
   const TGGC *GetDocumentBckgndGC() const { return fDocbgndGC; }

   const TGGC *GetSelectedGC()       const { return fSelGC; }
   const TGGC *GetSelectedBckgndGC() const { return fSelbgndGC; }

   const TGGC *GetTipGC()            const { return fTipGC; }

   // Pixmaps...

   Pixmap_t GetCheckeredPixmap() const { return fCheckered; }
   Pixmap_t GetCheckeredBitmap() const { return fCheckeredBitmap; }

   const TGPicture *GetFrameBckgndPicture() const
         { return fDefaultBackPicture; }
   const TGPicture *GetDocumentBckgndPicture() const
         { return fDefaultDocBackPicture; }

   // Cursors...

   Cursor_t GetDefaultCursor() const { return fDefaultCursor; }
   Cursor_t GetGrabCursor()    const { return fGrabCursor; }
   Cursor_t GetTextCursor()    const { return fTextCursor; }
   Cursor_t GetWaitCursor()    const { return fWaitCursor; }

   // Colormaps...

   Colormap_t GetDefaultColormap() const { return fDefaultColormap; }

   // Miscellaneous...

   TGMimeTypes *GetMimeTypes() const { return fMimeTypeList; }

   Atom_t       GetClipboard() const { return fClipboardAtom; }

   ClassDef(TGResourcePool,0)  // Graphics resource pool
};

#endif
