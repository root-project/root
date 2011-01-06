// $Id: TGHtml.h,v 1.1 2007/05/04 17:07:01 brun Exp $
// Author:  Valeriy Onuchin   03/05/2007

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    HTML widget for xclass. Based on tkhtml 1.28
    Copyright (C) 1997-2000 D. Richard Hipp <drh@acm.org>
    Copyright (C) 2002-2003 Hector Peraza.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

#ifndef ROOT_TGHtml
#define ROOT_TGHtml

#ifndef ROOT_TGView
#include "TGView.h"
#endif

#ifndef ROOT_TGHtmlTokens
#include "TGHtmlTokens.h"
#endif

class TGClient;
class TImage;
class TGFont;
class TGIdleHandler;
class THashTable;
class TTimer;

//----------------------------------------------------------------------

#define HTML_RELIEF_FLAT    0
#define HTML_RELIEF_SUNKEN  1
#define HTML_RELIEF_RAISED  2

//#define TABLE_TRIM_BLANK 1

// Debug must be turned on for testing to work.
//#define DEBUG

#define CANT_HAPPEN  \
  fprintf(stderr, \
          "Unplanned behavior in the HTML Widget in file %s line %d\n", \
          __FILE__, __LINE__)

#define UNTESTED  \
  fprintf(stderr, \
          "Untested code executed in the HTML Widget in file %s line %d\n", \
          __FILE__, __LINE__)

// Sanity checking macros.

#ifdef DEBUG
#define HtmlAssert(X) \
  if(!(X)){ \
    fprintf(stderr,"Assertion failed on line %d of %s\n",__LINE__,__FILE__); \
  }
#define HtmlCantHappen \
  fprintf(stderr,"Can't happen on line %d of %s\n",__LINE__,__FILE__);
#else
#define HtmlAssert(X)
#define HtmlCantHappen
#endif

// Bitmasks for the HtmlTraceMask global variable

#define HtmlTrace_Table1       0x00000001
#define HtmlTrace_Table2       0x00000002
#define HtmlTrace_Table3       0x00000004
#define HtmlTrace_Table4       0x00000008
#define HtmlTrace_Table5       0x00000010
#define HtmlTrace_Table6       0x00000020
#define HtmlTrace_GetLine      0x00000100
#define HtmlTrace_GetLine2     0x00000200
#define HtmlTrace_FixLine      0x00000400
#define HtmlTrace_BreakMarkup  0x00001000
#define HtmlTrace_Style        0x00002000
#define HtmlTrace_Input1       0x00004000

// The TRACE macro is used to print internal information about the
// HTML layout engine during testing and debugging. The amount of
// information printed is governed by a global variable named
// HtmlTraceMask. If bits in the first argument to the TRACE macro
// match any bits in HtmlTraceMask variable, then the trace message
// is printed.
//
// All of this is completely disabled, of course, if the DEBUG macro
// is not defined.

#ifdef DEBUG
extern int HtmlTraceMask;
extern int HtmlDepth;
# define TRACE_INDENT  printf("%*s",HtmlDepth-3,"")
# define TRACE(Flag, Args) \
    if( (Flag)&HtmlTraceMask ){ \
       TRACE_INDENT; printf Args; fflush(stdout); \
    }
# define TRACE_PUSH(Flag)  if( (Flag)&HtmlTraceMask ){ HtmlDepth+=3; }
# define TRACE_POP(Flag)   if( (Flag)&HtmlTraceMask ){ HtmlDepth-=3; }
#else
# define TRACE_INDENT
# define TRACE(Flag, Args)
# define TRACE_PUSH(Flag)
# define TRACE_POP(Flag)
#endif


//----------------------------------------------------------------------

// Various data types. This code is designed to run on a modern cached
// architecture where the CPU runs a lot faster than the memory bus. Hence
// we try to pack as much data into as small a space as possible so that it
// is more likely to fit in cache. The extra CPU instruction or two needed
// to unpack the data is not normally an issue since we expect the speed of
// the memory bus to be the limiting factor.

typedef unsigned char  Html_u8_t;      // 8-bit unsigned integer
typedef short          Html_16_t;      // 16-bit signed integer
typedef unsigned short Html_u16_t;     // 16-bit unsigned integer
typedef int            Html_32_t;      // 32-bit signed integer

// An instance of the following structure is used to record style
// information on each Html element.

struct SHtmlStyle_t {
  unsigned int fFont      : 6;      // Font to use for display
  unsigned int fColor     : 6;      // Foreground color
  signed int   fSubscript : 4;      // Positive for <sup>, negative for <sub>
  unsigned int fAlign     : 2;      // Horizontal alignment
  unsigned int fBgcolor   : 6;      // Background color
  unsigned int fExpbg     : 1;      // Set to 1 if bgcolor explicitely set
  unsigned int fFlags     : 7;      // the STY_ flags below
};


// We allow 8 different font families: Normal, Bold, Italic and Bold-Italic
// in either variable or constant width. Within each family there can be up
// to 7 font sizes from 1 (the smallest) up to 7 (the largest). Hence, the
// widget can use a maximum of 56 fonts. The ".font" field of the style is
// an integer between 0 and 55 which indicates which font to use.

// HP: we further subdivide the .font field into two 3-bit subfields (size
// and family). That makes easier to manipulate the family field.

#define N_FONT_FAMILY     8
#define N_FONT_SIZE       7
#define N_FONT            71
#define NormalFont(X)     (X)
#define BoldFont(X)       ((X) | 8)
#define ItalicFont(X)     ((X) | 16)
#define CWFont(X)         ((X) | 32)
#define FontSize(X)       ((X) & 007)
#define FontFamily(X)     ((X) & 070)
#define FONT_Any          -1
#define FONT_Default      3
#define FontSwitch(Size, Bold, Italic, Cw) \
                          ((Size) | ((Bold+(Italic)*2+(Cw)*4) << 3))

// Macros for manipulating the fontValid bitmap of an TGHtml object.

#define FontIsValid(I)     ((fFontValid[(I)>>3] &   (1<<((I)&3)))!=0)
#define FontSetValid(I)     (fFontValid[(I)>>3] |=  (1<<((I)&3)))
#define FontClearValid(I)   (fFontValid[(I)>>3] &= ~(1<<((I)&3)))


// Information about available colors.
//
// The widget will use at most N_COLOR colors. 4 of these colors are
// predefined. The rest are user selectable by options to various markups.
// (Ex: <font color=red>)
//
// All colors are stored in the apColor[] array of the main widget object.
// The ".color" field of the SHtmlStyle_t is an integer between 0 and
// N_COLOR-1 which indicates which of these colors to use.

#define N_COLOR             32      // Total number of colors

#define COLOR_Normal         0      // Index for normal color (black)
#define COLOR_Unvisited      1      // Index for unvisited hyperlinks
#define COLOR_Visited        2      // Color for visited hyperlinks
#define COLOR_Selection      3      // Background color for the selection
#define COLOR_Background     4      // Default background color
#define N_PREDEFINED_COLOR   5      // Number of predefined colors


// The "align" field of the style determines how text is justified
// horizontally. ALIGN_None means that the alignment is not specified.
// (It should probably default to ALIGN_Left in this case.)

#define ALIGN_Left   1
#define ALIGN_Right  2
#define ALIGN_Center 3
#define ALIGN_None   0


// Possible value of the "flags" field of SHtmlStyle_t are shown below.
//
//  STY_Preformatted       If set, the current text occurred within
//                         <pre>..</pre>
//
//  STY_StrikeThru         Draw a solid line thru the middle of this text.
//
//  STY_Underline          This text should drawn with an underline.
//
//  STY_NoBreak            This text occurs within <nobr>..</nobr>
//
//  STY_Anchor             This text occurs within <a href=X>..</a>.
//
//  STY_DT                 This text occurs within <dt>..</dt>.
//
//  STY_Invisible          This text should not appear in the main HTML
//                         window. (For example, it might be within
//                         <title>..</title> or <marquee>..</marquee>.)

#define STY_Preformatted    0x001
#define STY_StrikeThru      0x002
#define STY_Underline       0x004
#define STY_NoBreak         0x008
#define STY_Anchor          0x010
#define STY_DT              0x020
#define STY_Invisible       0x040
#define STY_FontMask        (STY_StrikeThru|STY_Underline)


//----------------------------------------------------------------------
// The first thing done with input HTML text is to parse it into
// TGHtmlElements. All sizing and layout is done using these elements.

// Every element contains at least this much information:

class TGHtmlElement : public TObject {
public:
   TGHtmlElement(int etype = 0);

   virtual int  IsMarkup() const { return (fType > Html_Block); }
   virtual const char *MarkupArg(const char * /*tag*/, const char * /*zDefault*/) { return 0; }
   virtual int  GetAlignment(int dflt) { return dflt; }
   virtual int  GetOrderedListType(int dflt) { return dflt; }
   virtual int  GetUnorderedListType(int dflt) { return dflt; }
   virtual int  GetVerticalAlignment(int dflt) { return dflt; }

public:
   TGHtmlElement *fPNext;        // Next input token in a list of them all
   TGHtmlElement *fPPrev;        // Previous token in a list of them all
   SHtmlStyle_t   fStyle;        // The rendering style for this token
   Html_u8_t      fType;         // The token type.
   Html_u8_t      fFlags;        // The HTML_ flags below
   Html_16_t      fCount;        // Various uses, depending on "type"
   int            fElId;         // Unique identifier
   int            fOffs;         // Offset within zText
};


// Bitmasks for the "flags" field of the TGHtmlElement

#define HTML_Visible   0x01   // This element produces "ink"
#define HTML_NewLine   0x02   // type == Html_Space and ends with newline
#define HTML_Selected  0x04   // Some or all of this Html_Block is selected
                              // Used by Html_Block elements only.


// Each text element holds additional information as shown here. Notice that
// extra space is allocated so that zText[] will be large enough to hold the
// complete text of the element. X and y coordinates are relative to the
// virtual canvas. The y coordinate refers to the baseline.

class TGHtmlTextElement : public TGHtmlElement {
public:
   TGHtmlTextElement(int size);
   virtual ~TGHtmlTextElement();

   Html_32_t    fY;                // y coordinate where text should be rendered
   Html_16_t    fX;                // x coordinate where text should be rendered
   Html_16_t    fW;                // width of this token in pixels
   Html_u8_t    fAscent;           // height above the baseline
   Html_u8_t    fDescent;          // depth below the baseline
   Html_u8_t    fSpaceWidth;       // Width of one space in the current font
   char        *fZText;            // Text for this element. Null terminated
};


// Each space element is represented like this:

class TGHtmlSpaceElement : public TGHtmlElement {
public:
   Html_16_t fW;                  // Width of a single space in current font
   Html_u8_t fAscent;             // height above the baseline
   Html_u8_t fDescent;            // depth below the baseline

public:
   TGHtmlSpaceElement() : TGHtmlElement(Html_Space), fW(0), fAscent(0), fDescent(0) {}
};

// Most markup uses this class. Some markup extends this class with
// additional information, but most use it as is, at the very least.
//
// If the markup doesn't have arguments (the "count" field of
// TGHtmlElement is 0) then the extra "argv" field of this class
// is not allocated and should not be used.

class TGHtmlMarkupElement : public TGHtmlElement {
public:
   TGHtmlMarkupElement(int type, int argc, int arglen[], char *argv[]);
   virtual ~TGHtmlMarkupElement();

   virtual const char *MarkupArg(const char *tag, const char *zDefault);
   virtual int  GetAlignment(int dflt);
   virtual int  GetOrderedListType(int dflt);
   virtual int  GetUnorderedListType(int dflt);
   virtual int  GetVerticalAlignment(int dflt);

public://protected:
   char **fArgv;
};


// The maximum number of columns allowed in a table. Any columns beyond
// this number are ignored.

#define HTML_MAX_COLUMNS 40


// This class is used for each <table> element.
//
// In the minW[] and maxW[] arrays, the [0] element is the overall
// minimum and maximum width, including cell padding, spacing and
// the "hspace". All other elements are the minimum and maximum
// width for the contents of individual cells without any spacing or
// padding.

class TGHtmlTable : public TGHtmlMarkupElement {
public:
   TGHtmlTable(int type, int argc, int arglen[], char *argv[]);
   ~TGHtmlTable();

public:
   Html_u8_t      fBorderWidth;              // Width of the border
   Html_u8_t      fNCol;                     // Number of columns
   Html_u16_t     fNRow;                     // Number of rows
   Html_32_t      fY;                        // top edge of table border
   Html_32_t      fH;                        // height of the table border
   Html_16_t      fX;                        // left edge of table border
   Html_16_t      fW;                        // width of the table border
   int            fMinW[HTML_MAX_COLUMNS+1]; // minimum width of each column
   int            fMaxW[HTML_MAX_COLUMNS+1]; // maximum width of each column
   TGHtmlElement *fPEnd;                     // Pointer to the end tag element
   TImage        *fBgImage;                  // A background for the entire table
   int            fHasbg;                    // 1 if a table above has bgImage
};


// Each <td> or <th> markup is represented by an instance of the
// following class.
//
// Drawing for a cell is a sunken 3D border with the border width given
// by the borderWidth field in the associated <table> object.

class TGHtmlCell : public TGHtmlMarkupElement {
public:
   TGHtmlCell(int type, int argc, int arglen[], char *argv[]);
   ~TGHtmlCell();

public:
   Html_16_t      fRowspan;      // Number of rows spanned by this cell
   Html_16_t      fColspan;      // Number of columns spanned by this cell
   Html_16_t      fX;            // X coordinate of left edge of border
   Html_16_t      fW;            // Width of the border
   Html_32_t      fY;            // Y coordinate of top of border indentation
   Html_32_t      fH;            // Height of the border
   TGHtmlTable   *fPTable;       // Pointer back to the <table>
   TGHtmlElement *fPRow;         // Pointer back to the <tr>
   TGHtmlElement *fPEnd;         // Element that ends this cell
   TImage        *fBgImage;      // Background for the cell
};


// This class is used for </table>, </td>, <tr>, </tr> and </th> elements.
// It points back to the <table> element that began the table. It is also
// used by </a> to point back to the original <a>. I'll probably think of
// other uses before all is said and done...

class TGHtmlRef : public TGHtmlMarkupElement {
public:
   TGHtmlRef(int type, int argc, int arglen[], char *argv[]);
   ~TGHtmlRef();

public:
   TGHtmlElement *fPOther;      // Pointer to some other Html element
   TImage        *fBgImage;     // A background for the entire row
};


// An instance of the following class is used to represent
// each <LI> markup.

class TGHtmlLi : public TGHtmlMarkupElement {
public:
   TGHtmlLi(int type, int argc, int arglen[], char *argv[]);

public:
   Html_u8_t fLtype;    // What type of list is this?
   Html_u8_t fAscent;   // height above the baseline
   Html_u8_t fDescent;  // depth below the baseline
   Html_16_t fCnt;      // Value for this element (if inside <OL>)
   Html_16_t fX;        // X coordinate of the bullet
   Html_32_t fY;        // Y coordinate of the bullet
};


// The ltype field of an TGHtmlLi or TGHtmlListStart object can take on
// any of the following values to indicate what type of bullet to draw.
// The value in TGHtmlLi will take precedence over the value in
// TGHtmlListStart if the two values differ.

#define LI_TYPE_Undefined 0     // If in TGHtmlLi, use the TGHtmlListStart value
#define LI_TYPE_Bullet1   1     // A solid circle
#define LI_TYPE_Bullet2   2     // A hollow circle
#define LI_TYPE_Bullet3   3     // A hollow square
#define LI_TYPE_Enum_1    4     // Arabic numbers
#define LI_TYPE_Enum_A    5     // A, B, C, ...
#define LI_TYPE_Enum_a    6     // a, b, c, ...
#define LI_TYPE_Enum_I    7     // Capitalized roman numerals
#define LI_TYPE_Enum_i    8     // Lower-case roman numerals


// An instance of this class is used for <UL> or <OL> markup.

class TGHtmlListStart : public TGHtmlMarkupElement {
public:
   TGHtmlListStart(int type, int argc, int arglen[], char *argv[]);

public:
   Html_u8_t         fLtype;     // One of the LI_TYPE_ defines above
   Html_u8_t         fCompact;   // True if the COMPACT flag is present
   Html_u16_t        fCnt;       // Next value for <OL>
   Html_u16_t        fWidth;     // How much space to allow for indentation
   TGHtmlListStart  *fLPrev;     // Next higher level list, or NULL
};


#define HTML_MAP_RECT    1
#define HTML_MAP_CIRCLE  2
#define HTML_MAP_POLY    3

class TGHtmlMapArea : public TGHtmlMarkupElement {
public:
   TGHtmlMapArea(int type, int argc, int arglen[], char *argv[]);

public:
   int    fMType;
   int   *fCoords;
   int    fNum;
};


//----------------------------------------------------------------------

// Structure to chain extension data onto.

struct SHtmlExtensions_t {
   void              *fExts;
   int                fTyp;
   int                fFlags;
   SHtmlExtensions_t *fNext;
};


//----------------------------------------------------------------------

// Information about each image on the HTML widget is held in an instance
// of the following class. All images are held on a list attached to the
// main widget object.
//
// This class is NOT an element. The <IMG> element is represented by an
// TGHtmlImageMarkup object below. There is one TGHtmlImageMarkup for each
// <IMG> in the source HTML. There is one of these objects for each unique
// image loaded. (If two <IMG> specify the same image, there are still two
// TGHtmlImageMarkup objects but only one TGHtmlImage object that is shared
// between them.)

class TGHtml;
class TGHtmlImageMarkup;

class TGHtmlImage : public TObject {
public:
   TGHtmlImage(TGHtml *htm, const char *url, const char *width,
               const char *height);
   virtual ~TGHtmlImage();

public:
   TGHtml            *fHtml;              // The owner of this image
   TImage            *fImage;             // The image token
   Html_32_t          fW;                 // Requested width of this image (0 if none)
   Html_32_t          fH;                 // Requested height of this image (0 if none)
   char              *fZUrl;              // The URL for this image.
   char              *fZWidth, *fZHeight; // Width and height in the <img> markup.
   TGHtmlImage       *fPNext;             // Next image on the list
   TGHtmlImageMarkup *fPList;             // List of all <IMG> markups that use this
                                          // same image
   TTimer            *fTimer;             // for animations
};

// Each <img> markup is represented by an instance of the following
// class.
//
// If pImage == 0, then we use the alternative text in zAlt.

class TGHtmlImageMarkup : public TGHtmlMarkupElement {
public:
   TGHtmlImageMarkup(int type, int argc, int arglen[], char *argv[]);

public:
   Html_u8_t          fAlign;          // Alignment. See IMAGE_ALIGN_ defines below
   Html_u8_t          fTextAscent;     // Ascent of text font in force at the <IMG>
   Html_u8_t          fTextDescent;    // Descent of text font in force at the <IMG>
   Html_u8_t          fRedrawNeeded;   // Need to redraw this image because the image
                                       // content changed.
   Html_16_t          fH;              // Actual height of the image
   Html_16_t          fW;              // Actual width of the image
   Html_16_t          fAscent;         // How far image extends above "y"
   Html_16_t          fDescent;        // How far image extends below "y"
   Html_16_t          fX;              // X coordinate of left edge of the image
   Html_32_t          fY;              // Y coordinate of image baseline
   const char        *fZAlt;           // Alternative text
   TGHtmlImage       *fPImage;         // Corresponding TGHtmlImage object
   TGHtmlElement     *fPMap;           // usemap
   TGHtmlImageMarkup *fINext;          // Next markup using the same TGHtmlImage object
};


// Allowed alignments for images. These represent the allowed arguments
// to the "align=" field of the <IMG> markup.

#define IMAGE_ALIGN_Bottom        0
#define IMAGE_ALIGN_Middle        1
#define IMAGE_ALIGN_Top           2
#define IMAGE_ALIGN_TextTop       3
#define IMAGE_ALIGN_AbsMiddle     4
#define IMAGE_ALIGN_AbsBottom     5
#define IMAGE_ALIGN_Left          6
#define IMAGE_ALIGN_Right         7


// All kinds of form markup, including <INPUT>, <TEXTAREA> and <SELECT>
// are represented by instances of the following class.
//
// (later...)  We also use this for the <APPLET> markup. That way,
// the window we create for an <APPLET> responds to the TGHtml::MapControls()
// and TGHtml::UnmapControls() function calls. For an <APPLET>, the
// pForm field is NULL. (Later still...) <EMBED> works just like
// <APPLET> so it uses this class too.

class TGHtmlForm;

class TGHtmlInput : public TGHtmlMarkupElement {
public:
   TGHtmlInput(int type, int argc, int arglen[], char *argv[]);

   void Empty();

public:
   TGHtmlForm     *fPForm;       // The <FORM> to which this belongs
   TGHtmlInput    *fINext;       // Next element in a list of all input elements
   TGFrame        *fFrame;       // The xclass window that implements this control
   TGHtml         *fHtml;        // The HTML widget this control is attached to
   TGHtmlElement  *fPEnd;        // End tag for <TEXTAREA>, etc.
   Html_u16_t      fInpId;       // Unique id for this element
   Html_u16_t      fSubId;       // For radio - an id, for select - option count
   Html_32_t       fY;           // Baseline for this input element
   Html_u16_t      fX;           // Left edge
   Html_u16_t      fW, fH;       // Width and height of this control
   Html_u8_t       fPadLeft;     // Extra padding on left side of the control
   Html_u8_t       fAlign;       // One of the IMAGE_ALIGN_xxx types
   Html_u8_t       fTextAscent;  // Ascent for the current font
   Html_u8_t       fTextDescent; // descent for the current font
   Html_u8_t       fItype;       // What type of input is this?
   Html_u8_t       fSized;       // True if this input has been sized already
   Html_u16_t      fCnt;         // Used to derive widget name. 0 if no widget
};


// An input control can be one of the following types. See the
// comment about <APPLET> on the TGHtmlInput class insight into
// INPUT_TYPE_Applet.

#define INPUT_TYPE_Unknown      0
#define INPUT_TYPE_Checkbox     1
#define INPUT_TYPE_File         2
#define INPUT_TYPE_Hidden       3
#define INPUT_TYPE_Image        4
#define INPUT_TYPE_Password     5
#define INPUT_TYPE_Radio        6
#define INPUT_TYPE_Reset        7
#define INPUT_TYPE_Select       8
#define INPUT_TYPE_Submit       9
#define INPUT_TYPE_Text        10
#define INPUT_TYPE_TextArea    11
#define INPUT_TYPE_Applet      12
#define INPUT_TYPE_Button      13


// There can be multiple <FORM> entries on a single HTML page.
// Each one must be given a unique number for identification purposes,
// and so we can generate unique state variable names for radiobuttons,
// checkbuttons, and entry boxes.

class TGHtmlForm : public TGHtmlMarkupElement {
public:
   TGHtmlForm(int type, int argc, int arglen[], char *argv[]);

public:
   Html_u16_t     fFormId;    // Unique number assigned to this form
   unsigned int   fElements;  // Number of elements
   unsigned int   fHasctl;    // Has controls
   TGHtmlElement *fPFirst;    // First form element
   TGHtmlElement *fPEnd;      // Pointer to end tag element
};


// Information used by a <HR> markup

class TGHtmlHr : public TGHtmlMarkupElement {
public:
   TGHtmlHr(int type, int argc, int arglen[], char *argv[]);

public:
   Html_32_t   fY;      // Baseline for this input element
   Html_u16_t  fX;      // Left edge
   Html_u16_t  fW, fH;  // Width and height of this control
   Html_u8_t   fIs3D;   // Is it drawn 3D?
};


// Information used by a <A> markup

class TGHtmlAnchor : public TGHtmlMarkupElement {
public:
   TGHtmlAnchor(int type, int argc, int arglen[], char *argv[]);

public:
   Html_32_t  fY;   // Top edge for this element
};


// Information about the <SCRIPT> markup. The parser treats <SCRIPT>
// specially. All text between <SCRIPT> and </SCRIPT> is captured and
// is indexed to by the nStart field of this class.
//
// The nStart field indexs to a spot in the zText field of the TGHtml object.
// The nScript field determines how long the script is.

class TGHtmlScript : public TGHtmlMarkupElement {
public:
   TGHtmlScript(int type, int argc, int arglen[], char *argv[]);

public:
   int   fNStart;    // Start of the script (index into TGHtml::zText)
   int   fNScript;   // Number of characters of text in zText holding
                     // the complete text of this script
};


// A block is a single unit of display information. This can be one or more
// text elements, or the border of table, or an image, etc.
//
// Blocks are used to improve display speed and to improve the speed of
// linear searchs through the token list. A single block will typically
// contain enough information to display a dozen or more Text and Space
// elements all with a single call to OXFont::DrawChars(). The blocks are
// linked together on their own list, so we can search them much faster than
// elements (since there are fewer of them.)
//
// Of course, you can construct pathological HTML that has as many Blocks as
// it has normal tokens. But you haven't lost anything. Using blocks just
// speeds things up in the common case.
//
// Much of the information needed for display is held in the original
// TGHtmlElement objects. "fPNext" points to the first object in the list
// which can be used to find the "style" "x" and "y".
//
// If n is zero, then "fPNext" might point to a special TGHtmlElement
// that defines some other kind of drawing, like <LI> or <IMG> or <INPUT>.

class TGHtmlBlock : public TGHtmlElement {
public:
   TGHtmlBlock();
   virtual ~TGHtmlBlock();

public:
   char        *fZ;                 // Space to hold text when n > 0
   int          fTop, fBottom;      // Extremes of y coordinates
   Html_u16_t   fLeft, fRight;      // Left and right boundry of this object
   Html_u16_t   fN;                 // Number of characters in z[]
   TGHtmlBlock *fBPrev, *fBNext;    // Linked list of all Blocks
};


// A stack of these structures is used to keep track of nested font and
// style changes. This allows us to easily revert to the previous style
// when we encounter and end-tag like </em> or </h3>.
//
// This stack is used to keep track of the current style while walking
// the list of elements. After all elements have been assigned a style,
// the information in this stack is no longer used.

struct SHtmlStyleStack_t {
   SHtmlStyleStack_t *fPNext;   // Next style on the stack
   int                fType;    // A markup that ends this style. Ex: Html_EndEM
   SHtmlStyle_t       fStyle;   // The currently active style.
};


// A stack of the following structures is used to remember the
// left and right margins within a layout context.

struct SHtmlMargin_t {
   int            fIndent;    // Size of the current margin
   int            fBottom;    // Y value at which this margin expires
   int            fTag;       // Markup that will cancel this margin
   SHtmlMargin_t *fPNext;     // Previous margin
};


// How much space (in pixels) used for a single level of indentation due
// to a <UL> or <DL> or <BLOCKQUOTE>, etc.

#define HTML_INDENT 36


//----------------------------------------------------------------------

// A layout context holds all state information used by the layout engine.

class TGHtmlLayoutContext : public TObject {
public:
   TGHtmlLayoutContext();

   void LayoutBlock();
   void Reset();

   void PopIndent();
   void PushIndent();

protected:
   void PushMargin(SHtmlMargin_t **ppMargin, int indent, int bottom, int tag);
   void PopOneMargin(SHtmlMargin_t **ppMargin);
   void PopMargin(SHtmlMargin_t **ppMargin, int tag);
   void PopExpiredMargins(SHtmlMargin_t **ppMarginStack, int y);
   void ClearMarginStack(SHtmlMargin_t **ppMargin);

   TGHtmlElement *GetLine(TGHtmlElement *pStart, TGHtmlElement *pEnd,
                          int width, int minX, int *actualWidth);

   void FixAnchors(TGHtmlElement *p, TGHtmlElement *pEnd, int y);
   int  FixLine(TGHtmlElement *pStart, TGHtmlElement *pEnd,
                int bottom, int width, int actualWidth, int leftMargin,
                int *maxX);
   void Paragraph(TGHtmlElement *p);
   void ComputeMargins(int *pX, int *pY, int *pW);
   void ClearObstacle(int mode);
   TGHtmlElement *DoBreakMarkup(TGHtmlElement *p);
   int  InWrapAround();
   void WidenLine(int reqWidth, int *pX, int *pY, int *pW);

   TGHtmlElement *TableLayout(TGHtmlTable *p);

public:
   TGHtml           *fHtml;            // The html widget undergoing layout
   TGHtmlElement    *fPStart;          // Start of elements to layout
   TGHtmlElement    *fPEnd;            // Stop when reaching this element
   int               fHeadRoom;        // Extra space wanted above this line
   int               fTop;             // Absolute top of drawing area
   int               fBottom;          // Bottom of previous line
   int               fLeft, fRight;    // Left and right extremes of drawing area
   int               fPageWidth;       // Width of the layout field, including
                                       // the margins
   int               fMaxX, fMaxY;     // Maximum X and Y values of paint
   SHtmlMargin_t    *fLeftMargin;      // Stack of left margins
   SHtmlMargin_t    *fRightMargin;     // Stack of right margins
};


// With 28 different fonts and 16 colors, we could in principle have
// as many as 448 different GCs. But in practice, a single page of
// HTML will typically have much less than this. So we won't try to
// keep all GCs on hand. Instead, we'll keep around the most recently
// used GCs and allocate new ones as necessary.
//
// The following structure is used to build a cache of GCs in the
// main widget object.

#define N_CACHE_GC 32

struct GcCache_t {
   GContext_t  fGc;        // The graphics context
   Html_u8_t   fFont;      // Font used for this context
   Html_u8_t   fColor;     // Color used for this context
   Html_u8_t   fIndex;     // Index used for LRU replacement
};


// An SHtmlIndex_t is a reference to a particular character within a
// particular Text or Space token.

struct SHtmlIndex_t {
   TGHtmlElement *fP;     // The token containing the character
   int            fI;     // Index of the character
};


// Used by the tokenizer

struct SHtmlTokenMap_t {
   const char       *fZName;        // Name of a markup
   Html_16_t         fType;         // Markup type code
   Html_16_t         fObjType;      // Which kind of TGHtml... object to alocate
   SHtmlTokenMap_t  *fPCollide;     // Hash table collision chain
};


// Markup element types to be allocated by the tokenizer.
// Do not confuse with .type field in TGHtmlElement

#define O_HtmlMarkupElement   0
#define O_HtmlCell            1
#define O_HtmlTable           2
#define O_HtmlRef             3
#define O_HtmlLi              4
#define O_HtmlListStart       5
#define O_HtmlImageMarkup     6
#define O_HtmlInput           7
#define O_HtmlForm            8
#define O_HtmlHr              9
#define O_HtmlAnchor          10
#define O_HtmlScript          11
#define O_HtmlMapArea         12


//----------------------------------------------------------------------

// The HTML widget. A derivate of TGView.

class TGListBox;
class THashTable;

class TGHtml : public TGView {
public:
   TGHtml(const TGWindow *p, int w, int h, int id = -1);
   virtual ~TGHtml();

   virtual Bool_t HandleFocusChange(Event_t *event);
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   virtual Bool_t HandleIdleEvent(TGIdleHandler *i);
   virtual Bool_t HandleTimer(TTimer *timer);

   virtual Bool_t ProcessMessage(Long_t, Long_t, Long_t);

   virtual void   DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t ItemLayout();

   Bool_t         HandleHtmlInput(TGHtmlInput *pr, Event_t *event);
   Bool_t         HandleRadioButton(TGHtmlInput *p);

public:   // user commands

   int  ParseText(char *text, const char *index = 0);

   void SetTableRelief(int relief);
   int  GetTableRelief() const { return fTableRelief; }

   void SetRuleRelief(int relief);
   int  GetRuleRelief() const { return fRuleRelief; }
   int  GetRulePadding() const { return fRulePadding; }

   void UnderlineLinks(int onoff);

   void SetBaseUri(const char *uri);
   const char *GetBaseUri() const { return fZBase; }

   int GotoAnchor(const char *name);

public:   // reloadable methods

   // called when the widget is cleared
   virtual void Clear(Option_t * = "");

   // User function to resolve URIs
   virtual char *ResolveUri(const char *uri);

   // User function to get an image from a URL
   virtual TImage *LoadImage(const char *uri, int w = 0, int h = 0) ;//
   //    { return 0; }

   // User function to tell if a hyperlink has already been visited
   virtual int IsVisited(const char * /*url*/)
      { return kFALSE; }

   // User function to to process tokens of the given type
   virtual int ProcessToken(TGHtmlElement * /*pElem*/, const char * /*name*/, int /*type*/)
      { return kFALSE; }

   virtual TGFont *GetFont(int iFont);

   // The HTML parser will invoke the following methods from time
   // to time to find out information it needs to complete formatting of
   // the document.

   // Method for handling <frameset> markup
   virtual int ProcessFrame()
      { return kFALSE; }

   // Method to process applets
   virtual TGFrame *ProcessApplet(TGHtmlInput * /*input*/)
      { return 0; }

   // Called when parsing forms
   virtual int FormCreate(TGHtmlForm * /*form*/, const char * /*zUrl*/, const char * /*args*/)
      { return kFALSE; }

   // Called when user presses Submit
   virtual int FormAction(TGHtmlForm * /*form*/, int /*id*/)
      { return kFALSE; }

   // Invoked to find font names
   virtual char *GetFontName()
      { return 0; }

   // Invoked for each <SCRIPT> markup
   virtual char *ProcessScript(TGHtmlScript * /*script*/)
      { return 0; }

public:
   const char *GetText() const { return fZText; }

   int GetMarginWidth() { return fMargins.fL + fMargins.fR; }
   int GetMarginHeight() { return fMargins.fT + fMargins.fB; }

   TGHtmlInput *GetInputElement(int x, int y);
   const char *GetHref(int x, int y, const char **target = 0);

   TGHtmlImage *GetImage(TGHtmlImageMarkup *p);

   int  InArea(TGHtmlMapArea *p, int left, int top, int x, int y);
   TGHtmlElement *GetMap(const char *name);

   void ResetBlocks() { fFirstBlock = fLastBlock = 0; }
   int  ElementCoords(TGHtmlElement *p, int i, int pct, int *coords);

   TGHtmlElement *TableDimensions(TGHtmlTable *pStart, int lineWidth);
   int  CellSpacing(TGHtmlElement *pTable);
   void MoveVertically(TGHtmlElement *p, TGHtmlElement *pLast, int dy);

   void PrintList(TGHtmlElement *first, TGHtmlElement *last);

   char *GetTokenName(TGHtmlElement *p);
   char *DumpToken(TGHtmlElement *p);

   void EncodeText(TGString *str, const char *z);

protected:
   void HClear();
   void ClearGcCache();
   void ResetLayoutContext();
   void Redraw();
   void ComputeVirtualSize();

   void ScheduleRedraw();

   void RedrawArea(int left, int top, int right, int bottom);
   void RedrawBlock(TGHtmlBlock *p);
   void RedrawEverything();
   void RedrawText(int y);

   float ColorDistance(ColorStruct_t *pA, ColorStruct_t *pB);
   int IsDarkColor(ColorStruct_t *p);
   int IsLightColor(ColorStruct_t *p);
   int GetColorByName(const char *zColor);
   int GetDarkShadowColor(int iBgColor);
   int GetLightShadowColor(int iBgColor);
   int GetColorByValue(ColorStruct_t *pRef);

   void FlashCursor();

   GContext_t GetGC(int color, int font);
   GContext_t GetAnyGC();

   void AnimateImage(TGHtmlImage *image);
   void ImageChanged(TGHtmlImage *image, int newWidth, int newHeight);
   int  GetImageAlignment(TGHtmlElement *p);
   int  GetImageAt(int x, int y);
   const char *GetPctWidth(TGHtmlElement *p, char *opt, char *ret);
   void TableBgndImage(TGHtmlElement *p);

   TGHtmlElement *FillOutBlock(TGHtmlBlock *p);
   void UnlinkAndFreeBlock(TGHtmlBlock *pBlock);
   void AppendBlock(TGHtmlElement *pToken, TGHtmlBlock *pBlock);

   void StringHW(const char *str, int *h, int *w);
   TGHtmlElement *MinMax(TGHtmlElement *p, int *pMin, int *pMax,
                         int lineWidth, int hasbg);

   void DrawSelectionBackground(TGHtmlBlock *pBlock, Drawable_t Drawable_t,
                                int x, int y);
   void DrawRect(Drawable_t drawable, TGHtmlElement *src,
                 int x, int y, int w, int h, int depth, int relief);
   void BlockDraw(TGHtmlBlock *pBlock, Drawable_t wid,
                  int left, int top,
                  int width, int height, Pixmap_t pixmap);
   void DrawImage(TGHtmlImageMarkup *image, Drawable_t wid,
                  int left, int top,
                  int right, int bottom);
   void DrawTableBgnd(int x, int y, int w, int h, Drawable_t d, TImage *image);

   TGHtmlElement *FindStartOfNextBlock(TGHtmlElement *p, int *pCnt);
   void FormBlocks();

   void AppendElement(TGHtmlElement *pElem);
   int  Tokenize();
   void AppToken(TGHtmlElement *pNew, TGHtmlElement *p, int offs);
   TGHtmlMarkupElement *MakeMarkupEntry(int objType, int type, int argc,
                                        int arglen[], char *argv[]);
   void TokenizerAppend(const char *text);
   TGHtmlElement *InsertToken(TGHtmlElement *pToken,
                              char *zType, char *zArgs, int offs);
   SHtmlTokenMap_t *NameToPmap(char *zType);
   int  NameToType(char *zType);
   const char *TypeToName(int type);
   int  TextInsertCmd(int argc, char **argv);
   SHtmlTokenMap_t* GetMarkupMap(int n);

   TGHtmlElement *TokenByIndex(int N, int flag);
   int  TokenNumber(TGHtmlElement *p);

   void MaxIndex(TGHtmlElement *p, int *pIndex, int isLast);
   int  IndexMod(TGHtmlElement **pp, int *ip, char *cp);
   void FindIndexInBlock(TGHtmlBlock *pBlock, int x,
                         TGHtmlElement **ppToken, int *pIndex);
   void IndexToBlockIndex(SHtmlIndex_t sIndex,
                          TGHtmlBlock **ppBlock, int *piIndex);
   int  DecodeBaseIndex(const char *zBase,
                        TGHtmlElement **ppToken, int *pIndex);
   int  GetIndex(const char *zIndex, TGHtmlElement **ppToken, int *pIndex);

   void LayoutDoc();

   int  MapControls();
   void UnmapControls();
   void DeleteControls();
   int  ControlSize(TGHtmlInput *p);
   void SizeAndLink(TGFrame *frame, TGHtmlInput *pElem);
   int  FormCount(TGHtmlInput *p, int radio);
   void AddFormInfo(TGHtmlElement *p);
   void AddSelectOptions(TGListBox *lb, TGHtmlElement *p, TGHtmlElement *pEnd);
   void AppendText(TGString *str, TGHtmlElement *pFirst, TGHtmlElement *pEnd);

   void UpdateSelection(int forceUpdate);
   void UpdateSelectionDisplay();
   void LostSelection();
   int  SelectionSet(const char *startIx, const char *endIx);
   void UpdateInsert();
   int  SetInsert(const char *insIx);

   const char *GetUid(const char *string);
   ColorStruct_t *AllocColor(const char *name);
   ColorStruct_t *AllocColorByValue(ColorStruct_t *color);
   void FreeColor(ColorStruct_t *color);

   SHtmlStyle_t GetCurrentStyle();
   void PushStyleStack(int tag, SHtmlStyle_t style);
   SHtmlStyle_t PopStyleStack(int tag);

   void MakeInvisible(TGHtmlElement *p_first, TGHtmlElement *p_last);
   int  GetLinkColor(const char *zURL);
   void AddStyle(TGHtmlElement *p);
   void Sizer();

   int  NextMarkupType(TGHtmlElement *p);

   TGHtmlElement *AttrElem(const char *name, char *value);

public:
   void AppendArglist(TGString *str, TGHtmlMarkupElement *pElem);
   TGHtmlElement *FindEndNest(TGHtmlElement *sp, int en, TGHtmlElement *lp);
   TGString *ListTokens(TGHtmlElement *p, TGHtmlElement *pEnd);
   TGString *TableText(TGHtmlTable *pTable, int flags);

   virtual void MouseOver(const char *uri) { Emit("MouseOver(const char *)",uri); } // *SIGNAL*
   virtual void MouseDown(const char *uri)  { Emit("MouseDown(const char *)",uri); } // *SIGNAL*
   virtual void ButtonClicked(const char *name, const char *val); // *SIGNAL*
   virtual void SubmitClicked(const char *val); // *SIGNAL*
   virtual void CheckToggled(const char *name, Bool_t on, const char *val); // *SIGNAL*
   virtual void RadioChanged(const char *name, const char *val); // *SIGNAL*
   virtual void InputSelected(const char *name, const char *val);   //*SIGNAL*
   virtual void SavePrimitive(ostream &out, Option_t * = "");

protected:
   virtual void UpdateBackgroundStart();

protected:
   TGHtmlElement *fPFirst;          // First HTML token on a list of them all
   TGHtmlElement *fPLast;           // Last HTML token on the list
   int            fNToken;          // Number of HTML tokens on the list.
                                    // Html_Block tokens don't count.
   TGHtmlElement *fLastSized;       // Last HTML element that has been sized
   TGHtmlElement *fNextPlaced;      // Next HTML element that needs to be
                                    // positioned on canvas.
   TGHtmlBlock   *fFirstBlock;      // List of all TGHtmlBlock tokens
   TGHtmlBlock   *fLastBlock;       // Last TGHtmlBlock in the list
   TGHtmlInput   *fFirstInput;      // First <INPUT> element
   TGHtmlInput   *fLastInput;       // Last <INPUT> element
   int            fNInput;          // The number of <INPUT> elements
   int            fNForm;           // The number of <FORM> elements
   int            fVarId;           // Used to construct a unique name for a
                                    // global array used by <INPUT> elements
   int            fInputIdx;        // Unique input index
   int            fRadioIdx;        // Unique radio index

   // Information about the selected region of text

   SHtmlIndex_t   fSelBegin;        // Start of the selection
   SHtmlIndex_t   fSelEnd;          // End of the selection
   TGHtmlBlock   *fPSelStartBlock;  // Block in which selection starts
   Html_16_t      fSelStartIndex;   // Index in pSelStartBlock of first selected
                                    // character
   Html_16_t      fSelEndIndex;     // Index of last selecte char in pSelEndBlock
   TGHtmlBlock   *fPSelEndBlock;    // Block in which selection ends

   // Information about the insertion cursor

   int            fInsOnTime;       // How long the cursor states one (millisec)
   int            fInsOffTime;      // How long it is off (milliseconds)
   int            fInsStatus;       // Is it visible?
   TTimer        *fInsTimer;        // Timer used to flash the insertion cursor
   SHtmlIndex_t   fIns;             // The insertion cursor position
   TGHtmlBlock   *fPInsBlock;       // The TGHtmlBlock containing the cursor
   int            fInsIndex;        // Index in pInsBlock of the cursor

   // The following fields hold state information used by the tokenizer.

   char          *fZText;           // Complete text of the unparsed HTML
   int            fNText;           // Number of characters in zText
   int            fNAlloc;          // Space allocated for zText
   int            fNComplete;       // How much of zText has actually been
                                    // converted into tokens
   int            fICol;            // The column in which zText[nComplete]
                                    // occurs. Used to resolve tabs in input
   int            fIPlaintext;      // If not zero, this is the token type that
                                    // caused us to go into plaintext mode. One
                                    // of Html_PLAINTEXT, Html_LISTING or
                                    // Html_XMP
   TGHtmlScript  *fPScript;         // <SCRIPT> currently being parsed

   TGIdleHandler *fIdle;

   // These fields hold state information used by the HtmlAddStyle routine.
   // We have to store this state information here since HtmlAddStyle
   // operates incrementally. This information must be carried from
   // one incremental execution to the next.

   SHtmlStyleStack_t *fStyleStack;     // The style stack
   int               fParaAlignment;   // Justification associated with <p>
   int               fRowAlignment;    // Justification associated with <tr>
   int               fAnchorFlags;     // Style flags associated with <A>...</A>
   int               fInDt;            // Style flags associated with <DT>...</DT>
   int               fInTr;            // True if within <tr>..</tr>
   int               fInTd;            // True if within <td>..</td> or <th>..</th>
   TGHtmlAnchor     *fAnchorStart;     // Most recent <a href=...>
   TGHtmlForm       *fFormStart;       // Most recent <form>
   TGHtmlInput      *fFormElemStart;   // Most recent <textarea> or <select>
   TGHtmlInput      *fFormElemLast;    // Most recent <input>, <textarea> or <select>
   TGHtmlListStart  *fInnerList;       // The inner most <OL> or <UL>
   TGHtmlElement    *fLoEndPtr;        // How far AddStyle has gone to
   TGHtmlForm       *fLoFormStart;     // For AddStyle

   // These fields are used to hold the state of the layout engine.
   // Because the layout is incremental, this state must be held for
   // the life of the widget.

   TGHtmlLayoutContext fLayoutContext;

   // Information used when displaying the widget:

   int               fHighlightWidth;        // Width in pixels of highlight to draw
                                             // around widget when it has the focus.
                                             // <= 0 means don't draw a highlight.
   TGInsets          fMargins;               // document margins (separation between the
                                             // edge of the clip window and rendered HTML).
   ColorStruct_t    *fHighlightBgColorPtr;   // Color for drawing traversal highlight
                                             // area when highlight is off.
   ColorStruct_t    *fHighlightColorPtr;     // Color for drawing traversal highlight.
   TGFont           *fAFont[N_FONT];         // Information about all screen fonts
   char              fFontValid[(N_FONT+7)/8]; // If bit N%8 of work N/8 of this field is 0
                                             // if aFont[N] needs to be reallocated before
                                             // being used.
   ColorStruct_t    *fApColor[N_COLOR];      // Information about all colors
   Long_t            fColorUsed;             // bit N is 1 if color N is in use. Only
                                             // applies to colors that aren't predefined
   int               fIDark[N_COLOR];        // Dark 3D shadow of color K is iDark[K]
   int               fILight[N_COLOR];       // Light 3D shadow of color K is iLight[K]
   ColorStruct_t    *fBgColor;               // Background color of the HTML document
   ColorStruct_t    *fFgColor;               // Color of normal text. apColor[0]
   ColorStruct_t    *fNewLinkColor;          // Color of unvisitied links. apColor[1]
   ColorStruct_t    *fOldLinkColor;          // Color of visitied links. apColor[2]
   ColorStruct_t    *fSelectionColor;        // Background color for selections
   GcCache_t         fAGcCache[N_CACHE_GC];  // A cache of GCs for general use
   int               fGcNextToFree;
   int               fLastGC;                // Index of recently used GC
   TGHtmlImage      *fImageList;             // A list of all images
   TImage           *fBgImage;               // Background image

   int               fFormPadding;           // Amount to pad form elements by
   int               fOverrideFonts;         // TRUE if we should override fonts
   int               fOverrideColors;        // TRUE if we should override colors
   int               fUnderlineLinks;        // TRUE if we should underline hyperlinks
   int               fHasScript;             // TRUE if we can do scripts for this page
   int               fHasFrames;             // TRUE if we can do frames for this page
   int               fAddEndTags;            // TRUE if we add /LI etc.
   int               fTableBorderMin;        // Force tables to have min border size
   int               fVarind;                // Index suffix for unique global var name

   // Information about the selection

   int               fExportSelection;       // True if the selection is automatically
                                             // exported to the clipboard

   // Miscellaneous information:

   int               fTableRelief;           // 3d effects on <TABLE>
   int               fRuleRelief;            // 3d effects on <HR>
   int               fRulePadding;           // extra pixels above and below <HR>
   const char       *fZBase;                 // The base URI
   char             *fZBaseHref;             // zBase as modified by <BASE HREF=..> markup
   Cursor_t          fCursor;                // Current cursor for window, or None.
   int               fMaxX, fMaxY;           // Maximum extent of any "paint" that appears
                                             // on the virtual canvas. Used to compute
                                             // scrollbar positions.
   int               fDirtyLeft, fDirtyTop;  // Top left corner of region to redraw. These
                                             // are physical screen coordinates relative to
                                             // the clip win, not main window.
   int               fDirtyRight, fDirtyBottom;  // Bottom right corner of region to redraw
   int               fFlags;                 // Various flags; see below for definitions.
   int               fIdind;
   int               fInParse;               // Prevent update if parsing
   char             *fZGoto;                 // Label to goto right after layout

   SHtmlExtensions_t *fExts;                 // Pointer to user extension data

   THashTable       *fUidTable;              // Hash table for some used string values
                                             // like color names, etc.
   const char       *fLastUri;               // Used in HandleMotion
   int               fExiting;               // True if the widget is being destroyed

   ClassDef(TGHtml, 0); // HTML widget
};


// Flag bits "flags" field of the Html widget:
//
// REDRAW_PENDING         An idle handler has already been queued to
//                        call the TGHtml::Redraw() method.
//
// GOT_FOCUS              This widget currently has input focus.
//
// HSCROLL                Horizontal scrollbar position needs to be
//                        recomputed.
//
// VSCROLL                Vertical scrollbar position needs to be
//                        recomputed.
//
// RELAYOUT               We need to reposition every element on the
//                        virtual canvas. (This happens, for example,
//                        when the size of the widget changes and we
//                        need to recompute the line breaks.)
//
// RESIZE_ELEMENTS        We need to recompute the size of every element.
//                        This happens, for example, when the fonts
//                        change.
//
// REDRAW_FOCUS           We need to repaint the focus highlight border.
//
// REDRAW_TEXT            Everything in the clipping window needs to be redrawn.
//
// STYLER_RUNNING         There is a call to HtmlAddStyle() in process.
//                        Used to prevent a recursive call to HtmlAddStyle().
//
// INSERT_FLASHING        True if there is a timer scheduled that will toggle
//                        the state of the insertion cursor.
//
// REDRAW_IMAGES          One or more TGHtmlImageMarkup objects have their
//                        redrawNeeded flag set.

#define REDRAW_PENDING       0x000001
#define GOT_FOCUS            0x000002
#define HSCROLL              0x000004
#define VSCROLL              0x000008
#define RELAYOUT             0x000010
#define RESIZE_ELEMENTS      0x000020
#define REDRAW_FOCUS         0x000040
#define REDRAW_TEXT          0x000080
#define EXTEND_LAYOUT        0x000100
#define STYLER_RUNNING       0x000200
#define INSERT_FLASHING      0x000400
#define REDRAW_IMAGES        0x000800
#define ANIMATE_IMAGES       0x001000


// Macros to set, clear or test bits of the "flags" field.

#define HtmlHasFlag(A,F)      (((A)->flags&(F))==(F))
#define HtmlHasAnyFlag(A,F)   (((A)->flags&(F))!=0)
#define HtmlSetFlag(A,F)      ((A)->flags|=(F))
#define HtmlClearFlag(A,F)    ((A)->flags&=~(F))


// No coordinate is ever as big as this number

#define LARGE_NUMBER 100000000


// Default values for configuration options

#define DEF_HTML_BG_COLOR             DEF_FRAME_BG_COLOR
#define DEF_HTML_BG_MONO              DEF_FRAME_BG_MONO
#define DEF_HTML_EXPORT_SEL           1
#define DEF_HTML_FG                   DEF_BUTTON_FG
#define DEF_HTML_HIGHLIGHT_BG         DEF_BUTTON_HIGHLIGHT_BG
#define DEF_HTML_HIGHLIGHT            DEF_BUTTON_HIGHLIGHT
#define DEF_HTML_HIGHLIGHT_WIDTH      "0"
#define DEF_HTML_INSERT_OFF_TIME      300
#define DEF_HTML_INSERT_ON_TIME       600
#define DEF_HTML_PADX                 (HTML_INDENT / 4)
#define DEF_HTML_PADY                 (HTML_INDENT / 4)
#define DEF_HTML_RELIEF               "raised"
#define DEF_HTML_SELECTION_COLOR      "skyblue"
#define DEF_HTML_TAKE_FOCUS           "0"
#define DEF_HTML_UNVISITED            "blue2"
#define DEF_HTML_VISITED              "purple4"

#ifdef NAVIGATOR_TABLES

#define DEF_HTML_TABLE_BORDER             "0"
#define DEF_HTML_TABLE_CELLPADDING        "2"
#define DEF_HTML_TABLE_CELLSPACING        "5"
#define DEF_HTML_TABLE_BORDER_LIGHT_COLOR "gray80"
#define DEF_HTML_TABLE_BORDER_DARK_COLOR  "gray40"

#endif  // NAVIGATOR_TABLES


//----------------------------------------------------------------------

// Messages generated by the HTML widget
/*
class TGHtmlMessage : public OWidgetMessage {
public:
  TGHtmlMessage(int t, int a, int i, const char *u, int rx, int ry) :
    OWidgetMessage(t, a, i) {
      uri = u;
      x_root = rx;
      y_root = ry;
    }

public:
  const char *uri;
  //ORectangle bbox;
  int x_root, y_root;
};
*/

#endif  // ROOT_TGHtml
