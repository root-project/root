/* GDK - The GIMP Drawing Kit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GTK+ Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GTK+ Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GTK+ at ftp://ftp.gtk.org/pub/gtk/. 
 */

#include "config.h"

#include <stdio.h>
#include <ctype.h>

#include "gdkfont.h"
#include "gdkwin32.h"

static GHashTable *font_name_hash = NULL;
static GHashTable *fontset_name_hash = NULL;

static float gfont_angle = 0.0;

static void
gdk_font_hash_insert(GdkFontType type,
                     GdkFont * font, const gchar * font_name)
{
   GdkFontPrivateWin32 *private = (GdkFontPrivateWin32 *) font;
   GHashTable **hashp = (type == GDK_FONT_FONT) ?
       &font_name_hash : &fontset_name_hash;

   if (!*hashp)
      *hashp = g_hash_table_new(g_str_hash, g_str_equal);

   private->names = g_slist_prepend(private->names, g_strdup(font_name));
   g_hash_table_insert(*hashp, private->names->data, font);
}

static void gdk_font_hash_remove(GdkFontType type, GdkFont * font)
{
   GdkFontPrivateWin32 *private = (GdkFontPrivateWin32 *) font;
   GSList *tmp_list;
   GHashTable *hash = (type == GDK_FONT_FONT) ?
       font_name_hash : fontset_name_hash;

   tmp_list = private->names;
   while (tmp_list) {
      g_hash_table_remove(hash, tmp_list->data);
      g_free(tmp_list->data);

      tmp_list = tmp_list->next;
   }

   g_slist_free(private->names);
   private->names = NULL;
}

static GdkFont *gdk_font_hash_lookup(GdkFontType type,
                                     const gchar * font_name)
{
   GdkFont *result;
   GHashTable *hash = (type == GDK_FONT_FONT) ?
       font_name_hash : fontset_name_hash;

   if (!hash)
      return NULL;
   else {
      result = g_hash_table_lookup(hash, font_name);
      if (result)
         gdk_font_ref(result);

      return result;
   }
}

static const char *charset_name(DWORD charset)
{
   switch (charset) {
   case ANSI_CHARSET:
      return "ansi";
   case DEFAULT_CHARSET:
      return "default";
   case SYMBOL_CHARSET:
      return "symbol";
   case SHIFTJIS_CHARSET:
      return "shiftjis";
   case HANGEUL_CHARSET:
      return "hangeul";
   case GB2312_CHARSET:
      return "gb2312";
   case CHINESEBIG5_CHARSET:
      return "big5";
   case JOHAB_CHARSET:
      return "johab";
   case HEBREW_CHARSET:
      return "hebrew";
   case ARABIC_CHARSET:
      return "arabic";
   case GREEK_CHARSET:
      return "greek";
   case TURKISH_CHARSET:
      return "turkish";
   case VIETNAMESE_CHARSET:
      return "vietnamese";
   case THAI_CHARSET:
      return "thai";
   case EASTEUROPE_CHARSET:
      return "easteurope";
   case RUSSIAN_CHARSET:
      return "russian";
   case MAC_CHARSET:
      return "mac";
   case BALTIC_CHARSET:
      return "baltic";
   }
   return "unknown";
}

static gint num_fonts;
static gint font_names_size;
static gchar **xfontnames;

static gchar *logfont_to_xlfd(const LOGFONT * lfp,
                              int size, int res, int avg_width)
{
   const gchar *weight;
   const gchar *registry, *encoding;
   int point_size;
   static int logpixelsy = 0;
   gchar facename[LF_FACESIZE * 5];
   gchar *utf8_facename;
   gchar *p;
   const gchar *q;

   if (logpixelsy == 0) {
      logpixelsy = GetDeviceCaps(gdk_DC, LOGPIXELSY);
   }

   if (lfp->lfWeight >= FW_HEAVY)
      weight = "heavy";
   else if (lfp->lfWeight >= FW_EXTRABOLD)
      weight = "extrabold";
   else if (lfp->lfWeight >= FW_BOLD)
      weight = "bold";
#ifdef FW_DEMIBOLD
   else if (lfp->lfWeight >= FW_DEMIBOLD)
      weight = "demibold";
#endif
   else if (lfp->lfWeight >= FW_MEDIUM)
      weight = "medium";
   else if (lfp->lfWeight >= FW_NORMAL)
      weight = "normal";
   else if (lfp->lfWeight >= FW_LIGHT)
      weight = "light";
   else if (lfp->lfWeight >= FW_EXTRALIGHT)
      weight = "extralight";
   else if (lfp->lfWeight >= FW_THIN)
      weight = "thin";
   else
      weight = "regular";

   switch (lfp->lfCharSet) {
   case ANSI_CHARSET:
      registry = "iso8859";
      encoding = "1";
      break;
   case SHIFTJIS_CHARSET:
      registry = "jisx0208.1983";
      encoding = "0";
      break;
   case HANGEUL_CHARSET:
      registry = "ksc5601.1987";
      encoding = "0";
      break;
   case GB2312_CHARSET:
      registry = "gb2312.1980";
      encoding = "0";
      break;
   case CHINESEBIG5_CHARSET:
      registry = "big5";
      encoding = "0";
      break;
   case GREEK_CHARSET:
      registry = "iso8859";
      encoding = "7";
      break;
   case TURKISH_CHARSET:
      registry = "iso8859";
      encoding = "9";
      break;
#if 0                           /* Not a good idea, I think, to use ISO8859-8 and -6 for the Windows
                                 * hebrew and arabic codepages, they differ too much.
                                 */
   case HEBREW_CHARSET:
      registry = "iso8859";
      encoding = "8";
      break;
   case ARABIC_CHARSET:
      registry = "iso8859";
      encoding = "6";
      break;
#endif
   default:
      registry = "microsoft";
      encoding = charset_name(lfp->lfCharSet);
   }

   point_size = (int) (((double) size / logpixelsy) * 720.);

   if (res == -1)
      res = logpixelsy;

   /* Convert the facename Windows fives us from the locale-dependent
    * codepage to UTF-8.
    */
   utf8_facename = g_filename_to_utf8(lfp->lfFaceName, -1, NULL, NULL, NULL);

   /* Replace characters illegal in an XLFD with hex escapes. */
   p = facename;
   q = utf8_facename;
   while (*q) {
      if (*q == '-' || *q == '*' || *q == '?' || *q == '%')
         p += sprintf(p, "%%%.02x", *q);
      else
         *p++ = *q;
      q++;
   }
   *p = '\0';
   g_free(utf8_facename);

   return g_strdup_printf
       ("-%s-%s-%s-%s-%s-%s-%d-%d-%d-%d-%s-%d-%s-%s",
        "unknown",
        facename,
        weight,
        (lfp->lfItalic ?
         ((lfp->lfPitchAndFamily & 0xF0) == FF_ROMAN
          || (lfp->lfPitchAndFamily & 0xF0) == FF_SCRIPT ?
          "i" : "o") : "r"),
        "normal",
        "",
        size,
        point_size,
        res,
        res,
        ((lfp->lfPitchAndFamily & 0x03) == FIXED_PITCH ? "m" : "p"),
        avg_width, registry, encoding);
}

gchar *gdk_font_full_name_get(GdkFont * font)
{
   GdkFontPrivateWin32 *private;
   GdkWin32SingleFont *singlefont;
   GSList *list;
   GString *string;
   gchar *result;
   gchar *xlfd;
   LOGFONT logfont;

   g_return_val_if_fail(font != NULL, NULL);

   private = (GdkFontPrivateWin32 *) font;

   list = private->fonts;
   string = g_string_new("");

   while (list) {
      singlefont = (GdkWin32SingleFont *) list->data;

      if (GetObject(singlefont->xfont, sizeof(LOGFONT), &logfont) == 0) {
         WIN32_GDI_FAILED("GetObject");
         return NULL;
      }

      xlfd = logfont_to_xlfd(&logfont, logfont.lfHeight, -1, 0);
      string = g_string_append(string, xlfd);
      g_free(xlfd);
      list = list->next;
      if (list)
         string = g_string_append_c(string, ',');
   }
   result = string->str;
   g_string_free(string, FALSE);
   return result;
}

void gdk_font_full_name_free(gchar * name)
{
   g_free(name);
}

static gboolean pattern_match(const gchar * pattern, const gchar * string)
{
   const gchar *p = pattern, *n = string;
   gchar c, c1;

   /* Common case first */
   if ((pattern[0] == '*' && pattern[1] == '\0')
       || (pattern[0] == '-' && pattern[1] == '*' && pattern[2] == '\0'))
      return TRUE;

   while ((c = *p++) != '\0') {
      c = tolower(c);

      switch (c) {
      case '?':
         if (*n == '\0')
            return FALSE;
         break;

      case '*':
         for (c = *p++; c == '?' || c == '*'; c = *p++, ++n)
            if (c == '?' && *n == '\0')
               return FALSE;

         if (c == '\0')
            return TRUE;

         c1 = tolower(c);
         for (--p; *n != '\0'; ++n)
            if (tolower(*n) == c1 && pattern_match(p, n))
               return TRUE;
         return FALSE;

      default:
         if (c != tolower(*n))
            return FALSE;
      }

      ++n;
   }

   if (*n == '\0')
      return TRUE;

   return FALSE;
}

static int CALLBACK
InnerEnumFontFamExProc(const LOGFONT * lfp,
                       const TEXTMETRIC * metrics,
                       DWORD fontType, LPARAM lParam)
{
   int size;
   gchar *xlfd;

   if (fontType == TRUETYPE_FONTTYPE) {
      size = 0;
   } else {
      size = lfp->lfHeight;
   }

   xlfd = logfont_to_xlfd(lfp, size, 0, 0);

   if (!pattern_match((gchar *) lParam, xlfd)) {
      g_free(xlfd);
      return 1;
   }

   num_fonts++;
   if (num_fonts == font_names_size) {
      font_names_size *= 2;
      xfontnames =
          g_realloc(xfontnames, font_names_size * sizeof(gchar *));
   }
   xfontnames[num_fonts - 1] = xlfd;

   return 1;
}

static int CALLBACK
EnumFontFamExProc(const LOGFONT * lfp,
                  const TEXTMETRIC * metrics,
                  DWORD fontType, LPARAM lParam)
{
   if (fontType == TRUETYPE_FONTTYPE) {
      LOGFONT lf;

      lf = *lfp;

      EnumFontFamiliesEx(gdk_DC, &lf, InnerEnumFontFamExProc, lParam, 0);
   } else
      InnerEnumFontFamExProc(lfp, metrics, fontType, lParam);

   return 1;
}

gchar **gdk_font_list_new(const gchar * font_pattern, gint * n_returned)
{
   LOGFONT logfont;
   gchar **result;

   num_fonts = 0;
   font_names_size = 100;
   xfontnames = g_new(gchar *, font_names_size);
   memset(&logfont, 0, sizeof(logfont));
   logfont.lfCharSet = DEFAULT_CHARSET;
   EnumFontFamiliesEx(gdk_DC, &logfont, EnumFontFamExProc,
                      (LPARAM) font_pattern, 0);

   result = g_new(gchar *, num_fonts + 1);
   memmove(result, xfontnames, num_fonts * sizeof(gchar *));
   result[num_fonts] = NULL;
   g_free(xfontnames);

   *n_returned = num_fonts;
   return result;
}

void gdk_font_list_free(gchar ** font_list)
{
   g_strfreev(font_list);
}

/* This table classifies Unicode characters according to the Microsoft
 * Unicode subset numbering. This is based on the table in "Developing
 * International Software for Windows 95 and Windows NT". This is almost,
 * but not quite, the same as the official Unicode block table in
 * Blocks.txt from ftp.unicode.org. The bit number field is the bitfield
 * number as in the FONTSIGNATURE struct's fsUsb field.
 * There are some grave bugs in the table in the books. For instance
 * it claims there are Hangul at U+3400..U+4DFF while this range in
 * fact contains CJK Unified Ideographs Extension A. Also, the whole
 * block of Hangul Syllables U+AC00..U+D7A3 is missing from the book.
 */

typedef enum {
   U_BASIC_LATIN = 0,
   U_LATIN_1_SUPPLEMENT = 1,
   U_LATIN_EXTENDED_A = 2,
   U_LATIN_EXTENDED_B = 3,
   U_IPA_EXTENSIONS = 4,
   U_SPACING_MODIFIER_LETTERS = 5,
   U_COMBINING_DIACRITICAL_MARKS = 6,
   U_BASIC_GREEK = 7,
   U_GREEK_SYMBOLS_AND_COPTIC = 8,
   U_CYRILLIC = 9,
   U_ARMENIAN = 10,
   U_HEBREW_EXTENDED = 12,
   U_BASIC_HEBREW = 11,
   U_BASIC_ARABIC = 13,
   U_ARABIC_EXTENDED = 14,
   U_DEVANAGARI = 15,
   U_BENGALI = 16,
   U_GURMUKHI = 17,
   U_GUJARATI = 18,
   U_ORIYA = 19,
   U_TAMIL = 20,
   U_TELUGU = 21,
   U_KANNADA = 22,
   U_MALAYALAM = 23,
   U_THAI = 24,
   U_LAO = 25,
   U_GEORGIAN_EXTENDED = 27,
   U_BASIC_GEORGIAN = 26,
   U_HANGUL_JAMO = 28,
   U_LATIN_EXTENDED_ADDITIONAL = 29,
   U_GREEK_EXTENDED = 30,
   U_GENERAL_PUNCTUATION = 31,
   U_SUPERSCRIPTS_AND_SUBSCRIPTS = 32,
   U_CURRENCY_SYMBOLS = 33,
   U_COMBINING_DIACRITICAL_MARKS_FOR_SYMBOLS = 34,
   U_LETTERLIKE_SYMBOLS = 35,
   U_NUMBER_FORMS = 36,
   U_ARROWS = 37,
   U_MATHEMATICAL_OPERATORS = 38,
   U_MISCELLANEOUS_TECHNICAL = 39,
   U_CONTROL_PICTURES = 40,
   U_OPTICAL_CHARACTER_RECOGNITION = 41,
   U_ENCLOSED_ALPHANUMERICS = 42,
   U_BOX_DRAWING = 43,
   U_BLOCK_ELEMENTS = 44,
   U_GEOMETRIC_SHAPES = 45,
   U_MISCELLANEOUS_SYMBOLS = 46,
   U_DINGBATS = 47,
   U_CJK_SYMBOLS_AND_PUNCTUATION = 48,
   U_HIRAGANA = 49,
   U_KATAKANA = 50,
   U_BOPOMOFO = 51,
   U_HANGUL_COMPATIBILITY_JAMO = 52,
   U_CJK_MISCELLANEOUS = 53,
   U_ENCLOSED_CJK = 54,
   U_CJK_COMPATIBILITY = 55,
   U_HANGUL = 56,
   U_HANGUL_SUPPLEMENTARY_A = 57,
   U_HANGUL_SUPPLEMENTARY_B = 58,
   U_CJK_UNIFIED_IDEOGRAPHS = 59,
   U_PRIVATE_USE_AREA = 60,
   U_CJK_COMPATIBILITY_IDEOGRAPHS = 61,
   U_ALPHABETIC_PRESENTATION_FORMS = 62,
   U_ARABIC_PRESENTATION_FORMS_A = 63,
   U_COMBINING_HALF_MARKS = 64,
   U_CJK_COMPATIBILITY_FORMS = 65,
   U_SMALL_FORM_VARIANTS = 66,
   U_ARABIC_PRESENTATION_FORMS_B = 67,
   U_SPECIALS = 69,
   U_HALFWIDTH_AND_FULLWIDTH_FORMS = 68,
   U_LAST_PLUS_ONE
} unicode_subset;

static struct {
   wchar_t low, high;
   unicode_subset bit;
   gchar *name;
} utab[] = {
   {
   0x0000, 0x007E, U_BASIC_LATIN, "Basic Latin"}, {
   0x00A0, 0x00FF, U_LATIN_1_SUPPLEMENT, "Latin-1 Supplement"}, {
   0x0100, 0x017F, U_LATIN_EXTENDED_A, "Latin Extended-A"}, {
   0x0180, 0x024F, U_LATIN_EXTENDED_B, "Latin Extended-B"}, {
   0x0250, 0x02AF, U_IPA_EXTENSIONS, "IPA Extensions"}, {
   0x02B0, 0x02FF,
          U_SPACING_MODIFIER_LETTERS, "Spacing Modifier Letters"}, {
   0x0300, 0x036F,
          U_COMBINING_DIACRITICAL_MARKS, "Combining Diacritical Marks"}, {
   0x0370, 0x03CF, U_BASIC_GREEK, "Basic Greek"}, {
   0x03D0, 0x03FF,
          U_GREEK_SYMBOLS_AND_COPTIC, "Greek Symbols and Coptic"}, {
   0x0400, 0x04FF, U_CYRILLIC, "Cyrillic"}, {
   0x0530, 0x058F, U_ARMENIAN, "Armenian"}, {
   0x0590, 0x05CF, U_HEBREW_EXTENDED, "Hebrew Extended"}, {
   0x05D0, 0x05FF, U_BASIC_HEBREW, "Basic Hebrew"}, {
   0x0600, 0x0652, U_BASIC_ARABIC, "Basic Arabic"}, {
   0x0653, 0x06FF, U_ARABIC_EXTENDED, "Arabic Extended"}, {
   0x0900, 0x097F, U_DEVANAGARI, "Devanagari"}, {
   0x0980, 0x09FF, U_BENGALI, "Bengali"}, {
   0x0A00, 0x0A7F, U_GURMUKHI, "Gurmukhi"}, {
   0x0A80, 0x0AFF, U_GUJARATI, "Gujarati"}, {
   0x0B00, 0x0B7F, U_ORIYA, "Oriya"}, {
   0x0B80, 0x0BFF, U_TAMIL, "Tamil"}, {
   0x0C00, 0x0C7F, U_TELUGU, "Telugu"}, {
   0x0C80, 0x0CFF, U_KANNADA, "Kannada"}, {
   0x0D00, 0x0D7F, U_MALAYALAM, "Malayalam"}, {
   0x0E00, 0x0E7F, U_THAI, "Thai"}, {
   0x0E80, 0x0EFF, U_LAO, "Lao"}, {
   0x10A0, 0x10CF, U_GEORGIAN_EXTENDED, "Georgian Extended"}, {
   0x10D0, 0x10FF, U_BASIC_GEORGIAN, "Basic Georgian"}, {
   0x1100, 0x11FF, U_HANGUL_JAMO, "Hangul Jamo"}, {
   0x1E00, 0x1EFF,
          U_LATIN_EXTENDED_ADDITIONAL, "Latin Extended Additional"}, {
   0x1F00, 0x1FFF, U_GREEK_EXTENDED, "Greek Extended"}, {
   0x2000, 0x206F, U_GENERAL_PUNCTUATION, "General Punctuation"}, {
   0x2070, 0x209F,
          U_SUPERSCRIPTS_AND_SUBSCRIPTS, "Superscripts and Subscripts"}, {
   0x20A0, 0x20CF, U_CURRENCY_SYMBOLS, "Currency Symbols"}, {
   0x20D0, 0x20FF,
          U_COMBINING_DIACRITICAL_MARKS_FOR_SYMBOLS,
          "Combining Diacritical Marks for Symbols"}, {
   0x2100, 0x214F, U_LETTERLIKE_SYMBOLS, "Letterlike Symbols"}, {
   0x2150, 0x218F, U_NUMBER_FORMS, "Number Forms"}, {
   0x2190, 0x21FF, U_ARROWS, "Arrows"}, {
   0x2200, 0x22FF, U_MATHEMATICAL_OPERATORS, "Mathematical Operators"}, {
   0x2300, 0x23FF, U_MISCELLANEOUS_TECHNICAL, "Miscellaneous Technical"}, {
   0x2400, 0x243F, U_CONTROL_PICTURES, "Control Pictures"}, {
   0x2440, 0x245F,
          U_OPTICAL_CHARACTER_RECOGNITION,
          "Optical Character Recognition"}, {
   0x2460, 0x24FF, U_ENCLOSED_ALPHANUMERICS, "Enclosed Alphanumerics"}, {
   0x2500, 0x257F, U_BOX_DRAWING, "Box Drawing"}, {
   0x2580, 0x259F, U_BLOCK_ELEMENTS, "Block Elements"}, {
   0x25A0, 0x25FF, U_GEOMETRIC_SHAPES, "Geometric Shapes"}, {
   0x2600, 0x26FF, U_MISCELLANEOUS_SYMBOLS, "Miscellaneous Symbols"}, {
   0x2700, 0x27BF, U_DINGBATS, "Dingbats"}, {
   0x3000, 0x303F,
          U_CJK_SYMBOLS_AND_PUNCTUATION, "CJK Symbols and Punctuation"}, {
   0x3040, 0x309F, U_HIRAGANA, "Hiragana"}, {
   0x30A0, 0x30FF, U_KATAKANA, "Katakana"}, {
   0x3100, 0x312F, U_BOPOMOFO, "Bopomofo"}, {
   0x3130, 0x318F,
          U_HANGUL_COMPATIBILITY_JAMO, "Hangul Compatibility Jamo"}, {
   0x3190, 0x319F, U_CJK_MISCELLANEOUS, "CJK Miscellaneous"}, {
   0x3200, 0x32FF, U_ENCLOSED_CJK, "Enclosed CJK"}, {
   0x3300, 0x33FF, U_CJK_COMPATIBILITY, "CJK Compatibility"},
       /* The book claims:
        * U+3400..U+3D2D Hangul
        * U+3D2E..U+44B7 Hangul Supplementary A
        * U+44B8..U+4DFF Hangul Supplementary B
        * but actually in Unicode
        * U+3400..U+4DB5 is CJK Unified Ideographs Extension A
        */
   {
   0x3400, 0x4DB5,
          U_CJK_UNIFIED_IDEOGRAPHS, "CJK Unified Ideographs Extension A"},
   {
   0x4E00, 0x9FFF, U_CJK_UNIFIED_IDEOGRAPHS, "CJK Unified Ideographs"},
       /* This was missing completely from the book's table. */
   {
   0xAC00, 0xD7A3, U_HANGUL, "Hangul Syllables"}, {
   0xE000, 0xF8FF, U_PRIVATE_USE_AREA, "Private Use Area"}, {
   0xF900, 0xFAFF,
          U_CJK_COMPATIBILITY_IDEOGRAPHS, "CJK Compatibility Ideographs"},
   {
   0xFB00, 0xFB4F,
          U_ALPHABETIC_PRESENTATION_FORMS,
          "Alphabetic Presentation Forms"}, {
   0xFB50, 0xFDFF, U_ARABIC_PRESENTATION_FORMS_A,
          "Arabic Presentation Forms-A"}, {
   0xFE20, 0xFE2F, U_COMBINING_HALF_MARKS, "Combining Half Marks"}, {
   0xFE30, 0xFE4F, U_CJK_COMPATIBILITY_FORMS, "CJK Compatibility Forms"}, {
   0xFE50, 0xFE6F, U_SMALL_FORM_VARIANTS, "Small Form Variants"}, {
   0xFE70, 0xFEFE,
          U_ARABIC_PRESENTATION_FORMS_B, "Arabic Presentation Forms-B"}, {
   0xFEFF, 0xFEFF, U_SPECIALS, "Specials"}, {
   0xFF00, 0xFFEF,
          U_HALFWIDTH_AND_FULLWIDTH_FORMS,
          "Halfwidth and Fullwidth Forms"}, {
   0xFFF0, 0xFFFD, U_SPECIALS, "Specials"}
};

static void print_unicode_subranges(FONTSIGNATURE * fsp)
{
   int i;
   gboolean checked[sizeof(utab) / sizeof(utab[0])];
   gboolean need_comma = FALSE;

   memset(checked, 0, sizeof(checked));

   for (i = 0; i < sizeof(utab) / sizeof(utab[0]); i++)
      if (!checked[i]
          && (fsp->fsUsb[utab[i].bit / 32] & (1 << (utab[i].bit % 32)))) {
         g_print("%s %s", (need_comma ? "," : ""), utab[i].name);
         need_comma = TRUE;
         checked[i] = TRUE;
      }
   if (!need_comma)
      g_print(" none!");
   g_print("\n");
}

static gboolean check_unicode_subranges(UINT charset, FONTSIGNATURE * fsp)
{
   gint i;
   gboolean retval = FALSE;

   /* If the fsUsb bit array has at least one of the bits set, trust it */
   for (i = 0; i < U_LAST_PLUS_ONE; i++)
      if (i != U_PRIVATE_USE_AREA
          && (fsp->fsUsb[i / 32] & (1 << (i % 32))))
         return FALSE;

   /* Otherwise, guess what subranges there should be in the font */
   fsp->fsUsb[0] = fsp->fsUsb[1] = fsp->fsUsb[2] = fsp->fsUsb[3] = 0;

#define set_bit(bitno) (fsp->fsUsb[(bitno)/32] |= (1 << ((bitno) % 32)))

   /* Set Unicode subrange bits based on code pages supported.
    * This is mostly just guesswork.
    */

#define check_cp(bit) (fsp->fsCsb[0] & (bit))

   if (check_cp(FS_LATIN1)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_CURRENCY_SYMBOLS);
      retval = TRUE;
   }
   if (check_cp(FS_LATIN2)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_CURRENCY_SYMBOLS);
      retval = TRUE;
   }
   if (check_cp(FS_CYRILLIC)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_CYRILLIC);
      retval = TRUE;
   }
   if (check_cp(FS_GREEK)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_BASIC_GREEK);
      retval = TRUE;
   }
   if (check_cp(FS_TURKISH)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_CURRENCY_SYMBOLS);
      retval = TRUE;
   }
   if (check_cp(FS_HEBREW)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_BASIC_HEBREW);
      set_bit(U_HEBREW_EXTENDED);
      retval = TRUE;
   }
   if (check_cp(FS_ARABIC)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_BASIC_ARABIC);
      set_bit(U_ARABIC_EXTENDED);
      retval = TRUE;
   }
   if (check_cp(FS_BALTIC)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      retval = TRUE;
   }
   if (check_cp(FS_VIETNAMESE)) {
      /* ??? */
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_LATIN_EXTENDED_ADDITIONAL);
      retval = TRUE;
   }
   if (check_cp(FS_THAI)) {
      set_bit(U_BASIC_LATIN);
      set_bit(U_THAI);
      retval = TRUE;
   }
   if (check_cp(FS_JISJAPAN)) {
      /* Based on MS Gothic */
      set_bit(U_BASIC_LATIN);
      set_bit(U_CJK_SYMBOLS_AND_PUNCTUATION);
      set_bit(U_HIRAGANA);
      set_bit(U_KATAKANA);
      set_bit(U_CJK_UNIFIED_IDEOGRAPHS);
      set_bit(U_HALFWIDTH_AND_FULLWIDTH_FORMS);
      retval = TRUE;
   }
   if (check_cp(FS_CHINESESIMP)) {
      /* Based on MS Hei */
      set_bit(U_BASIC_LATIN);
      set_bit(U_HIRAGANA);
      set_bit(U_KATAKANA);
      set_bit(U_BOPOMOFO);
      set_bit(U_CJK_UNIFIED_IDEOGRAPHS);
      retval = TRUE;
   }
   if (check_cp(FS_WANSUNG)
       || check_cp(FS_JOHAB)) { /* ??? */
      /* Based on GulimChe. I wonder if all Korean fonts
       * really support this large range of Unicode subranges?
       */
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_BASIC_GREEK);
      set_bit(U_CYRILLIC);
      set_bit(U_HANGUL_JAMO);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_NUMBER_FORMS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_ENCLOSED_ALPHANUMERICS);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      set_bit(U_CJK_SYMBOLS_AND_PUNCTUATION);
      set_bit(U_HIRAGANA);
      set_bit(U_KATAKANA);
      set_bit(U_HANGUL_COMPATIBILITY_JAMO);
      set_bit(U_ENCLOSED_CJK);
      set_bit(U_CJK_COMPATIBILITY_FORMS);
      set_bit(U_HANGUL);
      set_bit(U_CJK_UNIFIED_IDEOGRAPHS);
      set_bit(U_CJK_COMPATIBILITY_IDEOGRAPHS);
      set_bit(U_HALFWIDTH_AND_FULLWIDTH_FORMS);
      retval = TRUE;
   }
   if (check_cp(FS_CHINESETRAD)) {
      /* Based on MingLiU */
      set_bit(U_BASIC_LATIN);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_CJK_SYMBOLS_AND_PUNCTUATION);
      set_bit(U_BOPOMOFO);
      set_bit(U_CJK_UNIFIED_IDEOGRAPHS);
      set_bit(U_CJK_COMPATIBILITY_IDEOGRAPHS);
      set_bit(U_SMALL_FORM_VARIANTS);
      set_bit(U_HALFWIDTH_AND_FULLWIDTH_FORMS);
      retval = TRUE;
   }
   if (check_cp(FS_SYMBOL) || charset == MAC_CHARSET) {
      /* Non-Unicode encoding, I guess. Pretend it covers
       * the single-byte range of values.
       */
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      retval = TRUE;
   }

   if (retval)
      return TRUE;

   GDK_NOTE(MISC, g_print("... No code page bits set!\n"));

   /* Sigh. Not even any code page bits were set. Guess based on
    * charset, then. These somewhat optimistic guesses are based on the
    * table in Appendix M in the book "Developing ..."  mentioned
    * above.
    */
   switch (charset) {
   case ANSI_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_COMBINING_DIACRITICAL_MARKS);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_CURRENCY_SYMBOLS);
#if 0                           /* I find this too hard to believe... */
      set_bit(U_BASIC_GREEK);
      set_bit(U_CYRILLIC);
      set_bit(U_BASIC_HEBREW);
      set_bit(U_HEBREW_EXTENDED);
      set_bit(U_BASIC_ARABIC);
      set_bit(U_ARABIC_EXTENDED);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_NUMBER_FORMS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_ENCLOSED_ALPHANUMERICS);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      set_bit(U_HIRAGANA);
      set_bit(U_KATAKANA);
      set_bit(U_BOPOMOFO);
      set_bit(U_HANGUL_COMPATIBILITY_JAMO);
      set_bit(U_CJK_MISCELLANEOUS);
      set_bit(U_CJK_COMPATIBILITY);
      set_bit(U_HANGUL);
      set_bit(U_HANGUL_SUPPLEMENTARY_A);
      set_bit(U_CJK_COMPATIBILITY_IDEOGRAPHS);
      set_bit(U_ALPHABETIC_PRESENTATION_FORMS);
      set_bit(U_SMALL_FORM_VARIANTS);
      set_bit(U_ARABIC_PRESENTATION_FORMS_B);
      set_bit(U_HALFWIDTH_AND_FULLWIDTH_FORMS);
      set_bit(U_SPECIALS);
#endif
      retval = TRUE;
      break;
   case SYMBOL_CHARSET:
      /* Unggh */
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      retval = TRUE;
      break;
   case SHIFTJIS_CHARSET:
   case HANGEUL_CHARSET:
   case GB2312_CHARSET:
   case CHINESEBIG5_CHARSET:
   case JOHAB_CHARSET:
      /* The table really does claim these "locales" (it doesn't
       * talk about charsets per se) cover the same Unicode
       * subranges
       */
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_COMBINING_DIACRITICAL_MARKS_FOR_SYMBOLS);
      set_bit(U_BASIC_GREEK);
      set_bit(U_CYRILLIC);
      set_bit(U_HANGUL_JAMO);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_NUMBER_FORMS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_ENCLOSED_ALPHANUMERICS);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      set_bit(U_CJK_SYMBOLS_AND_PUNCTUATION);
      set_bit(U_HIRAGANA);
      set_bit(U_KATAKANA);
      set_bit(U_BOPOMOFO);
      set_bit(U_HANGUL_COMPATIBILITY_JAMO);
      set_bit(U_CJK_MISCELLANEOUS);
      set_bit(U_CJK_COMPATIBILITY);
      set_bit(U_HANGUL);
      set_bit(U_HANGUL_SUPPLEMENTARY_A);
      set_bit(U_CJK_UNIFIED_IDEOGRAPHS);
      set_bit(U_CJK_COMPATIBILITY_IDEOGRAPHS);
      set_bit(U_ALPHABETIC_PRESENTATION_FORMS);
      set_bit(U_SMALL_FORM_VARIANTS);
      set_bit(U_ARABIC_PRESENTATION_FORMS_B);
      set_bit(U_SPECIALS);
      retval = TRUE;
      break;
   case HEBREW_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_BASIC_HEBREW);
      set_bit(U_HEBREW_EXTENDED);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_LETTERLIKE_SYMBOLS);
      retval = TRUE;
      break;
   case ARABIC_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_BASIC_GREEK);
      set_bit(U_BASIC_ARABIC);
      set_bit(U_ARABIC_EXTENDED);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      set_bit(U_HALFWIDTH_AND_FULLWIDTH_FORMS);
      retval = TRUE;
      break;
   case GREEK_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_BASIC_GREEK);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      retval = TRUE;
      break;
   case TURKISH_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_BASIC_GREEK);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      retval = TRUE;
      break;
   case VIETNAMESE_CHARSET:
   case THAI_CHARSET:
      /* These are not in the table, so I have no idea */
      break;
   case BALTIC_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_BASIC_GREEK);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      retval = TRUE;
      break;
   case EASTEUROPE_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_LATIN_EXTENDED_A);
      set_bit(U_LATIN_EXTENDED_B);
      set_bit(U_SPACING_MODIFIER_LETTERS);
      set_bit(U_BASIC_GREEK);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_SUPERSCRIPTS_AND_SUBSCRIPTS);
      set_bit(U_CURRENCY_SYMBOLS);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      retval = TRUE;
      break;
   case RUSSIAN_CHARSET:
      set_bit(U_BASIC_LATIN);
      set_bit(U_LATIN_1_SUPPLEMENT);
      set_bit(U_CYRILLIC);
      set_bit(U_GENERAL_PUNCTUATION);
      set_bit(U_LETTERLIKE_SYMBOLS);
      set_bit(U_ARROWS);
      set_bit(U_MATHEMATICAL_OPERATORS);
      set_bit(U_MISCELLANEOUS_TECHNICAL);
      set_bit(U_BOX_DRAWING);
      set_bit(U_BLOCK_ELEMENTS);
      set_bit(U_GEOMETRIC_SHAPES);
      set_bit(U_MISCELLANEOUS_SYMBOLS);
      retval = TRUE;
      break;
   }

#undef set_bit
   return retval;
}

//-----------------------------------------------------------------------------

GdkWin32SingleFont *gdk_font_load_internal(const gchar * font_name)
{
   GdkWin32SingleFont *singlefont;
   HFONT hfont;
   LOGFONT logfont;
   CHARSETINFO csi;
   DWORD fdwItalic, fdwUnderline, fdwStrikeOut, fdwCharSet,
       fdwOutputPrecision, fdwClipPrecision, fdwQuality, fdwPitchAndFamily;
   HGDIOBJ oldfont;
   char *lpszFace;
   gchar face[100];

   int numfields, n1, n2, tries;
   char foundry[32], family[100], weight[32], slant[32], set_width[32],
       spacing[32], registry[32], encoding[32];
   char pixel_size[10], point_size[10], res_x[10], res_y[10],
       avg_width[10];
   int c;
   char *p;
   int nHeight, nWidth, nEscapement, nOrientation, fnWeight;
   int logpixelsy;

   g_return_val_if_fail(font_name != NULL, NULL);

   GDK_NOTE(MISC, g_print("gdk_font_load_internal: %s\n", font_name));

   numfields = sscanf(font_name,
                      "-%30[^-]-%100[^-]-%30[^-]-%30[^-]-%30[^-]-%n",
                      foundry, family, weight, slant, set_width, &n1);
   if (numfields == 0) {
      /* Probably a plain Windows font name */
      nHeight = 0;
      nWidth = 0;
//      nEscapement = 0;
//      nOrientation = 0;
      gfont_angle = atoi(set_width);
      nOrientation = (int) gfont_angle;
      nEscapement = (int) gfont_angle;
      fnWeight = FW_DONTCARE;
      fdwItalic = FALSE;
      fdwUnderline = FALSE;
      fdwStrikeOut = FALSE;
      fdwCharSet = ANSI_CHARSET;
      fdwOutputPrecision = OUT_TT_PRECIS;
      fdwClipPrecision = CLIP_DEFAULT_PRECIS;
      fdwQuality = PROOF_QUALITY;
      fdwPitchAndFamily = DEFAULT_PITCH;
      lpszFace = g_filename_from_utf8(font_name, -1, NULL, NULL, NULL);
   } else if (numfields != 5) {
      g_warning("gdk_font_load: font name %s illegal", font_name);
      return NULL;
   } else {
      /* It must be a XLFD name */

      /* Check for hex escapes in the font family,
       * put in there by logfont_to_xlfd. Convert them in-place.
       */
      p = family;
      while (*p) {
         if (*p == '%' && isxdigit(p[1]) && isxdigit(p[2])) {
            sscanf(p + 1, "%2x", &c);
            *p = c;
            strcpy(p + 1, p + 3);
         }
         p++;
      }

      /* Skip add_style which often is empty in the requested font name */
      while (font_name[n1] && font_name[n1] != '-')
         n1++;
      numfields++;

      numfields += sscanf(font_name + n1,
                          "-%8[^-]-%8[^-]-%8[^-]-%8[^-]-%30[^-]-%8[^-]-%30[^-]-%30[^-]%n",
                          pixel_size,
                          point_size,
                          res_x,
                          res_y,
                          spacing, avg_width, registry, encoding, &n2);

      if (numfields != 14 || font_name[n1 + n2] != '\0') {
         g_warning("gdk_font_load: font name %s illegal", font_name);
         return NULL;
      }

      logpixelsy = GetDeviceCaps(gdk_DC, LOGPIXELSY);

      if (strcmp(pixel_size, "*") == 0) {
         if (strcmp(point_size, "*") == 0) {
            nHeight = 0;
         }
         else {
            nHeight = -MulDiv(atoi(point_size), GetDeviceCaps(gdk_DC, LOGPIXELSY), 72);
//                (int) (((double) atoi(point_size)) / 720. * logpixelsy);
         }
      }
      else {
         nHeight = -1 * atoi(pixel_size);
      }

      nWidth = 0;
//      nEscapement = 0;
//      nOrientation = 0;
      gfont_angle = atoi(avg_width);
      nOrientation = (int) gfont_angle;	// bb change
      nEscapement = (int) gfont_angle;	// bb change

      if (g_strcasecmp(weight, "thin") == 0)
         fnWeight = FW_THIN;
      else if (g_strcasecmp(weight, "extralight") == 0)
         fnWeight = FW_EXTRALIGHT;
      else if (g_strcasecmp(weight, "ultralight") == 0)
#ifdef FW_ULTRALIGHT
         fnWeight = FW_ULTRALIGHT;
#else
         fnWeight = FW_EXTRALIGHT;	/* In fact, FW_ULTRALIGHT really is 
	 * defined as FW_EXTRALIGHT anyway.
	 */
#endif
      else if (g_strcasecmp(weight, "light") == 0)
         fnWeight = FW_LIGHT;
      else if (g_strcasecmp(weight, "normal") == 0)
         fnWeight = FW_NORMAL;
      else if (g_strcasecmp(weight, "regular") == 0)
         fnWeight = FW_REGULAR;
      else if (g_strcasecmp(weight, "medium") == 0)
         fnWeight = FW_MEDIUM;
      else if (g_strcasecmp(weight, "semibold") == 0)
         fnWeight = FW_SEMIBOLD;
      else if (g_strcasecmp(weight, "demibold") == 0)
#ifdef FW_DEMIBOLD
         fnWeight = FW_DEMIBOLD;
#else
         fnWeight = FW_SEMIBOLD;	/* As above */
#endif
      else if (g_strcasecmp(weight, "bold") == 0)
         fnWeight = FW_BOLD;
      else if (g_strcasecmp(weight, "extrabold") == 0)
         fnWeight = FW_EXTRABOLD;
      else if (g_strcasecmp(weight, "ultrabold") == 0)
#ifdef FW_ULTRABOLD
         fnWeight = FW_ULTRABOLD;
#else
         fnWeight = FW_EXTRABOLD;	/* As above */
#endif
      else if (g_strcasecmp(weight, "heavy") == 0)
         fnWeight = FW_HEAVY;
      else if (g_strcasecmp(weight, "black") == 0)
#ifdef FW_BLACK
         fnWeight = FW_BLACK;
#else
         fnWeight = FW_HEAVY;   /* As above */
#endif
      else
         fnWeight = FW_DONTCARE;

      if (g_strcasecmp(slant, "italic") == 0
          || g_strcasecmp(slant, "oblique") == 0
          || g_strcasecmp(slant, "i") == 0
          || g_strcasecmp(slant, "o") == 0)
         fdwItalic = TRUE;
      else
         fdwItalic = FALSE;
      fdwUnderline = FALSE;
      fdwStrikeOut = FALSE;
      if (g_strcasecmp(registry, "iso8859") == 0)
         if (strcmp(encoding, "1") == 0)
            fdwCharSet = ANSI_CHARSET;
         else if (strcmp(encoding, "2") == 0)
            fdwCharSet = EASTEUROPE_CHARSET;
         else if (strcmp(encoding, "7") == 0)
            fdwCharSet = GREEK_CHARSET;
         else if (strcmp(encoding, "8") == 0)
            fdwCharSet = HEBREW_CHARSET;
         else if (strcmp(encoding, "9") == 0)
            fdwCharSet = TURKISH_CHARSET;
         else
            fdwCharSet = ANSI_CHARSET;	/* XXX ??? */
      else if (g_strcasecmp(registry, "jisx0208.1983") == 0)
         fdwCharSet = SHIFTJIS_CHARSET;
      else if (g_strcasecmp(registry, "ksc5601.1987") == 0)
         fdwCharSet = HANGEUL_CHARSET;
      else if (g_strcasecmp(registry, "gb2312.1980") == 0)
         fdwCharSet = GB2312_CHARSET;
      else if (g_strcasecmp(registry, "big5") == 0)
         fdwCharSet = CHINESEBIG5_CHARSET;
      else if (g_strcasecmp(registry, "windows") == 0
               || g_strcasecmp(registry, "microsoft") == 0)
         if (g_strcasecmp(encoding, "symbol") == 0)
            fdwCharSet = SYMBOL_CHARSET;
         else if (g_strcasecmp(encoding, "shiftjis") == 0)
            fdwCharSet = SHIFTJIS_CHARSET;
         else if (g_strcasecmp(encoding, "gb2312") == 0)
            fdwCharSet = GB2312_CHARSET;
         else if (g_strcasecmp(encoding, "hangeul") == 0)
            fdwCharSet = HANGEUL_CHARSET;
         else if (g_strcasecmp(encoding, "big5") == 0)
            fdwCharSet = CHINESEBIG5_CHARSET;
         else if (g_strcasecmp(encoding, "johab") == 0)
            fdwCharSet = JOHAB_CHARSET;
         else if (g_strcasecmp(encoding, "hebrew") == 0)
            fdwCharSet = HEBREW_CHARSET;
         else if (g_strcasecmp(encoding, "arabic") == 0)
            fdwCharSet = ARABIC_CHARSET;
         else if (g_strcasecmp(encoding, "greek") == 0)
            fdwCharSet = GREEK_CHARSET;
         else if (g_strcasecmp(encoding, "turkish") == 0)
            fdwCharSet = TURKISH_CHARSET;
         else if (g_strcasecmp(encoding, "easteurope") == 0)
            fdwCharSet = EASTEUROPE_CHARSET;
         else if (g_strcasecmp(encoding, "russian") == 0)
            fdwCharSet = RUSSIAN_CHARSET;
         else if (g_strcasecmp(encoding, "mac") == 0)
            fdwCharSet = MAC_CHARSET;
         else if (g_strcasecmp(encoding, "baltic") == 0)
            fdwCharSet = BALTIC_CHARSET;
         else if (g_strcasecmp(encoding, "cp1251") == 0)
            fdwCharSet = RUSSIAN_CHARSET;
         else
            fdwCharSet = ANSI_CHARSET;	/* XXX ??? */
      else
         fdwCharSet = ANSI_CHARSET;	/* XXX ??? */
      fdwOutputPrecision = OUT_TT_PRECIS;	//OUT_TT_ONLY_PRECIS;
      fdwClipPrecision = CLIP_DEFAULT_PRECIS;
      fdwQuality = PROOF_QUALITY;	//ANTIALIASED_QUALITY;
      if (g_strcasecmp(spacing, "m") == 0)
         fdwPitchAndFamily = FIXED_PITCH;
      else if (g_strcasecmp(spacing, "p") == 0)
         fdwPitchAndFamily = VARIABLE_PITCH;
      else
         fdwPitchAndFamily = DEFAULT_PITCH;
      lpszFace = g_filename_from_utf8(family, -1, NULL, NULL, NULL);
   }

   for (tries = 0;; tries++) {
      GDK_NOTE(MISC, g_print("... trying CreateFont(%d,%d,%d,%d,"
                             "%d,%d,%d,%d,"
                             "%d,%d,%d,"
                             "%d,%#.02x,\"%s\")\n",
                             nHeight, nWidth, nEscapement, nOrientation,
                             fnWeight, fdwItalic, fdwUnderline,
                             fdwStrikeOut, fdwCharSet, fdwOutputPrecision,
                             fdwClipPrecision, fdwQuality,
                             fdwPitchAndFamily, lpszFace));
      hfont =
          CreateFont(nHeight, nWidth, nEscapement, nOrientation, fnWeight,
                     fdwItalic, fdwUnderline, fdwStrikeOut, fdwCharSet,
                     fdwOutputPrecision, fdwClipPrecision, fdwQuality,
                     fdwPitchAndFamily, lpszFace);
      /* After the first try lpszFace contains a return value
       * from g_filename_from_utf8(), so free it.
       */
      if (tries == 0)
         g_free(lpszFace);

      if (hfont != NULL)
         break;

      /* If we fail, try some similar fonts often found on Windows. */
      if (tries == 0) {
         if (g_strcasecmp(family, "helvetica") == 0)
            lpszFace = "arial";
         else if (g_strcasecmp(family, "new century schoolbook") == 0)
            lpszFace = "century schoolbook";
         else if (g_strcasecmp(family, "courier") == 0)
            lpszFace = "courier new";
         else if (g_strcasecmp(family, "lucida") == 0)
            lpszFace = "lucida sans unicode";
         else if (g_strcasecmp(family, "lucidatypewriter") == 0)
            lpszFace = "lucida console";
         else if (g_strcasecmp(family, "times") == 0)
            lpszFace = "times new roman";
      } else if (tries == 1) {
         if (g_strcasecmp(family, "courier") == 0) {
            lpszFace = "";
            fdwPitchAndFamily |= FF_MODERN;
         } else if (g_strcasecmp(family, "times new roman") == 0) {
            lpszFace = "";
            fdwPitchAndFamily |= FF_ROMAN;
         } else if (g_strcasecmp(family, "helvetica") == 0
                    || g_strcasecmp(family, "lucida") == 0) {
            lpszFace = "";
            fdwPitchAndFamily |= FF_SWISS;
         } else {
            lpszFace = "";
            fdwPitchAndFamily = (fdwPitchAndFamily & 0x0F) | FF_DONTCARE;
         }
      } else
         break;
      tries++;
   }

   if (!hfont)
      return NULL;

   singlefont = g_new(GdkWin32SingleFont, 1);
   singlefont->xfont = hfont;
   GetObject(singlefont->xfont, sizeof(logfont), &logfont);
   oldfont = SelectObject(gdk_DC, singlefont->xfont);
   memset(&singlefont->fs, 0, sizeof(singlefont->fs));
   singlefont->charset = GetTextCharsetInfo(gdk_DC, &singlefont->fs, 0);
   GetTextFace(gdk_DC, sizeof(face), face);
   SelectObject(gdk_DC, oldfont);
   if (TranslateCharsetInfo((DWORD *) singlefont->charset, &csi,
                            TCI_SRCCHARSET)
       && singlefont->charset != MAC_CHARSET)
      singlefont->codepage = csi.ciACP;
   else
      singlefont->codepage = 0;

   GDK_NOTE(MISC, (g_print("... = %#x %s cs %s cp%d\n",
                           singlefont->xfont, face,
                           charset_name(singlefont->charset),
                           singlefont->codepage),
                   g_print("... Unicode subranges:"),
                   print_unicode_subranges(&singlefont->fs)));
   if (check_unicode_subranges(singlefont->charset, &singlefont->fs))
      GDK_NOTE(MISC, (g_print("... Guesstimated Unicode subranges:"),
                      print_unicode_subranges(&singlefont->fs)));

   return singlefont;
}

GdkFont *gdk_font_load(const gchar * font_name)
{
   GdkFont *font;
   GdkFontPrivateWin32 *private;
   GdkWin32SingleFont *singlefont;
   HGDIOBJ oldfont;
   TEXTMETRIC textmetric;

   g_return_val_if_fail(font_name != NULL, NULL);

   font = gdk_font_hash_lookup(GDK_FONT_FONTSET, font_name);
   if (font)
      return font;

   singlefont = gdk_font_load_internal(font_name);

   private = g_new(GdkFontPrivateWin32, 1);
   font = (GdkFont *) private;

   private->base.ref_count = 1;
   private->names = NULL;
   private->fonts = g_slist_append(NULL, singlefont);

   /* Pretend all fonts are fontsets... Gtktext and gtkentry work better
    * that way, they use wide chars, which is necessary for non-ASCII
    * chars to work. (Yes, even Latin-1, as we use Unicode internally.)
    */
   font->type = GDK_FONT_FONTSET;
   oldfont = SelectObject(gdk_DC, singlefont->xfont);
   GetTextMetrics(gdk_DC, &textmetric);
   SelectObject(gdk_DC, oldfont);
   font->ascent = textmetric.tmAscent;
   font->descent = textmetric.tmDescent;

   GDK_NOTE(MISC, g_print("... asc %d desc %d\n",
                          font->ascent, font->descent));

   gdk_font_hash_insert(GDK_FONT_FONTSET, font, font_name);

   return font;
}

GdkFont *gdk_fontset_load(const gchar * fontset_name)
{
   GdkFont *font;
   GdkFontPrivateWin32 *private;
   GdkWin32SingleFont *singlefont;
   HGDIOBJ oldfont;
   TEXTMETRIC textmetric;
   GSList *base_font_list = NULL;
   gchar *fs;
   gchar *b, *p, *s;

   g_return_val_if_fail(fontset_name != NULL, NULL);

   font = gdk_font_hash_lookup(GDK_FONT_FONTSET, fontset_name);
   if (font)
      return font;

   s = fs = g_strdup(fontset_name);
   while (*s && isspace(*s))
      s++;

   g_return_val_if_fail(*s, NULL);

   private = g_new(GdkFontPrivateWin32, 1);
   font = (GdkFont *) private;

   private->base.ref_count = 1;
   private->names = NULL;
   private->fonts = NULL;

   font->type = GDK_FONT_FONTSET;
   font->ascent = 0;
   font->descent = 0;

   while (TRUE) {
      if ((p = strchr(s, ',')) != NULL)
         b = p;
      else
         b = s + strlen(s);

      while (isspace(b[-1]))
         b--;
      *b = '\0';
      singlefont = gdk_font_load_internal(s);
      if (singlefont) {
         private->fonts = g_slist_append(private->fonts, singlefont);
         oldfont = SelectObject(gdk_DC, singlefont->xfont);
         GetTextMetrics(gdk_DC, &textmetric);
         SelectObject(gdk_DC, oldfont);
         font->ascent = MAX(font->ascent, textmetric.tmAscent);
         font->descent = MAX(font->descent, textmetric.tmDescent);
      }
      if (p) {
         s = p + 1;
         while (*s && isspace(*s))
            s++;
      } else
         break;
      if (!*s)
         break;
   }

   g_free(fs);

   gdk_font_hash_insert(GDK_FONT_FONTSET, font, fontset_name);

   return font;
}

void _gdk_font_destroy(GdkFont * font)
{
   GdkFontPrivateWin32 *private = (GdkFontPrivateWin32 *) font;
   GdkWin32SingleFont *singlefont;
   GSList *list;

   singlefont = (GdkWin32SingleFont *) private->fonts->data;
   GDK_NOTE(MISC, g_print("_gdk_font_destroy %#x\n", singlefont->xfont));

   gdk_font_hash_remove(font->type, font);

   switch (font->type) {
   case GDK_FONT_FONT:
      DeleteObject(singlefont->xfont);
      break;

   case GDK_FONT_FONTSET:
      list = private->fonts;
      while (list) {
         singlefont = (GdkWin32SingleFont *) list->data;
         DeleteObject(singlefont->xfont);

         list = list->next;
      }
      g_slist_free(private->fonts);
      break;
   }
   g_free(font);
}

gint _gdk_font_strlen(GdkFont * font, const gchar * str)
{
   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(str != NULL, -1);

   return strlen(str);
}

glong gdk_font_id(const GdkFont * font)
{
   const GdkFontPrivateWin32 *private;

   g_return_val_if_fail(font != NULL, 0);

   private = (const GdkFontPrivateWin32 *) font;

   if (font->type == GDK_FONT_FONT)
      return (glong) ((GdkWin32SingleFont *) private->fonts->data)->xfont;
   else
      return 0;
}

gint gdk_font_equal(const GdkFont * fonta, const GdkFont * fontb)
{
   const GdkFontPrivateWin32 *privatea;
   const GdkFontPrivateWin32 *privateb;

   g_return_val_if_fail(fonta != NULL, FALSE);
   g_return_val_if_fail(fontb != NULL, FALSE);

   privatea = (const GdkFontPrivateWin32 *) fonta;
   privateb = (const GdkFontPrivateWin32 *) fontb;

   if (fonta->type == GDK_FONT_FONT && fontb->type == GDK_FONT_FONT)
      return (((GdkWin32SingleFont *) privatea->fonts->data)->xfont
              == ((GdkWin32SingleFont *) privateb->fonts->data)->xfont);
   else if (fonta->type == GDK_FONT_FONTSET
            && fontb->type == GDK_FONT_FONTSET) {
      GSList *lista = privatea->fonts;
      GSList *listb = privateb->fonts;

      while (lista && listb) {
         if (((GdkWin32SingleFont *) lista->data)->xfont
             != ((GdkWin32SingleFont *) listb->data)->xfont)
            return 0;
         lista = lista->next;
         listb = listb->next;
      }
      if (lista || listb)
         return 0;
      else
         return 1;
   } else
      return 0;
}

/* Return the Unicode Subset bitfield number for a Unicode character */

static int unicode_classify(wchar_t wc)
{
   int min = 0;
   int max = sizeof(utab) / sizeof(utab[0]) - 1;
   int mid;

   while (max >= min) {
      mid = (min + max) / 2;
      if (utab[mid].high < wc)
         min = mid + 1;
      else if (wc < utab[mid].low)
         max = mid - 1;
      else if (utab[mid].low <= wc && wc <= utab[mid].high)
         return utab[mid].bit;
      else
         return -1;
   }
   return -1;
}

void
gdk_wchar_text_handle(GdkFont * font,
                      const wchar_t * wcstr,
                      int wclen,
                      void (*handler) (GdkWin32SingleFont *,
                                       const wchar_t *,
                                       int, void *), void *arg)
{
   GdkFontPrivateWin32 *private;
   GdkWin32SingleFont *singlefont;
   GSList *list;
   int block;
   const wchar_t *start, *end, *wcp;

   wcp = wcstr;
   end = wcp + wclen;
   private = (GdkFontPrivateWin32 *) font;

   g_assert(private->base.ref_count > 0);

   GDK_NOTE(MISC, g_print("gdk_wchar_text_handle: "));

   while (wcp < end) {
      /* Split Unicode string into pieces of the same class */
      start = wcp;
      block = unicode_classify(*wcp);
      while (wcp + 1 < end && unicode_classify(wcp[1]) == block)
         wcp++;

      /* Find a font in the fontset that can handle this class */
      list = private->fonts;
      while (list) {
         singlefont = (GdkWin32SingleFont *) list->data;

         if (singlefont->fs.fsUsb[block / 32] & (1 << (block % 32)))
            break;

         list = list->next;
      }

      if (!list)
         singlefont = NULL;

      GDK_NOTE(MISC,
               g_print("%d:%d:%d ", start - wcstr, wcp - wcstr, block));

      /* Call the callback function */
      (*handler) (singlefont, start, wcp + 1 - start, arg);
      wcp++;
   }
   GDK_NOTE(MISC, g_print("\n"));
}

typedef struct {
   SIZE total;
} gdk_text_size_arg;

static void
gdk_text_size_handler(GdkWin32SingleFont * singlefont,
                      const wchar_t * wcstr, int wclen, void *argp)
{
   SIZE this_size;
   HGDIOBJ oldfont;
   gdk_text_size_arg *arg = (gdk_text_size_arg *) argp;

   if (!singlefont)
      return;

   if ((oldfont = SelectObject(gdk_DC, singlefont->xfont)) == NULL) {
      WIN32_GDI_FAILED("SelectObject");
      return;
   }
   GetTextExtentPoint32W(gdk_DC, wcstr, wclen, &this_size);
   SelectObject(gdk_DC, oldfont);

   arg->total.cx += this_size.cx;
   arg->total.cy = MAX(arg->total.cy, this_size.cy);
}

static gboolean
gdk_text_size(GdkFont * font,
              const gchar * text,
              gint text_length, gdk_text_size_arg * arg)
{
   gint wlen;
   wchar_t *wcstr;

   g_return_val_if_fail(font != NULL, FALSE);
   g_return_val_if_fail(text != NULL, FALSE);

   if (text_length == 0)
      return 0;

   g_assert(font->type == GDK_FONT_FONT || font->type == GDK_FONT_FONTSET);

   wcstr = g_new(wchar_t, text_length);
   if (text_length == 1) {
      /* For single characters, don't try to interpret as UTF-8.
       */
      wcstr[0] = (guchar) text[0];
      gdk_wchar_text_handle(font, wcstr, 1, gdk_text_size_handler, arg);
   } else {
      if ((wlen =
           gdk_nmbstowchar_ts(wcstr, text, text_length,
                              text_length)) == -1)
         g_warning("gdk_text_size: gdk_nmbstowchar_ts failed");
      else
         gdk_wchar_text_handle(font, wcstr, wlen, gdk_text_size_handler,
                               arg);
   }

   g_free(wcstr);

   return TRUE;
}

gint gdk_text_width(GdkFont * font, const gchar * text, gint text_length)
{
   gdk_text_size_arg arg;

   arg.total.cx = arg.total.cy = 0;

   if (!gdk_text_size(font, text, text_length, &arg))
      return -1;

   return arg.total.cx;
}

gint
gdk_text_width_wc(GdkFont * font, const GdkWChar * text, gint text_length)
{
   gdk_text_size_arg arg;
   wchar_t *wcstr;
   gint i;

   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(text != NULL, -1);

   if (text_length == 0)
      return 0;

   g_assert(font->type == GDK_FONT_FONT || font->type == GDK_FONT_FONTSET);

   if (sizeof(wchar_t) != sizeof(GdkWChar)) {
      wcstr = g_new(wchar_t, text_length);
      for (i = 0; i < text_length; i++)
         wcstr[i] = text[i];
   } else
      wcstr = (wchar_t *) text;

   arg.total.cx = arg.total.cy = 0;

   gdk_wchar_text_handle(font, wcstr, text_length,
                         gdk_text_size_handler, &arg);

   if (sizeof(wchar_t) != sizeof(GdkWChar))
      g_free(wcstr);

   return arg.total.cx;
}

void
gdk_text_extents(GdkFont * font,
                 const gchar * text,
                 gint text_length,
                 gint * lbearing,
                 gint * rbearing,
                 gint * width, gint * ascent, gint * descent)
{
   gdk_text_size_arg arg;
   gint wlen;
   wchar_t *wcstr;

   g_return_if_fail(font != NULL);
   g_return_if_fail(text != NULL);

   if (text_length == 0) {
      if (lbearing)
         *lbearing = 0;
      if (rbearing)
         *rbearing = 0;
      if (width)
         *width = 0;
      if (ascent)
         *ascent = 0;
      if (descent)
         *descent = 0;
      return;
   }

   g_assert(font->type == GDK_FONT_FONT || font->type == GDK_FONT_FONTSET);

   arg.total.cx = arg.total.cy = 0;

   wcstr = g_new(wchar_t, text_length);
   if (text_length == 1) {
      wcstr[0] = (guchar) text[0];
      gdk_wchar_text_handle(font, wcstr, 1, gdk_text_size_handler, &arg);
   } else {
      if ((wlen =
           gdk_nmbstowchar_ts(wcstr, text, text_length,
                              text_length)) == -1)
         g_warning("gdk_text_extents: gdk_nmbstowchar_ts failed");
      else
         gdk_wchar_text_handle(font, wcstr, wlen, gdk_text_size_handler,
                               &arg);
   }

   g_free(wcstr);

   /* XXX This is quite bogus */
   if (lbearing)
      *lbearing = 0;
   if (rbearing)
      *rbearing = arg.total.cx;
   /* What should be the difference between width and rbearing? */
   if (width)
      *width = arg.total.cx;
   if (ascent)
      *ascent = arg.total.cy + 1;
   if (descent)
      *descent = font->descent + 1;
}

void
gdk_text_extents_wc(GdkFont * font,
                    const GdkWChar * text,
                    gint text_length,
                    gint * lbearing,
                    gint * rbearing,
                    gint * width, gint * ascent, gint * descent)
{
   gdk_text_size_arg arg;
   wchar_t *wcstr;
   gint i;

   g_return_if_fail(font != NULL);
   g_return_if_fail(text != NULL);

   if (text_length == 0) {
      if (lbearing)
         *lbearing = 0;
      if (rbearing)
         *rbearing = 0;
      if (width)
         *width = 0;
      if (ascent)
         *ascent = 0;
      if (descent)
         *descent = 0;
      return;
   }

   g_assert(font->type == GDK_FONT_FONT || font->type == GDK_FONT_FONTSET);

   if (sizeof(wchar_t) != sizeof(GdkWChar)) {
      wcstr = g_new(wchar_t, text_length);
      for (i = 0; i < text_length; i++)
         wcstr[i] = text[i];
   } else
      wcstr = (wchar_t *) text;

   arg.total.cx = arg.total.cy = 0;

   gdk_wchar_text_handle(font, wcstr, text_length,
                         gdk_text_size_handler, &arg);

   if (sizeof(wchar_t) != sizeof(GdkWChar))
      g_free(wcstr);

   /* XXX This is quite bogus */
   if (lbearing)
      *lbearing = 0;
   if (rbearing)
      *rbearing = arg.total.cx;
   if (width)
      *width = arg.total.cx;
   if (ascent)
      *ascent = arg.total.cy + 1;
   if (descent)
      *descent = font->descent + 1;
}
